"""Cache the three physics maps (W_cnn, z*, R) per frame as float16 .npz files.

The 6-channel deepfake classifier (``forge_detect.baseline_cnn.build_physics_classifier``)
expects RGB + W_cnn + z* + R as 6 channels per image. Computing those maps
on-the-fly inside the training DataLoader is too slow — the PDE solver is
~1-5 s/frame even on the GPU cluster — so we materialise them once per
dataset and load from disk during training.

**Critical**: maps are stored as raw float16 numpy arrays, *not* as
viridis-coloured PNGs. Colormap quantisation introduces artefacts the CNN
will happily learn instead of the actual physics. Visualisation
(:func:`forge_detect.viz.panel`) is reserved for paper figures.

Layout (FF++)::

    <root>/original_sequences/youtube/<comp>/frames/<vid>/<f>.png
                              -> physics_heuristic/<vid>/<f>.npz
    <root>/manipulated_sequences/<method>/<comp>/frames/<vid>/<f>.png
                              -> physics_heuristic/<vid>/<f>.npz
                              -> physics_gtmask/<vid>/<f>.npz   (--variant gtmask)

Layout (Celeb-DF)::

    <root>/<subset>/frames/<vid>/<f>.png
                  -> physics_heuristic/<vid>/<f>.npz

Each .npz contains three float16 arrays of identical shape::

    wcnn      (H, W)     trust map in [0, 1]
    z_star    (H, W)     settled manifold (signed float)
    residual  (H, W)     z_star - z_ideal (signed float)

The two variants describe which trust map fed the PDE — they yield
different ``z_star`` and ``residual`` because W_cnn weights the
consistency term in the energy:

  - **heuristic** — chromatic-residual exp(-gain * ‖I - G_σ*I‖). Default;
    available for every frame; matches the inference-time distribution.
  - **gtmask** — 1 - binarise(GT_mask) for FF++ fakes with masks. Stronger
    train-time supervision; never used at inference (no ground truth).

Resume: each frame writes its own file, so re-running the script picks up
where the previous invocation left off — already-cached frames are
skipped before any I/O.

Usage:

    # Pass A: heuristic, FF++ all splits (this is what training reads)
    python scripts/cache_physics_maps.py \\
        --data-root /scratch/$USER/data/FaceForensics++ \\
        --dataset face-forensics \\
        --variant heuristic \\
        --image-size 256 --device cpu --num-workers 4

    # Pass B: GT-mask, FF++ TRAIN fakes only (for the gtmask experiment)
    python scripts/cache_physics_maps.py \\
        --data-root /scratch/$USER/data/FaceForensics++ \\
        --dataset face-forensics \\
        --variant gtmask --ff-split train \\
        --image-size 256 --device cpu --num-workers 4

    # CelebDF cross-dataset eval set (testing list only)
    python scripts/cache_physics_maps.py \\
        --data-root /scratch/$USER/data/Celeb-DF-v2 \\
        --dataset celeb-df --celeb-testing-list \\
        --variant heuristic \\
        --image-size 256 --device cpu --num-workers 4
"""

from __future__ import annotations

import argparse
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np

# Threading model:
#   - rayon (Rust) uses its default thread pool — all CPU cores. Each detect()
#     call internally parallelises the PDE / convolution kernels across cores.
#   - On top of that, a Python ThreadPoolExecutor runs N worker threads. The
#     Rust core releases the GIL via py.allow_threads(), and rayon's work-
#     stealing scheduler interleaves work from concurrent Python callers, so
#     a small thread pool (4-8) yields ~2-3x throughput by overlapping the
#     GIL-bound parts (PIL decode, numpy normalisation, npz compression+write)
#     with the rayon-bound parts.
#
# Empirical bench (M4 Pro 14C, 256x256, max_iter=200):
#   rayon=default + 1 worker  → 2.94 img/s (baseline)
#   rayon=default + 4 workers → 7.79 img/s (2.65x — sweet spot)
#   RAYON_NUM_THREADS=1 + 8 workers → 3.72 img/s (worse — all serialise on
#                                                  the single rayon worker)
#
# Setting RAYON_NUM_THREADS in the environment overrides the rayon default,
# but on a normal multi-core box you almost never want to.


def _build_dataset(args: argparse.Namespace) -> Any:
    """Build the dataset adapter; return raw frames (no physics-map loading)."""
    from forge_detect.datasets import CelebDFAdapter, FaceForensicsAdapter

    target_size = (args.image_size, args.image_size) if args.image_size else None
    if args.dataset == "celeb-df":
        return CelebDFAdapter(
            root=args.data_root,
            max_frames_per_video=args.frames_per_video,
            target_size=target_size,
            testing_list=args.celeb_testing_list,
        )
    return FaceForensicsAdapter(
        root=args.data_root,
        compression=args.compression,
        max_frames_per_video=args.frames_per_video,
        target_size=target_size,
        ff_split=args.ff_split,
    )


def _to_hwc(image_t: Any) -> np.ndarray:
    arr = image_t.detach().cpu().numpy() if hasattr(image_t, "detach") else image_t
    return np.transpose(arr, (1, 2, 0)).astype(np.float32, copy=False)


def _to_hw_mask(mask_t: Any | None) -> np.ndarray | None:
    if mask_t is None:
        return None
    arr = mask_t.detach().cpu().numpy() if hasattr(mask_t, "detach") else mask_t
    return arr.astype(np.float32, copy=False)


def _binarise_mask(mask: np.ndarray) -> np.ndarray:
    return (mask > 0.5).astype(np.float32)


def _trust_map(
    rgb: np.ndarray, mask: np.ndarray | None, label: int, variant: str,
) -> np.ndarray:
    """Build the trust map for the configured variant.

    For ``heuristic`` (always): chromatic-residual map.
    For ``gtmask`` on fakes: 1 - binarise(mask). On reals (no mask) or
    fakes whose mask is missing on disk, fall back to ``heuristic`` so the
    cache is still consistent with what the dataset adapter will read at
    train time.
    """
    from forge_detect.trust_map import heuristic_trust_map

    if variant == "gtmask" and label == 1 and mask is not None:
        return 1.0 - _binarise_mask(mask)
    return heuristic_trust_map(rgb)


def _save_npz(path: Path, wcnn: np.ndarray, z_star: np.ndarray, residual: np.ndarray) -> None:
    """Atomic-ish write: tmp file then rename, so a crash leaves no half-written .npz.

    Note: ``np.savez_compressed`` auto-appends ``.npz`` to a path argument that
    doesn't already end in it, which silently sabotages a tmp+rename pattern
    when the tmp name is e.g. ``foo.npz.tmp``. We pass a file-object instead
    so the suffix is honoured exactly as given.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as f:
        np.savez_compressed(
            f,
            wcnn=wcnn.astype(np.float16, copy=False),
            z_star=z_star.astype(np.float16, copy=False),
            residual=residual.astype(np.float16, copy=False),
        )
    tmp.replace(path)


def _process_index(
    idx: int,
    dataset: Any,
    args: argparse.Namespace,
    params: Any,
) -> int:
    """Load + math + save for a single dataset index; called from worker threads.

    The Rust core releases the GIL inside ``solve_and_extract``, so eight of
    these running concurrently *do* run in parallel — that's the whole point
    of the thread pool over here vs the previous single-thread DataLoader
    pipeline.
    """
    from forge_detect.datasets import physics_npz_path
    from forge_detect.pipeline import detect

    rec = dataset._records[idx]
    item = dataset[idx]
    if len(item) == 3:
        image_t, label, mask_t = item
    else:
        image_t, label = item[0], item[1]
        mask_t = None

    rgb = _to_hwc(image_t)
    mask = _to_hw_mask(mask_t)
    w = _trust_map(rgb, mask, int(label), args.variant)
    result = detect(rgb, params=params, trust_map=w, device=args.device)
    npz = physics_npz_path(rec.image_path, args.variant)
    _save_npz(
        npz,
        wcnn=w,
        z_star=result.solve.z_star,
        residual=result.solve.residual,
    )
    return idx


def _cache_one_dataset(args: argparse.Namespace) -> dict[str, int]:
    # Eagerly import torch in the main thread so the module-init machinery
    # doesn't race when worker threads first touch torch.from_numpy().
    import torch  # noqa: F401

    from forge_detect.config import PdeParams, PipelineParams
    from forge_detect.datasets import physics_npz_path

    dataset = _build_dataset(args)
    n_total = len(dataset)
    print(f"[cache] dataset has {n_total} frames")

    fakes_with_mask = 0
    fakes_total = 0
    for rec in dataset._records:
        if rec.label == 1:
            fakes_total += 1
            if getattr(rec, "mask_path", None) is not None:
                fakes_with_mask += 1
    coverage = fakes_with_mask / max(1, fakes_total)
    if args.variant == "gtmask":
        print(
            f"[cache] gtmask variant: {fakes_with_mask}/{fakes_total} = {coverage:.1%} "
            "of fake records carry a mask path",
        )
        if coverage == 0 and fakes_total > 0:
            print(
                "[cache] WARNING: no masks found for any fake; --variant gtmask is "
                "indistinguishable from heuristic. Check --include-masks on extraction.",
            )

    indices_to_do: list[int] = []
    indices_skip = 0
    for i, rec in enumerate(dataset._records):
        npz = physics_npz_path(rec.image_path, args.variant)
        if npz.exists():
            indices_skip += 1
            continue
        indices_to_do.append(i)

    print(f"[cache] {indices_skip} frames already cached, {len(indices_to_do)} new to process")
    print(
        f"[cache] threading: {args.num_workers} python worker threads x "
        f"rayon (RAYON_NUM_THREADS={os.environ.get('RAYON_NUM_THREADS', 'default=all-cores')})",
    )
    if not indices_to_do:
        return {"processed": 0, "failed": 0, "skipped": indices_skip, "total": n_total}

    params = PipelineParams(
        n_scales=args.n_scales,
        pde=PdeParams(max_iter=args.max_iter, log_every=0),
    )

    t0 = time.time()
    n_done = 0
    n_failed = 0
    with ThreadPoolExecutor(max_workers=args.num_workers) as ex:
        futures = {
            ex.submit(_process_index, i, dataset, args, params): i for i in indices_to_do
        }
        for fut in as_completed(futures):
            idx = futures[fut]
            try:
                fut.result()
                n_done += 1
            except Exception as e:
                n_failed += 1
                rec = dataset._records[idx]
                print(
                    f"  ERROR on index {idx} ({rec.image_path}): "
                    f"{type(e).__name__}: {e}",
                )

            seen = n_done + n_failed
            if args.log_every and seen % args.log_every == 0:
                elapsed = time.time() - t0
                rate = seen / max(elapsed, 1.0e-6)
                eta = (len(indices_to_do) - seen) / max(rate, 1.0e-6)
                fail_str = f" ({n_failed} failed)" if n_failed else ""
                print(
                    f"  [{seen}/{len(indices_to_do)}] {rate:.2f} img/s "
                    f"ETA {eta:.0f}s{fail_str}",
                )

    return {
        "processed": n_done,
        "failed": n_failed,
        "skipped": indices_skip,
        "total": n_total,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument(
        "--dataset",
        choices=("face-forensics", "celeb-df"),
        default="face-forensics",
    )
    parser.add_argument("--compression", choices=("raw", "c23", "c40"), default="c23")
    parser.add_argument(
        "--ff-split",
        choices=("train", "val", "test"),
        default=None,
        help="If set, restrict FF++ to this official video-disjoint split. "
        "For --variant gtmask, omit this to cache GT-mask physics for ALL "
        "FF++ splits' fakes (recommended for the FF++-only protocol where "
        "GT masks are equally available at eval time as at train time).",
    )
    parser.add_argument(
        "--celeb-testing-list",
        action="store_true",
        help="For --dataset celeb-df: restrict to the published 518-video benchmark.",
    )
    parser.add_argument(
        "--variant",
        choices=("heuristic", "gtmask"),
        default="heuristic",
        help="Trust-map source for the PDE. heuristic = chromatic-residual "
        "(every frame); gtmask = 1 - binarise(GT mask) for FF++ fakes (falls "
        "back to heuristic on reals / mask-less frames).",
    )
    parser.add_argument(
        "--frames-per-video",
        type=int,
        default=None,
        help="Cap frames per video. Default: cache all frames.",
    )
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--n-scales", type=int, default=3)
    parser.add_argument("--max-iter", type=int, default=200)
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of Python worker threads running detect() concurrently. "
        "Combined with rayon's default multi-threaded kernels, this is the "
        "sweet spot for overlapping GIL-bound work (PIL decode, npz save) "
        "with the Rust math. Empirically ~2.65x over a single thread; "
        "going higher gives diminishing returns and risks oversubscription.",
    )
    parser.add_argument("--log-every", type=int, default=25)
    args = parser.parse_args()

    if args.dataset == "celeb-df" and args.variant == "gtmask":
        parser.error(
            "Celeb-DF has no GT manipulation masks; --variant gtmask is not "
            "applicable for that dataset.",
        )

    print(f"[cache] data_root = {args.data_root}")
    print(f"[cache] variant = {args.variant}")
    print(f"[cache] image_size = {args.image_size}")
    print(f"[cache] device = {args.device}")

    t0 = time.time()
    summary = _cache_one_dataset(args)
    elapsed = time.time() - t0

    print()
    print("=== cache summary ===")
    print(f"  processed: {summary['processed']}")
    print(f"  failed: {summary.get('failed', 0)}")
    print(f"  skipped (already cached): {summary['skipped']}")
    print(f"  total in dataset: {summary['total']}")
    print(f"  elapsed: {elapsed:.0f}s")
    if summary["processed"] > 0:
        print(f"  rate: {summary['processed'] / max(elapsed, 1):.2f} img/s")
    return 0 if summary.get("failed", 0) == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
