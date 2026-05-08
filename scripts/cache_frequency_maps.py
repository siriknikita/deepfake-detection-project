"""Cache the three Phase-3 frequency maps per frame as float16 .npz files.

The 9-channel deepfake classifier (RGB + physics + frequency) reads its
spectral channels from this cache. Computing them on the fly inside the
training DataLoader would be cheap (~5 ms per 256² face) but caching
keeps train-time I/O uniform with the physics-map cache and lets the
``--channels`` flag in ``train_physics_cnn.py`` swap channel sets without
reshaping the dataloader.

Layout (FF++)::

    <root>/original_sequences/youtube/<comp>/frames/<vid>/<f>.png
                              -> frequency_default/<vid>/<f>.npz
    <root>/manipulated_sequences/<method>/<comp>/frames/<vid>/<f>.png
                              -> frequency_default/<vid>/<f>.npz

    <root>/original_sequences/youtube/<comp>/frames_faces/<vid>/<f>.png
                              -> frequency_faces_default/<vid>/<f>.npz

Layout (Celeb-DF)::

    <root>/<subset>/frames/<vid>/<f>.png
                  -> frequency_default/<vid>/<f>.npz

Each .npz contains three float16 arrays of identical shape::

    dct_block_energy    (H, W)     log of total AC energy per 8×8 block, tiled
    dct_high_ratio      (H, W)     upper-half AC energy ratio per block, tiled
    fft_radial_logmag   (H, W)     log-mag of the full-image 2D FFT, fftshifted

Resume: each frame writes its own file, so re-running the script picks
up where the previous invocation left off — already-cached frames are
skipped before any compute.

Usage:

    # FF++ face-cropped, all splits (this is what training reads):
    python scripts/cache_frequency_maps.py \\
        --data-root /scratch/$USER/data/FaceForensics++ \\
        --dataset face-forensics \\
        --frames-subdir frames_faces \\
        --image-size 256 --num-workers 4

    # Celeb-DF cross-dataset eval set (testing list only):
    python scripts/cache_frequency_maps.py \\
        --data-root /scratch/$USER/data/Celeb-DF-v2 \\
        --dataset celeb-df --celeb-testing-list \\
        --frames-subdir frames_faces \\
        --image-size 256 --num-workers 4
"""

from __future__ import annotations

import argparse
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np

# Threading: the per-frame work is pure NumPy under the GIL. The thread
# pool is still useful because PIL decode + npz compression release the
# GIL, so a small number of workers (4) overlaps decode/encode with the
# DCT/FFT compute. With more workers the GIL becomes the bottleneck —
# don't oversubscribe.


def _build_dataset(args: argparse.Namespace) -> Any:
    """Build the dataset adapter; return raw frames (no extra-channel loading)."""
    from forge_detect.datasets import CelebDFAdapter, FaceForensicsAdapter

    target_size = (args.image_size, args.image_size) if args.image_size else None
    if args.dataset == "celeb-df":
        return CelebDFAdapter(
            root=args.data_root,
            max_frames_per_video=args.frames_per_video,
            target_size=target_size,
            testing_list=args.celeb_testing_list,
            frames_subdir=args.frames_subdir,
        )
    return FaceForensicsAdapter(
        root=args.data_root,
        compression=args.compression,
        max_frames_per_video=args.frames_per_video,
        target_size=target_size,
        ff_split=args.ff_split,
        frames_subdir=args.frames_subdir,
    )


def _to_hwc(image_t: Any) -> np.ndarray:
    arr = image_t.detach().cpu().numpy() if hasattr(image_t, "detach") else image_t
    return np.transpose(arr, (1, 2, 0)).astype(np.float32, copy=False)


def _save_npz(
    path: Path,
    *,
    dct_block_energy: np.ndarray,
    dct_high_ratio: np.ndarray,
    fft_radial_logmag: np.ndarray,
) -> None:
    """Atomic-ish write: tmp file then rename, so a crash leaves no half-written .npz.

    Mirrors ``cache_physics_maps._save_npz``: the suffix-quirk of
    ``np.savez_compressed`` is sidestepped by passing a file object so the
    tmp path keeps its ``.tmp`` suffix.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as f:
        np.savez_compressed(
            f,
            dct_block_energy=dct_block_energy.astype(np.float16, copy=False),
            dct_high_ratio=dct_high_ratio.astype(np.float16, copy=False),
            fft_radial_logmag=fft_radial_logmag.astype(np.float16, copy=False),
        )
    tmp.replace(path)


def _process_index(
    idx: int,
    dataset: Any,
    args: argparse.Namespace,
) -> int:
    """Load one image, compute the three frequency maps, write to disk."""
    from forge_detect.datasets import frequency_npz_path
    from forge_detect.frequency_map import frequency_maps

    rec = dataset._records[idx]
    item = dataset[idx]
    image_t = item[0]
    rgb = _to_hwc(image_t)
    dct_e, dct_r, fft_m = frequency_maps(rgb)
    npz = frequency_npz_path(rec.image_path, args.variant)
    _save_npz(
        npz,
        dct_block_energy=dct_e,
        dct_high_ratio=dct_r,
        fft_radial_logmag=fft_m,
    )
    return idx


def _cache_one_dataset(args: argparse.Namespace) -> dict[str, int]:
    from forge_detect.datasets import frequency_npz_path

    dataset = _build_dataset(args)
    n_total = len(dataset)
    print(f"[cache-freq] dataset has {n_total} frames")

    indices_to_do: list[int] = []
    indices_skip = 0
    for i, rec in enumerate(dataset._records):
        npz = frequency_npz_path(rec.image_path, args.variant)
        if npz.exists():
            indices_skip += 1
            continue
        indices_to_do.append(i)

    print(
        f"[cache-freq] {indices_skip} frames already cached, "
        f"{len(indices_to_do)} new to process",
    )
    if not indices_to_do:
        return {"processed": 0, "failed": 0, "skipped": indices_skip, "total": n_total}

    t0 = time.time()
    n_done = 0
    n_failed = 0
    with ThreadPoolExecutor(max_workers=args.num_workers) as ex:
        futures = {
            ex.submit(_process_index, i, dataset, args): i for i in indices_to_do
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
        help="If set, restrict FF++ to this official video-disjoint split.",
    )
    parser.add_argument(
        "--celeb-testing-list",
        action="store_true",
        help="For --dataset celeb-df: restrict to the published 518-video benchmark.",
    )
    parser.add_argument(
        "--variant",
        choices=("default",),
        default="default",
        help="Frequency-map variant. Currently only 'default' (block-DCT + "
        "FFT recipe described in forge_detect.frequency_map).",
    )
    parser.add_argument(
        "--frames-per-video",
        type=int,
        default=None,
        help="Cap frames per video. Default: cache all frames.",
    )
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Python worker threads. The per-frame work is pure-Python "
        "DCT + FFT; PIL decode + npz compression release the GIL, so a "
        "small pool overlaps decode/encode with the math. Going higher "
        "than 4-8 saturates the GIL.",
    )
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument(
        "--frames-subdir",
        type=str,
        default="frames",
        help="Subdirectory holding the frames to read. Default 'frames'; "
        "pass 'frames_faces' to cache frequency maps for the face-cropped "
        "tree. The output cache directory name encodes this so caches "
        "from different sources don't collide: "
        "frames -> frequency_<variant>; frames_faces -> "
        "frequency_faces_<variant>.",
    )
    args = parser.parse_args()

    print(f"[cache-freq] data_root = {args.data_root}")
    print(f"[cache-freq] variant = {args.variant}")
    print(f"[cache-freq] image_size = {args.image_size}")

    t0 = time.time()
    summary = _cache_one_dataset(args)
    elapsed = time.time() - t0

    print()
    print("=== cache-freq summary ===")
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
