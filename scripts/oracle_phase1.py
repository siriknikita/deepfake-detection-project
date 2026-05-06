"""Oracle Phase-1: feature extraction with ground-truth manipulation masks.

The Phase-1 study (`scripts/quick_classifier.py`) uses the deterministic
*heuristic* trust map of `forge_detect.trust_map.heuristic_trust_map`.
That heuristic is a chromatic-residual function of the input image; it
has no notion of "this region is manipulated", so the energy functional
cannot concentrate residual where the manipulation actually lives. The
empirical Phase-1 result on FF++ c23 is therefore inconclusive about
whether the *mathematical core* of Hyperplane-Forge can separate real
from fake: failure could be the heuristic's localisation, the math
itself, or both.

This script disambiguates by replacing the heuristic with an *oracle*
trust map derived directly from the FF++ ground-truth manipulation
masks:

  - For real frames (no mask file present), W_cnn = 1 everywhere.
  - For fake frames, W_cnn = 1 - binarize(mask): trust drops to 0
    inside the manipulated region and stays at 1 outside.

If the resulting features cleanly separate real from fake, the math
pipeline works given a properly localised trust map and the diploma's
load-bearing claim survives — the Phase-2 ChromaticEfficientNet then
becomes the engineering question of approximating the oracle. If the
features still fail to separate, the framework's mathematical core does
not discriminate at FF++ c23 and Phase 2 cannot save it.

Usage:

    python scripts/oracle_phase1.py \\
        --data-root ~/data/FaceForensics++ \\
        --runs-dir ~/runs/oracle_phase1 \\
        --device cuda

The script mirrors `quick_classifier.py`'s split / training / evaluation
flow, so the resulting AUROC is directly comparable. Output schema is
identical: features-{train,val,test}.csv, classifier.pkl, report.json.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np


def _build_dataset(
    args: argparse.Namespace,
    *,
    subset_video_ids: set[str] | None,
    frames_per_video: int | None,
) -> Any:
    from forge_detect.datasets import FaceForensicsAdapter

    target_size = (args.image_size, args.image_size) if args.image_size else None
    return FaceForensicsAdapter(
        root=args.data_root,
        compression=args.compression,
        max_frames_per_video=frames_per_video,
        target_size=target_size,
        subset_video_ids=subset_video_ids,
    )


def _all_video_ids(args: argparse.Namespace) -> list[str]:
    full = _build_dataset(args, subset_video_ids=None, frames_per_video=1)
    return full.video_ids()


def _to_hwc(image_tensor: Any) -> np.ndarray:
    """Adapter returns (3, H, W) torch tensor in [0, 1]; pipeline.detect
    wants (H, W, 3) numpy."""
    arr = image_tensor.detach().cpu().numpy() if hasattr(image_tensor, "detach") else image_tensor
    return np.transpose(arr, (1, 2, 0)).astype(np.float32, copy=False)


def _to_hw_mask(mask_tensor: Any | None) -> np.ndarray | None:
    if mask_tensor is None:
        return None
    arr = mask_tensor.detach().cpu().numpy() if hasattr(mask_tensor, "detach") else mask_tensor
    return arr.astype(np.float32, copy=False)


def _oracle_trust_map(rgb_hwc: np.ndarray, mask_hw: np.ndarray | None) -> np.ndarray:
    """W_cnn = 1 everywhere for reals; W_cnn = 1 - binarize(mask) for fakes."""
    h, w = rgb_hwc.shape[:2]
    if mask_hw is None:
        return np.ones((h, w), dtype=np.float32)
    if mask_hw.shape != (h, w):
        msg = f"mask shape {mask_hw.shape} must match image H×W {(h, w)}"
        raise ValueError(msg)
    binarised = (mask_hw > 0.5).astype(np.float32)
    return 1.0 - binarised


class _MetadataDataset:
    """Wraps a FaceForensicsAdapter slice so each item also carries its
    ``video_id`` and ``image_path`` (the adapter's `__getitem__` returns
    only ``(image, label, mask)``). The wrapper plus an explicit index
    list is also our resume mechanism — skipping done records *before*
    the DataLoader sees them avoids paying their I/O cost.
    """

    def __init__(self, base: Any, indices: list[int]) -> None:
        self.base = base
        self.indices = indices

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> tuple[Any, int, Any | None, str, str]:
        real_idx = self.indices[idx]
        item = self.base[real_idx]
        rec = self.base._records[real_idx]
        if len(item) == 3:
            image, label, mask = item
        else:
            image, label = item[0], item[1]
            mask = None
        return image, int(label), mask, str(rec.video_id), str(rec.image_path)


def _collate_passthrough(batch: list[Any]) -> Any:
    """batch_size=1; the math kernels work one frame at a time, so we
    just unwrap rather than stacking. Default collate would crash on
    ``mask=None`` records (real frames) anyway."""
    return batch[0]


def _extract_oracle_features(
    dataset: Any,
    *,
    device: str,
    params: Any,
    cache_path: Path,
    log_every: int = 25,
    save_every: int = 100,
    num_workers: int = 4,
) -> Any:
    """Feature extraction with the oracle trust map, parallelised at the
    I/O layer.

    A ``DataLoader`` with ``num_workers`` worker processes overlaps PIL
    decode, resize, and tensorisation with the GPU's PDE work in the main
    thread. Frames are still processed one at a time on the GPU because
    ``pipeline.detect`` is not batched; the speedup comes from hiding
    image-loading latency behind compute. Crash-resumable: any prior
    rows already present in ``cache_path`` are skipped at the index
    level (workers don't re-decode them).
    """
    import pandas as pd
    import torch
    from torch.utils.data import DataLoader

    from forge_detect.eval import FeatureMatrix
    from forge_detect.features import FEATURE_NAMES, extract_features
    from forge_detect.pipeline import detect

    feats: list[np.ndarray] = []
    labels: list[int] = []
    paths: list[str] = []
    video_ids: list[str] = []
    done_paths: set[str] = set()

    if cache_path.exists():
        prior = pd.read_csv(cache_path)
        feats = list(prior[list(FEATURE_NAMES)].to_numpy(dtype=np.float64))
        labels = prior["label"].tolist()
        paths = prior["path"].tolist()
        if "video_id" in prior.columns:
            video_ids = prior["video_id"].astype(str).tolist()
        done_paths = set(paths)
        print(f"[oracle] resuming from {cache_path} ({len(done_paths)} done)")

    remaining_indices = [
        i
        for i, rec in enumerate(dataset._records)
        if str(rec.image_path) not in done_paths
    ]
    if not remaining_indices:
        print(f"[oracle] {cache_path.name} fully cached, skipping extraction")
        return FeatureMatrix(
            features=(
                np.stack(feats)
                if feats
                else np.empty((0, len(FEATURE_NAMES)), dtype=np.float64)
            ),
            labels=np.asarray(labels, dtype=np.int64),
            paths=paths,
            video_ids=video_ids,
        )

    n_total = len(remaining_indices) + len(done_paths)
    print(
        f"[oracle] {cache_path.name}: {len(remaining_indices)} new, "
        f"{len(done_paths)} cached, num_workers={num_workers}",
    )

    loader_kwargs: dict[str, Any] = {
        "batch_size": 1,
        "shuffle": False,
        "num_workers": num_workers,
        "collate_fn": _collate_passthrough,
        "pin_memory": False,
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True
    loader = DataLoader(_MetadataDataset(dataset, remaining_indices), **loader_kwargs)

    t0 = time.time()
    new_count = 0
    n_processed = len(done_paths)
    for image_t, label, mask_t, video_id, image_path in loader:
        rgb = _to_hwc(image_t)
        mask = _to_hw_mask(mask_t)
        w_cnn = _oracle_trust_map(rgb, mask)
        result = detect(rgb, params=params, trust_map=w_cnn, device=device)
        f = extract_features(result.solve)
        feats.append(f)
        labels.append(label)
        paths.append(image_path)
        video_ids.append(video_id)
        new_count += 1
        n_processed += 1

        if log_every and new_count % log_every == 0:
            elapsed = time.time() - t0
            rate = new_count / max(elapsed, 1e-6)
            eta = (n_total - n_processed) / max(rate, 1e-6)
            print(f"  [{n_processed}/{n_total}] {rate:.2f} images/s, ETA {eta:.0f}s")

        if save_every and new_count % save_every == 0:
            FeatureMatrix(
                features=np.stack(feats),
                labels=np.asarray(labels, dtype=np.int64),
                paths=paths,
                video_ids=video_ids,
            ).save(cache_path)

    # torch may keep the workers alive between dataloaders; release them.
    del loader
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    fm = FeatureMatrix(
        features=np.stack(feats)
        if feats
        else np.empty((0, len(FEATURE_NAMES)), dtype=np.float64),
        labels=np.asarray(labels, dtype=np.int64),
        paths=paths,
        video_ids=video_ids,
    )
    fm.save(cache_path)
    return fm


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--compression", choices=("raw", "c23", "c40"), default="c23")
    parser.add_argument("--frames-per-video-train", type=int, default=30)
    parser.add_argument("--frames-per-video-eval", type=int, default=10)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--n-scales", type=int, default=3)
    parser.add_argument("--max-iter", type=int, default=200)
    parser.add_argument("--runs-dir", type=Path, default=Path("runs/oracle_phase1"))
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--test-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="DataLoader worker processes for parallel image loading. "
        "0 = synchronous (no overlap). Default 4 typically gives 2–3× "
        "speedup over synchronous on the 256×256 + PDE workload.",
    )
    args = parser.parse_args()

    args.runs_dir.mkdir(parents=True, exist_ok=True)
    print(f"[oracle] runs_dir = {args.runs_dir}")
    print(f"[oracle] image-size = {args.image_size}")

    from forge_detect.classifier import (
        evaluate_classifier,
        save_classifier,
        train_classifier,
    )
    from forge_detect.config import PdeParams, PipelineParams
    from forge_detect.datasets import split_videos

    print("[oracle] enumerating videos for the disjoint split ...")
    all_video_ids = _all_video_ids(args)
    train_vids, val_vids, test_vids = split_videos(
        all_video_ids,
        seed=args.seed,
        val_fraction=args.val_fraction,
        test_fraction=args.test_fraction,
    )
    print(
        f"[oracle] split: train={len(train_vids)} val={len(val_vids)} test={len(test_vids)} "
        f"(of {len(all_video_ids)} total videos)",
    )

    train_ds = _build_dataset(
        args, subset_video_ids=train_vids, frames_per_video=args.frames_per_video_train,
    )
    val_ds = _build_dataset(
        args, subset_video_ids=val_vids, frames_per_video=args.frames_per_video_eval,
    )
    test_ds = _build_dataset(
        args, subset_video_ids=test_vids, frames_per_video=args.frames_per_video_eval,
    )
    print(f"[oracle] dataset sizes: train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}")

    # Sanity check: at least some fake records should carry a mask path —
    # otherwise the oracle is just an all-ones map and we're wasting time.
    n_with_mask = sum(1 for r in train_ds._records if r.mask_path is not None)
    n_fake_total = sum(1 for r in train_ds._records if r.label == 1)
    if n_fake_total == 0:
        print("[oracle] ERROR: no fake training rows; aborting.")
        return 2
    coverage = n_with_mask / n_fake_total
    print(f"[oracle] mask coverage on train fakes: {n_with_mask}/{n_fake_total} = {coverage:.1%}")
    if coverage < 0.5:
        print(
            "[oracle] WARNING: less than 50% of fake training frames have a mask file. "
            "Did you run extract_frames.py with --include-masks for this compression level?",
        )

    params = PipelineParams(
        n_scales=args.n_scales,
        pde=PdeParams(max_iter=args.max_iter, log_every=20),
    )

    t0 = time.time()
    print("[oracle] extracting features (val) ...")
    fm_val = _extract_oracle_features(
        val_ds,
        device=args.device,
        params=params,
        cache_path=args.runs_dir / "features-val.csv",
        num_workers=args.num_workers,
    )
    print("[oracle] extracting features (test) ...")
    fm_test = _extract_oracle_features(
        test_ds,
        device=args.device,
        params=params,
        cache_path=args.runs_dir / "features-test.csv",
        num_workers=args.num_workers,
    )
    print("[oracle] extracting features (train) ...")
    fm_train = _extract_oracle_features(
        train_ds,
        device=args.device,
        params=params,
        cache_path=args.runs_dir / "features-train.csv",
        num_workers=args.num_workers,
    )

    train_size = fm_train.features.shape[0]
    print(f"[oracle] training classifier on {train_size} frames ...")
    classifier = train_classifier(fm_train.features, fm_train.labels)
    save_classifier(classifier, args.runs_dir / "classifier.pkl")

    val_m = evaluate_classifier(
        classifier, fm_val.features, fm_val.labels, video_ids=fm_val.video_ids,
    )
    test_m = evaluate_classifier(
        classifier, fm_test.features, fm_test.labels, video_ids=fm_test.video_ids,
    )
    elapsed = time.time() - t0

    report = {
        "phase": "oracle phase-1 (ground-truth-mask trust map)",
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "elapsed_seconds": elapsed,
        "n_videos": {"train": len(train_vids), "val": len(val_vids), "test": len(test_vids)},
        "n_frames": {
            "train": train_size,
            "val": fm_val.features.shape[0],
            "test": fm_test.features.shape[0],
        },
        "mask_coverage_train_fakes": coverage,
        "val": {
            "frame_auroc": val_m.auroc,
            "frame_accuracy": val_m.accuracy,
            "video_auroc_mean": val_m.video_auroc_mean,
            "video_auroc_max": val_m.video_auroc_max,
            "n_videos": val_m.n_videos,
        },
        "test": {
            "frame_auroc": test_m.auroc,
            "frame_accuracy": test_m.accuracy,
            "video_auroc_mean": test_m.video_auroc_mean,
            "video_auroc_max": test_m.video_auroc_max,
            "n_videos": test_m.n_videos,
            "top_features": dict(
                sorted(test_m.feature_importances.items(), key=lambda kv: -kv[1])[:10],
            ),
        },
    }
    (args.runs_dir / "report.json").write_text(json.dumps(report, indent=2, default=str))

    print("\n=== Oracle Phase-1 result ===")
    print(f"  Frame AUROC  val={val_m.auroc:.4f}  test={test_m.auroc:.4f}")
    if not math.isnan(test_m.video_auroc_mean):
        print(
            f"  Video AUROC  val={val_m.video_auroc_mean:.4f}  "
            f"test={test_m.video_auroc_mean:.4f}  (mean-pool)",
        )
    print(f"  Mask coverage on train fakes: {coverage:.1%}")
    print(f"  Total time: {elapsed:.0f}s")
    print()
    print("Interpretation rubric:")
    print("  Oracle video AUROC ≥ 0.85  -> math is sound; heuristic was the bottleneck.")
    print("                                Phase 2 target = approximate the oracle with a CNN.")
    print("  Oracle video AUROC ~ 0.70  -> math has moderate signal under perfect localisation;")
    print("                                Phase 2 ceiling is bounded; CNN may help but won't win big.")
    print("  Oracle video AUROC ≤ 0.55  -> framework's load-bearing claim does not hold at this")
    print("                                compression. Phase 2 cannot save it; report honestly.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
