"""Phase-1 AUROC: heuristic W_cnn + Gradient Boosting classifier.

The fastest defendable number you can put in the diploma. Skips the
ChromaticEfficientNet training entirely — uses the deterministic
chromatic-residual trust map for every image and trains the binary
classifier on the resulting impact-map features. Total cluster time on
a single 4090 with default settings is ~2 hours on a 30-frame-per-video
FF++ slice, vs ~25 hours for the full pivot study.

Methodology (matches the published cross-dataset protocol):
- **Video-disjoint splits.** Train/val/test are split by *video id*,
  not by frame index, so adjacent frames from the same video can never
  leak across splits.
- **Asymmetric frame caps.** Training uses ``--frames-per-video-train``
  (default 30) for variety; eval uses ``--frames-per-video-eval``
  (default 10) to match the canonical academic protocol.
- **Video-level AUROC** is reported alongside frame-level. Mean-pool
  per video is the canonical metric; max-pool is a robustness check.

What you get:
- runs_dir/features-train.csv, features-val.csv, features-test.csv
- runs_dir/classifier.pkl                                 trained pipeline
- runs_dir/report.json                                    AUROCs + top features

Workflow:

    # Fast cluster run, defendable number in ~2 hours:
    python scripts/quick_classifier.py \\
        --data-root /scratch/$USER/data/FaceForensics++ \\
        --runs-dir runs/quick_phase1 \\
        --device cuda

    # Cross-dataset evaluation against Celeb-DF v2's official 518-video set:
    python scripts/quick_classifier.py \\
        --data-root /scratch/$USER/data/Celeb-DF-v2 \\
        --dataset celeb-df --celeb-testing-list \\
        --classifier-checkpoint runs/quick_phase1/classifier.pkl \\
        --runs-dir runs/quick_phase1_celebdf --device cuda

    # Wrap in continue.sh so a power outage does not kill the run:
    ./scripts/continue.sh -- python scripts/quick_classifier.py \\
        --data-root /scratch/$USER/data/FaceForensics++ \\
        --runs-dir runs/quick_phase1 --device cuda

The output is the *Phase-1 baseline* against which the full pivot study
later compares. If the heuristic-classifier AUROC is already strong, the
ChromaticEfficientNet upgrade may not be worth its training time; if
it is weak, that itself is informative ("the math pipeline alone is
not separating real from fake on this data").

Resume on crash:
- Each features-*.csv accumulates incrementally — restarts skip already-
  processed images.
- classifier.pkl is regenerated from features.csv on every run, so a
  classifier-stage crash just reruns sklearn (~seconds).
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any


def _build_dataset(
    args: argparse.Namespace,
    *,
    subset_video_ids: set[str] | None,
    frames_per_video: int | None,
) -> Any:
    """Construct the configured dataset adapter restricted to a video subset.

    Three flavors are supported via ``--dataset``:
      - ``face-forensics`` (default): canonical FF++ tree.
      - ``celeb-df``: Celeb-DF v1 / v2 — used for cross-dataset eval.
      - ``image-folder``: generic ``(real_dir, fake_dir)`` pairs.
    """
    from forge_detect.datasets import (
        CelebDFAdapter,
        FaceForensicsAdapter,
        ImageFolderDataset,
    )

    target_size = (args.image_size, args.image_size) if args.image_size else None
    if args.dataset == "image-folder":
        if not args.fake_dir:
            msg = "--fake-dir is required for --dataset=image-folder"
            raise SystemExit(msg)
        # subset_video_ids is unused for the bare image-folder case — the
        # adapter does not have a video concept beyond the parent directory.
        return ImageFolderDataset(
            real_dir=args.data_root,
            fake_dir=args.fake_dir,
            target_size=target_size,
        )
    if args.dataset == "celeb-df":
        return CelebDFAdapter(
            root=args.data_root,
            max_frames_per_video=frames_per_video,
            target_size=target_size,
            testing_list=args.celeb_testing_list,
            subset_video_ids=subset_video_ids,
        )
    return FaceForensicsAdapter(
        root=args.data_root,
        compression=args.compression,
        max_frames_per_video=frames_per_video,
        target_size=target_size,
        subset_video_ids=subset_video_ids,
    )


def _all_video_ids(args: argparse.Namespace) -> list[str]:
    """Build the dataset once with no frame cap to enumerate every video id."""
    full = _build_dataset(args, subset_video_ids=None, frames_per_video=1)
    if hasattr(full, "video_ids"):
        return full.video_ids()
    # ImageFolderDataset has video_id but no aggregator method — derive it.
    records = getattr(full, "_records", [])
    return sorted({rec.video_id for rec in records})


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    # ---- dataset ----
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument(
        "--dataset",
        choices=("face-forensics", "celeb-df", "image-folder"),
        default="face-forensics",
    )
    parser.add_argument("--fake-dir", type=Path, default=None)
    parser.add_argument("--compression", choices=("raw", "c23", "c40"), default="c23")
    parser.add_argument(
        "--celeb-testing-list",
        action="store_true",
        help=(
            "For --dataset=celeb-df: restrict to videos listed in "
            "List_of_testing_videos.txt (the published 518-video benchmark)."
        ),
    )
    parser.add_argument(
        "--frames-per-video-train",
        type=int,
        default=30,
        help=(
            "Frame cap per training video (default 30). More frames = more "
            "pose / lighting variety for the classifier to fit on."
        ),
    )
    parser.add_argument(
        "--frames-per-video-eval",
        type=int,
        default=10,
        help=(
            "Frame cap per val/test video (default 10). Matches the "
            "academic cross-dataset protocol for video-level AUROC."
        ),
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=256,
        help="Resize every image to this side (default 256, the FF++ standard).",
    )
    # ---- pipeline ----
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda"),
        default="cpu",
        help="Backend for the math kernels (default cpu — works everywhere).",
    )
    parser.add_argument(
        "--n-scales",
        type=int,
        default=3,
        help="DoG scales (default 3 — slightly faster than the full 4).",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=200,
        help="Jacobi iteration cap (default 200).",
    )
    # ---- output / split ----
    parser.add_argument("--runs-dir", type=Path, default=Path("runs/quick_phase1"))
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=0.15,
        help="Fraction of *videos* held out for validation (default 0.15).",
    )
    parser.add_argument(
        "--test-fraction",
        type=float,
        default=0.15,
        help="Fraction of *videos* held out for testing (default 0.15).",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--classifier-checkpoint",
        type=Path,
        default=None,
        help=(
            "Skip training; load this pre-trained classifier and only "
            "extract features + evaluate on val+test. Useful for cross-"
            "dataset eval against Celeb-DF after FF++ training."
        ),
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help=(
            "DataLoader worker processes for parallel image decoding. "
            "0 (default) preserves the synchronous behaviour. Set to 4 "
            "for ~2–3× speedup on the 256×256 + PDE workload."
        ),
    )
    args = parser.parse_args()

    args.runs_dir.mkdir(parents=True, exist_ok=True)
    print(f"[quick] runs_dir = {args.runs_dir}")
    print(f"[quick] image-size = {args.image_size}")
    print(f"[quick] frames-per-video train/eval = "
          f"{args.frames_per_video_train}/{args.frames_per_video_eval}")

    from forge_detect.classifier import (
        evaluate_classifier,
        load_classifier,
        save_classifier,
        train_classifier,
    )
    from forge_detect.config import PdeParams, PipelineParams
    from forge_detect.datasets import split_videos
    from forge_detect.eval import extract_features_over_dataset

    # 1. Enumerate all video ids and split *videos* (not frames) into
    #    train/val/test. Video-disjoint splits are a methodology
    #    requirement: frame-disjoint splits leak ~0.05–0.15 AUROC.
    print("[quick] enumerating videos for the disjoint split ...")
    all_video_ids = _all_video_ids(args)
    train_vids, val_vids, test_vids = split_videos(
        all_video_ids,
        seed=args.seed,
        val_fraction=args.val_fraction,
        test_fraction=args.test_fraction,
    )
    print(
        f"[quick] split: train={len(train_vids)} val={len(val_vids)} test={len(test_vids)} "
        f"(of {len(all_video_ids)} total videos)",
    )

    # 2. Build adapters only for non-empty splits (a checkpoint cross-dataset
    #    eval can omit train; an extreme test_fraction can omit val).
    train_ds = (
        _build_dataset(
            args, subset_video_ids=train_vids, frames_per_video=args.frames_per_video_train,
        )
        if train_vids
        else None
    )
    val_ds = (
        _build_dataset(
            args, subset_video_ids=val_vids, frames_per_video=args.frames_per_video_eval,
        )
        if val_vids
        else None
    )
    test_ds = (
        _build_dataset(
            args, subset_video_ids=test_vids, frames_per_video=args.frames_per_video_eval,
        )
        if test_vids
        else None
    )
    print(
        f"[quick] dataset sizes: "
        f"train={len(train_ds) if train_ds is not None else 0} "
        f"val={len(val_ds) if val_ds is not None else 0} "
        f"test={len(test_ds) if test_ds is not None else 0}",
    )

    params = PipelineParams(
        n_scales=args.n_scales,
        pde=PdeParams(max_iter=args.max_iter, log_every=20),
    )

    # 3. Extract features for each non-empty split. Each cache is
    #    crash-resumable; restarts skip already-processed paths.
    t0 = time.time()
    fm_val = None
    if val_ds is not None:
        print("[quick] extracting features (val) ...")
        fm_val = extract_features_over_dataset(
            val_ds,
            device=args.device,
            params=params,
            cache_path=args.runs_dir / "features-val.csv",
            num_workers=args.num_workers,
        )
    fm_test = None
    if test_ds is not None:
        print("[quick] extracting features (test) ...")
        fm_test = extract_features_over_dataset(
            test_ds,
            device=args.device,
            params=params,
            cache_path=args.runs_dir / "features-test.csv",
            num_workers=args.num_workers,
        )

    # 4. Either load a pre-trained classifier (cross-dataset eval) or
    #    extract train features and fit one.
    if args.classifier_checkpoint is not None:
        classifier = load_classifier(args.classifier_checkpoint)
        print(f"[quick] loaded classifier from {args.classifier_checkpoint}")
        train_size = 0
    else:
        if train_ds is None:
            msg = "no train videos in split and no --classifier-checkpoint provided"
            raise SystemExit(msg)
        print("[quick] extracting features (train) ...")
        fm_train = extract_features_over_dataset(
            train_ds,
            device=args.device,
            params=params,
            cache_path=args.runs_dir / "features-train.csv",
            num_workers=args.num_workers,
        )
        train_size = fm_train.features.shape[0]
        print(f"[quick] training classifier on {train_size} frames ...")
        classifier = train_classifier(fm_train.features, fm_train.labels)
        save_classifier(classifier, args.runs_dir / "classifier.pkl")

    val_m = (
        evaluate_classifier(classifier, fm_val.features, fm_val.labels, video_ids=fm_val.video_ids)
        if fm_val is not None
        else None
    )
    test_m = (
        evaluate_classifier(
            classifier, fm_test.features, fm_test.labels, video_ids=fm_test.video_ids,
        )
        if fm_test is not None
        else None
    )
    elapsed = time.time() - t0
    print(f"\n[quick] total time: {elapsed:.0f}s")

    def _split_block(m: Any | None, fm: Any | None, *, with_top_features: bool = False) -> Any:
        if m is None or fm is None:
            return None
        block = {
            "frame_auroc": m.auroc,
            "frame_accuracy": m.accuracy,
            "video_auroc_mean": m.video_auroc_mean,
            "video_auroc_max": m.video_auroc_max,
            "n_videos": m.n_videos,
        }
        if with_top_features:
            block["top_features"] = dict(
                sorted(m.feature_importances.items(), key=lambda kv: -kv[1])[:10],
            )
        return block

    report = {
        "phase": "phase-1 heuristic + classifier",
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "elapsed_seconds": elapsed,
        "n_videos": {
            "train": len(train_vids),
            "val": len(val_vids),
            "test": len(test_vids),
        },
        "n_frames": {
            "train": train_size,
            "val": fm_val.features.shape[0] if fm_val is not None else 0,
            "test": fm_test.features.shape[0] if fm_test is not None else 0,
        },
        "val": _split_block(val_m, fm_val),
        "test": _split_block(test_m, fm_test, with_top_features=True),
    }
    report_path = args.runs_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2, default=str))
    print(f"[quick] saved report -> {report_path}")

    def _fmt_auroc(m: Any | None, attr: str) -> str:
        if m is None:
            return "  n/a "
        v = getattr(m, attr)
        return f"{v:.4f}" if not math.isnan(v) else "  nan "

    print("\n=== Phase-1 result ===")
    print(
        f"  Frame AUROC  val={_fmt_auroc(val_m, 'auroc')}  "
        f"test={_fmt_auroc(test_m, 'auroc')}",
    )
    if test_m is not None and not math.isnan(test_m.video_auroc_mean):
        print(
            f"  Video AUROC  val={_fmt_auroc(val_m, 'video_auroc_mean')}  "
            f"test={_fmt_auroc(test_m, 'video_auroc_mean')}  (mean-pool)",
        )
        print(
            f"               val={_fmt_auroc(val_m, 'video_auroc_max')}  "
            f"test={_fmt_auroc(test_m, 'video_auroc_max')}  (max-pool)",
        )
    if test_m is not None:
        print(
            f"  Test split:  videos real={test_m.n_video_real} fake={test_m.n_video_fake}",
        )
        print("  Top 5 features (by GBC importance):")
        for name, imp in sorted(test_m.feature_importances.items(), key=lambda kv: -kv[1])[:5]:
            print(f"    {name:30s} {imp:.4f}")

    # Use video-level AUROC for the rubric — that is the metric the
    # diploma defense will quote and the literature compares against.
    headline = (
        test_m.video_auroc_mean
        if test_m is not None and not math.isnan(test_m.video_auroc_mean)
        else (test_m.auroc if test_m is not None else float("nan"))
    )
    print(f"\nHeadline test AUROC (video-level mean-pool): {headline:.4f}")
    print("\nWhat to do with this number:")
    print("  AUROC >= 0.85  -> strong baseline; the heuristic + classifier alone")
    print("                    is defendable; CNN training is an upgrade, not a")
    print("                    prerequisite. Phase 2 is optional.")
    print("  AUROC ~ 0.70   -> reasonable but not great; CNN training likely")
    print("                    helps; consider Phase 2 if you have cluster time.")
    print("  AUROC <= 0.55  -> near chance; the math pipeline + heuristic does")
    print("                    not separate this dataset; revisit assumptions")
    print("                    before committing to CNN training.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
