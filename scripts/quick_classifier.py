"""Phase-1 AUROC: heuristic W_cnn + Gradient Boosting classifier.

The fastest defendable number you can put in the diploma. Skips the
ChromaticEfficientNet training entirely — uses the deterministic
chromatic-residual trust map for every image and trains the binary
classifier on the resulting impact-map features. Total cluster time on
a single 4090 with default settings is ~2 hours on a 30-frame-per-video
FF++ slice, vs ~25 hours for the full pivot study.

What you get:
- runs_dir/features.csv               full impact-map feature matrix
- runs_dir/classifier.pkl             trained sklearn pipeline
- runs_dir/report.json                AUROC, accuracy, top features

Workflow:

    # Fast cluster run, defendable number in ~2 hours:
    python scripts/quick_classifier.py \\
        --data-root /scratch/$USER/data/FaceForensics++ \\
        --runs-dir runs/quick_phase1 \\
        --device cuda

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
- features.csv accumulates incrementally — restarts skip already-
  processed images.
- classifier.pkl is regenerated from features.csv on every run, so a
  classifier-stage crash just reruns sklearn (~seconds).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any


def _build_dataset(args: argparse.Namespace) -> Any:
    from forge_detect.datasets import FaceForensicsAdapter, ImageFolderDataset

    target_size = (args.image_size, args.image_size) if args.image_size else None
    if args.dataset == "image-folder":
        if not args.fake_dir:
            msg = "--fake-dir is required for --dataset=image-folder"
            raise SystemExit(msg)
        return ImageFolderDataset(
            real_dir=args.data_root,
            fake_dir=args.fake_dir,
            target_size=target_size,
        )
    return FaceForensicsAdapter(
        root=args.data_root,
        compression=args.compression,
        max_frames_per_video=args.max_frames_per_video,
        target_size=target_size,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    # ---- dataset ----
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument(
        "--dataset",
        choices=("face-forensics", "image-folder"),
        default="face-forensics",
    )
    parser.add_argument("--fake-dir", type=Path, default=None)
    parser.add_argument("--compression", choices=("raw", "c23", "c40"), default="c23")
    parser.add_argument(
        "--max-frames-per-video",
        type=int,
        default=10,
        help="FF++ stride-sample cap (default 10 — fast turnaround).",
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
    # ---- output ----
    parser.add_argument("--runs-dir", type=Path, default=Path("runs/quick_phase1"))
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=0.15,
        help="Stratified val slice (default 0.15).",
    )
    parser.add_argument(
        "--test-fraction",
        type=float,
        default=0.15,
        help="Stratified test slice (default 0.15).",
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    args.runs_dir.mkdir(parents=True, exist_ok=True)
    print(f"[quick] runs_dir = {args.runs_dir}")
    print(f"[quick] image-size = {args.image_size}")
    print(f"[quick] max-frames-per-video = {args.max_frames_per_video}")

    from forge_detect.classifier import save_classifier
    from forge_detect.config import PdeParams, PipelineParams
    from forge_detect.eval import evaluate_pipeline

    dataset = _build_dataset(args)
    print(f"[quick] dataset size = {len(dataset)}")

    params = PipelineParams(
        n_scales=args.n_scales,
        pde=PdeParams(max_iter=args.max_iter, log_every=20),
    )

    cache_path = args.runs_dir / "features.csv"

    t0 = time.time()
    val_m, test_m, classifier, _fm = evaluate_pipeline(
        dataset,
        device=args.device,
        params=params,
        val_fraction=args.val_fraction,
        test_fraction=args.test_fraction,
        seed=args.seed,
        cache_path=cache_path,
    )
    elapsed = time.time() - t0
    print(f"\n[quick] total time: {elapsed:.0f}s")

    # Save classifier and report.
    classifier_path = args.runs_dir / "classifier.pkl"
    save_classifier(classifier, classifier_path)
    print(f"[quick] saved classifier -> {classifier_path}")

    report = {
        "phase": "phase-1 heuristic + classifier",
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "elapsed_seconds": elapsed,
        "val": {
            "auroc": val_m.auroc,
            "accuracy": val_m.accuracy,
            "n_real": val_m.n_real,
            "n_fake": val_m.n_fake,
        },
        "test": {
            "auroc": test_m.auroc,
            "accuracy": test_m.accuracy,
            "n_real": test_m.n_real,
            "n_fake": test_m.n_fake,
            "top_features": dict(
                sorted(test_m.feature_importances.items(), key=lambda kv: -kv[1])[:10],
            ),
        },
    }
    report_path = args.runs_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2, default=str))
    print(f"[quick] saved report -> {report_path}")

    print("\n=== Phase-1 result ===")
    print(f"  Val  AUROC: {val_m.auroc:.4f}  Accuracy: {val_m.accuracy:.4f}")
    print(f"  Test AUROC: {test_m.auroc:.4f}  Accuracy: {test_m.accuracy:.4f}")
    print(f"  Test split: real={test_m.n_real} fake={test_m.n_fake}")
    print("  Top 5 features (by GBC importance):")
    for name, imp in sorted(test_m.feature_importances.items(), key=lambda kv: -kv[1])[:5]:
        print(f"    {name:30s} {imp:.4f}")

    print("\nWhat to do with this number:")
    print("  AUROC ≥ 0.85  → strong baseline; the heuristic + classifier alone")
    print("                  is defendable; CNN training is an upgrade, not a")
    print("                  prerequisite. Phase 2 is optional.")
    print("  AUROC ~ 0.70  → reasonable but not great; CNN training likely")
    print("                  helps; consider Phase 2 if you have cluster time.")
    print("  AUROC ≤ 0.55  → near chance; the math pipeline + heuristic does")
    print("                  not separate this dataset; revisit assumptions")
    print("                  before committing to CNN training.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
