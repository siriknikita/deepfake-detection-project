"""Pivot study: pure-CNN baseline vs heuristic pipeline vs full pipeline.

Trains all three approaches on the same labeled dataset with the same
train / val / test split and reports AUROC, accuracy, and approximate
runtime cost. The diploma's empirical chapter consumes this report:

  - **Pure CNN** wins by a clear margin → the math framework adds no
    value; pivot to a learned-classifier-only project.
  - **Full pipeline** wins by a clear margin → the physics signal
    is real; the framework is the contribution; defend it.
  - **Heuristic ≈ Full pipeline** but both ≈ Pure CNN → the math
    pipeline carries useful signal but the *trained* CNN trust map is
    not adding much over the deterministic heuristic; report and
    discuss.
  - **All three within noise** → no method has signal on this data
    split; revisit dataset assumptions or pipeline parameters.

Run on a small slice first:

    python scripts/pivot_study.py \\
        --data-root /scratch/data/FaceForensics++ \\
        --max-frames-per-video 1 --image-size 128 \\
        --epochs 3 --device cuda

Then on the full split for the diploma:

    python scripts/pivot_study.py \\
        --data-root /scratch/data/FaceForensics++ \\
        --max-frames-per-video 30 --image-size 256 \\
        --epochs 30 --device cuda \\
        --output runs/pivot_report.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any


def _build_dataset(args: argparse.Namespace, root: Path) -> Any:
    from forge_detect.datasets import FaceForensicsAdapter, ImageFolderDataset

    target_size = (args.image_size, args.image_size) if args.image_size else None
    if args.dataset == "image-folder":
        return ImageFolderDataset(
            real_dir=root,
            fake_dir=args.fake_dir,
            target_size=target_size,
        )
    return FaceForensicsAdapter(
        root=root,
        compression=args.compression,
        max_frames_per_video=args.max_frames_per_video,
        target_size=target_size,
    )


def _stratified_subset(dataset: Any, indices: list[int]) -> Any:
    from torch.utils.data import Subset

    return Subset(dataset, indices)


def _run_baseline_cnn(
    dataset: Any,
    train_idx: list[int],
    val_idx: list[int],
    test_idx: list[int],
    args: argparse.Namespace,
) -> dict[str, float]:
    """Train EfficientNet-B0 directly on (image, label)."""
    from forge_detect.baseline_cnn import (
        BaselineConfig,
        build_baseline_classifier,
        evaluate_baseline_cnn,
        train_baseline_cnn,
    )
    from forge_detect.cnn import load_weights

    train_ds = _stratified_subset(dataset, train_idx)
    val_ds = _stratified_subset(dataset, val_idx)
    test_ds = _stratified_subset(dataset, test_idx)
    cfg = BaselineConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        device=args.device,
        num_workers=args.num_workers,
        checkpoint_dir=args.runs_dir / "baseline",
    )
    t0 = time.time()
    out = train_baseline_cnn(train_ds, val_ds, cfg, pretrained=not args.no_pretrained)
    train_seconds = time.time() - t0
    # Reload best.pt and evaluate on test split.
    model = build_baseline_classifier(pretrained=False)
    load_weights(model, Path(out["run_dir"]) / "best.pt")
    test_metrics = evaluate_baseline_cnn(
        model,
        test_ds,
        device=args.device if args.device != "auto" else "cuda",
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    return {
        **test_metrics,
        "train_seconds": train_seconds,
        "run_dir": out["run_dir"],
    }


def _run_pipeline_eval(
    dataset: Any,
    train_idx: list[int],
    val_idx: list[int],
    test_idx: list[int],
    args: argparse.Namespace,
    *,
    cnn_checkpoint: Path | None,
    label: str,
) -> dict[str, float]:
    """Run the math pipeline → features → GBC on the supplied splits."""
    from forge_detect.classifier import evaluate_classifier, train_classifier
    from forge_detect.cnn import build_chromatic_efficientnet, load_weights
    from forge_detect.config import PdeParams, PipelineParams
    from forge_detect.eval import extract_features_over_dataset

    # Optionally load the trained CNN.
    cnn_model: Any | None = None
    cnn_device = args.cnn_device if args.cnn_device != "auto" else args.device
    if cnn_checkpoint is not None:
        cnn_model = build_chromatic_efficientnet(pretrained=False)
        load_weights(cnn_model, cnn_checkpoint)
        cnn_model = cnn_model.to(cnn_device).eval()

    # The pipeline runs once over every record; features are then sliced
    # by the precomputed split indices.
    cache_path = args.runs_dir / f"features-{label}.csv"
    if cache_path.exists():
        from forge_detect.eval import FeatureMatrix

        print(f"[{label}] loading cached features from {cache_path}")
        fm = FeatureMatrix.load(cache_path)
    else:
        print(f"[{label}] extracting features over {len(dataset)} records ...")
        params = PipelineParams(
            n_scales=args.n_scales,
            pde=PdeParams(max_iter=args.max_iter),
        )
        t0 = time.time()
        fm = extract_features_over_dataset(
            dataset,
            device=args.device,
            params=params,
            cnn_model=cnn_model,
            cnn_device=cnn_device,
        )
        print(f"[{label}] feature extraction took {time.time() - t0:.1f}s")
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        fm.save(cache_path)

    classifier = train_classifier(fm.features[train_idx], fm.labels[train_idx])
    test_m = evaluate_classifier(classifier, fm.features[test_idx], fm.labels[test_idx])
    val_m = evaluate_classifier(classifier, fm.features[val_idx], fm.labels[val_idx])
    return {
        "test_auroc": test_m.auroc,
        "test_accuracy": test_m.accuracy,
        "val_auroc": val_m.auroc,
        "val_accuracy": val_m.accuracy,
        "n_test_real": test_m.n_real,
        "n_test_fake": test_m.n_fake,
        "top_features": dict(
            sorted(test_m.feature_importances.items(), key=lambda kv: -kv[1])[:5],
        ),
    }


def _format_report(report: dict[str, Any]) -> str:
    lines = ["=" * 70, "Hyperplane-Forge — pivot study", "=" * 70]
    for name, metrics in report["baselines"].items():
        lines.append(f"\n[{name}]")
        for k, v in metrics.items():
            if isinstance(v, float):
                lines.append(f"  {k:24s} {v:.4f}")
            else:
                lines.append(f"  {k:24s} {v}")
    lines.append("\n" + "-" * 70)
    lines.append("Decision rubric:")
    lines.append("  Pure CNN > Full pipeline + 0.03 AUROC -> pivot to CNN-only")
    lines.append("  Full pipeline > Pure CNN + 0.02 AUROC -> physics framework wins")
    lines.append("  Heuristic ≈ Trained CNN -> CNN trust map is not adding value")
    lines.append("=" * 70)
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument(
        "--dataset",
        choices=("face-forensics", "image-folder"),
        default="face-forensics",
    )
    parser.add_argument("--fake-dir", type=Path, default=None)
    parser.add_argument("--compression", choices=("raw", "c23", "c40"), default="c23")
    parser.add_argument("--max-frames-per-video", type=int, default=None)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
    )
    parser.add_argument(
        "--cnn-device",
        choices=("auto", "cpu", "cuda", "mps"),
        default="auto",
    )
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--n-scales", type=int, default=4)
    parser.add_argument("--max-iter", type=int, default=300)
    parser.add_argument("--runs-dir", type=Path, default=Path("runs/pivot"))
    parser.add_argument("--output", type=Path, default=None, help="Optional JSON report path.")
    parser.add_argument(
        "--skip-trained-cnn",
        action="store_true",
        help="Skip training the ChromaticEfficientNet — only run pure-CNN and heuristic.",
    )
    args = parser.parse_args()

    from forge_detect.datasets import stratified_split

    args.runs_dir.mkdir(parents=True, exist_ok=True)
    dataset = _build_dataset(args, args.data_root)
    print(f"dataset size: {len(dataset)}")
    train_idx, val_idx, test_idx = stratified_split(
        len(dataset),
        seed=0,
        val_fraction=0.15,
        test_fraction=0.15,
    )
    print(f"split: train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}")

    report: dict[str, Any] = {"baselines": {}, "args": vars(args)}

    # Baseline 1: pure CNN end-to-end.
    print("\n--- Baseline 1: pure CNN ---")
    report["baselines"]["pure_cnn"] = _run_baseline_cnn(
        dataset,
        train_idx,
        val_idx,
        test_idx,
        args,
    )

    # Baseline 2: pipeline + heuristic trust map.
    print("\n--- Baseline 2: pipeline (heuristic W_cnn) ---")
    report["baselines"]["pipeline_heuristic"] = _run_pipeline_eval(
        dataset,
        train_idx,
        val_idx,
        test_idx,
        args,
        cnn_checkpoint=None,
        label="heuristic",
    )

    # Baseline 3: train ChromaticEfficientNet, then pipeline + trained trust map.
    if not args.skip_trained_cnn:
        print("\n--- Training ChromaticEfficientNet for trust map ---")
        from forge_detect.train import TrainingConfig, train_cnn

        cfg = TrainingConfig(
            epochs=args.epochs,
            batch_size=args.batch_size,
            device=args.device,
            num_workers=args.num_workers,
            checkpoint_dir=args.runs_dir / "trust_map",
        )
        train_ds = _stratified_subset(dataset, train_idx)
        val_ds = _stratified_subset(dataset, val_idx)
        out = train_cnn(train_ds, val_ds, cfg, pretrained=not args.no_pretrained)
        ckpt = Path(out["run_dir"]) / "best.pt"

        print("\n--- Baseline 3: pipeline (trained CNN W_cnn) ---")
        report["baselines"]["pipeline_trained_cnn"] = _run_pipeline_eval(
            dataset,
            train_idx,
            val_idx,
            test_idx,
            args,
            cnn_checkpoint=ckpt,
            label="trained_cnn",
        )

    pretty = _format_report(report)
    print("\n" + pretty)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, default=str))
        print(f"\nWrote JSON report to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
