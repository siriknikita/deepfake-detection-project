"""Pivot study: pure-CNN baseline vs heuristic pipeline vs full pipeline.

Trains all three approaches on the same labeled dataset with the same
*video-disjoint* train / val / test split and reports both frame-level
and video-level AUROC, accuracy, and approximate runtime cost. The
diploma's empirical chapter consumes this report:

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
        --frames-per-video-train 1 --frames-per-video-eval 1 \\
        --image-size 128 --epochs 3 --device cuda

Then on the full split for the diploma:

    python scripts/pivot_study.py \\
        --data-root /scratch/data/FaceForensics++ \\
        --frames-per-video-train 30 --frames-per-video-eval 10 \\
        --image-size 256 --epochs 30 --device cuda \\
        --output runs/pivot_report.json
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
    from forge_detect.datasets import (
        CelebDFAdapter,
        FaceForensicsAdapter,
        ImageFolderDataset,
    )

    target_size = (args.image_size, args.image_size) if args.image_size else None
    if args.dataset == "image-folder":
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
    full = _build_dataset(args, subset_video_ids=None, frames_per_video=1)
    if hasattr(full, "video_ids"):
        return full.video_ids()
    records = getattr(full, "_records", [])
    return sorted({rec.video_id for rec in records})


def _run_baseline_cnn(
    train_ds: Any,
    val_ds: Any,
    test_ds: Any,
    args: argparse.Namespace,
) -> dict[str, float]:
    """Train EfficientNet-B0 directly on (image, label).

    Crash-resumable: the run dir is a fixed name (``runs_dir/baseline_run``),
    so re-running picks up from the saved checkpoint.pt.
    """
    from forge_detect.baseline_cnn import (
        BaselineConfig,
        build_baseline_classifier,
        evaluate_baseline_cnn,
        train_baseline_cnn,
    )
    from forge_detect.cnn import load_weights

    run_dir = args.runs_dir / "baseline_run"
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg_kwargs: dict[str, Any] = {
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "device": args.device,
        "num_workers": args.num_workers,
        "checkpoint_dir": run_dir.parent,  # ignored — resume_dir wins
    }
    if args.lr is not None:
        cfg_kwargs["learning_rate"] = args.lr
    if args.amp:
        cfg_kwargs["mixed_precision"] = True
    cfg = BaselineConfig(**cfg_kwargs)
    t0 = time.time()
    out = train_baseline_cnn(
        train_ds,
        val_ds,
        cfg,
        pretrained=not args.no_pretrained,
        resume_dir=run_dir,
    )
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
    train_ds: Any,
    val_ds: Any,
    test_ds: Any,
    args: argparse.Namespace,
    *,
    cnn_checkpoint: Path | None,
    label: str,
) -> dict[str, Any]:
    """Run the math pipeline → features → GBC on the supplied splits.

    Each split's features are extracted independently (separate cache
    files) so a crash mid-stream resumes per-split. Video-level AUROC
    is reported alongside frame-level using the per-row video_id
    captured during extraction.
    """
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

    params = PipelineParams(
        n_scales=args.n_scales,
        pde=PdeParams(max_iter=args.max_iter),
    )

    def _extract(ds: Any, split_name: str) -> Any:
        cache_path = args.runs_dir / f"features-{label}-{split_name}.csv"
        print(f"[{label}] extracting features ({split_name}, {len(ds)} frames) ...")
        return extract_features_over_dataset(
            ds,
            device=args.device,
            params=params,
            cnn_model=cnn_model,
            cnn_device=cnn_device,
            cache_path=cache_path,
            num_workers=args.num_workers,
        )

    t0 = time.time()
    fm_train = _extract(train_ds, "train")
    fm_val = _extract(val_ds, "val")
    fm_test = _extract(test_ds, "test")
    extract_seconds = time.time() - t0

    classifier = train_classifier(fm_train.features, fm_train.labels)
    val_m = evaluate_classifier(
        classifier, fm_val.features, fm_val.labels, video_ids=fm_val.video_ids,
    )
    test_m = evaluate_classifier(
        classifier, fm_test.features, fm_test.labels, video_ids=fm_test.video_ids,
    )
    return {
        "frame_test_auroc": test_m.auroc,
        "frame_test_accuracy": test_m.accuracy,
        "frame_val_auroc": val_m.auroc,
        "frame_val_accuracy": val_m.accuracy,
        "video_test_auroc_mean": test_m.video_auroc_mean,
        "video_test_auroc_max": test_m.video_auroc_max,
        "video_val_auroc_mean": val_m.video_auroc_mean,
        "video_val_auroc_max": val_m.video_auroc_max,
        "n_test_real_frames": test_m.n_real,
        "n_test_fake_frames": test_m.n_fake,
        "n_test_real_videos": test_m.n_video_real,
        "n_test_fake_videos": test_m.n_video_fake,
        "extract_seconds": extract_seconds,
        "top_features": dict(
            sorted(test_m.feature_importances.items(), key=lambda kv: -kv[1])[:5],
        ),
    }


def _format_report(report: dict[str, Any]) -> str:
    lines = ["=" * 70, "Hyperplane-Forge — pivot study", "=" * 70]
    for name, metrics in report["baselines"].items():
        lines.append(f"\n[{name}]")
        for k, v in metrics.items():
            if isinstance(v, float) and not math.isnan(v):
                lines.append(f"  {k:24s} {v:.4f}")
            elif isinstance(v, float):  # NaN
                lines.append(f"  {k:24s} (n/a)")
            else:
                lines.append(f"  {k:24s} {v}")
    lines.append("\n" + "-" * 70)
    lines.append("Decision rubric (uses video-level AUROC, mean-pool):")
    lines.append("  Pure CNN > Full pipeline + 0.03 -> pivot to CNN-only")
    lines.append("  Full pipeline > Pure CNN + 0.02 -> physics framework wins")
    lines.append("  Heuristic ≈ Trained CNN          -> CNN trust map adds no value")
    lines.append("=" * 70)
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
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
        help="Frames per training video (default 30).",
    )
    parser.add_argument(
        "--frames-per-video-eval",
        type=int,
        default=10,
        help="Frames per val/test video (default 10, matches academic protocol).",
    )
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
    parser.add_argument("--seed", type=int, default=0)
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
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Learning rate override for the pure-CNN baseline. If omitted, "
        "uses BaselineConfig.learning_rate (1e-4 — the standard fine-tuning "
        "rate for pretrained EfficientNet-B0 on FF++).",
    )
    parser.add_argument(
        "--amp",
        action="store_true",
        help="Enable mixed-precision (fp16 autocast) training. Off by default; "
        "AMP with tight fine-tuning gradients was associated with cuDNN "
        "execution failures and stalled loss. Re-enable once a working "
        "baseline is in hand.",
    )
    parser.add_argument(
        "--baselines",
        type=str,
        default="all",
        help=(
            "Comma-separated subset of {pure-cnn, heuristic, trained-cnn} to run, "
            "or 'all' (default). Lets you run just the cheap stages first and "
            "add expensive ones later."
        ),
    )
    args = parser.parse_args()

    valid_baselines = {"pure-cnn", "heuristic", "trained-cnn"}
    if args.baselines == "all":
        active_baselines = set(valid_baselines)
    else:
        active_baselines = {b.strip() for b in args.baselines.split(",") if b.strip()}
        unknown = active_baselines - valid_baselines
        if unknown:
            parser.error(
                f"unknown baselines: {sorted(unknown)} — pick from {sorted(valid_baselines)}",
            )
    print(f"[pivot] active baselines: {sorted(active_baselines)}")

    from forge_detect.datasets import split_videos

    args.runs_dir.mkdir(parents=True, exist_ok=True)
    print("[pivot] enumerating videos for the disjoint split ...")
    all_video_ids = _all_video_ids(args)
    train_vids, val_vids, test_vids = split_videos(
        all_video_ids,
        seed=args.seed,
        val_fraction=args.val_fraction,
        test_fraction=args.test_fraction,
    )
    print(
        f"[pivot] split: train={len(train_vids)} val={len(val_vids)} test={len(test_vids)} "
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
    print(f"[pivot] frames: train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}")

    report: dict[str, Any] = {"baselines": {}, "args": vars(args)}
    partial_path = args.runs_dir / "report.partial.json"

    def _flush_partial() -> None:
        partial_path.parent.mkdir(parents=True, exist_ok=True)
        partial_path.write_text(json.dumps(report, indent=2, default=str))

    # Baseline 1: pure CNN end-to-end.
    if "pure-cnn" in active_baselines:
        print("\n--- Baseline 1: pure CNN ---")
        report["baselines"]["pure_cnn"] = _run_baseline_cnn(
            train_ds, val_ds, test_ds, args,
        )
        _flush_partial()

    # Baseline 2: pipeline + heuristic trust map.
    if "heuristic" in active_baselines:
        print("\n--- Baseline 2: pipeline (heuristic W_cnn) ---")
        report["baselines"]["pipeline_heuristic"] = _run_pipeline_eval(
            train_ds, val_ds, test_ds, args, cnn_checkpoint=None, label="heuristic",
        )
        _flush_partial()

    # Baseline 3: train ChromaticEfficientNet, then pipeline + trained trust map.
    if "trained-cnn" in active_baselines:
        print("\n--- Training ChromaticEfficientNet for trust map ---")
        from forge_detect.train import TrainingConfig, train_cnn

        trust_map_dir = args.runs_dir / "trust_map_run"
        trust_map_dir.mkdir(parents=True, exist_ok=True)
        cfg = TrainingConfig(
            epochs=args.epochs,
            batch_size=args.batch_size,
            device=args.device,
            num_workers=args.num_workers,
            checkpoint_dir=trust_map_dir.parent,  # ignored — resume_dir wins
        )
        out = train_cnn(
            train_ds,
            val_ds,
            cfg,
            pretrained=not args.no_pretrained,
            resume_dir=trust_map_dir,
        )
        ckpt = Path(out["run_dir"]) / "best.pt"

        print("\n--- Baseline 3: pipeline (trained CNN W_cnn) ---")
        report["baselines"]["pipeline_trained_cnn"] = _run_pipeline_eval(
            train_ds, val_ds, test_ds, args, cnn_checkpoint=ckpt, label="trained_cnn",
        )
        _flush_partial()

    pretty = _format_report(report)
    print("\n" + pretty)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, default=str))
        print(f"\nWrote JSON report to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
