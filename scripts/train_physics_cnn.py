"""Train EfficientNet-B0 on the 6-channel physics tensor (RGB + W_cnn + z* + R).

The math pipeline already produced strong but anti-correlated frame-level
signal on FF++ c23 (oracle AUROC 0.37 → 0.63 if you flip the sign). This
script drops the analytical classifier and feeds the raw spatial maps into
EfficientNet-B0, letting the CNN learn which polarity / pattern combination
discriminates fakes from reals — and how to weight the physics signal
against raw RGB.

Two variants of the physics cache feed two training runs:

  - ``--variant heuristic``  W_cnn = chromatic-residual heuristic everywhere.
                             Inference distribution matches training exactly.
  - ``--variant gtmask``     W_cnn = 1 - GT_mask for FF++ train fakes;
                             heuristic for reals and inference. Stronger
                             training signal at the cost of train/test shift.

Pre-requisite: the ``scripts/cache_physics_maps.py`` cache must exist for
*every* frame the dataset will read (training: FF++ all splits; eval: FF++
test split + CelebDF testing list). Missing maps raise loudly at
``__getitem__``.

Usage:

    python scripts/train_physics_cnn.py \\
        --data-root /scratch/$USER/data/FaceForensics++ \\
        --celeb-data-root /scratch/$USER/data/Celeb-DF-v2 \\
        --variant heuristic \\
        --runs-dir runs/physics_6ch_heuristic \\
        --epochs 30 --batch-size 32 --device cuda

A 3-channel RGB baseline can be run by also pointing
``scripts/pivot_study.py --baselines pure-cnn`` at the same data — that
gives the apples-to-apples control number this experiment compares against.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any


def _build_ff_dataset(
    args: argparse.Namespace,
    *,
    ff_split: str,
    physics_variant: str,
    frames_per_video: int | None,
) -> Any:
    from forge_detect.datasets import FaceForensicsAdapter

    target_size = (args.image_size, args.image_size) if args.image_size else None
    return FaceForensicsAdapter(
        root=args.data_root,
        compression=args.compression,
        max_frames_per_video=frames_per_video,
        target_size=target_size,
        ff_split=ff_split,
        load_physics_maps=True,
        physics_variant=physics_variant,
    )


def _build_celeb_dataset(args: argparse.Namespace) -> Any:
    from forge_detect.datasets import CelebDFAdapter

    target_size = (args.image_size, args.image_size) if args.image_size else None
    return CelebDFAdapter(
        root=args.celeb_data_root,
        max_frames_per_video=args.frames_per_video_eval,
        target_size=target_size,
        testing_list=True,  # canonical 518-video benchmark
        load_physics_maps=True,  # CelebDF only has heuristic variant
    )


def _train(
    args: argparse.Namespace,
    train_ds: Any,
    val_ds: Any,
) -> tuple[Path, dict[str, list[dict[str, float]] | str]]:
    from forge_detect.baseline_cnn import (
        BaselineConfig,
        build_physics_classifier,
        train_baseline_cnn,
    )

    run_dir = args.runs_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg = BaselineConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        device=args.device,
        num_workers=args.num_workers,
        checkpoint_dir=run_dir.parent,  # ignored — resume_dir wins
        balance_classes=True,
        mixed_precision=args.amp,
    )

    def factory(*, pretrained: bool) -> Any:
        return build_physics_classifier(in_channels=6, pretrained=pretrained)

    out = train_baseline_cnn(
        train_ds,
        val_ds,
        cfg,
        pretrained=not args.no_pretrained,
        resume_dir=run_dir,
        model_factory=factory,
        log_tag=f"physics-6ch-{args.variant}",
    )
    return run_dir, out


def _evaluate(
    args: argparse.Namespace,
    model: Any,
    dataset: Any,
    *,
    label: str,
) -> dict[str, float]:
    from forge_detect.baseline_cnn import evaluate_baseline_cnn

    print(f"[physics-6ch] evaluating on {label} ({len(dataset)} frames) ...")
    metrics = evaluate_baseline_cnn(
        model,
        dataset,
        device=args.device if args.device != "auto" else "cuda",
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-root", type=Path, required=True, help="FaceForensics++ root")
    parser.add_argument(
        "--celeb-data-root",
        type=Path,
        default=None,
        help="Celeb-DF v2 root for cross-dataset eval. If omitted, only FF++ "
        "test eval runs.",
    )
    parser.add_argument("--compression", choices=("raw", "c23", "c40"), default="c23")
    parser.add_argument(
        "--variant",
        choices=("heuristic", "gtmask"),
        default="heuristic",
        help="Physics-cache variant for FF++ fakes. Used for train AND eval "
        "uniformly — for FF++-only protocols, GT masks are equally available "
        "at eval time as at train time, so there is no train/test "
        "distribution shift. Reals always read the heuristic cache (no GT "
        "mask exists for them). For cross-dataset eval (e.g. on Celeb-DF), "
        "the trained model must be applied with heuristic cache only.",
    )
    parser.add_argument("--frames-per-video-train", type=int, default=30)
    parser.add_argument("--frames-per-video-eval", type=int, default=10)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1.0e-4,
        help="Default 1e-4 (standard fine-tuning rate for pretrained "
        "EfficientNet-B0). Higher rates stall the loss at log(2) and never "
        "recover on this task.",
    )
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument(
        "--amp",
        action="store_true",
        help="Enable fp16 autocast. Off by default — tight fine-tuning "
        "gradients caused cuDNN execution errors with AMP enabled.",
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--runs-dir", type=Path, required=True)
    parser.add_argument(
        "--skip-train",
        action="store_true",
        help="Skip training and only evaluate from --runs-dir/best.pt — useful "
        "when re-running cross-dataset eval after a model is trained.",
    )
    args = parser.parse_args()

    args.runs_dir.mkdir(parents=True, exist_ok=True)
    print(f"[physics-6ch] runs_dir = {args.runs_dir}")
    print(f"[physics-6ch] variant = {args.variant}")

    print("[physics-6ch] building FF++ datasets (load_physics_maps=True) ...")
    # All three splits use the configured variant. For the FF++-only protocol
    # this avoids the train/test distribution shift that would arise if train
    # used GT masks but val/test fell back to heuristic.
    train_ds = _build_ff_dataset(
        args, ff_split="train", physics_variant=args.variant,
        frames_per_video=args.frames_per_video_train,
    )
    val_ds = _build_ff_dataset(
        args, ff_split="val", physics_variant=args.variant,
        frames_per_video=args.frames_per_video_eval,
    )
    test_ds = _build_ff_dataset(
        args, ff_split="test", physics_variant=args.variant,
        frames_per_video=args.frames_per_video_eval,
    )
    print(
        f"[physics-6ch] frames: train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}",
    )

    if not args.skip_train:
        t_train_start = time.time()
        run_dir, _train_out = _train(args, train_ds, val_ds)
        train_seconds = time.time() - t_train_start
    else:
        run_dir = args.runs_dir
        train_seconds = 0.0

    # Reload best.pt for evaluation.
    from forge_detect.baseline_cnn import build_physics_classifier
    from forge_detect.cnn import load_weights

    best_path = run_dir / "best.pt"
    if not best_path.exists():
        msg = (
            f"best.pt not found at {best_path}; training may have died before "
            "the first val pass. Re-run without --skip-train, or point "
            "--runs-dir at a completed run."
        )
        raise FileNotFoundError(msg)
    model = build_physics_classifier(in_channels=6, pretrained=False)
    load_weights(model, best_path)

    report: dict[str, Any] = {
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "train_seconds": train_seconds,
        "n_frames": {
            "train": len(train_ds),
            "val": len(val_ds),
            "test_ff": len(test_ds),
        },
    }

    report["ff_test"] = _evaluate(args, model, test_ds, label="FF++ test")

    if args.celeb_data_root is not None:
        celeb_ds = _build_celeb_dataset(args)
        report["n_frames"]["test_celeb"] = len(celeb_ds)
        report["celeb_test"] = _evaluate(args, model, celeb_ds, label="CelebDF testing list")

    out_path = run_dir / "report.json"
    out_path.write_text(json.dumps(report, indent=2, default=str))
    print(f"\n[physics-6ch] wrote {out_path}")

    print("\n=== summary ===")
    for split, metrics in report.items():
        if not isinstance(metrics, dict) or "auroc" not in metrics:
            continue
        line = f"  {split}: frame_auroc={metrics['auroc']:.4f}"
        if "video_auroc_mean" in metrics:
            line += f" video_auroc_mean={metrics['video_auroc_mean']:.4f}"
            line += f" video_auroc_max={metrics['video_auroc_max']:.4f}"
        print(line)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
