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
    subset_video_ids: set[str] | None = None,
    ff_split: str | None = None,
    physics_variant: str,
    frames_per_video: int | None,
    channel_sources: Any,
) -> Any:
    """Build the FF++ adapter for a video subset.

    Pass ``ff_split`` (official FF++ splits/<name>.json) for identity-disjoint
    partitions — the only correct choice when the splits files exist. Falls
    back to ``subset_video_ids`` (random video-disjoint split via
    :func:`forge_detect.datasets.split_videos`) when the user hasn't
    downloaded the official splits; this random path is documented to leak
    identities between train and val and is included only for parity with
    pivot_study.py's fallback so the comparison stays apples-to-apples.

    When ``channel_sources`` is a non-empty list, the adapter is built with
    the multi-source channel API (Phase 3+) and ``physics_variant`` is
    ignored. When it is ``None``, the legacy ``load_physics_maps=True``
    code path with the configured variant is used (Phase 2 default).
    """
    from forge_detect.datasets import FaceForensicsAdapter

    target_size = (args.image_size, args.image_size) if args.image_size else None
    frames_subdir = "frames_faces" if args.use_face_crops else "frames"
    if channel_sources is not None:
        return FaceForensicsAdapter(
            root=args.data_root,
            compression=args.compression,
            max_frames_per_video=frames_per_video,
            target_size=target_size,
            subset_video_ids=subset_video_ids,
            ff_split=ff_split,
            channel_sources=channel_sources,
            frames_subdir=frames_subdir,
        )
    return FaceForensicsAdapter(
        root=args.data_root,
        compression=args.compression,
        max_frames_per_video=frames_per_video,
        target_size=target_size,
        subset_video_ids=subset_video_ids,
        ff_split=ff_split,
        load_physics_maps=True,
        physics_variant=physics_variant,
        frames_subdir=frames_subdir,
    )


def _check_ff_splits_present(data_root: Path) -> None:
    """See :func:`pivot_study._check_ff_splits_present`."""
    splits_dir = Path(data_root) / "splits"
    needed = ["train.json", "val.json", "test.json"]
    missing = [n for n in needed if not (splits_dir / n).exists()]
    if not missing:
        return
    base = "https://raw.githubusercontent.com/ondyari/FaceForensics/master/dataset/splits"
    cmds = "\n".join(
        f"  curl -L -o {splits_dir / n} {base}/{n}" for n in needed
    )
    msg = (
        f"Official FF++ split files missing under {splits_dir}: {missing}\n"
        f"Create the directory and download them:\n"
        f"  mkdir -p {splits_dir}\n"
        f"{cmds}\n"
        "Then re-run with --use-ff-splits."
    )
    raise FileNotFoundError(msg)


def _enumerate_ff_video_ids(args: argparse.Namespace) -> list[str]:
    from forge_detect.datasets import FaceForensicsAdapter

    target_size = (args.image_size, args.image_size) if args.image_size else None
    full = FaceForensicsAdapter(
        root=args.data_root,
        compression=args.compression,
        max_frames_per_video=1,
        target_size=target_size,
    )
    return full.video_ids()


def _build_celeb_dataset(args: argparse.Namespace, *, channel_sources: Any) -> Any:
    from forge_detect.datasets import CelebDFAdapter

    target_size = (args.image_size, args.image_size) if args.image_size else None
    frames_subdir = "frames_faces" if args.use_face_crops else "frames"
    if channel_sources is not None:
        return CelebDFAdapter(
            root=args.celeb_data_root,
            max_frames_per_video=args.frames_per_video_eval,
            target_size=target_size,
            testing_list=True,
            channel_sources=channel_sources,
            frames_subdir=frames_subdir,
        )
    return CelebDFAdapter(
        root=args.celeb_data_root,
        max_frames_per_video=args.frames_per_video_eval,
        target_size=target_size,
        testing_list=True,  # canonical 518-video benchmark
        load_physics_maps=True,  # CelebDF only has heuristic variant
        frames_subdir=frames_subdir,
    )


def _train(
    args: argparse.Namespace,
    train_ds: Any,
    val_ds: Any,
    *,
    in_channels: int,
    log_tag: str,
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
        balance_classes=not args.no_balance,
        augment_hflip=not args.no_augment,
        freeze_bn=args.freeze_bn,
        mixed_precision=args.amp,
    )

    def factory(*, pretrained: bool) -> Any:
        return build_physics_classifier(in_channels=in_channels, pretrained=pretrained)

    out = train_baseline_cnn(
        train_ds,
        val_ds,
        cfg,
        pretrained=not args.no_pretrained,
        resume_dir=run_dir,
        model_factory=factory,
        log_tag=log_tag,
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

    print(f"[eval] {label}: {len(dataset)} frames")
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
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed for the video-disjoint train/val/test split. Use the same "
        "value as the baseline_3ch pivot_study run so the comparison is "
        "fair (same train/val/test videos for both experiments).",
    )
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=0.15,
        help="Fraction of videos held out for validation. Default 0.15 matches "
        "pivot_study.py.",
    )
    parser.add_argument(
        "--test-fraction",
        type=float,
        default=0.15,
        help="Fraction of videos held out for test. Default 0.15 matches "
        "pivot_study.py.",
    )
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--learning-rate",
        "--lr",
        type=float,
        default=1.0e-4,
        dest="learning_rate",
        help="Default 1e-4 (standard fine-tuning rate for pretrained "
        "EfficientNet-B0). Both --learning-rate and --lr accepted (the "
        "shorter form matches pivot_study.py for parity in shell scripts).",
    )
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument(
        "--amp",
        action="store_true",
        help="Enable fp16 autocast. Off by default — tight fine-tuning "
        "gradients caused cuDNN execution errors with AMP enabled.",
    )
    parser.add_argument(
        "--no-balance",
        action="store_true",
        help="Disable WeightedRandomSampler. WRS is on by default — without "
        "it the natural FF++ ~20/80 real:fake imbalance makes BCE-on-logits "
        "collapse to majority-class predictions. Use only for ablations.",
    )
    parser.add_argument(
        "--no-augment",
        action="store_true",
        help="Disable random horizontal flip augmentation. Augmentation is on "
        "by default and works on all 6 channels uniformly — the spatial flip "
        "preserves RGB / physics-map alignment.",
    )
    parser.add_argument(
        "--freeze-bn",
        action="store_true",
        help="Force BatchNorm layers into eval mode during training. Off by "
        "default — kept as ablation flag (see pivot_study.py for the "
        "history).",
    )
    parser.add_argument(
        "--use-ff-splits",
        action="store_true",
        help="Use the official FF++ splits/{train,val,test}.json files. "
        "Strongly recommended — random splits over compound fake video_ids "
        "leak source identities between train and val. Pass the same flag to "
        "pivot_study.py for the baseline_3ch run so both experiments use the "
        "same partition.",
    )
    parser.add_argument(
        "--use-face-crops",
        action="store_true",
        help="Read face-cropped frames from frames_faces/ instead of full "
        "frames. Required for the published-recipe FF++ EfficientNet-B0 "
        "training regime; full frames at this scale plateau at AUROC ~0.5. "
        "Run scripts/extract_faces.py and scripts/cache_physics_maps.py "
        "--frames-subdir frames_faces first.",
    )
    parser.add_argument(
        "--channels",
        type=str,
        default=None,
        help="Comma-separated list of input-channel families (Phase 3+). "
        "Tokens (case-insensitive): 'rgb' (implicit base, optional), "
        "'physics' or 'physics:<variant>', 'frequency' or "
        "'frequency:<variant>'. Examples: 'rgb,physics,frequency' for the "
        "9-channel Phase-3 configuration, 'rgb,physics' for the Phase-2 "
        "6-channel configuration. When omitted, falls back to the legacy "
        "physics-only path driven by --variant (Phase-2 default). The "
        "frequency cache must already exist — run "
        "scripts/cache_frequency_maps.py before training.",
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

    # Resolve channel sources up front so train + eval share the same spec.
    if args.channels is not None:
        from forge_detect.datasets import parse_channel_spec, total_channels

        channel_sources = parse_channel_spec(
            args.channels, physics_variant=args.variant,
        )
        in_channels = total_channels(channel_sources)
        log_tag = f"physics-{in_channels}ch-{args.channels.replace(',', '+')}"
    else:
        channel_sources = None
        in_channels = 6
        log_tag = f"physics-6ch-{args.variant}"

    print(f"[{log_tag}] runs_dir = {args.runs_dir}")
    print(f"[{log_tag}] variant = {args.variant}")
    print(f"[{log_tag}] in_channels = {in_channels}")
    if channel_sources is not None:
        print(
            f"[{log_tag}] channel sources: "
            f"{[s.name for s in channel_sources]}",
        )

    print(f"[{log_tag}] building FF++ datasets ...")
    # All three splits use the configured variant. For the FF++-only protocol
    # this avoids the train/test distribution shift that would arise if train
    # used GT masks but val/test fell back to heuristic.
    if args.use_ff_splits:
        _check_ff_splits_present(args.data_root)
        print(f"[{log_tag}] using official FF++ splits/<name>.json")
        train_ds = _build_ff_dataset(
            args, ff_split="train", physics_variant=args.variant,
            frames_per_video=args.frames_per_video_train,
            channel_sources=channel_sources,
        )
        val_ds = _build_ff_dataset(
            args, ff_split="val", physics_variant=args.variant,
            frames_per_video=args.frames_per_video_eval,
            channel_sources=channel_sources,
        )
        test_ds = _build_ff_dataset(
            args, ff_split="test", physics_variant=args.variant,
            frames_per_video=args.frames_per_video_eval,
            channel_sources=channel_sources,
        )
    else:
        from forge_detect.datasets import split_videos

        print(f"[{log_tag}] enumerating videos for the disjoint split ...")
        all_video_ids = _enumerate_ff_video_ids(args)
        train_vids, val_vids, test_vids = split_videos(
            all_video_ids,
            seed=args.seed,
            val_fraction=args.val_fraction,
            test_fraction=args.test_fraction,
        )
        print(
            f"[{log_tag}] split: train={len(train_vids)} val={len(val_vids)} "
            f"test={len(test_vids)} (of {len(all_video_ids)} total videos)",
        )
        print(
            f"[{log_tag}] WARNING: random split over FF++ video_ids leaks "
            "identities between train and val (see --use-ff-splits).",
        )
        train_ds = _build_ff_dataset(
            args, subset_video_ids=train_vids, physics_variant=args.variant,
            frames_per_video=args.frames_per_video_train,
            channel_sources=channel_sources,
        )
        val_ds = _build_ff_dataset(
            args, subset_video_ids=val_vids, physics_variant=args.variant,
            frames_per_video=args.frames_per_video_eval,
            channel_sources=channel_sources,
        )
        test_ds = _build_ff_dataset(
            args, subset_video_ids=test_vids, physics_variant=args.variant,
            frames_per_video=args.frames_per_video_eval,
            channel_sources=channel_sources,
        )
    print(
        f"[{log_tag}] frames: train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}",
    )

    if not args.skip_train:
        t_train_start = time.time()
        run_dir, _train_out = _train(
            args, train_ds, val_ds, in_channels=in_channels, log_tag=log_tag,
        )
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
    model = build_physics_classifier(in_channels=in_channels, pretrained=False)
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
        celeb_ds = _build_celeb_dataset(args, channel_sources=channel_sources)
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
