"""Cross-dataset eval of FF++-trained models on Celeb-DF v2.

Loads a trained ``best.pt`` checkpoint (either the 3-channel baseline
from ``pivot_study.py`` or the 6-channel physics model from
``train_physics_cnn.py``) and evaluates it on the published Celeb-DF v2
518-video testing list. Reports frame and video AUROC under both
mean-pool and max-pool, plus accuracies and class counts. Stand-alone
script — does not require the FF++ data to be present on the eval
machine, which is what the existing training scripts assume.

Pre-requisites on the eval machine:

    - Celeb-DF v2 with face-cropped frames at <celeb-root>/<subset>/frames_faces/
      (run scripts/extract_faces.py --dataset celeb-df ... first)
    - For the 6-channel model: physics maps cached under
      <celeb-root>/<subset>/physics_faces_heuristic/ (run
      scripts/cache_physics_maps.py --dataset celeb-df --frames-subdir
      frames_faces ...)
    - The trained model weights (best.pt). 20 MB; scp from the training
      cluster.

Usage:

    # Evaluate the 3-channel RGB baseline:
    python scripts/eval_celebdf.py \\
        --celeb-data-root ~/Development/personal-projects/datasets/Celeb-DF-v2 \\
        --weights runs/baseline_3ch_faces/baseline_run/best.pt \\
        --model baseline --device cuda --use-face-crops \\
        --output runs/baseline_3ch_faces/celeb_test.json

    # Evaluate the 6-channel physics model (requires physics maps cached):
    python scripts/eval_celebdf.py \\
        --celeb-data-root ~/Development/personal-projects/datasets/Celeb-DF-v2 \\
        --weights runs/physics_6ch_faces_heuristic/best.pt \\
        --model physics --device cuda --use-face-crops \\
        --output runs/physics_6ch_faces_heuristic/celeb_test.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _build_dataset(
    args: argparse.Namespace,
    *,
    load_physics_maps: bool,
    channel_sources: Any | None = None,
) -> Any:
    from forge_detect.datasets import CelebDFAdapter

    target_size = (args.image_size, args.image_size) if args.image_size else None
    frames_subdir = "frames_faces" if args.use_face_crops else "frames"
    if channel_sources is not None:
        return CelebDFAdapter(
            root=args.celeb_data_root,
            max_frames_per_video=args.frames_per_video,
            target_size=target_size,
            testing_list=True,
            channel_sources=channel_sources,
            frames_subdir=frames_subdir,
        )
    return CelebDFAdapter(
        root=args.celeb_data_root,
        max_frames_per_video=args.frames_per_video,
        target_size=target_size,
        testing_list=True,
        load_physics_maps=load_physics_maps,
        frames_subdir=frames_subdir,
    )


def _build_model(args: argparse.Namespace, *, in_channels: int) -> Any:
    from forge_detect.baseline_cnn import (
        build_baseline_classifier,
        build_physics_classifier,
    )
    from forge_detect.cnn import load_weights

    if args.model == "baseline":
        model = build_baseline_classifier(pretrained=False)
    elif args.model == "physics":
        model = build_physics_classifier(in_channels=in_channels, pretrained=False)
    else:
        msg = f"--model must be 'baseline' or 'physics', got {args.model!r}"
        raise ValueError(msg)
    load_weights(model, args.weights)
    return model


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--celeb-data-root", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument(
        "--model",
        choices=("baseline", "physics"),
        required=True,
        help="'baseline' for the 3-channel RGB classifier, 'physics' for "
        "the 6-channel physics-tensor classifier.",
    )
    parser.add_argument(
        "--use-face-crops",
        action="store_true",
        help="Read frames from <subset>/frames_faces/ instead of "
        "<subset>/frames/. Should match how the model was trained.",
    )
    parser.add_argument("--frames-per-video", type=int, default=10)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda", "mps"),
        default="cuda",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Where to write the JSON report. Default: print only.",
    )
    parser.add_argument(
        "--channels",
        type=str,
        default=None,
        help="Channel spec for --model physics (Phase 3+). Must match the "
        "spec used at training time. Tokens: 'rgb', 'physics', "
        "'physics:<variant>', 'frequency', 'frequency:<variant>'. When "
        "omitted with --model physics, the legacy 6-channel Phase-2 build "
        "is assumed. Ignored for --model baseline.",
    )
    args = parser.parse_args()

    if not args.weights.exists():
        msg = f"--weights path does not exist: {args.weights}"
        raise FileNotFoundError(msg)
    if not args.celeb_data_root.exists():
        msg = f"--celeb-data-root does not exist: {args.celeb_data_root}"
        raise FileNotFoundError(msg)

    needs_physics = args.model == "physics"

    if needs_physics and args.channels is not None:
        from forge_detect.datasets import parse_channel_spec, total_channels

        channel_sources: Any = parse_channel_spec(args.channels)
        in_channels = total_channels(channel_sources)
    elif needs_physics:
        channel_sources = None
        in_channels = 6
    else:
        channel_sources = None
        in_channels = 3

    print(f"[eval-celeb] model = {args.model}")
    print(f"[eval-celeb] weights = {args.weights}")
    print(f"[eval-celeb] celeb_data_root = {args.celeb_data_root}")
    print(
        f"[eval-celeb] frames_subdir = "
        f"{'frames_faces' if args.use_face_crops else 'frames'}",
    )
    print(f"[eval-celeb] in_channels = {in_channels}")
    if channel_sources is not None:
        print(
            f"[eval-celeb] channel sources: "
            f"{[s.name for s in channel_sources]}",
        )
    elif needs_physics:
        print("[eval-celeb] dataset will load physics maps (heuristic variant)")

    print("[eval-celeb] building dataset (Celeb-DF testing list, 518 videos) ...")
    ds = _build_dataset(
        args, load_physics_maps=needs_physics, channel_sources=channel_sources,
    )
    print(f"[eval-celeb] dataset: {len(ds)} frames across {len(ds.video_ids())} videos")

    print("[eval-celeb] building model and loading weights ...")
    model = _build_model(args, in_channels=in_channels)

    print(f"[eval-celeb] evaluating on {args.device} ...")
    from forge_detect.baseline_cnn import evaluate_baseline_cnn

    metrics = evaluate_baseline_cnn(
        model,
        ds,
        device=args.device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    print()
    print("=== Celeb-DF v2 cross-dataset evaluation ===")
    print(f"  weights:                {args.weights}")
    print(f"  model:                  {args.model}")
    print(f"  frames evaluated:       {int(metrics.get('n_real', 0) + metrics.get('n_fake', 0))}")
    print(f"  frame AUROC:            {metrics.get('auroc', float('nan')):.4f}")
    print(f"  frame accuracy:         {metrics.get('accuracy', float('nan')):.4f}")
    if "video_auroc_mean" in metrics:
        print(f"  video AUROC (mean):     {metrics['video_auroc_mean']:.4f}")
        print(f"  video AUROC (max):      {metrics['video_auroc_max']:.4f}")
        print(f"  video accuracy (mean):  {metrics['video_accuracy_mean']:.4f}")
        print(
            f"  test split: "
            f"{int(metrics.get('n_video_real', 0))} real videos, "
            f"{int(metrics.get('n_video_fake', 0))} fake videos",
        )

    if args.output is not None:
        payload = {
            "model": args.model,
            "weights": str(args.weights),
            "celeb_data_root": str(args.celeb_data_root),
            "use_face_crops": args.use_face_crops,
            **metrics,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2, default=str))
        print()
        print(f"[eval-celeb] wrote {args.output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
