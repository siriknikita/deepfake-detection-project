"""Per-method FF++ evaluation of both trained models.

Loads the 3-channel baseline (`pivot_study.py` output) and the 6-channel
physics model (`train_physics_cnn.py` output) and evaluates each on the
FF++ c23 test split, *one manipulation method at a time*. For each
method m in {Deepfakes, Face2Face, FaceSwap, NeuralTextures}, the test
set is restricted to (all real test videos) ∪ (m's test fakes only),
matching the standard per-method protocol used in published FF++
ablations.

Both models are also evaluated on the union ("Combined") for sanity —
those numbers should reproduce the run-time `report.json` to 4 decimals.

Output: a Markdown-style table that you can paste straight into the
diploma's empirical chapter.

Usage:

    python scripts/eval_per_method.py \\
        --data-root ~/data/FaceForensics++ \\
        --baseline-weights runs/baseline_3ch_faces/baseline_run/best.pt \\
        --physics-weights  runs/physics_6ch_faces_heuristic/best.pt \\
        --device cuda --use-face-crops --use-ff-splits \\
        --output runs/per_method_comparison.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

FF_METHODS: tuple[str, ...] = ("Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures")


def _build_dataset(
    args: argparse.Namespace,
    *,
    methods: tuple[str, ...],
    load_physics_maps: bool,
    channel_sources: Any | None = None,
) -> Any:
    """Build the FF++ test-split adapter restricted to ``methods``.

    When ``channel_sources`` is provided (multi-source / Phase 3+ path),
    the adapter ignores ``load_physics_maps`` and reads from each source's
    cache. When ``channel_sources`` is ``None``, the legacy
    ``load_physics_maps`` boolean controls the 3-channel vs 6-channel
    behaviour.
    """
    from forge_detect.datasets import FaceForensicsAdapter

    target_size = (args.image_size, args.image_size) if args.image_size else None
    frames_subdir = "frames_faces" if args.use_face_crops else "frames"
    common_kwargs: dict[str, Any] = {
        "root": args.data_root,
        "methods": methods,
        "compression": args.compression,
        "max_frames_per_video": args.frames_per_video,
        "target_size": target_size,
        "frames_subdir": frames_subdir,
    }
    if channel_sources is not None:
        common_kwargs["channel_sources"] = channel_sources
    else:
        common_kwargs["load_physics_maps"] = load_physics_maps
        common_kwargs["physics_variant"] = "heuristic"
    if args.use_ff_splits:
        return FaceForensicsAdapter(ff_split="test", **common_kwargs)
    # Random-split fallback: only useful for debugging — use --use-ff-splits.
    from forge_detect.datasets import split_videos

    full = FaceForensicsAdapter(
        root=args.data_root,
        methods=methods,
        compression=args.compression,
        max_frames_per_video=1,
        target_size=target_size,
        frames_subdir=frames_subdir,
    )
    _train, _val, test_vids = split_videos(
        full.video_ids(),
        seed=args.seed,
        val_fraction=args.val_fraction,
        test_fraction=args.test_fraction,
    )
    return FaceForensicsAdapter(subset_video_ids=test_vids, **common_kwargs)


def _load_model(args: argparse.Namespace, kind: str, *, in_channels: int) -> Any:
    from forge_detect.baseline_cnn import (
        build_baseline_classifier,
        build_physics_classifier,
    )
    from forge_detect.cnn import load_weights

    if kind == "baseline":
        weights = args.baseline_weights
        model = build_baseline_classifier(pretrained=False)
    elif kind == "physics":
        weights = args.physics_weights
        model = build_physics_classifier(in_channels=in_channels, pretrained=False)
    else:
        msg = f"unknown kind {kind!r}"
        raise ValueError(msg)
    # `argparse(type=Path)` turns an empty string into Path(".") which is
    # truthy and exists() == True. Use is_file() so the documented "set to ''
    # to skip" behaviour actually works for both empty strings and bogus paths.
    if not weights or not Path(weights).is_file():
        return None
    load_weights(model, weights)
    return model


def _evaluate(
    model: Any,
    dataset: Any,
    args: argparse.Namespace,
) -> dict[str, float]:
    from forge_detect.baseline_cnn import evaluate_baseline_cnn

    return evaluate_baseline_cnn(
        model,
        dataset,
        device=args.device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )


def _print_metric_table(
    title: str,
    key: str,
    results: dict[str, dict[str, dict[str, float] | None]],
) -> None:
    """Print one metric across all rows (per-method + Combined) of `results`."""
    print()
    print(title)
    print(f"{'method':<18} {'baseline':>10} {'physics':>10} {'delta':>10}")
    print("-" * 50)
    for m, mdict in results.items():
        bl = mdict.get("baseline") or {}
        ph = mdict.get("physics") or {}
        bl_v = bl.get(key)
        ph_v = ph.get(key)
        bl_str = f"{bl_v:.4f}" if isinstance(bl_v, (int, float)) else "  n/a "
        ph_str = f"{ph_v:.4f}" if isinstance(ph_v, (int, float)) else "  n/a "
        delta_str = (
            f"{(ph_v - bl_v):+.4f}"
            if isinstance(bl_v, (int, float)) and isinstance(ph_v, (int, float))
            else "  n/a "
        )
        print(f"{m:<18} {bl_str:>10} {ph_str:>10} {delta_str:>10}")


def main() -> int:  # noqa: PLR0912
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument(
        "--baseline-weights",
        type=Path,
        default=Path("runs/baseline_3ch_faces/baseline_run/best.pt"),
        help="Path to the 3-channel RGB baseline best.pt. Set to '' to skip.",
    )
    parser.add_argument(
        "--physics-weights",
        type=Path,
        default=Path("runs/physics_6ch_faces_heuristic/best.pt"),
        help="Path to the 6-channel physics model best.pt. Set to '' to skip.",
    )
    parser.add_argument("--compression", choices=("raw", "c23", "c40"), default="c23")
    parser.add_argument("--frames-per-video", type=int, default=10)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--use-face-crops", action="store_true")
    parser.add_argument("--use-ff-splits", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--test-fraction", type=float, default=0.15)
    parser.add_argument("--device", choices=("cpu", "cuda", "mps"), default="cuda")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--methods",
        type=str,
        default=",".join(FF_METHODS),
        help="Comma-separated subset of FF++ methods. Default: all four.",
    )
    parser.add_argument(
        "--channels",
        type=str,
        default=None,
        help="Channel spec for the physics model (Phase 3+). Must match "
        "the spec used at training time. Tokens: 'rgb', 'physics', "
        "'physics:<variant>', 'frequency', 'frequency:<variant>'. When "
        "omitted, the physics model is treated as the legacy 6-channel "
        "Phase-2 build (RGB + heuristic physics).",
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    methods = tuple(m.strip() for m in args.methods.split(",") if m.strip())
    invalid = [m for m in methods if m not in FF_METHODS]
    if invalid:
        msg = f"unknown methods: {invalid}; pick from {FF_METHODS}"
        raise ValueError(msg)

    if args.channels is not None:
        from forge_detect.datasets import parse_channel_spec, total_channels

        channel_sources: Any = parse_channel_spec(args.channels)
        physics_in_channels = total_channels(channel_sources)
    else:
        channel_sources = None
        physics_in_channels = 6

    print(f"[per-method] data_root      = {args.data_root}")
    print(f"[per-method] baseline best  = {args.baseline_weights}")
    print(f"[per-method] physics  best  = {args.physics_weights}")
    print(f"[per-method] frames_subdir  = {'frames_faces' if args.use_face_crops else 'frames'}")
    split_label = "official FF++" if args.use_ff_splits else "random video-disjoint"
    print(f"[per-method] split source   = {split_label}")
    print(f"[per-method] methods        = {methods}")
    print(f"[per-method] physics in_channels = {physics_in_channels}")
    if channel_sources is not None:
        print(
            f"[per-method] channel sources: "
            f"{[s.name for s in channel_sources]}",
        )

    bl_model = _load_model(args, "baseline", in_channels=3)
    ph_model = _load_model(args, "physics", in_channels=physics_in_channels)
    if bl_model is None and ph_model is None:
        print("[per-method] ERROR: neither weights file found; nothing to evaluate")
        return 1

    results: dict[str, dict[str, dict[str, float] | None]] = {}

    # Per-method
    for m in methods:
        print()
        print(f"[per-method] === {m} ===")
        # 3ch baseline does not need physics/frequency maps
        if bl_model is not None:
            ds = _build_dataset(args, methods=(m,), load_physics_maps=False)
            print(f"  baseline test set: {len(ds)} frames")
            bl_metrics = _evaluate(bl_model, ds, args)
        else:
            bl_metrics = None
        if ph_model is not None:
            ds = _build_dataset(
                args,
                methods=(m,),
                load_physics_maps=channel_sources is None,
                channel_sources=channel_sources,
            )
            print(f"  physics  test set: {len(ds)} frames")
            ph_metrics = _evaluate(ph_model, ds, args)
        else:
            ph_metrics = None
        results[m] = {"baseline": bl_metrics, "physics": ph_metrics}
        if bl_metrics:
            print(
                f"  baseline: frame={bl_metrics.get('auroc'):.4f} "
                f"video_mean={bl_metrics.get('video_auroc_mean'):.4f} "
                f"video_max={bl_metrics.get('video_auroc_max'):.4f}",
            )
        if ph_metrics:
            print(
                f"  physics : frame={ph_metrics.get('auroc'):.4f} "
                f"video_mean={ph_metrics.get('video_auroc_mean'):.4f} "
                f"video_max={ph_metrics.get('video_auroc_max'):.4f}",
            )

    # Combined (all 4 methods together) for sanity-check vs report.json
    print()
    print("[per-method] === Combined (all methods) ===")
    if bl_model is not None:
        ds = _build_dataset(args, methods=tuple(FF_METHODS), load_physics_maps=False)
        print(f"  baseline test set: {len(ds)} frames")
        bl_combined = _evaluate(bl_model, ds, args)
    else:
        bl_combined = None
    if ph_model is not None:
        ds = _build_dataset(
            args,
            methods=tuple(FF_METHODS),
            load_physics_maps=channel_sources is None,
            channel_sources=channel_sources,
        )
        print(f"  physics  test set: {len(ds)} frames")
        ph_combined = _evaluate(ph_model, ds, args)
    else:
        ph_combined = None
    results["Combined"] = {"baseline": bl_combined, "physics": ph_combined}

    # Print summary table
    print()
    print("=" * 88)
    print("FF++ c23 per-method test-set comparison: baseline_3ch vs physics_6ch")
    print("=" * 88)

    _print_metric_table("frame AUROC",             "auroc",            results)
    _print_metric_table("video AUROC (mean-pool)", "video_auroc_mean", results)
    _print_metric_table("video AUROC (max-pool)",  "video_auroc_max",  results)

    print()
    print("=" * 88)

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(results, indent=2, default=str))
        print(f"[per-method] wrote {args.output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
