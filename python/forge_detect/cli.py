"""Command-line interface.

Subcommands:
- ``forge-detect detect <image>`` — run the pipeline on a single image.
- ``forge-detect train --data-root <path>`` — train ChromaticEfficientNet.
- ``forge-detect eval --data-root <path>`` — train + evaluate the binary
  classifier on a labeled dataset; reports AUROC and feature importances.
- ``forge-detect bench --data-root <path> --out features.csv`` — run the
  pipeline over a dataset and dump the feature matrix to disk.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from forge_detect.config import PdeParams, PipelineParams
from forge_detect.pipeline import detect

# ----------------- detect ----------------- #


def _add_detect_subparser(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    p = sub.add_parser("detect", help="Run end-to-end detection on an image.")
    p.add_argument("image", type=Path, help="Path to the image file (PNG/JPEG/...).")
    p.add_argument(
        "--device",
        choices=("cpu", "cuda"),
        default="cpu",
        help="Backend selection (default: cpu).",
    )
    p.add_argument(
        "--cnn-checkpoint",
        type=Path,
        default=None,
        help="Path to a trained ChromaticEfficientNet .pt; uses the heuristic if omitted.",
    )
    p.add_argument(
        "--cnn-device",
        choices=("auto", "cpu", "cuda", "mps"),
        default="auto",
        help="Device for the CNN forward pass (default: auto).",
    )
    p.add_argument(
        "--n-scales",
        type=int,
        default=PipelineParams().n_scales,
        help="Number of DoG bands (default: %(default)s).",
    )
    p.add_argument(
        "--max-iter",
        type=int,
        default=PdeParams().max_iter,
        help="Jacobi PDE iteration cap (default: %(default)s).",
    )
    p.add_argument(
        "--visualize",
        type=Path,
        default=None,
        metavar="PATH",
        help="If given, save the 6-panel diagnostic figure to PATH (PNG).",
    )
    p.add_argument(
        "--print-features",
        action="store_true",
        help="Print every feature name and value to stdout.",
    )


def _load_cnn_if_given(args: argparse.Namespace) -> tuple[Any | None, str]:
    """Load a CNN checkpoint (if `--cnn-checkpoint`) and return (model, device)."""
    ckpt: Path | None = getattr(args, "cnn_checkpoint", None)
    if ckpt is None:
        return None, "cpu"
    from forge_detect.cnn import build_chromatic_efficientnet, load_weights

    cnn_device = _resolve_cnn_device(getattr(args, "cnn_device", "auto"))
    model = build_chromatic_efficientnet(pretrained=False)
    load_weights(model, ckpt)
    model.eval()
    return model.to(cnn_device), cnn_device


def _resolve_cnn_device(prefer: str) -> str:
    if prefer != "auto":
        return prefer
    try:
        import torch
    except ImportError:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _run_detect(args: argparse.Namespace) -> int:
    params = PipelineParams(
        n_scales=args.n_scales,
        pde=PdeParams(max_iter=args.max_iter),
    )
    cnn_model, cnn_device = _load_cnn_if_given(args)
    result = detect(
        args.image,
        device=args.device,
        params=params,
        cnn_model=cnn_model,
        cnn_device=cnn_device,
    )
    print(f"Image:        {result.image_path}")
    print(f"Iterations:   {result.solve.iterations}")
    print(f"Converged:    {result.solve.converged}")
    print(f"Energy:       total={result.solve.energy_total:.4f}")
    print(f"              data={result.solve.energy_data:.4f}")
    print(f"              smooth={result.solve.energy_smoothness:.4f}")
    print(f"              cons={result.solve.energy_consistency:.4f}")
    if result.deepfake_probability is not None:
        print(f"Deepfake prob: {result.deepfake_probability:.4f}")
    if args.print_features:
        print("\nFeatures:")
        for name, value in zip(result.feature_names, result.features, strict=True):
            print(f"  {name:30s} {value:.6f}")
    if args.visualize is not None:
        from forge_detect.pipeline import load_image
        from forge_detect.trust_map import heuristic_trust_map
        from forge_detect.viz import save_panel

        rgb = load_image(args.image)
        w_cnn = heuristic_trust_map(rgb)
        save_panel(result, rgb, w_cnn, args.visualize)
        print(f"Saved panel to {args.visualize}")
    return 0


# ----------------- dataset helper ----------------- #


def _build_dataset(args: argparse.Namespace) -> object:
    """Build a torch Dataset from `--data-root` and the format flag."""
    from forge_detect.datasets import FaceForensicsAdapter, ImageFolderDataset

    root = args.data_root
    target_size = (args.image_size, args.image_size) if args.image_size else None
    if args.dataset == "image-folder":
        if not args.fake_dir:
            msg = "--fake-dir is required for --dataset=image-folder"
            raise SystemExit(msg)
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


def _add_dataset_args(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--data-root",
        type=Path,
        required=True,
        help=(
            "FaceForensics++ root for --dataset=face-forensics, or the real-frames "
            "directory for --dataset=image-folder."
        ),
    )
    p.add_argument(
        "--dataset",
        choices=("face-forensics", "image-folder"),
        default="face-forensics",
        help="Dataset adapter to use (default: face-forensics).",
    )
    p.add_argument(
        "--fake-dir",
        type=Path,
        default=None,
        help="Fake-frames directory (image-folder only).",
    )
    p.add_argument(
        "--compression",
        choices=("raw", "c23", "c40"),
        default="c23",
        help="FF++ compression level (default: c23).",
    )
    p.add_argument(
        "--max-frames-per-video",
        type=int,
        default=None,
        help="FF++ stride-sample cap per video.",
    )
    p.add_argument(
        "--image-size",
        type=int,
        default=256,
        help="Resize every image to this size (default: 256).",
    )


# ----------------- train ----------------- #


def _add_train_subparser(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    p = sub.add_parser("train", help="Train the trust-map CNN.")
    _add_dataset_args(p)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--learning-rate", type=float, default=1.0e-3)
    p.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    p.add_argument("--checkpoint-dir", type=Path, default=Path("runs"))
    p.add_argument("--no-pretrained", action="store_true", help="Skip ImageNet weights.")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument(
        "--resume",
        type=Path,
        default=None,
        metavar="RUN_DIR",
        help=(
            "Resume training from RUN_DIR/checkpoint.pt. The model, optimizer, "
            "scheduler, scaler, and epoch are all restored. Use this after a "
            "crash / preemption to pick up where the run left off."
        ),
    )


def _run_train(args: argparse.Namespace) -> int:
    from torch.utils.data import Subset

    from forge_detect.datasets import stratified_split
    from forge_detect.train import TrainingConfig, train_cnn

    full = _build_dataset(args)
    train_idx, val_idx, _ = stratified_split(
        len(full),  # type: ignore[arg-type]
        seed=0,
        val_fraction=0.1,
        test_fraction=0.0,
    )
    train_ds = Subset(full, train_idx)  # type: ignore[arg-type]
    val_ds = Subset(full, val_idx)  # type: ignore[arg-type]
    cfg = TrainingConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        device=args.device,
        checkpoint_dir=args.checkpoint_dir,
        num_workers=args.num_workers,
    )
    out = train_cnn(
        train_ds,
        val_ds,
        cfg,
        pretrained=not args.no_pretrained,
        resume_dir=args.resume,
    )
    print(f"Training complete. Run dir: {out['run_dir']}")
    return 0


# ----------------- eval ----------------- #


def _add_eval_subparser(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    p = sub.add_parser("eval", help="Train the binary classifier and report AUROC.")
    _add_dataset_args(p)
    p.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    p.add_argument(
        "--cnn-checkpoint",
        type=Path,
        default=None,
        help="Path to a trained ChromaticEfficientNet .pt; heuristic trust map if omitted.",
    )
    p.add_argument(
        "--cnn-device",
        choices=("auto", "cpu", "cuda", "mps"),
        default="auto",
    )
    p.add_argument(
        "--cache",
        type=Path,
        default=None,
        help="Cache features here (loaded if exists, written otherwise).",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Save the trained classifier (pickle).",
    )
    p.add_argument("--max-iter", type=int, default=PdeParams().max_iter)
    p.add_argument("--n-scales", type=int, default=PipelineParams().n_scales)


def _run_eval(args: argparse.Namespace) -> int:
    from forge_detect.classifier import save_classifier
    from forge_detect.eval import evaluate_pipeline

    dataset = _build_dataset(args)
    params = PipelineParams(n_scales=args.n_scales, pde=PdeParams(max_iter=args.max_iter))
    cnn_model, cnn_device = _load_cnn_if_given(args)
    val_m, test_m, classifier, _fm = evaluate_pipeline(
        dataset,
        device=args.device,
        params=params,
        cache_path=args.cache,
        cnn_model=cnn_model,
        cnn_device=cnn_device,
    )
    print("\n--- Validation ---")
    print(f"  AUROC:    {val_m.auroc:.4f}")
    print(f"  Accuracy: {val_m.accuracy:.4f}")
    print(f"  Real / Fake: {val_m.n_real} / {val_m.n_fake}")
    print("\n--- Test ---")
    print(f"  AUROC:    {test_m.auroc:.4f}")
    print(f"  Accuracy: {test_m.accuracy:.4f}")
    print(f"  Real / Fake: {test_m.n_real} / {test_m.n_fake}")
    print("\nTop 10 features by importance:")
    top = sorted(test_m.feature_importances.items(), key=lambda kv: -kv[1])[:10]
    for name, importance in top:
        print(f"  {name:30s} {importance:.4f}")
    if args.output is not None:
        save_classifier(classifier, args.output)
        print(f"\nSaved classifier to {args.output}")
    return 0


# ----------------- bench ----------------- #


def _add_bench_subparser(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    p = sub.add_parser("bench", help="Extract features over a dataset and write CSV.")
    _add_dataset_args(p)
    p.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    p.add_argument("--cnn-checkpoint", type=Path, default=None)
    p.add_argument("--cnn-device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    p.add_argument("--out", type=Path, required=True, help="Output CSV path.")
    p.add_argument("--max-iter", type=int, default=PdeParams().max_iter)
    p.add_argument("--n-scales", type=int, default=PipelineParams().n_scales)
    p.add_argument("--summary", type=Path, default=None, help="Optional JSON summary path.")


def _run_bench(args: argparse.Namespace) -> int:
    from forge_detect.eval import extract_features_over_dataset

    dataset = _build_dataset(args)
    params = PipelineParams(n_scales=args.n_scales, pde=PdeParams(max_iter=args.max_iter))
    cnn_model, cnn_device = _load_cnn_if_given(args)
    fm = extract_features_over_dataset(
        dataset,
        device=args.device,
        params=params,
        cnn_model=cnn_model,
        cnn_device=cnn_device,
    )
    fm.save(args.out)
    print(f"Saved {fm.features.shape[0]} feature rows to {args.out}")
    if args.summary is not None:
        summary = {
            "n_records": fm.features.shape[0],
            "n_real": int((fm.labels == 0).sum()),
            "n_fake": int((fm.labels == 1).sum()),
            "feature_means": dict(
                zip(
                    [str(c) for c in range(fm.features.shape[1])],
                    fm.features.mean(axis=0).tolist(),
                    strict=True,
                ),
            ),
        }
        args.summary.write_text(json.dumps(summary, indent=2))
        print(f"Wrote summary to {args.summary}")
    return 0


# ----------------- argparse plumbing ----------------- #


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="forge-detect",
        description="Hyperplane-Forge: deepfake detection via physical-manifold settlement.",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    _add_detect_subparser(sub)
    _add_train_subparser(sub)
    _add_eval_subparser(sub)
    _add_bench_subparser(sub)
    return parser


_DISPATCH = {
    "detect": _run_detect,
    "train": _run_train,
    "eval": _run_eval,
    "bench": _run_bench,
}


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    handler = _DISPATCH.get(args.command)
    if handler is None:
        parser.error(f"unsupported command: {args.command}")
    return handler(args)


if __name__ == "__main__":
    sys.exit(main())
