"""Command-line interface: ``python -m forge_detect detect <image>``."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from forge_detect.config import PdeParams, PipelineParams
from forge_detect.pipeline import detect


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="forge-detect",
        description="Hyperplane-Forge: deepfake detection via physical-manifold settlement.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    detect_p = sub.add_parser("detect", help="Run end-to-end detection on an image.")
    detect_p.add_argument("image", type=Path, help="Path to the image file (PNG/JPEG/...).")
    detect_p.add_argument(
        "--device",
        choices=("cpu", "cuda"),
        default="cpu",
        help="Backend selection (default: cpu).",
    )
    detect_p.add_argument(
        "--n-scales",
        type=int,
        default=PipelineParams().n_scales,
        help="Number of DoG bands (default: %(default)s).",
    )
    detect_p.add_argument(
        "--max-iter",
        type=int,
        default=PdeParams().max_iter,
        help="Jacobi PDE iteration cap (default: %(default)s).",
    )
    detect_p.add_argument(
        "--visualize",
        type=Path,
        default=None,
        metavar="PATH",
        help="If given, save the 6-panel diagnostic figure to PATH (PNG).",
    )
    detect_p.add_argument(
        "--print-features",
        action="store_true",
        help="Print every feature name and value to stdout.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command != "detect":
        parser.error(f"unsupported command: {args.command}")

    params = PipelineParams(
        n_scales=args.n_scales,
        pde=PdeParams(max_iter=args.max_iter),
    )
    result = detect(args.image, device=args.device, params=params)

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


if __name__ == "__main__":
    sys.exit(main())
