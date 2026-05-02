"""Single-image Hyperplane-Forge demo.

Usage:
    uv run --python .venv/bin/python examples/detect_demo.py <image_path> [--visualize]

Loads an image, runs the full Phase 1 → Phase 6 pipeline on the CPU
backend, prints the energy decomposition and feature summary, and
optionally writes a 6-panel diagnostic figure to disk.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from forge_detect.config import PdeParams, PipelineParams
from forge_detect.pipeline import detect, load_image
from forge_detect.trust_map import heuristic_trust_map


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("image", type=Path, help="Path to an RGB image (PNG/JPEG/...).")
    parser.add_argument(
        "--visualize",
        type=Path,
        default=None,
        metavar="PATH",
        help="If given, render the 6-panel figure and save it to PATH.",
    )
    parser.add_argument(
        "--n-scales",
        type=int,
        default=4,
        help="Number of DoG bands (default: 4).",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=300,
        help="Jacobi PDE iteration cap (default: 300).",
    )
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    if not args.image.exists():
        print(f"image not found: {args.image}", file=sys.stderr)
        return 2

    params = PipelineParams(
        n_scales=args.n_scales,
        pde=PdeParams(max_iter=args.max_iter),
    )
    print(f"Loading {args.image} ...")
    rgb = load_image(args.image)
    print(f"Image:       {rgb.shape[:2][0]} x {rgb.shape[:2][1]} pixels")

    print("Computing heuristic trust map ...")
    w_cnn = heuristic_trust_map(rgb)
    trust_low = float((w_cnn < 0.5).mean()) * 100.0
    print(f"Low-trust pixels: {trust_low:.1f} %")

    print("Running settlement pipeline ...")
    result = detect(rgb, params=params, trust_map=w_cnn)

    print("\n--- Settled manifold ---")
    print(f"Iterations:       {result.solve.iterations}")
    print(f"Converged:        {result.solve.converged}")
    print(f"Energy total:     {result.solve.energy_total:.4f}")
    print(f"  data:           {result.solve.energy_data:.4f}")
    print(f"  smoothness:     {result.solve.energy_smoothness:.4f}")
    print(f"  consistency:    {result.solve.energy_consistency:.4f}")

    print("\n--- Impact map ---")
    abs_l = result.solve.laplacian
    abs_l_max = float(abs(abs_l).max())
    r_max = float(abs(result.solve.residual).max())
    print(f"max |R|:          {r_max:.4f}")
    print(f"max |L|:          {abs_l_max:.4f}")

    if args.visualize is not None:
        from forge_detect.viz import save_panel

        print(f"\nWriting panel to {args.visualize} ...")
        save_panel(result, rgb, w_cnn, args.visualize)
        print("done.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
