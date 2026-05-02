"""Visualization helpers for the impact map.

Renders the side-by-side panel that the diploma defense uses: original
image, CNN trust map, forged manifold, settled manifold, residual, and
crack overlay. Matplotlib is the only dependency and is required at
import time.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from forge_detect.pipeline import DetectResult


def crack_overlay(rgb: np.ndarray, laplacian: np.ndarray, threshold_pct: float = 95.0) -> Any:
    """Overlay |L| above its ``threshold_pct`` percentile on the RGB image.

    Args:
        rgb: ``(H, W, 3)`` float32 image in ``[0, 1]``.
        laplacian: ``(H, W)`` float32 ``Δz*`` map.
        threshold_pct: percentile of ``|L|`` above which pixels are flagged.

    Returns:
        A matplotlib ``Figure`` with a single axes showing the overlay.
    """
    import matplotlib.pyplot as plt

    abs_l = np.abs(laplacian)
    threshold = float(np.percentile(abs_l, threshold_pct))
    mask = abs_l > threshold

    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    ax.imshow(rgb)
    overlay = np.zeros((*rgb.shape[:2], 4), dtype=np.float32)
    overlay[..., 0] = 1.0  # red
    overlay[..., 3] = np.where(mask, 0.6, 0.0)
    ax.imshow(overlay)
    ax.set_title(f"Geometric cracks (|L| > p{int(threshold_pct)})")
    ax.axis("off")
    return fig


def panel(result: DetectResult, rgb: np.ndarray, w_cnn: np.ndarray) -> Any:
    """6-panel diagnostic figure: original, W_cnn, z_forged, z_star, R, L overlay."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    axes = axes.ravel()

    axes[0].imshow(rgb)
    axes[0].set_title("Input image")
    axes[0].axis("off")

    axes[1].imshow(w_cnn, cmap="viridis", vmin=0.0, vmax=1.0)
    axes[1].set_title("W_cnn (trust map)")
    axes[1].axis("off")

    axes[2].imshow(result.solve.z_forged, cmap="magma")
    axes[2].set_title("z_forged (initial manifold)")
    axes[2].axis("off")

    axes[3].imshow(result.solve.z_star, cmap="magma")
    axes[3].set_title("z* (settled manifold)")
    axes[3].axis("off")

    axes[4].imshow(result.solve.residual, cmap="seismic")
    axes[4].set_title("R = z* - z_ideal")
    axes[4].axis("off")

    abs_l = np.abs(result.solve.laplacian)
    threshold = float(np.percentile(abs_l, 95))
    axes[5].imshow(rgb)
    overlay = np.zeros((*rgb.shape[:2], 4), dtype=np.float32)
    overlay[..., 0] = 1.0
    overlay[..., 3] = np.where(abs_l > threshold, 0.6, 0.0)
    axes[5].imshow(overlay)
    axes[5].set_title("Geometric cracks")
    axes[5].axis("off")

    if result.deepfake_probability is not None:
        fig.suptitle(f"Deepfake probability: {result.deepfake_probability:.2%}")
    fig.tight_layout()
    return fig


def save_panel(result: DetectResult, rgb: np.ndarray, w_cnn: np.ndarray, path: str | Path) -> None:
    """Render :func:`panel` and save it to ``path``."""
    fig = panel(result, rgb, w_cnn)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    import matplotlib.pyplot as plt

    plt.close(fig)


__all__ = ["crack_overlay", "panel", "save_panel"]
