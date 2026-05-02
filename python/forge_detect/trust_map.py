"""Trust map producers (`W_cnn` for the Phase 4 energy functional).

Two implementations live here:

- :func:`heuristic_trust_map` — a deterministic, training-free fallback
  that computes a per-pixel chromatic-residual map (local color variance,
  scaled by the absolute deviation from a smoothed reference). This makes
  the pipeline runnable end-to-end on any image without a trained CNN. The
  values land in ``[0, 1]`` with high values flagging "trusted" pixels (low
  chromatic anomaly) and low values flagging suspect pixels.

- :class:`forge_detect.cnn.ChromaticTrustNet` — the trained-CNN path. The
  scaffolded architecture is in :mod:`forge_detect.cnn`; weights and a
  training loop are tracked as future work.
"""

from __future__ import annotations

import numpy as np


def heuristic_trust_map(rgb: np.ndarray, sigma: float = 2.0, gain: float = 8.0) -> np.ndarray:
    """Build a chromatic-residual trust map from an RGB image.

    The heuristic flags as low-trust the pixels whose color differs strongly
    from a low-pass-filtered reference of themselves. Real face pixels under
    natural illumination produce smoothly varying color; GAN/diffusion
    artefacts often introduce chromatic high-frequency components inconsistent
    with the local smooth region. The map is therefore

        d(x, y) = ‖I(x, y) − G_σ * I (x, y)‖₂
        W_cnn(x, y) = exp(−gain · d(x, y))                    (clipped to [0, 1])

    Args:
        rgb: ``(H, W, 3)`` float32 image in ``[0, 1]``.
        sigma: standard deviation of the Gaussian smoothing kernel that
            defines the chromatic reference.
        gain: rate at which trust falls off with chromatic deviation.

    Returns:
        ``(H, W)`` float32 trust map.
    """
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        msg = f"rgb must be (H, W, 3), got {rgb.shape}"
        raise ValueError(msg)
    rgb = rgb.astype(np.float32, copy=False)

    smoothed = np.stack(
        [_gaussian_blur_2d(rgb[..., c], sigma) for c in range(3)],
        axis=-1,
    )
    diff = rgb - smoothed
    dist = np.sqrt(np.sum(diff * diff, axis=-1))  # per-pixel ‖ΔRGB‖
    trust = np.exp(-gain * dist)
    return trust.clip(0.0, 1.0).astype(np.float32, copy=False)


def _gaussian_blur_2d(arr: np.ndarray, sigma: float) -> np.ndarray:
    """Separable 2D Gaussian blur with reflect-mirror boundaries.

    Implemented in pure NumPy so this module has no dependency on the Rust
    extension or on SciPy. Performance is adequate for trust-map preview;
    production runs on the GPU cluster use the PyTorch path in
    :class:`forge_detect.backends.cuda.CudaBackend` instead.
    """
    if sigma <= 0:
        return arr
    radius = int(np.ceil(3.0 * sigma))
    xs = np.arange(-radius, radius + 1, dtype=np.float32)
    kernel = np.exp(-(xs * xs) / (2.0 * sigma * sigma))
    kernel /= kernel.sum()

    h, w = arr.shape
    # Horizontal pass: pad along axis 1, convolve with a sliding window.
    padded_h = np.pad(arr, ((0, 0), (radius, radius)), mode="reflect")
    horiz = np.zeros((h, w), dtype=arr.dtype)
    for k, kv in enumerate(kernel):
        horiz += kv * padded_h[:, k : k + w]
    # Vertical pass: pad along axis 0.
    padded_v = np.pad(horiz, ((radius, radius), (0, 0)), mode="reflect")
    out = np.zeros((h, w), dtype=arr.dtype)
    for k, kv in enumerate(kernel):
        out += kv * padded_v[k : k + h, :]
    return out


__all__ = ["heuristic_trust_map"]
