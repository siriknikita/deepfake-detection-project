"""Frequency-domain channel producers (Phase 3).

Three ``(H, W)`` float32 maps per image, normalised to ``[0, 1]``:

1. ``dct_block_energy`` — log total AC (non-DC) energy in each $8 times 8$
   block-DCT, tiled to image resolution. A per-block "how much non-flat
   content lives here" signature.
2. ``dct_high_ratio`` — per-block ratio of upper-half-band AC energy to
   total AC energy (zigzag positions $32{-}63$ over $1{-}63$). Sensitive
   to upsampling / interpolation artefacts that put energy into the
   higher frequencies of a block.
3. ``fft_radial_logmag`` — log magnitude of the full-image 2D FFT,
   fftshifted to centre DC. Captures the global radial frequency
   landscape Frank et al. (ICML 2020) and Durall et al. (CVPR 2020)
   identified as a GAN/diffusion fingerprint. Tiled at the image's own
   resolution rather than radial-projected to keep the channel a 2D map
   the CNN's stem can consume directly.

The geometric-chromatic Phase-2 channels ($W_"cnn"$, $z^*$, $R$) help on
autoencoder-based manipulations and hurt on parametric / graphics-based
ones (FF++ Face2Face, FaceSwap) — see Phase 2's per-method ablation.
The hypothesis behind these frequency channels is that synthesis
pipelines leave systematic spectral signatures even when they preserve
geometric coherence by construction, so adding frequency channels should
recover signal on exactly the methods where the manifold-settlement
channels are silent.

Implementation is pure NumPy: no SciPy, no Torch. The block-DCT goes
through a constant $8 times 8$ orthonormal DCT-II matrix and matrix
multiplication broadcast across the block grid; the FFT uses
``numpy.fft``. Throughput is adequate for cache-time use; the
``scripts/cache_frequency_maps.py`` script overlaps these with I/O via
the same threading model as the physics-map cache.
"""

from __future__ import annotations

import numpy as np

# Rec. 709 luminance weights, matching the convention in the Rust
# luminance.rs kernel so the spectral channel sees the same Y as the
# geometric channels do.
_LUMA = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)

_BLOCK_SIZE = 8
# Zigzag position ≥ 32 marks the upper half of the AC band. 31 of the 63
# AC coefficients sit at zigzag positions 32-63; this band catches the
# upsampling / interpolation residuals that distinguish synthesised
# textures from natural high-frequency content.
_HIGH_BAND_ZZ_START = 32
_FREQ_EPS = 1.0e-6


def _dct_matrix(n: int) -> np.ndarray:
    """Orthonormal DCT-II matrix of shape ``(n, n)``.

    Constructed once at import time for the $n = 8$ case. A 2D DCT of an
    $n times n$ block ``X`` is then ``M @ X @ M.T``; broadcast across a
    block grid ``(n_h, n_w, n, n)`` it stays vectorised.
    """
    k = np.arange(n, dtype=np.float32).reshape(-1, 1)
    j = np.arange(n, dtype=np.float32).reshape(1, -1)
    m = np.cos(np.pi * (2.0 * j + 1.0) * k / (2.0 * n))
    # alpha(k): sqrt(1/N) for the DC row, sqrt(2/N) for all AC rows.
    m[0] *= float(np.sqrt(1.0 / n))
    m[1:] *= float(np.sqrt(2.0 / n))
    return m


_DCT8 = _dct_matrix(_BLOCK_SIZE)


def _zigzag_indices(n: int) -> np.ndarray:
    """Flat indices into an ``n × n`` block in JPEG zigzag order.

    Returns a length-``n²`` array. ``zigzag[k]`` is the row-major index of
    the $k$-th coefficient in zigzag order, so a flattened DCT block can
    be re-ordered by ``flat[zigzag]``.
    """
    out = np.empty(n * n, dtype=np.int64)
    k = 0
    for s in range(2 * n - 1):
        if s % 2 == 0:
            r = min(s, n - 1)
            c = s - r
            while r >= 0 and c < n:
                out[k] = r * n + c
                k += 1
                r -= 1
                c += 1
        else:
            c = min(s, n - 1)
            r = s - c
            while c >= 0 and r < n:
                out[k] = r * n + c
                k += 1
                c -= 1
                r += 1
    return out


_ZIGZAG_8 = _zigzag_indices(_BLOCK_SIZE)
_AC_INDICES = _ZIGZAG_8[1:]  # drop the DC coefficient
# Index into _AC_INDICES at which the high band starts. _AC_INDICES[k] is
# zigzag position k+1, so high-band start at zigzag 32 means k = 31.
_HIGH_BAND_OFFSET = _HIGH_BAND_ZZ_START - 1


def _luminance(rgb: np.ndarray) -> np.ndarray:
    """``(H, W, 3)`` RGB in ``[0, 1]`` → ``(H, W)`` luminance in ``[0, 1]``."""
    return (rgb * _LUMA).sum(axis=-1).astype(np.float32, copy=False)


def _to_blocks(luma: np.ndarray, block: int = _BLOCK_SIZE) -> tuple[np.ndarray, int, int]:
    """Reflect-pad and reshape ``(H, W)`` luminance to ``(n_h, n_w, b, b)``.

    Returns the block grid plus the original ``(H, W)`` so callers can crop
    the per-pixel tiled map back to the input shape after the tiling step.
    """
    h, w = luma.shape
    pad_h = (-h) % block
    pad_w = (-w) % block
    if pad_h or pad_w:
        luma = np.pad(luma, ((0, pad_h), (0, pad_w)), mode="reflect")
    h2, w2 = luma.shape
    n_h = h2 // block
    n_w = w2 // block
    blocks = luma.reshape(n_h, block, n_w, block).transpose(0, 2, 1, 3)
    return blocks, h, w


def _block_dct(blocks: np.ndarray) -> np.ndarray:
    """2D DCT-II of every block. ``blocks`` has shape ``(n_h, n_w, b, b)``."""
    return _DCT8 @ blocks @ _DCT8.T


def _tile_per_block(values: np.ndarray, block: int, h: int, w: int) -> np.ndarray:
    """Per-block scalar ``(n_h, n_w)`` → per-pixel map ``(H, W)`` by tiling.

    The repeat-then-crop sequence handles reflect-padding on the right /
    bottom edges: any block grid past the original ``(H, W)`` is dropped.
    """
    tiled = np.repeat(np.repeat(values, block, axis=0), block, axis=1)
    return tiled[:h, :w]


def _per_image_minmax(arr: np.ndarray) -> np.ndarray:
    """Per-image min-max to ``[0, 1]``, matching the Phase 2 convention.

    The absolute scale of the spectrum changes with content (a flat
    photograph and a textured one differ by orders of magnitude); only
    the spatial pattern is informative for the classifier. Per-image
    normalisation strips the cross-image drift the same way Phase 2 does
    for $z^*$ and $R$.
    """
    lo, hi = float(arr.min()), float(arr.max())
    if hi - lo < _FREQ_EPS:
        return np.zeros_like(arr, dtype=np.float32)
    return ((arr - lo) / (hi - lo)).astype(np.float32, copy=False)


def dct_block_maps(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Build the two block-DCT maps from an RGB image.

    Args:
        rgb: ``(H, W, 3)`` float32 image in ``[0, 1]``.

    Returns:
        ``(dct_block_energy, dct_high_ratio)`` both ``(H, W)`` float32 in
        ``[0, 1]``. Per-block scalars are tiled to image resolution so
        early CNN layers can localise the per-block frequency signature.
    """
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        msg = f"rgb must be (H, W, 3), got {rgb.shape}"
        raise ValueError(msg)
    rgb = rgb.astype(np.float32, copy=False)
    luma = _luminance(rgb)
    blocks, h, w = _to_blocks(luma)
    dct = _block_dct(blocks)
    flat = dct.reshape(*dct.shape[:2], -1)
    ac = flat[..., _AC_INDICES]
    ac_sq = ac * ac
    total_ac = ac_sq.sum(axis=-1)
    high_ac = ac_sq[..., _HIGH_BAND_OFFSET:].sum(axis=-1)
    log_total = np.log1p(total_ac)
    high_ratio = high_ac / (total_ac + _FREQ_EPS)
    energy_map = _tile_per_block(log_total, _BLOCK_SIZE, h, w)
    ratio_map = _tile_per_block(high_ratio, _BLOCK_SIZE, h, w)
    return _per_image_minmax(energy_map), _per_image_minmax(ratio_map)


def fft_radial_logmag(rgb: np.ndarray) -> np.ndarray:
    """Log-magnitude of the full-image 2D FFT, fftshifted, in ``[0, 1]``.

    Args:
        rgb: ``(H, W, 3)`` float32 image in ``[0, 1]``.

    Returns:
        ``(H, W)`` float32. ``np.fft.fft2`` on luminance, ``log1p`` of the
        magnitude, ``fftshift`` so DC sits at the centre, then per-image
        min-max so the channel composes with the others.
    """
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        msg = f"rgb must be (H, W, 3), got {rgb.shape}"
        raise ValueError(msg)
    luma = _luminance(rgb.astype(np.float32, copy=False))
    spectrum = np.fft.fftshift(np.fft.fft2(luma))
    mag = np.abs(spectrum).astype(np.float32, copy=False)
    log_mag = np.log1p(mag)
    return _per_image_minmax(log_mag)


def frequency_maps(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute all three frequency maps at once.

    Convenience wrapper used by ``scripts/cache_frequency_maps.py``.

    Returns:
        ``(dct_block_energy, dct_high_ratio, fft_radial_logmag)``, each
        ``(H, W)`` float32 in ``[0, 1]``.
    """
    dct_energy, dct_ratio = dct_block_maps(rgb)
    fft_map = fft_radial_logmag(rgb)
    return dct_energy, dct_ratio, fft_map


__all__ = [
    "dct_block_maps",
    "fft_radial_logmag",
    "frequency_maps",
]
