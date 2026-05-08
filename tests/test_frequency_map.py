"""Tests for the Phase-3 frequency-domain channel producers."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.fft import dctn

from forge_detect.frequency_map import (
    _AC_INDICES,
    _DCT8,
    _ZIGZAG_8,
    _block_dct,
    _to_blocks,
    dct_block_maps,
    fft_radial_logmag,
    frequency_maps,
)


def _rand_image(seed: int = 0, shape: tuple[int, int, int] = (64, 64, 3)) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.random(shape).astype(np.float32)


def test_dct_matrix_is_orthonormal() -> None:
    """The 8×8 DCT-II matrix is orthonormal up to single-precision noise."""
    err = float(np.abs(_DCT8 @ _DCT8.T - np.eye(8, dtype=np.float32)).max())
    assert err < 1.0e-5, f"DCT8 @ DCT8.T deviates from I by {err:.2e}"


def test_block_dct_matches_scipy() -> None:
    """Vectorised block DCT agrees with scipy's reference DCT-II per block."""
    rng = np.random.default_rng(1)
    blocks = rng.random((4, 4, 8, 8), dtype=np.float32)
    ours = _block_dct(blocks)
    for i in range(4):
        for j in range(4):
            ref = dctn(blocks[i, j], type=2, norm="ortho")
            err = float(np.abs(ours[i, j] - ref).max())
            assert err < 1.0e-4, f"block ({i}, {j}) DCT mismatch: {err:.2e}"


def test_zigzag_indices_cover_all_positions() -> None:
    """Zigzag is a permutation of [0, n²)."""
    assert set(_ZIGZAG_8.tolist()) == set(range(64))
    # AC = zigzag minus the DC at position 0.
    assert _AC_INDICES.shape == (63,)
    assert 0 not in _AC_INDICES.tolist()


def test_to_blocks_reflect_pads_non_divisible() -> None:
    """Non-multiple-of-8 input is reflect-padded so the block grid tiles."""
    luma = np.zeros((63, 65), dtype=np.float32)
    blocks, h, w = _to_blocks(luma)
    assert (h, w) == (63, 65)
    assert blocks.shape == (8, 9, 8, 8)  # padded to (64, 72) -> 8x9 blocks


def test_dct_block_maps_shape_and_range() -> None:
    img = _rand_image(seed=2)
    energy, ratio = dct_block_maps(img)
    assert energy.shape == (64, 64)
    assert ratio.shape == (64, 64)
    assert energy.dtype == np.float32
    assert ratio.dtype == np.float32
    assert energy.min() >= 0.0 and energy.max() <= 1.0
    assert ratio.min() >= 0.0 and ratio.max() <= 1.0


def test_dct_block_maps_non_divisible_size() -> None:
    img = _rand_image(seed=3, shape=(63, 65, 3))
    energy, ratio = dct_block_maps(img)
    assert energy.shape == (63, 65)
    assert ratio.shape == (63, 65)


def test_dct_block_maps_constant_image_zero() -> None:
    """A flat image has zero AC energy in every block; both maps degenerate to 0.

    The per-image min-max protects against ε-divisions by collapsing the
    map to all zeros when the dynamic range is below the epsilon.
    """
    img = np.full((32, 32, 3), 0.5, dtype=np.float32)
    energy, ratio = dct_block_maps(img)
    assert (energy == 0.0).all()
    assert (ratio == 0.0).all()


def test_fft_radial_logmag_shape_and_range() -> None:
    img = _rand_image(seed=4)
    fft_map = fft_radial_logmag(img)
    assert fft_map.shape == (64, 64)
    assert fft_map.dtype == np.float32
    assert fft_map.min() >= 0.0 and fft_map.max() <= 1.0


def test_fft_radial_logmag_dc_at_centre() -> None:
    """A pure DC image has its FFT magnitude concentrated at the centre after fftshift.

    The exact centre pixel should hit the per-image max (1.0) once the
    log-magnitude is min-max'd.
    """
    img = np.full((32, 32, 3), 0.5, dtype=np.float32)
    fft_map = fft_radial_logmag(img)
    # All-DC image: only the centre pixel of the shifted spectrum is non-zero,
    # so the per-image min-max produces exactly one max-valued cell at the
    # centre and zeros everywhere else.
    cy, cx = fft_map.shape[0] // 2, fft_map.shape[1] // 2
    assert fft_map[cy, cx] == pytest.approx(1.0)
    mask = np.ones_like(fft_map, dtype=bool)
    mask[cy, cx] = False
    assert (fft_map[mask] == 0.0).all()


def test_frequency_maps_returns_three_channels() -> None:
    img = _rand_image(seed=5)
    e, r, f = frequency_maps(img)
    assert e.shape == r.shape == f.shape == (64, 64)
    # All three channels are independently produced (not aliased to each other).
    assert not np.allclose(e, r)
    assert not np.allclose(e, f)


def test_frequency_maps_rejects_non_rgb_input() -> None:
    img = np.zeros((32, 32), dtype=np.float32)
    with pytest.raises(ValueError, match="rgb must be"):
        dct_block_maps(img)
    with pytest.raises(ValueError, match="rgb must be"):
        fft_radial_logmag(img)


def test_dct_high_ratio_localises_high_frequency_content() -> None:
    """Mixed-frequency image: the diagonal-high-frequency half ranks above
    the smooth half on the dct_high_ratio map.

    The high band by our zigzag definition starts at position 32 — i.e. the
    upper diagonal half of the 8×8 spectrum. A 2×2 checkerboard concentrates
    energy at zigzag 63 (the true Nyquist diagonal), so it is the cleanest
    test-case for the high-band path; alternating-column patterns put energy
    at zigzag 28, which falls in the mid band and would not light up the
    map by design.
    """
    h, w = 64, 64
    img = np.zeros((h, w, 3), dtype=np.float32)
    # Left half: smooth horizontal gradient (low-frequency).
    img[:, : w // 2, :] = np.tile(
        np.linspace(0.0, 1.0, w // 2, dtype=np.float32), (h, 1),
    )[..., None]
    # Right half: 2x2 checkerboard (Nyquist diagonal frequency).
    yy, xx = np.indices((h, w // 2))
    img[:, w // 2 :, :] = ((yy + xx) % 2).astype(np.float32)[..., None]
    _, ratio = dct_block_maps(img)
    left_mean = ratio[:, : w // 2].mean()
    right_mean = ratio[:, w // 2 :].mean()
    assert right_mean > left_mean, (
        f"high-freq half should outrank low-freq half: "
        f"left={left_mean:.4f} right={right_mean:.4f}"
    )
