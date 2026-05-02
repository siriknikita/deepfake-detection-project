"""Smoke tests for the high-level Python pipeline (trust map, features, detect)."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from forge_detect.config import PdeParams, PipelineParams
from forge_detect.features import FEATURE_NAMES, extract_features
from forge_detect.pipeline import detect, load_image
from forge_detect.trust_map import heuristic_trust_map


def _save_temp_image(rgb: np.ndarray) -> Path:
    f = tempfile.NamedTemporaryFile(suffix=".png", delete=False)  # noqa: SIM115
    arr_uint8 = (rgb * 255.0).clip(0, 255).astype(np.uint8)
    Image.fromarray(arr_uint8).save(f.name)
    f.close()
    return Path(f.name)


def _disk(shape: tuple[int, int]) -> np.ndarray:
    h, w = shape
    yy, xx = np.indices((h, w))
    cy, cx = h / 2.0, w / 2.0
    r2 = (yy - cy) ** 2 + (xx - cx) ** 2
    bright = (r2 < (min(h, w) / 3.0) ** 2).astype(np.float32) * 0.6 + 0.2
    return np.stack([bright] * 3, axis=-1)


def _quick_params() -> PipelineParams:
    return PipelineParams(n_scales=2, pde=PdeParams(max_iter=50, log_every=5))


def test_heuristic_trust_map_shape_and_range() -> None:
    rgb = _disk((24, 24))
    w = heuristic_trust_map(rgb)
    assert w.shape == (24, 24)
    assert w.dtype == np.float32
    assert (w >= 0.0).all() and (w <= 1.0).all()


def test_heuristic_trust_map_constant_input_is_one() -> None:
    rgb = np.full((16, 16, 3), 0.5, dtype=np.float32)
    w = heuristic_trust_map(rgb)
    # No chromatic deviation ⇒ trust = exp(0) = 1.
    assert np.allclose(w, 1.0, atol=1e-6)


def test_heuristic_trust_map_rejects_non_rgb() -> None:
    bad = np.zeros((10, 10, 4), dtype=np.float32)
    with pytest.raises(ValueError, match="must be"):
        heuristic_trust_map(bad)


def test_features_length_and_names() -> None:
    rgb = _disk((24, 24))
    result = detect(rgb, params=_quick_params())
    assert result.features.shape == (24,)
    assert len(result.feature_names) == 24
    assert tuple(result.feature_names) == FEATURE_NAMES


def test_features_are_finite() -> None:
    rgb = _disk((24, 24))
    result = detect(rgb, params=_quick_params())
    assert np.isfinite(result.features).all()


def test_extract_features_smoke() -> None:
    # Run on a precomputed SolveResult.
    rgb = _disk((24, 24))
    result = detect(rgb, params=_quick_params())
    feats = extract_features(result.solve)
    assert feats.shape == (24,)
    np.testing.assert_array_equal(feats, result.features)


def test_detect_loads_image_from_path() -> None:
    rgb = _disk((24, 24))
    p = _save_temp_image(rgb)
    try:
        result = detect(p, params=_quick_params())
        assert result.image_path == p
        assert result.solve.z_star.shape == (24, 24)
    finally:
        p.unlink(missing_ok=True)


def test_detect_in_memory_array() -> None:
    rgb = _disk((24, 24))
    result = detect(rgb, params=_quick_params())
    assert str(result.image_path) == "(in-memory)"


def test_detect_rejects_wrong_shape_array() -> None:
    bad = np.zeros((24, 24, 4), dtype=np.float32)
    with pytest.raises(ValueError, match="image must be"):
        detect(bad, params=_quick_params())


def test_load_image_round_trip() -> None:
    rgb = (_disk((16, 16)) * 255.0).astype(np.uint8)
    p = _save_temp_image(rgb.astype(np.float32) / 255.0)
    try:
        loaded = load_image(p)
        assert loaded.shape == (16, 16, 3)
        assert loaded.dtype == np.float32
        assert (loaded.min() >= 0.0) and (loaded.max() <= 1.0)
    finally:
        p.unlink(missing_ok=True)


def test_detect_with_trained_cnn_model() -> None:
    """detect() routes the CNN forward pass into the trust map when one is supplied."""
    pytest.importorskip("torch")
    from forge_detect.cnn import build_chromatic_efficientnet

    rgb = _disk((32, 32))
    model = build_chromatic_efficientnet(pretrained=False)
    model.eval()
    result = detect(rgb, params=_quick_params(), cnn_model=model)
    # Output shape and finiteness are the contract; semantic correctness
    # of the trust map depends on training, which is not our concern here.
    assert result.solve.z_star.shape == (32, 32)
    assert np.isfinite(result.features).all()
