"""Tests for the source-pluggable channel-loading API (Phase 3+)."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from forge_detect.datasets import (
    FREQUENCY_NPZ_KEYS,
    PHYSICS_NPZ_KEYS,
    ChannelSource,
    frequency_channel_source,
    frequency_npz_path,
    load_channels_concat,
    parse_channel_spec,
    physics_channel_source,
    physics_npz_path,
    total_channels,
)


def _make_image_tree(
    root: Path,
    *,
    frames_subdir: str = "frames",
    video_id: str = "v0",
    frame_stem: str = "0001",
) -> Path:
    image_dir = root / frames_subdir / video_id
    image_dir.mkdir(parents=True, exist_ok=True)
    image_path = image_dir / f"{frame_stem}.png"
    image_path.touch()
    return image_path


def _save_physics_npz(path: Path, hw: tuple[int, int]) -> None:
    h, w = hw
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        wcnn=np.full((h, w), 0.7, dtype=np.float16),
        z_star=np.linspace(0, 1, h * w, dtype=np.float16).reshape(h, w),
        residual=np.zeros((h, w), dtype=np.float16),
    )


def _save_frequency_npz(path: Path, hw: tuple[int, int]) -> None:
    h, w = hw
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        dct_block_energy=np.full((h, w), 0.1, dtype=np.float16),
        dct_high_ratio=np.full((h, w), 0.4, dtype=np.float16),
        fft_radial_logmag=np.linspace(0, 1, h * w, dtype=np.float16).reshape(h, w),
    )


def test_parse_channel_spec_recognised_tokens() -> None:
    assert parse_channel_spec("rgb") == []
    sources = parse_channel_spec("rgb,physics")
    assert [s.name for s in sources] == ["physics:heuristic"]
    sources = parse_channel_spec("rgb,physics,frequency")
    assert [s.name for s in sources] == ["physics:heuristic", "frequency:default"]
    sources = parse_channel_spec("physics:gtmask,frequency")
    assert [s.name for s in sources] == ["physics:gtmask", "frequency:default"]


def test_parse_channel_spec_is_case_insensitive_and_strips_whitespace() -> None:
    sources = parse_channel_spec(" RGB , Physics , Frequency ")
    assert [s.name for s in sources] == ["physics:heuristic", "frequency:default"]


def test_parse_channel_spec_rejects_duplicate_family() -> None:
    with pytest.raises(ValueError, match="appears twice"):
        parse_channel_spec("physics,physics")


def test_parse_channel_spec_rejects_unknown_token() -> None:
    with pytest.raises(ValueError, match="unknown channel token"):
        parse_channel_spec("rgb,specular")


def test_total_channels() -> None:
    assert total_channels([]) == 3
    assert total_channels(parse_channel_spec("physics")) == 6
    assert total_channels(parse_channel_spec("frequency")) == 6
    assert total_channels(parse_channel_spec("physics,frequency")) == 9


def test_physics_channel_source_validates_variant() -> None:
    physics_channel_source("heuristic")
    physics_channel_source("gtmask")
    with pytest.raises(ValueError, match="physics variant"):
        physics_channel_source("nope")


def test_frequency_channel_source_validates_variant() -> None:
    frequency_channel_source("default")
    with pytest.raises(ValueError, match="frequency variant"):
        frequency_channel_source("nope")


def test_load_channels_concat_empty_sources_is_passthrough() -> None:
    rgb = np.full((3, 8, 8), 0.5, dtype=np.float32)
    out = load_channels_concat(rgb, Path("/tmp/x.png"), [])
    assert out.shape == (3, 8, 8)
    assert out.dtype == np.float32
    assert (out == 0.5).all()


def test_load_channels_concat_rgb_plus_physics_plus_frequency() -> None:
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        image_path = _make_image_tree(root)
        _save_physics_npz(physics_npz_path(image_path, "heuristic"), (8, 8))
        _save_frequency_npz(frequency_npz_path(image_path, "default"), (8, 8))
        rgb = np.full((3, 8, 8), 0.5, dtype=np.float32)
        sources = parse_channel_spec("rgb,physics,frequency")
        out = load_channels_concat(rgb, image_path, sources)
        assert out.shape == (9, 8, 8)
        assert out.dtype == np.float32
        # First three channels remain RGB.
        assert (out[:3] == 0.5).all()
        # Physics channels follow physics-source order: wcnn, z_star, residual.
        assert out[3].mean() == pytest.approx(0.7, rel=1e-2)


def test_load_channels_concat_missing_cache_raises() -> None:
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        image_path = _make_image_tree(root)
        rgb = np.full((3, 8, 8), 0.5, dtype=np.float32)
        with pytest.raises(FileNotFoundError, match="channel cache"):
            load_channels_concat(rgb, image_path, [physics_channel_source()])


def test_load_channels_concat_shape_mismatch_raises() -> None:
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        image_path = _make_image_tree(root)
        # Cache is 8x8, image is 16x16 -> shape mismatch.
        _save_physics_npz(physics_npz_path(image_path, "heuristic"), (8, 8))
        rgb = np.full((3, 16, 16), 0.5, dtype=np.float32)
        with pytest.raises(ValueError, match="shape"):
            load_channels_concat(rgb, image_path, [physics_channel_source()])


def test_load_channels_concat_missing_npz_key_raises() -> None:
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        image_path = _make_image_tree(root)
        npz = physics_npz_path(image_path, "heuristic")
        npz.parent.mkdir(parents=True, exist_ok=True)
        # Save with a partial set of keys to trigger the missing-key path.
        np.savez_compressed(npz, wcnn=np.zeros((8, 8), dtype=np.float16))
        rgb = np.full((3, 8, 8), 0.5, dtype=np.float32)
        with pytest.raises(ValueError, match="missing key"):
            load_channels_concat(rgb, image_path, [physics_channel_source()])


def test_channel_source_is_pluggable() -> None:
    """A user-defined ChannelSource composes with the built-in ones."""

    def normalize(arrays: dict[str, np.ndarray]) -> np.ndarray:
        return np.stack([arrays["x"], arrays["x"]], axis=0).astype(np.float32, copy=False)

    custom = ChannelSource(
        name="custom",
        n_channels=2,
        cache_path_fn=lambda p: p.parent / f"{p.stem}.custom.npz",
        npz_keys=("x",),
        normalize=normalize,
    )

    with tempfile.TemporaryDirectory() as d:
        image_path = Path(d) / "img.png"
        image_path.touch()
        cache = custom.cache_path_fn(image_path)
        np.savez_compressed(cache, x=np.full((4, 4), 0.25, dtype=np.float16))
        rgb = np.full((3, 4, 4), 0.5, dtype=np.float32)
        out = load_channels_concat(rgb, image_path, [custom])
        assert out.shape == (5, 4, 4)
        assert (out[3] == 0.25).all()
        assert (out[4] == 0.25).all()


def test_npz_key_constants_are_aligned() -> None:
    """Module-level constants stay in sync with the source keys."""
    assert physics_channel_source().npz_keys == PHYSICS_NPZ_KEYS
    assert frequency_channel_source().npz_keys == FREQUENCY_NPZ_KEYS
