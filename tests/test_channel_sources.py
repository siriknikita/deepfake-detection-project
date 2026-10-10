"""Tests for the source-pluggable channel-loading API (Phase 3+)."""

from __future__ import annotations

import pickle
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


def test_built_in_sources_are_picklable() -> None:
    """torch.utils.data.DataLoader spawns worker processes for num_workers>0
    and pickles the dataset (and through it the channel sources) into each
    worker. Lambdas in cache_path_fn would break this — the built-in
    physics / frequency sources use functools.partial instead. This test
    locks in pickleability so the regression can't return.
    """
    for src in [
        physics_channel_source("heuristic"),
        physics_channel_source("gtmask"),
        frequency_channel_source("default"),
    ]:
        blob = pickle.dumps(src)
        restored = pickle.loads(blob)
        assert restored.name == src.name
        assert restored.n_channels == src.n_channels
        assert restored.npz_keys == src.npz_keys
        # The cache_path_fn must remain callable after the round-trip and
        # produce the same path the original source would.
        sample = Path("/data/X/c23/frames/v0/0001.png")
        assert restored.cache_path_fn(sample) == src.cache_path_fn(sample)


def test_uint8_physics_cache_dequantises_to_float16_equivalent_inputs(tmp_path: Path) -> None:
    """A uint8 physics cache (per-map lo/hi codes) must load to the same normalised
    channels as the float16 cache up to about one code step (< 0.01 after tanh/min-max)."""
    rng = np.random.default_rng(0)
    h = w = 16
    rgb = rng.random((3, h, w)).astype(np.float32)
    wcnn = rng.random((h, w)).astype(np.float32)
    z_star = (rng.standard_normal((h, w)) * 0.3).astype(np.float32)
    residual = (rng.standard_normal((h, w)) * 0.05).astype(np.float32)
    frames = tmp_path / "frames_faces" / "vid"
    frames.mkdir(parents=True)
    img = frames / "0001.png"
    img.write_bytes(b"")
    src = physics_channel_source("heuristic")
    npz = src.cache_path_fn(img)
    npz.parent.mkdir(parents=True)
    np.savez_compressed(
        npz,
        wcnn=wcnn.astype(np.float16),
        z_star=z_star.astype(np.float16),
        residual=residual.astype(np.float16),
    )
    ref = load_channels_concat(rgb, img, [src])

    def q(a: np.ndarray) -> tuple[np.ndarray, float, float]:
        lo, hi = float(a.min()), float(a.max())
        return np.round((a - lo) / (hi - lo) * 255).astype(np.uint8), lo, hi

    qw, lw, hw = q(wcnn)
    qz, lz, hz = q(z_star)
    qr, lr, hr = q(residual)
    np.savez_compressed(
        npz,
        wcnn=qw,
        z_star=qz,
        residual=qr,
        lo=np.array([lw, lz, lr], np.float32),
        hi=np.array([hw, hz, hr], np.float32),
    )
    got = load_channels_concat(rgb, img, [src])
    assert got.shape == ref.shape == (6, h, w)
    assert np.abs(got - ref).max() < 0.01


def test_frequency_compute_matches_cached_frequency_maps(tmp_path: Path) -> None:
    pytest.importorskip("scipy")
    from forge_detect.frequency_map import frequency_maps

    rng = np.random.default_rng(1)
    h = w = 32
    rgb_hwc = rng.random((h, w, 3)).astype(np.float32)
    rgb = np.transpose(rgb_hwc, (2, 0, 1))
    frames = tmp_path / "frames_faces" / "vid"
    frames.mkdir(parents=True)
    img = frames / "0001.png"
    img.write_bytes(b"")
    cached = frequency_channel_source("default")
    npz = cached.cache_path_fn(img)
    npz.parent.mkdir(parents=True)
    e, r, m = frequency_maps(rgb_hwc)
    np.savez_compressed(
        npz,
        dct_block_energy=e.astype(np.float16),
        dct_high_ratio=r.astype(np.float16),
        fft_radial_logmag=m.astype(np.float16),
    )
    ref = load_channels_concat(rgb, img, [cached])
    computed = parse_channel_spec("rgb,frequency:compute")
    assert computed[0].compute_fn is not None and computed[0].name == "frequency:compute"
    got = load_channels_concat(rgb, img, computed)
    assert got.shape == ref.shape == (6, h, w)
    assert np.abs(got - ref).max() < 2e-3
    # pickling survives (DataLoader workers)
    assert pickle.loads(pickle.dumps(computed[0])).compute_fn is not None
