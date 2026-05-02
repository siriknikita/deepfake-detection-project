"""Tests for dataset adapters."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from forge_detect.datasets import (
    FF_COMPRESSIONS,
    FF_METHODS,
    FaceForensicsAdapter,
    ImageFolderDataset,
    load_image_chw,
    stratified_split,
)


def _save_synthetic(path: Path, size: tuple[int, int] = (32, 32), seed: int = 0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    arr = (rng.random((*size, 3)) * 255).astype(np.uint8)
    Image.fromarray(arr).save(path)


def test_load_image_chw_shape_and_range() -> None:
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "x.png"
        _save_synthetic(p)
        arr = load_image_chw(p)
        assert arr.shape == (3, 32, 32)
        assert arr.dtype == np.float32
        assert (arr >= 0.0).all() and (arr <= 1.0).all()


def test_load_image_chw_resize() -> None:
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "x.png"
        _save_synthetic(p, size=(64, 96))
        arr = load_image_chw(p, target_size=(48, 80))
        assert arr.shape == (3, 48, 80)


def test_image_folder_dataset_balances_real_and_fake() -> None:
    pytest.importorskip("torch")
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        for i in range(3):
            _save_synthetic(root / "real" / f"r{i}.png", seed=i)
        for i in range(2):
            _save_synthetic(root / "fake" / f"f{i}.png", seed=10 + i)

        ds = ImageFolderDataset(root / "real", root / "fake")
        assert len(ds) == 5
        labels = [ds[i][1] for i in range(len(ds))]
        assert sum(labels) == 2
        assert labels.count(0) == 3


def test_image_folder_dataset_returns_tensor() -> None:
    pytest.importorskip("torch")
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        _save_synthetic(root / "real" / "x.png")
        _save_synthetic(root / "fake" / "y.png")
        ds = ImageFolderDataset(root / "real", root / "fake", target_size=(16, 16))
        img, label = ds[0]
        import torch

        assert isinstance(img, torch.Tensor)
        assert img.shape == (3, 16, 16)
        assert label in (0, 1)


def test_image_folder_dataset_rejects_empty_folders() -> None:
    pytest.importorskip("torch")
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        (root / "real").mkdir()
        (root / "fake").mkdir()
        with pytest.raises(FileNotFoundError):
            ImageFolderDataset(root / "real", root / "fake")


def test_face_forensics_adapter_full_layout() -> None:
    pytest.importorskip("torch")
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        # Real frames.
        for i in range(2):
            _save_synthetic(
                root / "original_sequences" / "youtube" / "c23" / "frames" / f"v{i}" / "0001.png"
            )
        # Fake frames for two methods.
        for method in ("Deepfakes", "Face2Face"):
            for i in range(2):
                _save_synthetic(
                    root
                    / "manipulated_sequences"
                    / method
                    / "c23"
                    / "frames"
                    / f"v{i}"
                    / "0001.png"
                )
        ds = FaceForensicsAdapter(root, methods=("Deepfakes", "Face2Face"))
        assert len(ds) == 6  # 2 real + 2*2 fake
        labels = [ds[i][1] for i in range(len(ds))]
        assert labels.count(0) == 2
        assert labels.count(1) == 4


def test_face_forensics_adapter_includes_optional_mask() -> None:
    pytest.importorskip("torch")
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        frame_path = (
            root / "manipulated_sequences" / "Deepfakes" / "c23" / "frames" / "v0" / "0001.png"
        )
        mask_path = (
            root / "manipulated_sequences" / "Deepfakes" / "c23" / "masks" / "v0" / "0001.png"
        )
        _save_synthetic(frame_path)
        # Save a binary mask.
        mask_path.parent.mkdir(parents=True, exist_ok=True)
        m = (np.random.default_rng(0).random((32, 32)) > 0.5).astype(np.uint8) * 255
        Image.fromarray(m).save(mask_path)

        ds = FaceForensicsAdapter(root, methods=("Deepfakes",))
        _img, label, returned_mask = ds[0]
        import torch

        assert label == 1
        assert returned_mask is not None
        assert isinstance(returned_mask, torch.Tensor)
        assert returned_mask.shape == (32, 32)


def test_face_forensics_adapter_rejects_bad_compression() -> None:
    with (
        tempfile.TemporaryDirectory() as d,
        pytest.raises(ValueError, match="compression must be"),
    ):
        FaceForensicsAdapter(d, compression="c10")


def test_face_forensics_adapter_max_frames_per_video() -> None:
    pytest.importorskip("torch")
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        for i in range(10):
            _save_synthetic(
                root / "original_sequences" / "youtube" / "c23" / "frames" / "v0" / f"{i:04d}.png"
            )
        ds = FaceForensicsAdapter(root, methods=(), max_frames_per_video=3)
        assert len(ds) == 3


def test_stratified_split_partitions_indices() -> None:
    train, val, test = stratified_split(100, seed=42, val_fraction=0.2, test_fraction=0.1)
    all_idx = sorted(train + val + test)
    assert all_idx == list(range(100))
    assert len(test) == 10
    assert len(val) == 20
    assert len(train) == 70


def test_stratified_split_is_deterministic() -> None:
    a = stratified_split(50, seed=7)
    b = stratified_split(50, seed=7)
    assert a == b


def test_known_constants_present() -> None:
    assert "Deepfakes" in FF_METHODS
    assert "c23" in FF_COMPRESSIONS
