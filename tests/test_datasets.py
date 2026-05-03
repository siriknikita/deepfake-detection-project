"""Tests for dataset adapters."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from forge_detect.datasets import (
    FF_COMPRESSIONS,
    FF_METHODS,
    CelebDFAdapter,
    FaceForensicsAdapter,
    ImageFolderDataset,
    load_celeb_testing_list,
    load_ff_split,
    load_image_chw,
    split_videos,
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


# ---------- video_id, splits, CelebDFAdapter --------------------------------


def test_face_forensics_records_carry_video_id() -> None:
    pytest.importorskip("torch")
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        for vid in ("000", "017"):
            _save_synthetic(
                root / "original_sequences" / "youtube" / "c23" / "frames" / vid / "0001.png"
            )
        ds = FaceForensicsAdapter(root, methods=())
        ids = [rec.video_id for rec in ds._records]
        assert sorted(set(ids)) == ["000", "017"]


def test_face_forensics_video_ids_method() -> None:
    pytest.importorskip("torch")
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        for vid in ("a", "b"):
            _save_synthetic(
                root / "original_sequences" / "youtube" / "c23" / "frames" / vid / "0001.png"
            )
        for vid in ("a_b",):
            _save_synthetic(
                root / "manipulated_sequences" / "Deepfakes" / "c23" / "frames" / vid / "0001.png"
            )
        ds = FaceForensicsAdapter(root, methods=("Deepfakes",))
        assert ds.video_ids() == ["a", "a_b", "b"]


def test_face_forensics_subset_video_ids_filters() -> None:
    pytest.importorskip("torch")
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        for vid in ("000", "017", "042"):
            _save_synthetic(
                root / "original_sequences" / "youtube" / "c23" / "frames" / vid / "0001.png"
            )
        ds = FaceForensicsAdapter(
            root, methods=(), subset_video_ids={"000", "042"},
        )
        assert ds.video_ids() == ["000", "042"]


def test_load_ff_split_flattens_pairs() -> None:
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        (root / "splits").mkdir()
        (root / "splits" / "test.json").write_text(json.dumps([["001", "023"], ["005", "100"]]))
        ids = load_ff_split(root, "test")
        assert ids == {"001", "023", "005", "100"}


def test_load_ff_split_rejects_unknown_split() -> None:
    with tempfile.TemporaryDirectory() as d, pytest.raises(ValueError, match="ff_split"):
        load_ff_split(d, "holdout")


def test_load_ff_split_missing_file() -> None:
    with tempfile.TemporaryDirectory() as d, pytest.raises(FileNotFoundError, match="splits"):
        load_ff_split(d, "test")


def test_face_forensics_ff_split_keeps_only_listed_videos() -> None:
    pytest.importorskip("torch")
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        for vid in ("001", "023", "999"):
            _save_synthetic(
                root / "original_sequences" / "youtube" / "c23" / "frames" / vid / "0001.png"
            )
        # Manipulated dirs use composite ids; pair (001,023) is in-split, (001,999) is not.
        for vid in ("001_023", "001_999"):
            _save_synthetic(
                root / "manipulated_sequences" / "Deepfakes" / "c23" / "frames" / vid / "0001.png"
            )
        (root / "splits").mkdir()
        (root / "splits" / "test.json").write_text(json.dumps([["001", "023"]]))
        ds = FaceForensicsAdapter(root, methods=("Deepfakes",), ff_split="test")
        # Real: 001, 023 (999 dropped).  Fake: 001_023 (composite of in-split pair).
        assert sorted(ds.video_ids()) == ["001", "001_023", "023"]


def test_split_videos_partitions_ids() -> None:
    train, val, test = split_videos(
        [f"v{i:03d}" for i in range(100)],
        seed=42,
        val_fraction=0.2,
        test_fraction=0.1,
    )
    assert len(test) == 10
    assert len(val) == 20
    assert len(train) == 70
    assert (train | val | test) == {f"v{i:03d}" for i in range(100)}
    assert train.isdisjoint(val) and val.isdisjoint(test) and train.isdisjoint(test)


def test_split_videos_is_deterministic() -> None:
    a = split_videos([f"v{i}" for i in range(50)], seed=7)
    b = split_videos([f"v{i}" for i in range(50)], seed=7)
    assert a == b


def _save_celeb(root: Path, subset: str, vid: str, n: int = 1) -> None:
    for i in range(n):
        _save_synthetic(root / subset / "frames" / vid / f"{i:04d}.png", seed=hash((vid, i)) & 0xFF)


def test_celeb_df_adapter_basic_layout() -> None:
    pytest.importorskip("torch")
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        _save_celeb(root, "Celeb-real", "id0_0001")
        _save_celeb(root, "YouTube-real", "00170")
        _save_celeb(root, "Celeb-synthesis", "id0_id1_0009")
        ds = CelebDFAdapter(root)
        assert len(ds) == 3
        labels = sorted(ds._records, key=lambda r: r.video_id)
        kinds = {(r.video_id, r.label) for r in ds._records}
        assert ("Celeb-real/id0_0001", 0) in kinds
        assert ("YouTube-real/00170", 0) in kinds
        assert ("Celeb-synthesis/id0_id1_0009", 1) in kinds
        del labels  # silence unused


def test_celeb_df_adapter_returns_tensor_and_label() -> None:
    pytest.importorskip("torch")
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        _save_celeb(root, "Celeb-real", "v0")
        ds = CelebDFAdapter(root, target_size=(16, 16))
        img, label = ds[0]
        import torch

        assert isinstance(img, torch.Tensor)
        assert img.shape == (3, 16, 16)
        assert label == 0


def test_celeb_df_adapter_max_frames_per_video() -> None:
    pytest.importorskip("torch")
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        _save_celeb(root, "Celeb-real", "v0", n=20)
        ds = CelebDFAdapter(root, max_frames_per_video=5)
        assert len(ds) == 5


def test_celeb_df_adapter_empty_root_errors() -> None:
    pytest.importorskip("torch")
    with tempfile.TemporaryDirectory() as d, pytest.raises(FileNotFoundError, match="no frames"):
        CelebDFAdapter(d)


def test_load_celeb_testing_list_parses_format() -> None:
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "List_of_testing_videos.txt"
        p.write_text(
            "1 YouTube-real/00170.mp4\n"
            "0 Celeb-synthesis/id0_id16_0009.mp4\n"
            "\n"  # blank line — should be skipped
            "1 Celeb-real/id1_0003.mp4\n",
        )
        ids = load_celeb_testing_list(p)
        assert ids == {
            "YouTube-real/00170",
            "Celeb-synthesis/id0_id16_0009",
            "Celeb-real/id1_0003",
        }


def test_load_celeb_testing_list_missing_errors() -> None:
    with tempfile.TemporaryDirectory() as d, pytest.raises(FileNotFoundError, match="testing_videos"):
        load_celeb_testing_list(Path(d) / "missing.txt")


def test_celeb_df_adapter_testing_list_filters() -> None:
    pytest.importorskip("torch")
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        _save_celeb(root, "Celeb-real", "kept_video")
        _save_celeb(root, "Celeb-real", "dropped_video")
        _save_celeb(root, "Celeb-synthesis", "kept_synth")
        (root / "List_of_testing_videos.txt").write_text(
            "1 Celeb-real/kept_video.mp4\n"
            "0 Celeb-synthesis/kept_synth.mp4\n",
        )
        ds = CelebDFAdapter(root, testing_list=True)
        assert sorted(ds.video_ids()) == [
            "Celeb-real/kept_video",
            "Celeb-synthesis/kept_synth",
        ]


def test_celeb_df_adapter_testing_list_explicit_path() -> None:
    pytest.importorskip("torch")
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        _save_celeb(root, "Celeb-real", "v_keep")
        _save_celeb(root, "Celeb-real", "v_drop")
        # Put the list outside the root.
        list_path = Path(d) / "external_list.txt"
        list_path.write_text("1 Celeb-real/v_keep.mp4\n")
        ds = CelebDFAdapter(root, testing_list=list_path)
        assert ds.video_ids() == ["Celeb-real/v_keep"]


def test_image_folder_records_carry_video_id() -> None:
    pytest.importorskip("torch")
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        # Layout: real/<video_id>/<frame>.png — tests that parent.name is captured.
        _save_synthetic(root / "real" / "vid_a" / "0001.png")
        _save_synthetic(root / "real" / "vid_a" / "0002.png")
        _save_synthetic(root / "fake" / "vid_b" / "0001.png")
        ds = ImageFolderDataset(root / "real", root / "fake")
        ids = sorted({rec.video_id for rec in ds._records})
        assert ids == ["vid_a", "vid_b"]
