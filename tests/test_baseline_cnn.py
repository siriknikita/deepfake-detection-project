"""Smoke tests for the pure-CNN baseline used in the pivot study."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image


def _make_tiny_dataset(root: Path, n_per_class: int = 4, size: tuple[int, int] = (64, 64)) -> None:
    rng = np.random.default_rng(0)
    for label in ("real", "fake"):
        (root / label).mkdir(exist_ok=True, parents=True)
        for i in range(n_per_class):
            arr = (rng.random((*size, 3)) * 255).astype(np.uint8)
            Image.fromarray(arr).save(root / label / f"{i}.png")


def test_build_baseline_classifier_forward() -> None:
    pytest.importorskip("torch")
    import torch

    from forge_detect.baseline_cnn import build_baseline_classifier

    model = build_baseline_classifier(pretrained=False)
    x = torch.rand(2, 3, 64, 64)
    with torch.no_grad():
        y = model(x)
    assert y.shape == (2, 1)
    assert torch.isfinite(y).all()


def test_train_baseline_cnn_runs() -> None:
    pytest.importorskip("torch")
    import torch

    from forge_detect.baseline_cnn import (
        BaselineConfig,
        build_baseline_classifier,
        evaluate_baseline_cnn,
        train_baseline_cnn,
    )
    from forge_detect.datasets import ImageFolderDataset

    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        _make_tiny_dataset(root)
        train_ds = ImageFolderDataset(root / "real", root / "fake", target_size=(64, 64))
        val_ds = ImageFolderDataset(root / "real", root / "fake", target_size=(64, 64))
        cfg = BaselineConfig(
            epochs=2,
            batch_size=2,
            num_workers=0,
            device="cpu",
            mixed_precision=False,
            checkpoint_dir=root / "runs",
        )
        out = train_baseline_cnn(train_ds, val_ds, cfg, pretrained=False)
        assert len(out["history"]) == 2
        run_dir = Path(out["run_dir"])  # type: ignore[arg-type]
        assert (run_dir / "best.pt").exists()
        # Loaded weights run forward and produce valid metrics.
        model = build_baseline_classifier(pretrained=False)
        model.load_state_dict(torch.load(run_dir / "best.pt", map_location="cpu"))
        metrics = evaluate_baseline_cnn(model, val_ds, device="cpu", num_workers=0)
        assert metrics["accuracy"] >= 0.0
        assert metrics["n_real"] + metrics["n_fake"] == len(val_ds)


def test_baseline_loss_decreases_or_stays_finite() -> None:
    """Loss must be finite at every epoch — the training-mechanics invariant."""
    pytest.importorskip("torch")

    from forge_detect.baseline_cnn import BaselineConfig, train_baseline_cnn
    from forge_detect.datasets import ImageFolderDataset

    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        _make_tiny_dataset(root)
        train_ds = ImageFolderDataset(root / "real", root / "fake", target_size=(64, 64))
        cfg = BaselineConfig(
            epochs=2,
            batch_size=2,
            num_workers=0,
            device="cpu",
            mixed_precision=False,
            checkpoint_dir=root / "runs",
        )
        out = train_baseline_cnn(train_ds, None, cfg, pretrained=False)
        for entry in out["history"]:
            assert np.isfinite(entry["train_loss"])
            assert entry["train_loss"] > 0
