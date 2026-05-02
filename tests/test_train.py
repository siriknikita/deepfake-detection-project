"""Smoke tests for the CNN training loop and trust-map model.

These tests verify the *mechanics* of training (loss is finite, gradients
flow, checkpoints round-trip, model produces a valid trust map) — they do
**not** assert any specific accuracy. The validation accuracy on tiny
random-noise datasets is at chance by design; what matters is that the
plumbing works end-to-end.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from forge_detect.datasets import ImageFolderDataset


def _make_tiny_dataset(root: Path, n_per_class: int = 4, size: tuple[int, int] = (64, 64)) -> None:
    rng = np.random.default_rng(0)
    for label in ("real", "fake"):
        (root / label).mkdir(exist_ok=True, parents=True)
        for i in range(n_per_class):
            arr = (rng.random((*size, 3)) * 255).astype(np.uint8)
            Image.fromarray(arr).save(root / label / f"{i}.png")


def test_chromatic_inputs_shape() -> None:
    pytest.importorskip("torch")
    import torch

    from forge_detect.cnn import chromatic_inputs

    rgb = torch.rand(2, 3, 32, 32)
    out = chromatic_inputs(rgb)
    assert out.shape == (2, 6, 32, 32)


def test_chromatic_inputs_rejects_wrong_shape() -> None:
    pytest.importorskip("torch")
    import torch

    from forge_detect.cnn import chromatic_inputs

    bad = torch.rand(2, 4, 32, 32)
    with pytest.raises(ValueError, match="expected"):
        chromatic_inputs(bad)


def test_build_chromatic_efficientnet_forward() -> None:
    pytest.importorskip("torch")
    import torch

    from forge_detect.cnn import build_chromatic_efficientnet

    model = build_chromatic_efficientnet(pretrained=False)
    x = torch.rand(1, 3, 64, 64)
    with torch.no_grad():
        y = model(x)
    assert y.shape == (1, 64, 64)
    assert (y >= 0).all() and (y <= 1).all()
    assert torch.isfinite(y).all()


def test_save_and_load_weights_round_trip() -> None:
    pytest.importorskip("torch")
    import torch

    from forge_detect.cnn import build_chromatic_efficientnet, load_weights, save_weights

    model = build_chromatic_efficientnet(pretrained=False)
    with tempfile.TemporaryDirectory() as d:
        ckpt = Path(d) / "x.pt"
        save_weights(model, ckpt)
        model2 = build_chromatic_efficientnet(pretrained=False)
        # Sanity: random init differs.
        p1 = next(model.parameters()).detach().clone()
        p2 = next(model2.parameters()).detach().clone()
        if p1.numel() == p2.numel():
            # Almost surely different due to random init.
            differs = not torch.allclose(p1, p2)
            assert differs
        load_weights(model2, ckpt)
        # After loading, parameters should match exactly.
        for a, b in zip(model.parameters(), model2.parameters(), strict=True):
            assert torch.equal(a, b)


def test_predict_trust_map_returns_valid_array() -> None:
    pytest.importorskip("torch")

    from forge_detect.cnn import build_chromatic_efficientnet, predict_trust_map

    model = build_chromatic_efficientnet(pretrained=False)
    rng = np.random.default_rng(0)
    rgb = rng.random((48, 64, 3)).astype(np.float32)
    w = predict_trust_map(model, rgb, device="cpu")
    assert w.shape == (48, 64)
    assert w.dtype == np.float32
    assert (w >= 0.0).all() and (w <= 1.0).all()


def test_train_cnn_runs_and_writes_checkpoints() -> None:
    pytest.importorskip("torch")
    import torch

    from forge_detect.train import TrainingConfig, train_cnn

    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        _make_tiny_dataset(root)
        train_ds = ImageFolderDataset(root / "real", root / "fake", target_size=(64, 64))
        val_ds = ImageFolderDataset(root / "real", root / "fake", target_size=(64, 64))
        cfg = TrainingConfig(
            epochs=2,
            batch_size=2,
            num_workers=0,
            device="cpu",
            mixed_precision=False,
            log_every=0,
            checkpoint_dir=root / "runs",
        )
        out = train_cnn(train_ds, val_ds, cfg, pretrained=False)
        assert len(out["history"]) == 2
        run_dir = Path(out["run_dir"])
        # Both checkpoints written and loadable.
        assert (run_dir / "best.pt").exists()
        assert (run_dir / "last.pt").exists()
        # Loaded weights produce valid output.
        from forge_detect.cnn import build_chromatic_efficientnet, load_weights

        model = build_chromatic_efficientnet(pretrained=False)
        load_weights(model, run_dir / "best.pt")
        with torch.no_grad():
            x = torch.rand(1, 3, 64, 64)
            y = model(x)
        assert y.shape == (1, 64, 64)
        assert (y >= 0).all() and (y <= 1).all()


def test_train_cnn_loss_is_finite_each_epoch() -> None:
    pytest.importorskip("torch")

    from forge_detect.train import TrainingConfig, train_cnn

    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        _make_tiny_dataset(root)
        train_ds = ImageFolderDataset(root / "real", root / "fake", target_size=(64, 64))
        cfg = TrainingConfig(
            epochs=2,
            batch_size=2,
            num_workers=0,
            device="cpu",
            mixed_precision=False,
            log_every=0,
            checkpoint_dir=root / "runs",
        )
        out = train_cnn(train_ds, None, cfg, pretrained=False)
        for entry in out["history"]:
            assert np.isfinite(entry["train_loss"])
            assert entry["train_loss"] > 0  # BCE on a real binary problem is positive


def test_train_cnn_resumes_from_checkpoint() -> None:
    """Training stopped mid-run should resume from the last completed epoch."""
    pytest.importorskip("torch")

    from forge_detect.train import TrainingConfig, train_cnn

    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        _make_tiny_dataset(root)
        train_ds = ImageFolderDataset(root / "real", root / "fake", target_size=(64, 64))
        # First run: 1 epoch.
        cfg1 = TrainingConfig(
            epochs=1,
            batch_size=2,
            num_workers=0,
            device="cpu",
            mixed_precision=False,
            log_every=0,
            checkpoint_dir=root / "runs",
        )
        out1 = train_cnn(train_ds, None, cfg1, pretrained=False)
        run_dir = Path(out1["run_dir"])
        assert (run_dir / "checkpoint.pt").exists()
        assert len(out1["history"]) == 1

        # Second run: same run_dir, increased epochs to 3 — should run only
        # epochs 1 and 2 (epoch 0 was already completed).
        cfg2 = TrainingConfig(
            epochs=3,
            batch_size=2,
            num_workers=0,
            device="cpu",
            mixed_precision=False,
            log_every=0,
            checkpoint_dir=root / "runs",
        )
        out2 = train_cnn(train_ds, None, cfg2, pretrained=False, resume_dir=run_dir)
        # History from the resumed run should contain all 3 epochs (1 from
        # the original + 2 added) and the run_dir is the same.
        assert Path(out2["run_dir"]) == run_dir
        epochs_in_history = [e["epoch"] for e in out2["history"]]
        assert epochs_in_history == [0.0, 1.0, 2.0]


def test_train_cnn_resume_already_complete() -> None:
    """Resuming a run that already hit `epochs` completes immediately."""
    pytest.importorskip("torch")

    from forge_detect.train import TrainingConfig, train_cnn

    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        _make_tiny_dataset(root)
        train_ds = ImageFolderDataset(root / "real", root / "fake", target_size=(64, 64))
        cfg = TrainingConfig(
            epochs=1,
            batch_size=2,
            num_workers=0,
            device="cpu",
            mixed_precision=False,
            log_every=0,
            checkpoint_dir=root / "runs",
        )
        out1 = train_cnn(train_ds, None, cfg, pretrained=False)
        run_dir = Path(out1["run_dir"])
        # Resume with the same epoch count: nothing to do.
        out2 = train_cnn(train_ds, None, cfg, pretrained=False, resume_dir=run_dir)
        assert Path(out2["run_dir"]) == run_dir
        assert len(out2["history"]) == 1
