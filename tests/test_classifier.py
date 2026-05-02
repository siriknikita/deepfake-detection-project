"""Tests for the binary classifier and end-to-end evaluation flow."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from forge_detect.classifier import (
    evaluate_classifier,
    load_classifier,
    save_classifier,
    train_classifier,
)
from forge_detect.datasets import ImageFolderDataset
from forge_detect.eval import FeatureMatrix, evaluate_pipeline, extract_features_over_dataset
from forge_detect.features import FEATURE_NAMES


def test_train_classifier_smoke() -> None:
    rng = np.random.default_rng(0)
    n = 80
    f = len(FEATURE_NAMES)
    # Build a dataset with a learnable signal: real (label=0) has the first
    # feature low; fake (label=1) has it high.
    features = rng.standard_normal((n, f))
    labels = (features[:, 0] > 0).astype(np.int64)
    pipeline = train_classifier(features, labels, n_estimators=20, random_state=0)
    proba = pipeline.predict_proba(features)
    assert proba.shape == (n, 2)
    assert np.allclose(proba.sum(axis=1), 1.0)


def test_train_classifier_rejects_wrong_shape() -> None:
    rng = np.random.default_rng(0)
    bad_features = rng.standard_normal((10, len(FEATURE_NAMES) - 1))
    labels = rng.integers(0, 2, 10)
    with pytest.raises(ValueError, match="must have"):
        train_classifier(bad_features, labels)


def test_train_classifier_rejects_mismatched_lengths() -> None:
    rng = np.random.default_rng(0)
    features = rng.standard_normal((10, len(FEATURE_NAMES)))
    labels = rng.integers(0, 2, 5)
    with pytest.raises(ValueError, match="agree"):
        train_classifier(features, labels)


def test_evaluate_classifier_perfectly_separable() -> None:
    rng = np.random.default_rng(0)
    n = 60
    f = len(FEATURE_NAMES)
    features = rng.standard_normal((n, f))
    labels = (features[:, 0] > 0).astype(np.int64)
    pipeline = train_classifier(features, labels, n_estimators=50, random_state=0)
    metrics = evaluate_classifier(pipeline, features, labels)
    assert metrics.auroc > 0.95
    assert metrics.accuracy > 0.95
    assert metrics.n_real + metrics.n_fake == n
    assert sum(metrics.feature_importances.values()) == pytest.approx(1.0, abs=1e-6)


def test_save_and_load_classifier_round_trip() -> None:
    rng = np.random.default_rng(0)
    features = rng.standard_normal((30, len(FEATURE_NAMES)))
    labels = (features[:, 0] > 0).astype(np.int64)
    pipeline = train_classifier(features, labels, n_estimators=10, random_state=0)
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "clf.pkl"
        save_classifier(pipeline, path)
        loaded = load_classifier(path)
        np.testing.assert_array_equal(
            pipeline.predict_proba(features),
            loaded.predict_proba(features),
        )


def test_feature_matrix_csv_round_trip() -> None:
    pytest.importorskip("pandas")
    rng = np.random.default_rng(0)
    features = rng.standard_normal((5, len(FEATURE_NAMES)))
    labels = rng.integers(0, 2, 5).astype(np.int64)
    paths = [f"img{i}.png" for i in range(5)]
    fm = FeatureMatrix(features, labels, paths)
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "feats.csv"
        fm.save(p)
        loaded = FeatureMatrix.load(p)
        np.testing.assert_allclose(loaded.features, features)
        np.testing.assert_array_equal(loaded.labels, labels)
        assert loaded.paths == paths


def _make_dataset(root: Path, n_per_class: int = 4) -> ImageFolderDataset:
    pytest.importorskip("torch")
    rng = np.random.default_rng(0)
    for label in ("real", "fake"):
        (root / label).mkdir(exist_ok=True, parents=True)
        for i in range(n_per_class):
            arr = (rng.random((48, 48, 3)) * 255).astype(np.uint8)
            Image.fromarray(arr).save(root / label / f"{i}.png")
    return ImageFolderDataset(root / "real", root / "fake", target_size=(32, 32))


def test_extract_features_over_dataset_smoke() -> None:
    pytest.importorskip("torch")
    with tempfile.TemporaryDirectory() as d:
        ds = _make_dataset(Path(d))
        from forge_detect.config import PdeParams, PipelineParams

        fm = extract_features_over_dataset(
            ds,
            params=PipelineParams(n_scales=2, pde=PdeParams(max_iter=15, log_every=5)),
            log_every=0,
        )
        assert fm.features.shape == (8, len(FEATURE_NAMES))
        assert fm.labels.shape == (8,)
        assert np.isfinite(fm.features).all()
        assert (fm.labels == 0).sum() == 4
        assert (fm.labels == 1).sum() == 4


def test_evaluate_pipeline_smoke() -> None:
    pytest.importorskip("torch")
    pytest.importorskip("pandas")
    with tempfile.TemporaryDirectory() as d:
        ds = _make_dataset(Path(d), n_per_class=8)
        from forge_detect.config import PdeParams, PipelineParams

        val_m, test_m, _pipeline, fm = evaluate_pipeline(
            ds,
            params=PipelineParams(n_scales=2, pde=PdeParams(max_iter=15, log_every=5)),
            val_fraction=0.25,
            test_fraction=0.25,
            seed=0,
        )
        assert fm.features.shape == (16, len(FEATURE_NAMES))
        # Random-noise input ⇒ AUROC at chance; what matters is the metrics
        # exist and are finite numbers in [0, 1].
        for m in (val_m, test_m):
            assert 0.0 <= m.accuracy <= 1.0
            # auroc may be NaN if a tiny split lands all-one-class.
            assert (np.isfinite(m.auroc) and 0.0 <= m.auroc <= 1.0) or np.isnan(m.auroc)
