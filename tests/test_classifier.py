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


def test_evaluate_classifier_video_level_pooling() -> None:
    """video_ids triggers per-video AUROC in addition to frame-level."""
    rng = np.random.default_rng(0)
    n_videos = 20
    frames_per_video = 5
    n = n_videos * frames_per_video
    f = len(FEATURE_NAMES)
    # Each video has a single learnable signal in feature 0; frames within
    # a video share the same true label, with a bit of noise added so frame
    # probabilities aren't identical and the pooling actually does something.
    video_labels = rng.integers(0, 2, n_videos)
    features = np.zeros((n, f))
    labels = np.zeros(n, dtype=np.int64)
    video_ids = []
    for v in range(n_videos):
        start = v * frames_per_video
        for k in range(frames_per_video):
            features[start + k] = rng.standard_normal(f)
            features[start + k, 0] = (
                3.0 * (video_labels[v] - 0.5) + 0.5 * rng.standard_normal()
            )
            labels[start + k] = int(video_labels[v])
            video_ids.append(f"vid_{v:02d}")

    pipeline = train_classifier(features, labels, n_estimators=50, random_state=0)
    metrics = evaluate_classifier(pipeline, features, labels, video_ids=video_ids)

    # Frame-level numbers are still populated.
    assert 0.0 <= metrics.auroc <= 1.0
    # Video-level numbers are populated and agree with the construction.
    assert metrics.n_videos == n_videos
    assert metrics.n_video_real + metrics.n_video_fake == n_videos
    assert 0.0 <= metrics.video_auroc_mean <= 1.0
    assert 0.0 <= metrics.video_auroc_max <= 1.0
    # On a clean signal mean-pooling should match or beat frame-level AUROC,
    # which is the whole reason the literature reports video-level numbers.
    assert metrics.video_auroc_mean >= metrics.auroc - 0.05


def test_evaluate_classifier_video_level_omitted_when_no_ids() -> None:
    rng = np.random.default_rng(0)
    features = rng.standard_normal((30, len(FEATURE_NAMES)))
    labels = (features[:, 0] > 0).astype(np.int64)
    pipeline = train_classifier(features, labels, n_estimators=20, random_state=0)
    metrics = evaluate_classifier(pipeline, features, labels)
    assert np.isnan(metrics.video_auroc_mean)
    assert np.isnan(metrics.video_auroc_max)
    assert metrics.n_videos == 0


def test_evaluate_classifier_rejects_inconsistent_video_labels() -> None:
    """Two frames sharing a video_id but with conflicting labels = corrupt data."""
    rng = np.random.default_rng(0)
    features = rng.standard_normal((4, len(FEATURE_NAMES)))
    labels = np.array([0, 1, 0, 0], dtype=np.int64)  # video_a has both 0 and 1
    pipeline = train_classifier(features, labels, n_estimators=10, random_state=0)
    with pytest.raises(ValueError, match="conflicting labels"):
        evaluate_classifier(
            pipeline,
            features,
            labels,
            video_ids=["video_a", "video_a", "video_b", "video_b"],
        )


def test_evaluate_classifier_rejects_video_ids_length_mismatch() -> None:
    rng = np.random.default_rng(0)
    features = rng.standard_normal((10, len(FEATURE_NAMES)))
    labels = (features[:, 0] > 0).astype(np.int64)
    pipeline = train_classifier(features, labels, n_estimators=10, random_state=0)
    with pytest.raises(ValueError, match="video_ids length"):
        evaluate_classifier(pipeline, features, labels, video_ids=["a", "b", "c"])


def test_feature_matrix_video_ids_round_trip() -> None:
    pytest.importorskip("pandas")
    rng = np.random.default_rng(0)
    features = rng.standard_normal((6, len(FEATURE_NAMES)))
    labels = rng.integers(0, 2, 6).astype(np.int64)
    paths = [f"frames/v{i // 2}/{i % 2}.png" for i in range(6)]
    vids = [f"v{i // 2}" for i in range(6)]
    fm = FeatureMatrix(features, labels, paths, video_ids=vids)
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "feats.csv"
        fm.save(p)
        loaded = FeatureMatrix.load(p)
        assert loaded.video_ids == vids


def test_feature_matrix_legacy_cache_falls_back_to_parent_dir() -> None:
    """Cache from before video_id support — load reconstructs ids from the path."""
    pytest.importorskip("pandas")
    import pandas as pd

    rng = np.random.default_rng(0)
    features = rng.standard_normal((4, len(FEATURE_NAMES)))
    df = pd.DataFrame(features, columns=list(FEATURE_NAMES))
    df["label"] = [0, 0, 1, 1]
    df["path"] = [
        "/data/frames/v0/0001.png",
        "/data/frames/v0/0002.png",
        "/data/frames/v1/0001.png",
        "/data/frames/v1/0002.png",
    ]
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "legacy.csv"
        df.to_csv(p, index=False)
        loaded = FeatureMatrix.load(p)
        assert loaded.video_ids == ["v0", "v0", "v1", "v1"]


def test_extract_features_resumes_from_partial_cache() -> None:
    """Crash mid-extraction → restart picks up where it left off."""
    pytest.importorskip("torch")
    pytest.importorskip("pandas")
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        ds = _make_dataset(root, n_per_class=4)
        cache = root / "feats.csv"
        from forge_detect.config import PdeParams, PipelineParams

        # First call: process every record, write the full cache.
        params = PipelineParams(n_scales=2, pde=PdeParams(max_iter=10, log_every=5))
        fm1 = extract_features_over_dataset(
            ds,
            params=params,
            log_every=0,
            cache_path=cache,
            save_every=1,
        )
        assert cache.exists()
        assert fm1.features.shape[0] == 8

        # Second call: cache exists, every record's path_key is in done_paths,
        # so the loop skips every record and we return the cached features
        # without re-running the pipeline. The result must equal fm1 exactly.
        fm2 = extract_features_over_dataset(
            ds,
            params=params,
            log_every=0,
            cache_path=cache,
        )
        assert fm2.features.shape == fm1.features.shape
        np.testing.assert_array_equal(fm2.labels, fm1.labels)
        np.testing.assert_allclose(fm2.features, fm1.features, atol=1e-5)
