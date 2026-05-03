"""Binary classifier on the impact-map feature vector.

The classifier sits at the very end of the pipeline. Its input is the
24-D feature vector produced by :func:`forge_detect.features.extract_features`;
its output is ``Pr(deepfake | image) ∈ [0, 1]``.

By design we keep this model small and interpretable — a
:class:`sklearn.ensemble.GradientBoostingClassifier` wrapped in a
``StandardScaler`` pipeline. Feature importances from the trained
booster tell us *which* forensic signal — large residual percentiles,
sharp Laplacian energies, anomalous energy ratios, slow convergence —
the classifier finds discriminative on a given dataset, which is the
kind of explanation a forensic tool actually needs.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from forge_detect.features import FEATURE_NAMES


@dataclass
class ClassifierMetrics:
    """Output of :func:`evaluate_classifier`.

    Frame-level metrics (``auroc``, ``accuracy``) are computed on every
    individual frame. When ``video_ids`` is supplied to
    :func:`evaluate_classifier`, the additional ``video_*`` fields hold
    the *video-level* numbers obtained by pooling frame probabilities
    per source video — the standard cross-dataset metric used by every
    published deepfake-detection benchmark. ``video_auroc_mean`` averages
    frame probabilities per video, ``video_auroc_max`` takes the maximum
    (more sensitive but noisier). When pooling is not requested, the
    video fields are ``NaN`` and ``n_videos`` is ``0``.
    """

    auroc: float
    accuracy: float
    n_real: int
    n_fake: int
    feature_importances: dict[str, float]
    video_auroc_mean: float = float("nan")
    video_auroc_max: float = float("nan")
    video_accuracy_mean: float = float("nan")
    n_videos: int = 0
    n_video_real: int = 0
    n_video_fake: int = 0


def train_classifier(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    n_estimators: int = 200,
    max_depth: int = 3,
    learning_rate: float = 0.1,
    random_state: int = 0,
) -> Any:
    """Fit a Gradient Boosting + StandardScaler pipeline on ``(features, labels)``.

    Args:
        features: ``(N, F)`` array of float64 features. ``F`` must equal
            ``len(FEATURE_NAMES)``.
        labels: ``(N,)`` binary labels — 0 = real, 1 = fake.
        n_estimators / max_depth / learning_rate / random_state: standard
            sklearn GBC hyperparameters.

    Returns:
        A fitted :class:`sklearn.pipeline.Pipeline`. Use ``.predict_proba``
        for the deepfake probability or ``.predict`` for hard labels.
    """
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    if features.ndim != 2:
        msg = f"features must be 2D, got {features.shape}"
        raise ValueError(msg)
    if features.shape[1] != len(FEATURE_NAMES):
        msg = (
            f"features must have {len(FEATURE_NAMES)} columns "
            f"(matching FEATURE_NAMES), got {features.shape[1]}"
        )
        raise ValueError(msg)
    if labels.shape[0] != features.shape[0]:
        msg = f"labels and features rows must agree: {labels.shape} vs {features.shape}"
        raise ValueError(msg)

    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "gbc",
                GradientBoostingClassifier(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    learning_rate=learning_rate,
                    random_state=random_state,
                ),
            ),
        ],
    )
    pipeline.fit(features, labels)
    return pipeline


def evaluate_classifier(
    pipeline: Any,
    features: np.ndarray,
    labels: np.ndarray,
    *,
    video_ids: list[str] | np.ndarray | None = None,
) -> ClassifierMetrics:
    """Compute AUROC, accuracy, importances, and (optionally) video-level AUROC.

    Args:
        pipeline: Fitted classifier from :func:`train_classifier`.
        features: ``(N, F)`` feature matrix.
        labels: ``(N,)`` binary labels (0 = real, 1 = fake).
        video_ids: Optional ``(N,)`` source-video id per row. When given,
            frame probabilities are pooled per video to produce
            ``video_auroc_mean`` (mean-pool) and ``video_auroc_max``
            (max-pool). Both pooling rules are reported because mean is
            the default convention but max is a useful robustness check
            — if max-pool blows up but mean-pool holds, the per-frame
            scores have a few outliers that shouldn't dominate.
    """
    from sklearn.metrics import accuracy_score, roc_auc_score

    proba = pipeline.predict_proba(features)[:, 1]
    pred = (proba >= 0.5).astype(int)
    importances_arr = pipeline.named_steps["gbc"].feature_importances_
    importances = dict(zip(FEATURE_NAMES, importances_arr.tolist(), strict=True))

    metrics = ClassifierMetrics(
        auroc=float(roc_auc_score(labels, proba)) if len(np.unique(labels)) > 1 else float("nan"),
        accuracy=float(accuracy_score(labels, pred)),
        n_real=int((labels == 0).sum()),
        n_fake=int((labels == 1).sum()),
        feature_importances=importances,
    )

    if video_ids is not None:
        v_ids = np.asarray(video_ids)
        if v_ids.shape[0] != labels.shape[0]:
            msg = (
                f"video_ids length {v_ids.shape[0]} does not match "
                f"labels length {labels.shape[0]}"
            )
            raise ValueError(msg)
        v_metrics = _video_level_metrics(proba, labels, v_ids)
        metrics.video_auroc_mean = v_metrics["auroc_mean"]
        metrics.video_auroc_max = v_metrics["auroc_max"]
        metrics.video_accuracy_mean = v_metrics["accuracy_mean"]
        metrics.n_videos = v_metrics["n_videos"]
        metrics.n_video_real = v_metrics["n_video_real"]
        metrics.n_video_fake = v_metrics["n_video_fake"]
    return metrics


def _video_level_metrics(
    proba: np.ndarray,
    labels: np.ndarray,
    video_ids: np.ndarray,
) -> dict[str, Any]:
    """Pool frame probabilities per video; return AUROC under mean+max pooling.

    Each video gets a single score and a single label. The label is
    deterministic — every frame from a given source video shares it
    by construction, so we take the first occurrence and assert
    uniqueness as a sanity check.
    """
    from sklearn.metrics import accuracy_score, roc_auc_score

    unique_videos = np.unique(video_ids)
    pooled_proba_mean = np.empty(len(unique_videos), dtype=np.float64)
    pooled_proba_max = np.empty(len(unique_videos), dtype=np.float64)
    pooled_labels = np.empty(len(unique_videos), dtype=np.int64)
    for k, vid in enumerate(unique_videos):
        mask = video_ids == vid
        frame_probs = proba[mask]
        frame_labels = labels[mask]
        if not np.all(frame_labels == frame_labels[0]):
            msg = f"video {vid!r} has frames with conflicting labels — data is corrupt"
            raise ValueError(msg)
        pooled_proba_mean[k] = float(frame_probs.mean())
        pooled_proba_max[k] = float(frame_probs.max())
        pooled_labels[k] = int(frame_labels[0])

    n_classes = len(np.unique(pooled_labels))
    auroc_mean = (
        float(roc_auc_score(pooled_labels, pooled_proba_mean)) if n_classes > 1 else float("nan")
    )
    auroc_max = (
        float(roc_auc_score(pooled_labels, pooled_proba_max)) if n_classes > 1 else float("nan")
    )
    pred_mean = (pooled_proba_mean >= 0.5).astype(int)
    return {
        "auroc_mean": auroc_mean,
        "auroc_max": auroc_max,
        "accuracy_mean": float(accuracy_score(pooled_labels, pred_mean)),
        "n_videos": int(len(unique_videos)),
        "n_video_real": int((pooled_labels == 0).sum()),
        "n_video_fake": int((pooled_labels == 1).sum()),
    }


def save_classifier(pipeline: Any, path: str | Path) -> None:
    """Pickle the trained pipeline to ``path``."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("wb") as f:
        pickle.dump(pipeline, f)


def load_classifier(path: str | Path) -> Any:
    """Load a previously :func:`save_classifier`-ed pipeline."""
    with Path(path).open("rb") as f:
        return pickle.load(f)


__all__ = [
    "ClassifierMetrics",
    "evaluate_classifier",
    "load_classifier",
    "save_classifier",
    "train_classifier",
]
