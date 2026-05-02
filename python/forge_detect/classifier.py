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
    """Output of :func:`evaluate_classifier`."""

    auroc: float
    accuracy: float
    n_real: int
    n_fake: int
    feature_importances: dict[str, float]


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
) -> ClassifierMetrics:
    """Compute AUROC, accuracy, and feature importances on a held-out set."""
    from sklearn.metrics import accuracy_score, roc_auc_score

    proba = pipeline.predict_proba(features)[:, 1]
    pred = (proba >= 0.5).astype(int)
    importances_arr = pipeline.named_steps["gbc"].feature_importances_
    importances = dict(zip(FEATURE_NAMES, importances_arr.tolist(), strict=True))

    return ClassifierMetrics(
        auroc=float(roc_auc_score(labels, proba)) if len(np.unique(labels)) > 1 else float("nan"),
        accuracy=float(accuracy_score(labels, pred)),
        n_real=int((labels == 0).sum()),
        n_fake=int((labels == 1).sum()),
        feature_importances=importances,
    )


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
