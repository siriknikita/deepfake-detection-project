"""End-to-end evaluation: pipeline → features → classifier → AUROC.

Two related entry points:

- :func:`extract_features_over_dataset` runs the full settlement pipeline
  on every image in a dataset and produces an ``(N, F)`` feature matrix
  plus an ``(N,)`` label vector. This is the time-consuming step
  (typically seconds per image on CPU; faster on GPU once the
  PyTorch backend matures). Output is cached as a CSV to disk so
  classifier training does not re-run the pipeline.

- :func:`evaluate_pipeline` runs the full chain on a labeled dataset
  (extract features → train classifier → evaluate on held-out set) and
  prints AUROC, accuracy, and the top feature importances.

The CLI ``forge-detect eval`` subcommand is the user-facing wrapper.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from forge_detect.classifier import (
    ClassifierMetrics,
    evaluate_classifier,
    train_classifier,
)
from forge_detect.config import PipelineParams
from forge_detect.datasets import stratified_split
from forge_detect.features import FEATURE_NAMES
from forge_detect.pipeline import detect


@dataclass
class FeatureMatrix:
    """Cached features extracted over a dataset."""

    features: np.ndarray  # (N, F)
    labels: np.ndarray  # (N,)
    paths: list[str]  # source image paths

    def save(self, path: str | Path) -> None:
        import pandas as pd

        df = pd.DataFrame(self.features, columns=list(FEATURE_NAMES))
        df["label"] = self.labels
        df["path"] = self.paths
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)

    @classmethod
    def load(cls, path: str | Path) -> FeatureMatrix:
        import pandas as pd

        df = pd.read_csv(path)
        feat_cols = list(FEATURE_NAMES)
        return cls(
            features=df[feat_cols].to_numpy(dtype=np.float64),
            labels=df["label"].to_numpy(dtype=np.int64),
            paths=df["path"].tolist(),
        )


def extract_features_over_dataset(
    dataset: Any,
    *,
    device: str = "cpu",
    params: PipelineParams | None = None,
    log_every: int = 25,
    cnn_model: Any | None = None,
    cnn_device: str = "cpu",
) -> FeatureMatrix:
    """Run the full pipeline over every record of ``dataset`` and return features.

    The dataset must yield ``(image_chw_or_path, label[, mask])`` records;
    image tensors / numpy arrays are converted to ``(H, W, 3)`` float32
    before being passed to :func:`forge_detect.pipeline.detect`.

    Args:
        dataset: Iterable / torch Dataset of (image, label[, ...]).
        device: ``"cpu"`` or ``"cuda"`` — passed to
            :func:`forge_detect.pipeline.detect`.
        params: Pipeline configuration, defaults to PipelineParams().
        log_every: Print a progress line every N records (set to 0 to silence).
        cnn_model: Optional trained ChromaticEfficientNet to compute the
            trust map. If ``None``, the heuristic fallback is used.
        cnn_device: PyTorch device the CNN lives on.

    Returns:
        :class:`FeatureMatrix` with ``(features, labels, paths)``.
    """
    feats: list[np.ndarray] = []
    labels: list[int] = []
    paths: list[str] = []
    t0 = time.time()
    for i in range(len(dataset)):  # type: ignore[arg-type]
        record = dataset[i]
        image = record[0]
        label = record[1]
        rgb = _to_hwc_float(image)
        result = detect(
            rgb,
            device=device,
            params=params,
            cnn_model=cnn_model,
            cnn_device=cnn_device,
        )
        feats.append(result.features)
        labels.append(int(label))
        path = getattr(getattr(dataset, "_records", [None])[i], "path", None)
        if path is None:
            path = getattr(getattr(dataset, "_records", [None])[i], "image_path", None)
        paths.append(str(path) if path is not None else f"<idx={i}>")
        if log_every > 0 and (i + 1) % log_every == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (len(dataset) - (i + 1)) / max(rate, 1e-9)  # type: ignore[arg-type]
            print(f"  [{i + 1}/{len(dataset)}] {rate:.2f} images/s, ETA {eta:.0f}s")
    return FeatureMatrix(
        features=np.stack(feats),
        labels=np.asarray(labels, dtype=np.int64),
        paths=paths,
    )


def _to_hwc_float(image: Any) -> np.ndarray:
    """Convert an image record (path / numpy / torch tensor) to (H, W, 3) float32."""
    if isinstance(image, (str, Path)):
        from forge_detect.pipeline import load_image

        return load_image(image)
    arr = np.asarray(image)
    if arr.ndim == 3 and arr.shape[0] == 3:
        # CHW -> HWC.
        arr = np.transpose(arr, (1, 2, 0))
    if arr.dtype != np.float32:
        arr = arr.astype(np.float32)
    if arr.max() > 1.5:  # likely 0-255 input
        arr = arr / 255.0
    return arr


def evaluate_pipeline(
    dataset: Any,
    *,
    device: str = "cpu",
    params: PipelineParams | None = None,
    val_fraction: float = 0.2,
    test_fraction: float = 0.2,
    seed: int = 0,
    cache_path: str | Path | None = None,
    cnn_model: Any | None = None,
    cnn_device: str = "cpu",
) -> tuple[ClassifierMetrics, ClassifierMetrics, Any, FeatureMatrix]:
    """Train + evaluate the binary classifier on a labeled dataset.

    Steps:
      1. Extract features over the entire dataset (cached if
         ``cache_path`` exists).
      2. Stratified split into train / val / test.
      3. Train the binary classifier on train.
      4. Evaluate on val (for tuning) and test (the reported number).

    Returns:
        ``(val_metrics, test_metrics, fitted_pipeline, FeatureMatrix)``.
    """
    if cache_path is not None and Path(cache_path).exists():
        print(f"loading cached features from {cache_path}")
        fm = FeatureMatrix.load(cache_path)
    else:
        print(f"extracting features over {len(dataset)} records ...")  # type: ignore[arg-type]
        fm = extract_features_over_dataset(
            dataset,
            device=device,
            params=params,
            cnn_model=cnn_model,
            cnn_device=cnn_device,
        )
        if cache_path is not None:
            fm.save(cache_path)
            print(f"cached features to {cache_path}")

    n = fm.features.shape[0]
    train_idx, val_idx, test_idx = stratified_split(
        n,
        seed=seed,
        val_fraction=val_fraction,
        test_fraction=test_fraction,
    )
    print(f"split: train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}")

    pipeline = train_classifier(fm.features[train_idx], fm.labels[train_idx])
    val_metrics = evaluate_classifier(pipeline, fm.features[val_idx], fm.labels[val_idx])
    test_metrics = evaluate_classifier(pipeline, fm.features[test_idx], fm.labels[test_idx])
    return val_metrics, test_metrics, pipeline, fm


__all__ = [
    "FeatureMatrix",
    "evaluate_pipeline",
    "extract_features_over_dataset",
]
