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
    """Cached features extracted over a dataset.

    ``video_ids`` is optional — older caches written before video-level
    pooling was added do not have a ``video_id`` column. When loading
    such a cache, the field falls back to the parent directory of each
    path, which matches how the FF++ and Celeb-DF adapters assign ids.
    """

    features: np.ndarray  # (N, F)
    labels: np.ndarray  # (N,)
    paths: list[str]  # source image paths
    video_ids: list[str] | None = None  # source video id per row

    def save(self, path: str | Path) -> None:
        import pandas as pd

        df = pd.DataFrame(self.features, columns=list(FEATURE_NAMES))
        df["label"] = self.labels
        df["path"] = self.paths
        if self.video_ids is not None:
            df["video_id"] = self.video_ids
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)

    @classmethod
    def load(cls, path: str | Path) -> FeatureMatrix:
        import pandas as pd

        df = pd.read_csv(path)
        feat_cols = list(FEATURE_NAMES)
        if "video_id" in df.columns:
            video_ids: list[str] | None = df["video_id"].astype(str).tolist()
        else:
            # Fallback: parent dir name of each frame path. Matches the
            # FaceForensicsAdapter / CelebDFAdapter video_id convention,
            # so old caches still pool correctly.
            video_ids = [Path(p).parent.name for p in df["path"].tolist()]
        return cls(
            features=df[feat_cols].to_numpy(dtype=np.float64),
            labels=df["label"].to_numpy(dtype=np.int64),
            paths=df["path"].tolist(),
            video_ids=video_ids,
        )


def extract_features_over_dataset(
    dataset: Any,
    *,
    device: str = "cpu",
    params: PipelineParams | None = None,
    log_every: int = 25,
    cnn_model: Any | None = None,
    cnn_device: str = "cpu",
    cache_path: str | Path | None = None,
    save_every: int = 100,
) -> FeatureMatrix:
    """Run the pipeline over every record of ``dataset`` and return features.

    Crash-resumable: when ``cache_path`` is given, the function appends to
    a partial CSV every ``save_every`` records. On restart, already-
    processed paths are loaded from the cache and skipped, so re-running
    after an outage continues from the last saved checkpoint instead of
    starting over.

    Args:
        dataset: Iterable / torch Dataset of (image, label[, ...]).
        device: ``"cpu"`` or ``"cuda"`` — passed to
            :func:`forge_detect.pipeline.detect`.
        params: Pipeline configuration, defaults to PipelineParams().
        log_every: Print a progress line every N records (set to 0 to silence).
        cnn_model: Optional trained ChromaticEfficientNet to compute the
            trust map. If ``None``, the heuristic fallback is used.
        cnn_device: PyTorch device the CNN lives on.
        cache_path: If given, write progress here every `save_every`
            records and skip already-processed paths on resume.
        save_every: Flush cache every this many records.

    Returns:
        :class:`FeatureMatrix` with ``(features, labels, paths)``.
    """
    cache_p = Path(cache_path) if cache_path is not None else None

    feats: list[np.ndarray] = []
    labels: list[int] = []
    paths: list[str] = []
    video_ids: list[str] = []
    done_paths: set[str] = set()

    if cache_p is not None and cache_p.exists():
        existing = FeatureMatrix.load(cache_p)
        feats = list(existing.features)
        labels = list(existing.labels.tolist())
        paths = list(existing.paths)
        video_ids = list(existing.video_ids) if existing.video_ids is not None else []
        done_paths = set(paths)
        print(f"  resuming from cache: {len(done_paths)} records already processed")

    t0 = time.time()
    n = len(dataset)  # type: ignore[arg-type]
    processed_in_session = 0
    for i in range(n):
        # Determine the record's stable path key BEFORE running the pipeline,
        # so we can short-circuit cached entries without paying for the load.
        path_key = _record_path_key(dataset, i)
        if path_key in done_paths:
            continue

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
        paths.append(path_key)
        video_ids.append(_record_video_id(dataset, i, fallback=path_key))
        done_paths.add(path_key)
        processed_in_session += 1

        if log_every > 0 and processed_in_session % log_every == 0:
            elapsed = time.time() - t0
            rate = processed_in_session / max(elapsed, 1e-9)
            eta = (n - len(done_paths)) / max(rate, 1e-9)
            print(
                f"  [{len(done_paths)}/{n}] {rate:.2f} images/s, ETA {eta:.0f}s",
            )
        if cache_p is not None and processed_in_session % save_every == 0:
            FeatureMatrix(
                features=np.stack(feats),
                labels=np.asarray(labels, dtype=np.int64),
                paths=paths,
                video_ids=video_ids,
            ).save(cache_p)

    fm = FeatureMatrix(
        features=np.stack(feats) if feats else np.empty((0, 0), dtype=np.float64),
        labels=np.asarray(labels, dtype=np.int64),
        paths=paths,
        video_ids=video_ids,
    )
    if cache_p is not None:
        fm.save(cache_p)
    return fm


def _record_path_key(dataset: Any, idx: int) -> str:
    """Stable identifier for record ``idx`` of ``dataset``.

    Falls back to ``<idx=N>`` when the dataset does not expose ``_records``
    with a path field — only ImageFolderDataset and FaceForensicsAdapter
    currently do.
    """
    records = getattr(dataset, "_records", None)
    if records is not None and idx < len(records):
        rec = records[idx]
        path = getattr(rec, "path", None) or getattr(rec, "image_path", None)
        if path is not None:
            return str(path)
    # Subset wraps another Dataset; recurse through the indices attribute.
    if hasattr(dataset, "indices") and hasattr(dataset, "dataset"):
        return _record_path_key(dataset.dataset, dataset.indices[idx])
    return f"<idx={idx}>"


def _record_video_id(dataset: Any, idx: int, *, fallback: str) -> str:
    """Stable per-video grouping key for record ``idx``.

    Looks up ``_records[idx].video_id`` if the adapter exposes it
    (FF++, Celeb-DF, ImageFolder all do post-refactor). For arbitrary
    iterable datasets we fall back to the parent directory of the
    record's image path — which matches the convention adapters use,
    so video-level pooling still works on bespoke datasets.
    """
    records = getattr(dataset, "_records", None)
    if records is not None and idx < len(records):
        vid = getattr(records[idx], "video_id", None)
        if vid is not None:
            return str(vid)
    if hasattr(dataset, "indices") and hasattr(dataset, "dataset"):
        return _record_video_id(dataset.dataset, dataset.indices[idx], fallback=fallback)
    return Path(fallback).parent.name if "/" in fallback or "\\" in fallback else fallback


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
    # extract_features_over_dataset itself handles loading + appending the
    # cache when cache_path is supplied, so we always call it. The "fully
    # cached" case (every record already in the cache) returns immediately
    # because every dataset[i] gets short-circuited by done_paths.
    print(f"extracting features over {len(dataset)} records ...")  # type: ignore[arg-type]
    fm = extract_features_over_dataset(
        dataset,
        device=device,
        params=params,
        cnn_model=cnn_model,
        cnn_device=cnn_device,
        cache_path=cache_path,
    )

    n = fm.features.shape[0]
    train_idx, val_idx, test_idx = stratified_split(
        n,
        seed=seed,
        val_fraction=val_fraction,
        test_fraction=test_fraction,
    )
    print(f"split: train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}")

    pipeline = train_classifier(fm.features[train_idx], fm.labels[train_idx])
    # Pass video_ids when available so video-level AUROC is reported alongside
    # frame-level. Frame-level numbers stay computed regardless.
    vids = np.asarray(fm.video_ids) if fm.video_ids is not None else None
    val_metrics = evaluate_classifier(
        pipeline,
        fm.features[val_idx],
        fm.labels[val_idx],
        video_ids=vids[val_idx] if vids is not None else None,
    )
    test_metrics = evaluate_classifier(
        pipeline,
        fm.features[test_idx],
        fm.labels[test_idx],
        video_ids=vids[test_idx] if vids is not None else None,
    )
    return val_metrics, test_metrics, pipeline, fm


__all__ = [
    "FeatureMatrix",
    "evaluate_pipeline",
    "extract_features_over_dataset",
]
