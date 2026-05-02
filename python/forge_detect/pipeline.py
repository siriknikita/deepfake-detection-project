"""High-level detection pipeline glueing the CNN, math core, and classifier."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from forge_detect.backends import Backend, make_backend
from forge_detect.config import PipelineParams
from forge_detect.features import FEATURE_NAMES, extract_features
from forge_detect.trust_map import heuristic_trust_map
from forge_detect.types import SolveResult


@dataclass(frozen=True)
class DetectResult:
    """Output of :func:`detect`."""

    image_path: Path
    solve: SolveResult
    features: np.ndarray
    feature_names: tuple[str, ...]
    deepfake_probability: float | None


def load_image(path: str | Path) -> np.ndarray:
    """Load an image file and return a ``(H, W, 3)`` float32 array in ``[0, 1]``."""
    img = Image.open(path).convert("RGB")
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return arr


def detect(
    image: str | Path | np.ndarray,
    *,
    device: str = "cpu",
    params: PipelineParams | None = None,
    classifier: object | None = None,
    trust_map: np.ndarray | None = None,
    cnn_model: object | None = None,
    cnn_device: str = "cpu",
) -> DetectResult:
    """Run end-to-end detection on a single image.

    Args:
        image: Path to an image file, or a pre-loaded ``(H, W, 3)`` array.
        device: ``"cpu"`` (Rust core) or ``"cuda"`` (PyTorch reimplementation).
        params: Pipeline configuration; defaults to :class:`PipelineParams` defaults.
        classifier: Optional trained binary classifier exposing
            ``predict_proba(features_2d)``. If ``None``, the result has
            ``deepfake_probability=None`` and only raw features are returned.
        trust_map: Optional pre-computed ``W_cnn``. Takes precedence over
            ``cnn_model`` when both are supplied.
        cnn_model: Optional trained ChromaticEfficientNet (or any callable
            ``model(rgb_BCHW_tensor) -> (B, H, W) tensor``). When supplied
            and no explicit ``trust_map`` is given, the model produces the
            trust map; otherwise the heuristic fallback is used.
        cnn_device: PyTorch device the model lives on (``"cpu"``,
            ``"cuda"``, ``"mps"``).

    Returns:
        A :class:`DetectResult` with the impact map, feature vector, and
        (optionally) the deepfake probability.
    """
    image_path = Path(image) if isinstance(image, (str, Path)) else Path("(in-memory)")
    rgb = (
        load_image(image) if isinstance(image, (str, Path)) else np.asarray(image, dtype=np.float32)
    )
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        msg = f"image must be (H, W, 3), got {rgb.shape}"
        raise ValueError(msg)

    if trust_map is not None:
        w_cnn = trust_map
    elif cnn_model is not None:
        from forge_detect.cnn import predict_trust_map

        w_cnn = predict_trust_map(cnn_model, rgb, device=cnn_device)
    else:
        w_cnn = heuristic_trust_map(rgb)
    if w_cnn.shape != rgb.shape[:2]:
        msg = f"trust_map shape {w_cnn.shape} must match image H × W {rgb.shape[:2]}"
        raise ValueError(msg)

    backend: Backend = make_backend(device)
    params = params or PipelineParams()
    solve = backend.solve(rgb, w_cnn, params)
    features = extract_features(solve)

    proba: float | None = None
    if classifier is not None:
        prediction = classifier.predict_proba(features.reshape(1, -1))  # type: ignore[attr-defined]
        # Standard sklearn convention: proba[:, 1] is the positive class.
        proba = float(prediction[0, 1])

    return DetectResult(
        image_path=image_path,
        solve=solve,
        features=features,
        feature_names=FEATURE_NAMES,
        deepfake_probability=proba,
    )


__all__ = ["DetectResult", "detect", "load_image"]
