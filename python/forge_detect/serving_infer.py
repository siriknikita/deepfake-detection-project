"""Inference adapter for the web serving layer.

Single seam between the research core (``forge_detect``) and the FastAPI
app under ``serving/``. The HTTP layer only parses requests and
serialises whatever this module returns.

Responsibilities:

1. A declarative **model registry** (:data:`MODEL_REGISTRY`). Adding a
   future checkpoint is one tuple entry, not an API change.
2. :func:`infer_image` — turn one uploaded image + a list of selected
   model ids into a comparison payload: the shared face crop, each
   model's score / verdict, the heuristic 6-panel manifold figure, the
   per-model input-channel previews (what each model "sees"), and the
   green **peak score** (strongest deepfake probability + which model).

The CNN single-image path is assembled here from existing, tested
producers so it matches the training-time data pipeline exactly:

- face crop via MTCNN (mirrors ``scripts/extract_faces.py``)
- physics maps from one ``detect()`` solve (mirrors
  ``scripts/cache_physics_maps.py`` — heuristic trust map → PDE solve)
- frequency maps from :func:`forge_detect.frequency_map.frequency_maps`
- channel assembly + per-image normalisation through the *public*
  :class:`forge_detect.datasets.ChannelSource` contract, so the tensor
  is byte-for-byte what the cached-npz dataset would have produced.

The physics solve is the expensive step; it runs **once per request**
and is reused for the heuristic figure, the physics channels, and the
channel previews.
"""

from __future__ import annotations

import base64
import io
import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np

# Lock matplotlib to the headless Agg backend before pyplot is imported
# anywhere. FastAPI runs sync handlers in a worker-thread pool; a GUI
# backend (macOS picks "MacOSX" by default) crashes off the main thread.
os.environ.setdefault("MPLBACKEND", "Agg")

# python/forge_detect/serving_infer.py -> repo root is three parents up.
_REPO_ROOT = Path(__file__).resolve().parents[2]

_CROP_SIZE = 256
_CROP_MARGIN = 0.3


class InferenceError(RuntimeError):
    """Raised for any inference failure the API should surface to the client.

    The HTTP layer maps this to a 4xx with the message intact (unknown
    model id, missing checkpoint, unreadable image, …).
    """


@dataclass(frozen=True)
class ModelSpec:
    """One selectable model.

    Attributes:
        id: Stable identifier used by the API and the frontend selector.
        label: Human-facing name shown in the UI.
        description: One-line explanation of what the model is.
        kind: ``"heuristic"`` (math pipeline, no weights) or ``"cnn"``.
        model_type: Builder selector — ``"heuristic"``, ``"baseline"``
            (3-ch RGB EfficientNet) or ``"physics"`` (n-ch stem-surgery
            EfficientNet).
        weights: Repo-relative checkpoint path, or ``None`` for the
            heuristic model.
        channel_spec: ``--channels`` spec understood by
            :func:`forge_detect.datasets.parse_channel_spec`.
        in_channels: Input channel count the checkpoint's stem expects.
    """

    id: str
    label: str
    description: str
    kind: str
    model_type: str
    weights: str | None
    channel_spec: str
    in_channels: int

    def weights_path(self) -> Path | None:
        """Absolute checkpoint path, or ``None`` for the heuristic model."""
        if self.weights is None:
            return None
        return _REPO_ROOT / self.weights

    def is_available(self) -> bool:
        """True when the model can actually run on this checkout."""
        if self.kind == "heuristic":
            return True
        path = self.weights_path()
        return path is not None and path.is_file()


MODEL_REGISTRY: tuple[ModelSpec, ...] = (
    ModelSpec(
        id="heuristic",
        label="Heuristic (math pipeline)",
        description=(
            "Physical-manifold settlement with the chromatic-residual trust "
            "map. No training; produces the 6-panel diagnostic figure."
        ),
        kind="heuristic",
        model_type="heuristic",
        weights=None,
        channel_spec="rgb",
        in_channels=3,
    ),
    ModelSpec(
        id="baseline_3ch",
        label="Baseline (RGB EfficientNet-B0)",
        description="Phase-2 3-channel RGB control. Pure learned classifier.",
        kind="cnn",
        model_type="baseline",
        weights="runs_mac/baseline_3ch_faces/baseline_run/best.pt",
        channel_spec="rgb",
        in_channels=3,
    ),
    ModelSpec(
        id="physics_6ch",
        label="Phase 2 (RGB + physics, 6-ch)",
        description=(
            "EfficientNet-B0 with RGB plus the three physics maps "
            "(W_cnn, z*, R) stacked as input channels."
        ),
        kind="cnn",
        model_type="physics",
        weights="runs_mac/physics_6ch_faces_heuristic/best.pt",
        channel_spec="rgb,physics",
        in_channels=6,
    ),
    ModelSpec(
        id="physics_9ch",
        label="Phase 3 (RGB + physics + frequency, 9-ch)",
        description=(
            "Phase-2 stack plus three frequency-domain channels "
            "(DCT energy, DCT high-band ratio, FFT log-magnitude)."
        ),
        kind="cnn",
        model_type="physics",
        weights="runs_mac/physics_9ch_freq/last.pt",
        channel_spec="rgb,physics,frequency",
        in_channels=9,
    ),
)

_BY_ID: dict[str, ModelSpec] = {spec.id: spec for spec in MODEL_REGISTRY}


def list_models() -> list[dict[str, object]]:
    """Serialisable registry view for ``GET /api/models``."""
    return [
        {
            "id": spec.id,
            "label": spec.label,
            "description": spec.description,
            "kind": spec.kind,
            "in_channels": spec.in_channels,
            "available": spec.is_available(),
        }
        for spec in MODEL_REGISTRY
    ]


def get_model_spec(model_id: str) -> ModelSpec:
    """Look up a model by id, raising :class:`InferenceError` if unknown."""
    spec = _BY_ID.get(model_id)
    if spec is None:
        known = ", ".join(sorted(_BY_ID))
        msg = f"unknown model id {model_id!r}; known ids: {known}"
        raise InferenceError(msg)
    return spec


def _resolve_model_ids(model_ids: list[str]) -> list[ModelSpec]:
    """Validate the selection: non-empty, known, available, de-duplicated."""
    if not model_ids:
        raise InferenceError("select at least one model")
    specs: list[ModelSpec] = []
    seen: set[str] = set()
    for model_id in model_ids:
        if model_id in seen:
            continue
        seen.add(model_id)
        spec = get_model_spec(model_id)
        if not spec.is_available():
            msg = (
                f"model {spec.id!r} is unavailable — checkpoint "
                f"{spec.weights!r} not found on this checkout"
            )
            raise InferenceError(msg)
        specs.append(spec)
    return specs


# --------------------------------------------------------------------------- #
# Image decoding + face crop (mirrors scripts/extract_faces.py)
# --------------------------------------------------------------------------- #


def _decode_image(image_bytes: bytes) -> Any:
    """Bytes -> RGB ``PIL.Image``. Raises :class:`InferenceError` on garbage."""
    from PIL import Image, UnidentifiedImageError

    try:
        return Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except (UnidentifiedImageError, OSError) as exc:
        msg = f"could not decode the uploaded file as an image: {exc}"
        raise InferenceError(msg) from exc


# Module-level caches. Plain dict mutation (no `global`) so the detector
# and models are built once and reused across requests.
_MTCNN_CACHE: dict[str, Any] = {}


def _get_mtcnn() -> Any:
    """Lazily build and cache the MTCNN detector (CPU; fast for one image)."""
    if "d" not in _MTCNN_CACHE:
        from facenet_pytorch import MTCNN

        _MTCNN_CACHE["d"] = MTCNN(
            image_size=_CROP_SIZE,
            margin=0,
            min_face_size=20,
            thresholds=[0.6, 0.7, 0.7],
            factor=0.709,
            post_process=False,
            device="cpu",
            keep_all=False,
        )
    return _MTCNN_CACHE["d"]


def _square_box(
    box: tuple[float, float, float, float], img_w: int, img_h: int, margin: float
) -> tuple[int, int, int, int]:
    """MTCNN ``(x1,y1,x2,y2)`` -> clipped square crop box with margin.

    Same geometry as ``scripts/extract_faces.py``'s
    ``_square_box_from_detection`` so crops match training.
    """
    x1, y1, x2, y2 = box
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    side = max(x2 - x1, y2 - y1) * (1.0 + margin)
    half = side / 2.0
    return (
        max(0, int(round(cx - half))),
        max(0, int(round(cy - half))),
        min(img_w, int(round(cx + half))),
        min(img_h, int(round(cy + half))),
    )


def _face_crop(pil_image: Any) -> tuple[np.ndarray, bool]:
    """Detect the largest face and return ``(256x256x3 float32 [0,1], found)``.

    Falls back to a centre square crop when no face is detected — the
    same completeness contract the training preprocessing uses.
    """
    from PIL import Image

    mtcnn = _get_mtcnn()
    boxes, _probs = mtcnn.detect(pil_image)
    if boxes is not None and len(boxes) > 0:
        box_arr = boxes if boxes.ndim == 1 else boxes[0]
        box = _square_box(
            (float(box_arr[0]), float(box_arr[1]), float(box_arr[2]), float(box_arr[3])),
            pil_image.width,
            pil_image.height,
            _CROP_MARGIN,
        )
        found = True
    else:
        side = min(pil_image.width, pil_image.height)
        x1 = (pil_image.width - side) // 2
        y1 = (pil_image.height - side) // 2
        box = (x1, y1, x1 + side, y1 + side)
        found = False
    crop = pil_image.crop(box).resize((_CROP_SIZE, _CROP_SIZE), Image.BILINEAR)
    rgb = np.asarray(crop, dtype=np.float32) / 255.0
    return rgb, found


# --------------------------------------------------------------------------- #
# Physics / frequency maps (computed once per request, reused everywhere)
# --------------------------------------------------------------------------- #


@dataclass
class _Maps:
    """In-memory equivalents of the cached physics/frequency npz contents.

    ``detect_result`` is the single PDE solve; physics channels and the
    heuristic 6-panel both read from it. Frequency maps are lazy (only
    the 9-ch model needs them).
    """

    crop_rgb: np.ndarray
    w_cnn: np.ndarray
    detect_result: Any
    _freq: dict[str, np.ndarray] | None = None

    def physics_arrays(self) -> dict[str, np.ndarray]:
        """``{wcnn, z_star, residual}`` keyed for the physics ChannelSource.

        Round-tripped through float16 so the values match the cached-npz
        precision the model was trained on (the cache writes float16).
        """
        solve = self.detect_result.solve
        return {
            "wcnn": _as_cached(self.w_cnn),
            "z_star": _as_cached(solve.z_star),
            "residual": _as_cached(solve.residual),
        }

    def frequency_arrays(self) -> dict[str, np.ndarray]:
        """``{dct_block_energy, dct_high_ratio, fft_radial_logmag}``."""
        if self._freq is None:
            from forge_detect.frequency_map import frequency_maps

            dct_e, dct_r, fft_m = frequency_maps(self.crop_rgb)
            self._freq = {
                "dct_block_energy": _as_cached(dct_e),
                "dct_high_ratio": _as_cached(dct_r),
                "fft_radial_logmag": _as_cached(fft_m),
            }
        return self._freq


def _as_cached(arr: np.ndarray) -> np.ndarray:
    """Mimic the float16 npz cache round-trip (removes train/infer skew)."""
    return arr.astype(np.float16).astype(np.float32, copy=False)


def _compute_maps(crop_rgb: np.ndarray) -> _Maps:
    """One heuristic trust map + one PDE solve, shared by all models."""
    from forge_detect.config import PipelineParams
    from forge_detect.pipeline import detect
    from forge_detect.trust_map import heuristic_trust_map

    w_cnn = heuristic_trust_map(crop_rgb)
    # Mirrors scripts/cache_physics_maps.py: detect() with the heuristic
    # trust map injected so the PDE is not re-seeded from a recompute.
    result = detect(crop_rgb, params=PipelineParams(), trust_map=w_cnn, device="cpu")
    return _Maps(crop_rgb=crop_rgb, w_cnn=w_cnn, detect_result=result)


# --------------------------------------------------------------------------- #
# CNN model cache + scoring
# --------------------------------------------------------------------------- #

_MODELS: dict[str, Any] = {}


def _select_device() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _get_cnn_model(spec: ModelSpec) -> tuple[Any, str]:
    """Build + load a checkpoint once; cache it. Returns ``(model, device)``."""
    import torch

    from forge_detect.baseline_cnn import (
        build_baseline_classifier,
        build_physics_classifier,
    )
    from forge_detect.cnn import load_weights

    device = _select_device()
    if spec.id not in _MODELS:
        if spec.model_type == "baseline":
            model = build_baseline_classifier(pretrained=False)
        else:
            model = build_physics_classifier(
                in_channels=spec.in_channels, pretrained=False
            )
        weights = spec.weights_path()
        assert weights is not None  # guaranteed by _resolve_model_ids
        load_weights(model, weights)
        model.eval()
        try:
            model = model.to(device)
        except (RuntimeError, NotImplementedError):
            device = "cpu"  # e.g. an MPS op gap — degrade gracefully
            model = model.to(device)
        _MODELS[spec.id] = (model, device)
    model, device = _MODELS[spec.id]
    _ = torch  # keep the import meaningful for the type checker
    return model, device


def _channel_tensor(spec: ModelSpec, maps: _Maps) -> np.ndarray:
    """Assemble the ``(C, H, W)`` float32 input tensor for ``spec``.

    RGB first, then each ChannelSource's contribution normalised through
    its own public ``normalize`` callable — identical to what the cached
    dataset feeds the model at training time.
    """
    from forge_detect.datasets import parse_channel_spec

    chw = np.transpose(maps.crop_rgb, (2, 0, 1)).astype(np.float32, copy=False)
    parts: list[np.ndarray] = [chw]
    for src in parse_channel_spec(spec.channel_spec):
        family = src.extra.get("family")
        if family == "physics":
            available = maps.physics_arrays()
        elif family == "frequency":
            available = maps.frequency_arrays()
        else:  # pragma: no cover - registry only uses physics/frequency
            msg = f"unsupported channel family {family!r} for {spec.id!r}"
            raise InferenceError(msg)
        parts.append(src.normalize({k: available[k] for k in src.npz_keys}))
    tensor = np.concatenate(parts, axis=0)
    if tensor.shape[0] != spec.in_channels:
        msg = (
            f"assembled {tensor.shape[0]} channels for {spec.id!r}, "
            f"expected {spec.in_channels}"
        )
        raise InferenceError(msg)
    return tensor


def _cnn_score(spec: ModelSpec, maps: _Maps) -> float:
    """Sigmoid(model(tensor)) — the deepfake probability for one image."""
    import torch

    model, device = _get_cnn_model(spec)
    tensor = _channel_tensor(spec, maps)
    x = torch.from_numpy(tensor[None]).to(device)
    with torch.no_grad():
        logit = model(x).squeeze()
        prob = torch.sigmoid(logit)
    return float(prob.item())


# --------------------------------------------------------------------------- #
# Visualisations
# --------------------------------------------------------------------------- #


def _png_b64(fig: Any) -> str:
    """Render a matplotlib figure to a base64 data URI and close it."""
    import matplotlib.pyplot as plt

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def _rgb_b64(rgb: np.ndarray) -> str:
    """``(H,W,3) [0,1]`` -> base64 PNG data URI."""
    from PIL import Image

    arr = (np.clip(rgb, 0.0, 1.0) * 255.0).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def _map_b64(arr: np.ndarray, cmap: str) -> str:
    """Colormap a ``(H,W)`` map to a base64 PNG (matplotlib Agg)."""
    import matplotlib.colors as mcolors
    from matplotlib import colormaps
    from PIL import Image

    finite = np.nan_to_num(arr.astype(np.float32))
    lo, hi = float(finite.min()), float(finite.max())
    norm = mcolors.Normalize(vmin=lo, vmax=hi if hi > lo else lo + 1e-6)
    rgba = colormaps[cmap](norm(finite))
    img = (rgba[..., :3] * 255.0).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def _heuristic_panel_b64(maps: _Maps) -> str:
    """The 6-panel diagnostic figure for the heuristic model."""
    from forge_detect.viz import panel

    fig = panel(maps.detect_result, maps.crop_rgb, maps.w_cnn)
    return _png_b64(fig)


def _channel_previews(spec: ModelSpec, maps: _Maps) -> list[dict[str, str]]:
    """The input channels each model "sees", as labelled preview images."""
    previews: list[dict[str, str]] = [
        {"label": "Face crop (RGB input)", "png": _rgb_b64(maps.crop_rgb)}
    ]
    families = {
        src.extra.get("family") for src in _safe_parse(spec.channel_spec)
    }
    if "physics" in families:
        solve = maps.detect_result.solve
        previews += [
            {"label": "W_cnn (trust map)", "png": _map_b64(maps.w_cnn, "viridis")},
            {"label": "z* (settled manifold)", "png": _map_b64(solve.z_star, "magma")},
            {"label": "R (flow break)", "png": _map_b64(solve.residual, "seismic")},
        ]
    if "frequency" in families:
        freq = maps.frequency_arrays()
        previews += [
            {"label": "DCT block energy", "png": _map_b64(freq["dct_block_energy"], "inferno")},
            {"label": "DCT high-band ratio", "png": _map_b64(freq["dct_high_ratio"], "inferno")},
            {"label": "FFT log-magnitude", "png": _map_b64(freq["fft_radial_logmag"], "inferno")},
        ]
    return previews


def _safe_parse(spec_str: str) -> list[Any]:
    from forge_detect.datasets import parse_channel_spec

    return parse_channel_spec(spec_str)


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #


def _analyse_crop(
    specs: list[ModelSpec], crop_rgb: np.ndarray, face_found: bool
) -> dict[str, object]:
    """Score + visualise every selected model on one prepared face crop."""
    needs_maps = any(s.kind == "heuristic" or s.model_type == "physics" for s in specs)
    maps = _compute_maps(crop_rgb) if needs_maps else _Maps(crop_rgb, crop_rgb[..., 0], None)

    models_out: list[dict[str, object]] = []
    peak: dict[str, object] | None = None
    for spec in specs:
        entry: dict[str, object] = {
            "id": spec.id,
            "label": spec.label,
            "kind": spec.kind,
        }
        if spec.kind == "heuristic":
            entry["score"] = None
            entry["verdict"] = None
            entry["note"] = "diagnostic only — no calibrated score on this checkout"
            entry["panel"] = _heuristic_panel_b64(maps)
            entry["channels"] = []
        else:
            score = _cnn_score(spec, maps)
            verdict = "fake" if score >= 0.5 else "real"
            entry["score"] = score
            entry["verdict"] = verdict
            entry["note"] = None
            entry["panel"] = None
            entry["channels"] = _channel_previews(spec, maps)
            if peak is None or score > float(peak["score"]):  # type: ignore[arg-type]
                peak = {
                    "model_id": spec.id,
                    "model_label": spec.label,
                    "score": score,
                    "verdict": verdict,
                }
        models_out.append(entry)

    return {
        "input": _rgb_b64(crop_rgb),
        "face_detected": face_found,
        "kind": "image",
        "peak": peak,
        "models": models_out,
    }


def infer_image(image_bytes: bytes, model_ids: list[str]) -> dict[str, object]:
    """Run every selected model on one image and return a comparison."""
    specs = _resolve_model_ids(model_ids)
    pil = _decode_image(image_bytes)
    crop_rgb, face_found = _face_crop(pil)
    return _analyse_crop(specs, crop_rgb, face_found)


# --------------------------------------------------------------------------- #
# Video — sampled frames, per-frame scoring, mean-pooled verdict
# --------------------------------------------------------------------------- #

_VIDEO_FRAMES = 12


def _score_crop(specs: list[ModelSpec], crop_rgb: np.ndarray) -> dict[str, float]:
    """CNN deepfake probability per model for one crop (no visuals)."""
    cnn = [s for s in specs if s.kind == "cnn"]
    if not cnn:
        return {}
    needs_maps = any(s.model_type == "physics" for s in cnn)
    maps = _compute_maps(crop_rgb) if needs_maps else _Maps(crop_rgb, crop_rgb[..., 0], None)
    return {s.id: _cnn_score(s, maps) for s in cnn}


def _extract_frames(video_bytes: bytes, n: int) -> list[Any]:
    """Sample up to ``n`` evenly-spaced frames from a video via ffmpeg.

    Mirrors the repo's ffmpeg-CLI convention (scripts/extract_frames.py)
    rather than adding a decoder dependency. Seeks to ``n`` evenly-spaced
    timestamps; falls back to fps sampling when the duration is unknown.
    """
    import shutil
    import subprocess
    import tempfile

    from PIL import Image

    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        raise InferenceError("video input requires ffmpeg/ffprobe on PATH")

    tmp = Path(tempfile.mkdtemp(prefix="forge_vid_"))
    try:
        src = tmp / "in.bin"
        src.write_bytes(video_bytes)

        duration = 0.0
        try:
            probe = subprocess.run(
                [
                    "ffprobe", "-v", "error", "-show_entries", "format=duration",
                    "-of", "default=nokey=1:noprint_wrappers=1", str(src),
                ],
                capture_output=True, text=True, timeout=30, check=False,
            )
            duration = float(probe.stdout.strip())
        except (ValueError, OSError, subprocess.SubprocessError):
            duration = 0.0

        produced: list[Path] = []
        if duration > 0.0:
            for i in range(n):
                t = duration * (i + 0.5) / n
                out = tmp / f"f{i:03d}.jpg"
                subprocess.run(
                    [
                        "ffmpeg", "-v", "error", "-ss", f"{t:.3f}", "-i", str(src),
                        "-frames:v", "1", "-q:v", "3", "-y", str(out),
                    ],
                    capture_output=True, timeout=60, check=False,
                )
                if out.exists():
                    produced.append(out)
        if not produced:
            subprocess.run(
                [
                    "ffmpeg", "-v", "error", "-i", str(src), "-vf", "fps=2",
                    "-frames:v", str(n), "-q:v", "3", "-y", str(tmp / "g%03d.jpg"),
                ],
                capture_output=True, timeout=120, check=False,
            )
            produced = sorted(tmp.glob("g*.jpg"))
        if not produced:
            raise InferenceError("could not decode any frames from the video")

        frames: list[Any] = []
        for p in produced:
            with Image.open(p) as im:
                frames.append(im.convert("RGB").copy())
        return frames
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def infer_video(
    video_bytes: bytes,
    model_ids: list[str],
    progress: Callable[[int, int], None] | None = None,
) -> dict[str, object]:
    """Sample frames, score each, mean-pool to a video-level verdict.

    Visuals (6-panel, channel previews) are rendered only for the single
    most-suspicious frame; per-frame P(fake) is returned for the timeline.
    """
    specs = _resolve_model_ids(model_ids)
    frames = _extract_frames(video_bytes, _VIDEO_FRAMES)
    total = len(frames)

    cnn_ids = [s.id for s in specs if s.kind == "cnn"]
    per_frame: list[dict[str, float]] = []
    crops: list[np.ndarray] = []
    faces: list[bool] = []
    for idx, pil in enumerate(frames):
        crop_rgb, found = _face_crop(pil)
        crops.append(crop_rgb)
        faces.append(found)
        per_frame.append(_score_crop(specs, crop_rgb))
        if progress is not None:
            progress(idx + 1, total)

    def frame_alarm(i: int) -> float:
        vals = [per_frame[i][m] for m in cnn_ids if m in per_frame[i]]
        return sum(vals) / len(vals) if vals else -1.0

    rep = max(range(total), key=frame_alarm) if cnn_ids else total // 2

    result = _analyse_crop(specs, crops[rep], faces[rep])
    models = cast("list[dict[str, object]]", result["models"])

    peak: dict[str, object] | None = None
    peak_s = -1.0
    for entry in models:
        mid = cast("str", entry["id"])
        if mid in cnn_ids:
            series = [per_frame[i][mid] for i in range(total) if mid in per_frame[i]]
            mean = sum(series) / len(series) if series else 0.0
            verdict = "fake" if mean >= 0.5 else "real"
            entry["score"] = mean
            entry["verdict"] = verdict
            entry["frame_scores"] = series
            if mean > peak_s:
                peak_s = mean
                peak = {
                    "model_id": mid,
                    "model_label": entry["label"],
                    "score": mean,
                    "verdict": verdict,
                }

    result["peak"] = peak
    result["kind"] = "video"
    result["n_frames"] = total
    return result


__all__ = [
    "MODEL_REGISTRY",
    "InferenceError",
    "ModelSpec",
    "get_model_spec",
    "infer_image",
    "infer_video",
    "list_models",
]
