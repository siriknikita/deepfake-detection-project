"""Dataset adapters for training and evaluation.

Three data sources are supported out of the box:

- :class:`ImageFolderDataset` — generic two-folder layout
  ``(real_dir/, fake_dir/)`` of frame-level images. Use this when you
  pre-extracted frames to disk in any layout.

- :class:`FaceForensicsAdapter` — the canonical FaceForensics++ tree:
  ``original_sequences/youtube/<compression>/frames/<video_id>/<frame>.png``
  and ``manipulated_sequences/<method>/<compression>/frames/<video_id>/<frame>.png``.
  Optionally honors the official ``splits/{train,val,test}.json``.

- :class:`CelebDFAdapter` — Celeb-DF v1 / v2 layout
  ``<subset>/frames/<video_id>/<frame>.png`` for ``subset`` in
  ``{Celeb-real, Celeb-synthesis, YouTube-real}``. Optionally honors
  ``List_of_testing_videos.txt`` (the published 518-video benchmark).

All adapters return ``(image, label[, mask])`` with ``image`` an
``(3, H, W)`` float32 tensor in ``[0, 1]`` and ``label`` ``0`` (real)
or ``1`` (fake). Each record also carries ``video_id``, which lets us
pool frame-level scores into video-level scores at evaluation time.

Where to get the data:

- *FaceForensics++*: https://github.com/ondyari/FaceForensics — sign the
  EULA, run their download script. ~1.5 TB at full quality.
- *Celeb-DF (v1, v2)*: https://github.com/yuezunli/celeb-deepfakeforensics
  — ~50 GB combined, EULA required. Used cross-dataset to test
  generalization of FF++-trained detectors.
- *DFDC*: https://www.kaggle.com/c/deepfake-detection-challenge/data —
  ~470 GB, larger and noisier than FF++.
- *FFHQ* (real-only): https://github.com/NVlabs/ffhq-dataset — 70 k
  high-quality 1024² faces, useful for real-only pretraining or as the
  "real" half against generated faces.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from PIL import Image

if TYPE_CHECKING:
    import torch
    from torch.utils.data import Dataset
else:
    try:
        from torch.utils.data import Dataset
    except ImportError:  # pragma: no cover — torch is optional at import
        Dataset = object  # type: ignore[assignment, misc]


_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}

# FaceForensics++ canonical manipulation methods.
FF_METHODS: tuple[str, ...] = (
    "Deepfakes",
    "Face2Face",
    "FaceSwap",
    "NeuralTextures",
)

FF_COMPRESSIONS: tuple[str, ...] = ("raw", "c23", "c40")

# Physics-map cache variants. Keep in sync with scripts/cache_physics_maps.py.
PHYSICS_VARIANTS: tuple[str, ...] = ("heuristic", "gtmask")
PHYSICS_NPZ_KEYS: tuple[str, ...] = ("wcnn", "z_star", "residual")
_PHYSICS_EPS = 1.0e-6

# Frequency-map cache variants (Phase 3). Keep in sync with
# scripts/cache_frequency_maps.py.
FREQUENCY_VARIANTS: tuple[str, ...] = ("default",)
FREQUENCY_NPZ_KEYS: tuple[str, ...] = (
    "dct_block_energy",
    "dct_high_ratio",
    "fft_radial_logmag",
)


def physics_npz_path(image_path: Path, variant: str) -> Path:
    """Locate the physics npz for an image laid out as ``.../<frames_dir>/<vid>/<frame>.<ext>``.

    Maps that path to a sibling cache directory whose name encodes both the
    frames directory and the variant, so caches from different sources
    don't collide:

    - ``frames/X/0001.png`` -> ``physics_<variant>/X/0001.npz``
    - ``frames_faces/X/0001.png`` -> ``physics_faces_<variant>/X/0001.npz``

    The caching script writes here; the adapters read from here.
    """
    if variant not in PHYSICS_VARIANTS:
        msg = f"physics variant must be one of {PHYSICS_VARIANTS}, got {variant!r}"
        raise ValueError(msg)
    frames_dir = image_path.parent.parent
    frames_dir_name = frames_dir.name
    if frames_dir_name == "frames":
        physics_dir_name = f"physics_{variant}"
    elif frames_dir_name.startswith("frames_"):
        # e.g. "frames_faces" -> "physics_faces_<variant>"
        suffix = frames_dir_name[len("frames_") :]
        physics_dir_name = f"physics_{suffix}_{variant}"
    else:
        msg = (
            f"image path {image_path} has parent directory name "
            f"{frames_dir_name!r}, expected 'frames' or 'frames_<...>' — "
            "cannot derive the physics-map cache location"
        )
        raise ValueError(msg)
    return (
        frames_dir.parent
        / physics_dir_name
        / image_path.parent.name
        / (image_path.stem + ".npz")
    )


def frequency_npz_path(image_path: Path, variant: str) -> Path:
    """Locate the frequency npz for an image, mirroring :func:`physics_npz_path`.

    - ``frames/X/0001.png`` -> ``frequency_<variant>/X/0001.npz``
    - ``frames_faces/X/0001.png`` -> ``frequency_faces_<variant>/X/0001.npz``

    The caching script ``scripts/cache_frequency_maps.py`` writes here; the
    adapters read from here through :class:`ChannelSource`.
    """
    if variant not in FREQUENCY_VARIANTS:
        msg = f"frequency variant must be one of {FREQUENCY_VARIANTS}, got {variant!r}"
        raise ValueError(msg)
    frames_dir = image_path.parent.parent
    frames_dir_name = frames_dir.name
    if frames_dir_name == "frames":
        cache_dir_name = f"frequency_{variant}"
    elif frames_dir_name.startswith("frames_"):
        suffix = frames_dir_name[len("frames_") :]
        cache_dir_name = f"frequency_{suffix}_{variant}"
    else:
        msg = (
            f"image path {image_path} has parent directory name "
            f"{frames_dir_name!r}, expected 'frames' or 'frames_<...>' — "
            "cannot derive the frequency-map cache location"
        )
        raise ValueError(msg)
    return (
        frames_dir.parent
        / cache_dir_name
        / image_path.parent.name
        / (image_path.stem + ".npz")
    )


@dataclass(frozen=True)
class ChannelSource:
    """Pluggable extra-channel source loaded from an on-disk npz cache.

    Each source contributes ``n_channels`` to the input tensor. Callers
    pre-build the cache with the matching ``scripts/cache_*.py`` script;
    the adapter reads + normalises at ``__getitem__`` time and concats the
    result to the RGB image.

    The contract is intentionally small: a name (for diagnostics + the
    spec parser), a channel count (for early-fail validation), a
    ``cache_path_fn`` mapping image path to npz path, the npz keys to
    read, and a ``normalize`` callable producing a ``(C, H, W)`` float32
    array from the loaded raw arrays.
    """

    name: str
    n_channels: int
    cache_path_fn: Callable[[Path], Path]
    npz_keys: tuple[str, ...]
    normalize: Callable[[dict[str, np.ndarray]], np.ndarray]
    extra: dict[str, str] = field(default_factory=dict)


def _normalize_physics(arrays: dict[str, np.ndarray]) -> np.ndarray:
    """Per-image normalisation for the Phase-2 physics maps.

    Matches the original Phase-2 normalisation exactly: trust map clipped
    to ``[0, 1]``, ``z*`` min-max scaled, residual squashed through
    ``tanh(R/std(R))`` and shifted to ``[0, 1]``. The scale is per-image
    on purpose — see the §11.1 paper rationale for why absolute scale is
    discarded.
    """
    wcnn = np.clip(arrays["wcnn"], 0.0, 1.0)
    z_star = arrays["z_star"]
    z_min, z_max = float(z_star.min()), float(z_star.max())
    z_norm = (z_star - z_min) / max(z_max - z_min, _PHYSICS_EPS)
    residual = arrays["residual"]
    r_std = max(float(residual.std()), _PHYSICS_EPS)
    r_norm = (np.tanh(residual / r_std) + 1.0) * 0.5
    return np.stack([wcnn, z_norm, r_norm], axis=0).astype(np.float32, copy=False)


def _normalize_frequency(arrays: dict[str, np.ndarray]) -> np.ndarray:
    """Frequency maps are already in ``[0, 1]`` from :mod:`forge_detect.frequency_map`.

    The cache writes float16 — clip back to ``[0, 1]`` to absorb any
    quantisation drift outside the range.
    """
    dct_e = np.clip(arrays["dct_block_energy"], 0.0, 1.0)
    dct_r = np.clip(arrays["dct_high_ratio"], 0.0, 1.0)
    fft_m = np.clip(arrays["fft_radial_logmag"], 0.0, 1.0)
    return np.stack([dct_e, dct_r, fft_m], axis=0).astype(np.float32, copy=False)


def physics_channel_source(variant: str = "heuristic") -> ChannelSource:
    """Build a :class:`ChannelSource` for the Phase-2 physics maps.

    The cache_path_fn is a ``functools.partial`` (not a lambda) so the
    returned source survives pickling across DataLoader worker processes
    spawned by multiprocessing.
    """
    if variant not in PHYSICS_VARIANTS:
        msg = f"physics variant must be one of {PHYSICS_VARIANTS}, got {variant!r}"
        raise ValueError(msg)
    return ChannelSource(
        name=f"physics:{variant}",
        n_channels=3,
        cache_path_fn=partial(physics_npz_path, variant=variant),
        npz_keys=PHYSICS_NPZ_KEYS,
        normalize=_normalize_physics,
        extra={"family": "physics", "variant": variant},
    )


def frequency_channel_source(variant: str = "default") -> ChannelSource:
    """Build a :class:`ChannelSource` for the Phase-3 frequency maps.

    See :func:`physics_channel_source` for the partial-vs-lambda rationale.
    """
    if variant not in FREQUENCY_VARIANTS:
        msg = f"frequency variant must be one of {FREQUENCY_VARIANTS}, got {variant!r}"
        raise ValueError(msg)
    return ChannelSource(
        name=f"frequency:{variant}",
        n_channels=3,
        cache_path_fn=partial(frequency_npz_path, variant=variant),
        npz_keys=FREQUENCY_NPZ_KEYS,
        normalize=_normalize_frequency,
        extra={"family": "frequency", "variant": variant},
    )


def parse_channel_spec(
    spec: str,
    *,
    physics_variant: str = "heuristic",
    frequency_variant: str = "default",
) -> list[ChannelSource]:
    """Parse a ``--channels`` CSV into an ordered list of channel sources.

    Token grammar (case-insensitive, comma-separated)::

        rgb                        — implicit base, no source produced
        physics                    — physics_channel_source(physics_variant)
        physics:<v>                — physics_channel_source(<v>)
        frequency                  — frequency_channel_source(frequency_variant)
        frequency:<v>              — frequency_channel_source(<v>)

    Empty / whitespace-only tokens are ignored. ``rgb`` is always implicit
    so callers can write ``"rgb,physics,frequency"`` for a 9-channel build
    or ``"physics,frequency"`` and get the same result. Each channel
    family may appear at most once.
    """
    sources: list[ChannelSource] = []
    seen: set[str] = set()
    for raw in spec.split(","):
        token = raw.strip().lower()
        if not token or token == "rgb":
            continue
        family, _, variant = token.partition(":")
        if family in seen:
            msg = f"channel family {family!r} appears twice in spec {spec!r}"
            raise ValueError(msg)
        if family == "physics":
            sources.append(physics_channel_source(variant or physics_variant))
        elif family == "frequency":
            sources.append(frequency_channel_source(variant or frequency_variant))
        else:
            msg = (
                f"unknown channel token {raw!r} in spec {spec!r}. "
                "Recognised families: rgb, physics, physics:<variant>, "
                "frequency, frequency:<variant>"
            )
            raise ValueError(msg)
        seen.add(family)
    return sources


def total_channels(sources: Iterable[ChannelSource]) -> int:
    """3 for RGB plus the sum of per-source channel counts.

    Use as ``in_channels=total_channels(sources)`` when constructing
    :func:`forge_detect.baseline_cnn.build_physics_classifier`.
    """
    return 3 + sum(s.n_channels for s in sources)


def load_channels_concat(
    image_chw: np.ndarray,
    image_path: Path,
    sources: Iterable[ChannelSource],
) -> np.ndarray:
    """Concat one or more channel-source contributions to an RGB image.

    Returns a ``(3 + sum(s.n_channels), H, W)`` float32 array. Each source
    in ``sources`` contributes its npz cache normalised through its own
    ``normalize`` callable. Missing caches raise :class:`FileNotFoundError`
    with an actionable message; mismatched shapes raise
    :class:`ValueError`. With ``sources=[]`` the function is a no-op cast
    to float32.
    """
    parts: list[np.ndarray] = [image_chw.astype(np.float32, copy=False)]
    _, h, w = image_chw.shape
    for src in sources:
        npz = src.cache_path_fn(image_path)
        if not npz.exists():
            msg = (
                f"channel cache for source '{src.name}' not found at {npz}. "
                "Run the matching cache_*.py script first; the script writes "
                "to this exact path."
            )
            raise FileNotFoundError(msg)
        with np.load(npz) as f:
            try:
                arrays = {k: np.asarray(f[k], dtype=np.float32) for k in src.npz_keys}
            except KeyError as e:
                msg = (
                    f"channel cache at {npz} is missing key {e}; "
                    f"expected keys: {src.npz_keys}"
                )
                raise ValueError(msg) from e
        for key, arr in arrays.items():
            if arr.shape != (h, w):
                msg = (
                    f"source '{src.name}' key {key!r} in {npz} has shape "
                    f"{arr.shape} but image is ({h}, {w}); rebuild the cache "
                    "with the same image_size"
                )
                raise ValueError(msg)
        normalized = src.normalize(arrays)
        if normalized.shape != (src.n_channels, h, w):
            msg = (
                f"source '{src.name}' normalize() returned shape "
                f"{normalized.shape}, expected ({src.n_channels}, {h}, {w})"
            )
            raise ValueError(msg)
        parts.append(normalized.astype(np.float32, copy=False))
    return np.concatenate(parts, axis=0)


def load_physics_maps_concat(
    image_chw: np.ndarray, npz_path: Path,
) -> np.ndarray:
    """Load `(wcnn, z_star, residual)` from npz, normalize each, concat to RGB.

    Returns a ``(6, H, W)`` float32 array. ``image_chw`` must be the
    ``(3, H, W)`` RGB tensor in ``[0, 1]`` produced by
    :func:`load_image_chw` so the spatial dimensions match the cached maps.
    Raises ``FileNotFoundError`` with an actionable message if the cache is
    missing, and ``ValueError`` if the cached maps don't match the image
    resolution (the cache must have been built at the same target_size).
    """
    if not npz_path.exists():
        msg = (
            f"physics-map cache not found at {npz_path}. Run "
            "`python scripts/cache_physics_maps.py` first; the script writes "
            "to this exact path"
        )
        raise FileNotFoundError(msg)
    _, h, w = image_chw.shape
    with np.load(npz_path) as f:
        try:
            wcnn = np.asarray(f["wcnn"], dtype=np.float32)
            z_star = np.asarray(f["z_star"], dtype=np.float32)
            residual = np.asarray(f["residual"], dtype=np.float32)
        except KeyError as e:
            msg = (
                f"physics-map cache at {npz_path} is missing key {e}; "
                f"expected keys: {PHYSICS_NPZ_KEYS}"
            )
            raise ValueError(msg) from e
    if wcnn.shape != (h, w) or z_star.shape != (h, w) or residual.shape != (h, w):
        msg = (
            f"physics maps in {npz_path} have shape {wcnn.shape} but image is "
            f"({h}, {w}); rebuild the cache with the same image_size"
        )
        raise ValueError(msg)
    # Per-image normalisation. Pre-image scale info is intentionally dropped:
    # for these maps the spatial pattern is informative, the absolute scale is
    # not, and per-image scaling cancels cross-image drift from compression /
    # solver convergence variance.
    wcnn = np.clip(wcnn, 0.0, 1.0)
    z_min, z_max = float(z_star.min()), float(z_star.max())
    z_norm = (z_star - z_min) / max(z_max - z_min, _PHYSICS_EPS)
    r_std = max(float(residual.std()), _PHYSICS_EPS)
    r_norm = (np.tanh(residual / r_std) + 1.0) * 0.5
    physics = np.stack([wcnn, z_norm, r_norm], axis=0).astype(np.float32, copy=False)
    return np.concatenate([image_chw, physics], axis=0)


def _list_images(root: Path) -> list[Path]:
    return sorted(
        p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in _IMAGE_EXTENSIONS
    )


def load_image_chw(path: Path, target_size: tuple[int, int] | None = None) -> np.ndarray:
    """Load an image and return ``(3, H, W)`` float32 in ``[0, 1]``."""
    img = Image.open(path).convert("RGB")
    if target_size is not None:
        img = img.resize((target_size[1], target_size[0]), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return np.transpose(arr, (2, 0, 1))  # HWC -> CHW


@dataclass
class _ImageFolderRecord:
    path: Path
    label: int  # 0 = real, 1 = fake
    video_id: str  # parent directory name, used as a coarse grouping key


class ImageFolderDataset(Dataset):
    """Generic two-folder ``(real, fake)`` layout returning ``(image, label)``.

    Both directories are walked recursively; any image file with a
    standard extension is included.

    Args:
        real_dir: Directory of authentic-image frames.
        fake_dir: Directory of synthesized / manipulated frames.
        target_size: Optional ``(H, W)`` to resize every image to. Use
            this when you want batch-friendly fixed shapes.
    """

    def __init__(
        self,
        real_dir: str | Path,
        fake_dir: str | Path,
        target_size: tuple[int, int] | None = None,
    ) -> None:
        self.real_dir = Path(real_dir)
        self.fake_dir = Path(fake_dir)
        self.target_size = target_size
        self._records: list[_ImageFolderRecord] = [
            _ImageFolderRecord(p, 0, p.parent.name) for p in _list_images(self.real_dir)
        ] + [_ImageFolderRecord(p, 1, p.parent.name) for p in _list_images(self.fake_dir)]
        if not self._records:
            msg = (
                f"no images found under {self.real_dir} or {self.fake_dir}; "
                "check the paths and that frames are extracted"
            )
            raise FileNotFoundError(msg)

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        rec = self._records[idx]
        arr = load_image_chw(rec.path, self.target_size)
        import torch

        return torch.from_numpy(arr), rec.label


@dataclass
class _FFRecord:
    image_path: Path
    mask_path: Path | None
    label: int
    video_id: str  # source video the frame was extracted from


class FaceForensicsAdapter(Dataset):
    """Adapter for the canonical FaceForensics++ on-disk layout.

    Expected directory tree under ``root``::

        original_sequences/youtube/<compression>/frames/<video_id>/<frame>.png
        manipulated_sequences/<method>/<compression>/frames/<video_id>/<frame>.png
        manipulated_sequences/<method>/<compression>/masks/<video_id>/<frame>.png   (optional)

    Frame extraction is the user's responsibility — see
    ``scripts/extract_frames.py`` for a ffmpeg-driven helper. Most published
    FF++ leaderboards use compression ``c23`` (visually-lossless H.264) so
    that is the default here.

    Args:
        root: Path to the FF++ root.
        methods: Which manipulation families to include
            (default: all four).
        compression: ``"raw"``, ``"c23"``, or ``"c40"``.
        max_frames_per_video: If set, sample at most this many frames per
            video via uniform stride-sampling.
        target_size: Optional resize.
        subset_video_ids: If given, drop frames whose source video is not
            in this set. Apply *after* loading the official splits, so
            you can intersect e.g. ``test_videos & {known-good ids}``.
        ff_split: One of ``"train" | "val" | "test"`` to load the
            official ``splits/<split>.json`` (the canonical 720/140/140
            video-disjoint split used by every FF++ paper). Reads from
            ``<root>/splits/<split>.json``.
    """

    def __init__(
        self,
        root: str | Path,
        methods: tuple[str, ...] = FF_METHODS,
        compression: str = "c23",
        max_frames_per_video: int | None = None,
        target_size: tuple[int, int] | None = None,
        subset_video_ids: Iterable[str] | None = None,
        ff_split: str | None = None,
        load_physics_maps: bool = False,
        physics_variant: str = "heuristic",
        frames_subdir: str = "frames",
        channel_sources: Iterable[ChannelSource] | None = None,
    ) -> None:
        """Args (extension):

        load_physics_maps: If True, ``__getitem__`` returns a 6-channel image
            ``(3 RGB + 3 physics)``. Cached physics maps (W_cnn, z*, R) must
            already exist alongside the frames — see
            ``scripts/cache_physics_maps.py``. Missing maps raise loudly.
            Legacy API; new callers should pass ``channel_sources`` instead.
        physics_variant: ``"heuristic"`` (default) loads the chromatic-residual
            trust-map variant; ``"gtmask"`` loads the GT-mask-derived variant
            for fakes (and falls back to heuristic for reals, which have no
            mask). Used only when ``load_physics_maps=True``.
        frames_subdir: Subdirectory under ``<compression>/`` that contains
            the frame images. Default ``"frames"`` reads the full extracted
            frames from ``scripts/extract_frames.py``. Set to
            ``"frames_faces"`` to read the face-cropped variant produced by
            ``scripts/extract_faces.py`` — required for any FF++ baseline
            that hopes to match the published EfficientNet-B0 numbers.
        channel_sources: Iterable of :class:`ChannelSource` for Phase-3+
            multi-source channel composition (e.g. RGB + physics +
            frequency = 9 channels). Mutually exclusive with the legacy
            ``load_physics_maps`` flag — passing both is rejected. The
            cache for every listed source must already exist; missing
            caches raise loudly at ``__getitem__``. Use
            :func:`parse_channel_spec` to build this list from a
            ``--channels`` CSV.
        """
        self.root = Path(root)
        if compression not in FF_COMPRESSIONS:
            msg = f"compression must be one of {FF_COMPRESSIONS}, got {compression!r}"
            raise ValueError(msg)
        if physics_variant not in PHYSICS_VARIANTS:
            msg = f"physics_variant must be one of {PHYSICS_VARIANTS}, got {physics_variant!r}"
            raise ValueError(msg)
        if channel_sources is not None and load_physics_maps:
            msg = (
                "FaceForensicsAdapter: channel_sources and load_physics_maps "
                "are mutually exclusive — pass one or the other, not both."
            )
            raise ValueError(msg)
        self.compression = compression
        self.target_size = target_size
        self.load_physics_maps = load_physics_maps
        self.physics_variant = physics_variant
        self.frames_subdir = frames_subdir
        self._channel_sources: list[ChannelSource] = (
            list(channel_sources) if channel_sources is not None else []
        )

        # Resolve the subset filter: if `ff_split` is set, intersect its
        # videos with `subset_video_ids` (when both are given).
        keep_videos: set[str] | None = None
        if ff_split is not None:
            keep_videos = load_ff_split(self.root, ff_split)
        if subset_video_ids is not None:
            extra = set(subset_video_ids)
            keep_videos = extra if keep_videos is None else keep_videos & extra

        records: list[_FFRecord] = []
        # Real frames.
        real_root = self.root / "original_sequences" / "youtube" / compression / frames_subdir
        if real_root.exists():
            records.extend(
                self._collect(
                    real_root,
                    label=0,
                    mask_root=None,
                    cap=max_frames_per_video,
                    keep_videos=keep_videos,
                ),
            )
        # Fake frames per method.
        for method in methods:
            method_root = (
                self.root / "manipulated_sequences" / method / compression / frames_subdir
            )
            # Masks are not face-cropped; gtmask physics from face-cropped
            # frames would need a separate mask-cropping pass. For now, only
            # attach masks when reading the standard "frames" tree.
            mask_root = (
                self.root / "manipulated_sequences" / method / compression / "masks"
                if frames_subdir == "frames"
                else None
            )
            if not method_root.exists():
                continue
            records.extend(
                self._collect(
                    method_root,
                    label=1,
                    mask_root=mask_root if mask_root and mask_root.exists() else None,
                    cap=max_frames_per_video,
                    keep_videos=keep_videos,
                ),
            )
        if not records:
            msg = (
                f"no frames found under {self.root}. Did you run frame "
                f"extraction? Expected layout: "
                f"<root>/original_sequences/youtube/{compression}/{frames_subdir}/<video>/*.png"
            )
            raise FileNotFoundError(msg)
        self._records = records

    @staticmethod
    def _collect(
        frames_root: Path,
        label: int,
        mask_root: Path | None,
        cap: int | None,
        keep_videos: set[str] | None,
    ) -> list[_FFRecord]:
        out: list[_FFRecord] = []
        for video_dir in sorted(p for p in frames_root.iterdir() if p.is_dir()):
            if keep_videos is not None and not _video_in_set(video_dir.name, keep_videos):
                continue
            frames = sorted(p for p in video_dir.iterdir() if p.suffix.lower() in _IMAGE_EXTENSIONS)
            if cap is not None:
                # Stride-sample to keep early/late frames represented.
                stride = max(1, len(frames) // cap)
                frames = frames[::stride][:cap]
            for f in frames:
                mask_path = None
                if mask_root is not None:
                    candidate = mask_root / video_dir.name / f.name
                    if candidate.exists():
                        mask_path = candidate
                out.append(
                    _FFRecord(
                        image_path=f,
                        mask_path=mask_path,
                        label=label,
                        video_id=video_dir.name,
                    ),
                )
        return out

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int, torch.Tensor | None]:
        rec = self._records[idx]
        arr = load_image_chw(rec.image_path, self.target_size)
        if self._channel_sources:
            arr = load_channels_concat(arr, rec.image_path, self._channel_sources)
        elif self.load_physics_maps:
            # Reals have no GT mask, so the gtmask variant has nothing to
            # offer them — they always read from the heuristic cache. Fakes
            # honor the configured variant.
            variant = (
                self.physics_variant if rec.label == 1 else "heuristic"
            )
            arr = load_physics_maps_concat(
                arr, physics_npz_path(rec.image_path, variant),
            )
        import torch

        image = torch.from_numpy(arr)
        mask: torch.Tensor | None = None
        if rec.mask_path is not None:
            m = Image.open(rec.mask_path).convert("L")
            if self.target_size is not None:
                m = m.resize((self.target_size[1], self.target_size[0]), Image.NEAREST)
            mask_np = (np.asarray(m, dtype=np.float32) > 0).astype(np.float32)
            mask = torch.from_numpy(mask_np)
        return image, rec.label, mask

    def video_ids(self) -> list[str]:
        """Sorted unique video ids across all records (real + fake)."""
        return sorted({rec.video_id for rec in self._records})


def _video_in_set(video_id: str, keep: set[str]) -> bool:
    """Match a frame directory's video_id against a set of canonical ids.

    FF++ manipulated-video directories are named ``<target>_<source>``
    (e.g. ``001_023``). The canonical splits list individual identifiers
    (``001``, ``023``); a manipulated video belongs to a split iff *both*
    of its component ids are in that split's set. Real video ids match
    directly. This helper handles both cases.
    """
    if video_id in keep:
        return True
    parts = video_id.split("_")
    return len(parts) == 2 and parts[0] in keep and parts[1] in keep


def load_ff_split(root: Path | str, split: str) -> set[str]:
    """Load the official FF++ ``splits/<split>.json`` as a flat set of ids.

    The on-disk format is a JSON array of two-element string arrays —
    e.g. ``[["001", "023"], ...]`` — listing source / target identifier
    pairs. We flatten that to a set so it can intersect a video filter.
    """
    if split not in {"train", "val", "test"}:
        msg = f"ff_split must be train|val|test, got {split!r}"
        raise ValueError(msg)
    path = Path(root) / "splits" / f"{split}.json"
    if not path.exists():
        msg = (
            f"FF++ split file not found: {path}. "
            "These ship with the FF++ downloader; if missing, fetch via "
            "`download-FaceForensics++.py <root> -d original_youtube_videos -t info`."
        )
        raise FileNotFoundError(msg)
    raw = json.loads(path.read_text())
    out: set[str] = set()
    for pair in raw:
        out.update(str(x) for x in pair)
    return out


# ---------- Celeb-DF (v1, v2) -----------------------------------------------

CELEB_SUBSETS_REAL: tuple[str, ...] = ("Celeb-real", "YouTube-real")
CELEB_SUBSETS_FAKE: tuple[str, ...] = ("Celeb-synthesis",)


@dataclass
class _CelebRecord:
    image_path: Path
    label: int
    video_id: str  # of the form "<subset>/<video_stem>" — globally unique


class CelebDFAdapter(Dataset):
    """Adapter for the Celeb-DF (v1 / v2) on-disk layout.

    Expected directory tree under ``root`` (after frame extraction)::

        Celeb-real/frames/<video_id>/<frame>.png        # real
        YouTube-real/frames/<video_id>/<frame>.png      # real (v2 only)
        Celeb-synthesis/frames/<video_id>/<frame>.png   # fake

    Use ``testing_list`` to honor the published 518-video benchmark
    (``List_of_testing_videos.txt``, shipped with v2). Every cross-
    dataset paper reports against this exact subset, so it is the only
    methodologically defensible eval split for cross-dataset numbers.

    Args:
        root: Path to the Celeb-DF root.
        max_frames_per_video: Optional uniform-stride frame cap.
        target_size: Optional resize.
        testing_list: Path to ``List_of_testing_videos.txt``. When set,
            only frames belonging to those 518 videos are kept. If you
            pass ``True``, the file is auto-located at
            ``<root>/List_of_testing_videos.txt``.
        subset_video_ids: Optional set of ``"<subset>/<stem>"`` ids to
            keep (post-``testing_list`` filter).
    """

    def __init__(
        self,
        root: str | Path,
        max_frames_per_video: int | None = None,
        target_size: tuple[int, int] | None = None,
        testing_list: bool | str | Path = False,
        subset_video_ids: Iterable[str] | None = None,
        load_physics_maps: bool = False,
        frames_subdir: str = "frames",
        channel_sources: Iterable[ChannelSource] | None = None,
    ) -> None:
        """Args (extension):

        load_physics_maps: If True, return 6-channel images with cached
            physics maps concatenated to RGB. Celeb-DF has no GT manipulation
            masks, so only the ``heuristic`` variant is supported here —
            see :class:`FaceForensicsAdapter` for the gtmask variant.
            Legacy API; new callers should pass ``channel_sources`` instead.
        frames_subdir: Subdirectory name under each subset (``Celeb-real``,
            ``Celeb-synthesis``, ``YouTube-real``) that holds the frame
            images. Default ``"frames"``; pass ``"frames_faces"`` to read
            face crops produced by ``scripts/extract_faces.py``.
        channel_sources: Iterable of :class:`ChannelSource` for Phase-3+
            multi-source channel composition. Mutually exclusive with
            ``load_physics_maps``. See
            :class:`FaceForensicsAdapter` for full semantics.
        """
        self.root = Path(root)
        if channel_sources is not None and load_physics_maps:
            msg = (
                "CelebDFAdapter: channel_sources and load_physics_maps are "
                "mutually exclusive — pass one or the other, not both."
            )
            raise ValueError(msg)
        self.target_size = target_size
        self.load_physics_maps = load_physics_maps
        self.frames_subdir = frames_subdir
        self._channel_sources: list[ChannelSource] = (
            list(channel_sources) if channel_sources is not None else []
        )

        keep_videos: set[str] | None = None
        if testing_list:
            list_path = (
                self.root / "List_of_testing_videos.txt"
                if testing_list is True
                else Path(testing_list)
            )
            keep_videos = load_celeb_testing_list(list_path)
        if subset_video_ids is not None:
            extra = set(subset_video_ids)
            keep_videos = extra if keep_videos is None else keep_videos & extra

        records: list[_CelebRecord] = []
        for subset in CELEB_SUBSETS_REAL:
            records.extend(
                self._collect_subset(
                    subset,
                    label=0,
                    cap=max_frames_per_video,
                    keep_videos=keep_videos,
                ),
            )
        for subset in CELEB_SUBSETS_FAKE:
            records.extend(
                self._collect_subset(
                    subset,
                    label=1,
                    cap=max_frames_per_video,
                    keep_videos=keep_videos,
                ),
            )
        if not records:
            msg = (
                f"no frames found under {self.root}. Expected layout: "
                f"<root>/{{Celeb-real,Celeb-synthesis,YouTube-real}}/{frames_subdir}/<v>/*.png. "
                "Did you run scripts/extract_frames.py --dataset celeb-df "
                "(and scripts/extract_faces.py for frames_faces)?"
            )
            raise FileNotFoundError(msg)
        self._records = records

    def _collect_subset(
        self,
        subset: str,
        label: int,
        cap: int | None,
        keep_videos: set[str] | None,
    ) -> list[_CelebRecord]:
        frames_root = self.root / subset / self.frames_subdir
        if not frames_root.exists():
            return []
        out: list[_CelebRecord] = []
        for video_dir in sorted(p for p in frames_root.iterdir() if p.is_dir()):
            video_id = f"{subset}/{video_dir.name}"
            if keep_videos is not None and video_id not in keep_videos:
                continue
            frames = sorted(p for p in video_dir.iterdir() if p.suffix.lower() in _IMAGE_EXTENSIONS)
            if cap is not None:
                stride = max(1, len(frames) // cap)
                frames = frames[::stride][:cap]
            for f in frames:
                out.append(_CelebRecord(image_path=f, label=label, video_id=video_id))
        return out

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        rec = self._records[idx]
        arr = load_image_chw(rec.image_path, self.target_size)
        if self._channel_sources:
            arr = load_channels_concat(arr, rec.image_path, self._channel_sources)
        elif self.load_physics_maps:
            arr = load_physics_maps_concat(
                arr, physics_npz_path(rec.image_path, "heuristic"),
            )
        import torch

        return torch.from_numpy(arr), rec.label

    def video_ids(self) -> list[str]:
        return sorted({rec.video_id for rec in self._records})


def load_celeb_testing_list(path: Path | str) -> set[str]:
    """Parse ``List_of_testing_videos.txt`` into ``{<subset>/<stem>}`` ids.

    The file ships with Celeb-DF v2 and lists the canonical 518-video
    benchmark, one entry per line, in the form::

        1 YouTube-real/00170.mp4
        0 Celeb-synthesis/id0_id16_0009.mp4

    The first column is the dataset's own real/fake flag (``1`` real,
    ``0`` fake — *opposite* to our ``0=real / 1=fake`` convention) and
    we ignore it: real/fake labels come from the directory the video
    sits in, not from the testing-list flag. We only keep the second
    column, stripped of the ``.mp4`` suffix, so it matches the
    ``<subset>/<video_stem>`` ids the adapter assigns.
    """
    p = Path(path)
    if not p.exists():
        msg = (
            f"List_of_testing_videos.txt not found at {p}. It ships with "
            "the Celeb-DF v2 archive; without it we cannot reproduce the "
            "published cross-dataset benchmark."
        )
        raise FileNotFoundError(msg)
    out: set[str] = set()
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        # `1 YouTube-real/00170.mp4` -> ('1', 'YouTube-real/00170.mp4')
        parts = line.split(maxsplit=1)
        if len(parts) != 2:
            continue
        rel = parts[1].strip()
        if rel.endswith(".mp4"):
            rel = rel[:-4]
        out.add(rel)
    return out


def stratified_split(
    n: int,
    seed: int = 0,
    val_fraction: float = 0.1,
    test_fraction: float = 0.1,
) -> tuple[list[int], list[int], list[int]]:
    """Return ``(train_idx, val_idx, test_idx)`` index lists for a dataset of size ``n``.

    Uses a fixed random seed so the split is reproducible across runs.
    Assumes the underlying dataset already balances its real/fake split;
    if it does not, see scikit-learn's ``train_test_split(stratify=labels)``.

    Note: this splits *records* (frames). For deepfake detection, prefer
    :func:`split_videos` so frames from the same video do not leak across
    train/val/test — frame-level splits inflate AUROC by ~5–15 points
    because adjacent frames are nearly identical.
    """
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_test = int(test_fraction * n)
    n_val = int(val_fraction * n)
    test_idx = perm[:n_test].tolist()
    val_idx = perm[n_test : n_test + n_val].tolist()
    train_idx = perm[n_test + n_val :].tolist()
    return train_idx, val_idx, test_idx


def split_videos(
    video_ids: Iterable[str],
    *,
    seed: int = 0,
    val_fraction: float = 0.15,
    test_fraction: float = 0.15,
) -> tuple[set[str], set[str], set[str]]:
    """Return ``(train, val, test)`` video-id sets for a video-disjoint split.

    Methodologically correct for deepfake detection: frames from the same
    source video must not appear in more than one split, otherwise
    train→test leakage inflates the reported AUROC. The split is by *id*,
    so a downstream adapter can be re-built per split with whatever
    frame-per-video cap is appropriate for that split (e.g. 30 for
    training to give the model variety, 10 for evaluation per the
    academic protocol).
    """
    ids = sorted(set(video_ids))
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(ids))
    n = len(ids)
    n_test = int(test_fraction * n)
    n_val = int(val_fraction * n)
    test_set = {ids[i] for i in perm[:n_test]}
    val_set = {ids[i] for i in perm[n_test : n_test + n_val]}
    train_set = {ids[i] for i in perm[n_test + n_val :]}
    return train_set, val_set, test_set


__all__ = [
    "CELEB_SUBSETS_FAKE",
    "CELEB_SUBSETS_REAL",
    "FF_COMPRESSIONS",
    "FF_METHODS",
    "FREQUENCY_NPZ_KEYS",
    "FREQUENCY_VARIANTS",
    "PHYSICS_NPZ_KEYS",
    "PHYSICS_VARIANTS",
    "CelebDFAdapter",
    "ChannelSource",
    "FaceForensicsAdapter",
    "ImageFolderDataset",
    "frequency_channel_source",
    "frequency_npz_path",
    "load_celeb_testing_list",
    "load_channels_concat",
    "load_ff_split",
    "load_image_chw",
    "load_physics_maps_concat",
    "parse_channel_spec",
    "physics_channel_source",
    "physics_npz_path",
    "split_videos",
    "stratified_split",
    "total_channels",
]
