"""Dataset adapters for training and evaluation.

Two data sources are supported out of the box:

- :class:`ImageFolderDataset` — generic two-folder layout
  ``(real_dir/, fake_dir/)`` of frame-level images. Use this when you
  pre-extracted frames to disk in any layout.

- :class:`FaceForensicsAdapter` — the canonical FaceForensics++ tree:
  ``original_sequences/youtube/<compression>/frames/<video_id>/<frame>.png``
  and ``manipulated_sequences/<method>/<compression>/frames/<video_id>/<frame>.png``.
  Frame extraction from videos is delegated to
  ``scripts/extract_frames.py``; the adapter assumes frames already on disk.

Both adapters return ``(image, label)`` where ``image`` is an
``(3, H, W)`` float32 tensor in ``[0, 1]`` and ``label`` is ``0`` for
real / ``1`` for fake. ``FaceForensicsAdapter`` optionally also returns
the pixel-level fake mask when one is available alongside the frame.

Where to get the data:

- *FaceForensics++*: https://github.com/ondyari/FaceForensics — sign the
  EULA, run their download script. ~1.5 TB at full quality.
- *Celeb-DF (v2)*: https://github.com/yuezunli/celeb-deepfakeforensics —
  ~50 GB, EULA required.
- *DFDC*: https://www.kaggle.com/c/deepfake-detection-challenge/data —
  ~470 GB, larger and noisier than FF++.
- *FFHQ* (real-only): https://github.com/NVlabs/ffhq-dataset — 70 k
  high-quality 1024² faces, useful for real-only pretraining or as the
  "real" half against generated faces.
"""

from __future__ import annotations

from dataclasses import dataclass
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
            _ImageFolderRecord(p, 0) for p in _list_images(self.real_dir)
        ] + [_ImageFolderRecord(p, 1) for p in _list_images(self.fake_dir)]
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
            video — useful for keeping epoch sizes manageable.
        target_size: Optional resize.
    """

    def __init__(
        self,
        root: str | Path,
        methods: tuple[str, ...] = FF_METHODS,
        compression: str = "c23",
        max_frames_per_video: int | None = None,
        target_size: tuple[int, int] | None = None,
    ) -> None:
        self.root = Path(root)
        if compression not in FF_COMPRESSIONS:
            msg = f"compression must be one of {FF_COMPRESSIONS}, got {compression!r}"
            raise ValueError(msg)
        self.compression = compression
        self.target_size = target_size

        records: list[_FFRecord] = []
        # Real frames.
        real_root = self.root / "original_sequences" / "youtube" / compression / "frames"
        if real_root.exists():
            records.extend(
                self._collect(real_root, label=0, mask_root=None, cap=max_frames_per_video)
            )
        # Fake frames per method.
        for method in methods:
            method_root = self.root / "manipulated_sequences" / method / compression / "frames"
            mask_root = self.root / "manipulated_sequences" / method / compression / "masks"
            if not method_root.exists():
                continue
            records.extend(
                self._collect(
                    method_root,
                    label=1,
                    mask_root=mask_root if mask_root.exists() else None,
                    cap=max_frames_per_video,
                ),
            )
        if not records:
            msg = (
                f"no frames found under {self.root}. Did you run frame "
                f"extraction? Expected layout: "
                f"<root>/original_sequences/youtube/{compression}/frames/<video>/*.png"
            )
            raise FileNotFoundError(msg)
        self._records = records

    @staticmethod
    def _collect(
        frames_root: Path,
        label: int,
        mask_root: Path | None,
        cap: int | None,
    ) -> list[_FFRecord]:
        out: list[_FFRecord] = []
        for video_dir in sorted(p for p in frames_root.iterdir() if p.is_dir()):
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
                out.append(_FFRecord(image_path=f, mask_path=mask_path, label=label))
        return out

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int, torch.Tensor | None]:
        rec = self._records[idx]
        arr = load_image_chw(rec.image_path, self.target_size)
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
    """
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_test = int(test_fraction * n)
    n_val = int(val_fraction * n)
    test_idx = perm[:n_test].tolist()
    val_idx = perm[n_test : n_test + n_val].tolist()
    train_idx = perm[n_test + n_val :].tolist()
    return train_idx, val_idx, test_idx


__all__ = [
    "FF_COMPRESSIONS",
    "FF_METHODS",
    "FaceForensicsAdapter",
    "ImageFolderDataset",
    "load_image_chw",
    "stratified_split",
]
