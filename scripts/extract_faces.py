"""Extract square face crops from FF++ / Celeb-DF frames.

The FF++ leaderboard convention is to train classifiers on tightly cropped
face regions, not full frames — without face cropping a vanilla EfficientNet-
B0 baseline plateaus at AUROC ~0.50 on this task. This script runs MTCNN
(facenet-pytorch) over every extracted frame, takes the highest-confidence
face, crops a square around it with a configurable margin, and writes the
result to a sibling ``frames_faces/`` directory.

Layout (FF++)::

    <root>/original_sequences/youtube/<comp>/frames/<vid>/<frame>.png
                              -> frames_faces/<vid>/<frame>.png
    <root>/manipulated_sequences/<method>/<comp>/frames/<vid>/<frame>.png
                              -> frames_faces/<vid>/<frame>.png

Layout (Celeb-DF)::

    <root>/<subset>/frames/<vid>/<frame>.png
                  -> frames_faces/<vid>/<frame>.png

Resumable — frames whose output already exists are skipped at the index
level. Falls back to a centre crop when MTCNN finds no face (logged in the
summary), so the output set has the same shape as the input set and
downstream code can iterate without missing-file branches.

Pipeline:

    extract_frames.py               -> .../frames/<vid>/<frame>.png
    extract_faces.py                -> .../frames_faces/<vid>/<frame>.png
    cache_physics_maps.py           -> .../physics_faces_<variant>/<vid>/<frame>.npz
        --frames-subdir frames_faces
    pivot_study.py / train_physics_cnn.py
        --use-face-crops            -> reads from frames_faces/

Usage:

    python scripts/extract_faces.py \\
        --data-root ~/data/FaceForensics++ \\
        --dataset face-forensics --compression c23 \\
        --output-size 256 --margin 0.3 --device cuda
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

from PIL import Image

_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def _build_mtcnn(device: str, output_size: int) -> Any:
    """Construct the MTCNN detector. Lazy import keeps the module loadable
    on machines without facenet-pytorch installed (e.g. for --help)."""
    from facenet_pytorch import MTCNN

    return MTCNN(
        image_size=output_size,
        margin=0,
        min_face_size=20,
        thresholds=[0.6, 0.7, 0.7],
        factor=0.709,
        post_process=False,
        device=device,
        keep_all=False,  # return only the highest-confidence detection
    )


def _find_frame_dirs(
    data_root: Path, dataset: str, compression: str,
) -> list[Path]:
    """Enumerate every ``frames/`` directory in the dataset that should
    receive face crops. Missing directories are silently filtered later."""
    if dataset == "celeb-df":
        return [
            data_root / "Celeb-real" / "frames",
            data_root / "Celeb-synthesis" / "frames",
            data_root / "YouTube-real" / "frames",
        ]
    methods = ("Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures")
    return [
        data_root / "original_sequences" / "youtube" / compression / "frames",
        *[
            data_root / "manipulated_sequences" / m / compression / "frames"
            for m in methods
        ],
    ]


def _square_box_from_detection(
    box: tuple[float, float, float, float],
    img_w: int,
    img_h: int,
    margin: float,
) -> tuple[int, int, int, int]:
    """Convert an MTCNN ``(x1, y1, x2, y2)`` detection into a square crop
    box with ``margin`` extra padding around the face. The returned box is
    clipped to image bounds, so the actual aspect ratio may not be exactly
    1:1 if the face sits near an edge — the subsequent resize forces
    output_size × output_size regardless."""
    x1, y1, x2, y2 = box
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    side = max(x2 - x1, y2 - y1) * (1.0 + margin)
    half = side / 2.0
    nx1 = max(0, int(round(cx - half)))
    ny1 = max(0, int(round(cy - half)))
    nx2 = min(img_w, int(round(cx + half)))
    ny2 = min(img_h, int(round(cy + half)))
    return nx1, ny1, nx2, ny2


def _center_square_box(img_w: int, img_h: int) -> tuple[int, int, int, int]:
    """Centre-crop fallback used when no face is detected — keeps the
    output set complete so downstream code doesn't need to branch."""
    side = min(img_w, img_h)
    x1 = (img_w - side) // 2
    y1 = (img_h - side) // 2
    return x1, y1, x1 + side, y1 + side


def _crop_and_resize(
    image: Image.Image,
    box: tuple[int, int, int, int],
    output_size: int,
) -> Image.Image:
    cropped = image.crop(box)
    return cropped.resize((output_size, output_size), Image.BILINEAR)


def _list_frames_in_video(
    video_dir: Path, max_frames: int | None,
) -> list[Path]:
    """List image files in ``video_dir``, optionally stride-sampled to
    ``max_frames``. Stride sampling matches the dataset adapter's
    behaviour so that the cropped subset corresponds exactly to the
    frames training will read."""
    frames = sorted(
        p for p in video_dir.iterdir()
        if p.suffix.lower() in _IMAGE_EXTENSIONS
    )
    if max_frames is not None and len(frames) > max_frames:
        stride = max(1, len(frames) // max_frames)
        frames = frames[::stride][:max_frames]
    return frames


def _process_one_dir(
    frames_dir: Path,
    faces_dir: Path,
    mtcnn: Any,
    args: argparse.Namespace,
) -> dict[str, int]:
    """Walk every video subdirectory under ``frames_dir`` and write face
    crops to the parallel ``faces_dir`` tree."""
    n_total = 0
    n_face = 0
    n_fallback = 0
    n_skipped = 0
    n_failed = 0

    t0 = time.time()
    for video_dir in sorted(frames_dir.iterdir()):
        if not video_dir.is_dir():
            continue
        out_video_dir = faces_dir / video_dir.name
        frames = _list_frames_in_video(video_dir, args.max_frames_per_video)
        for frame_path in frames:
            n_total += 1
            out_path = out_video_dir / frame_path.name
            if out_path.exists():
                n_skipped += 1
                continue

            try:
                image = Image.open(frame_path).convert("RGB")
                boxes, _probs = mtcnn.detect(image)
                if boxes is not None and len(boxes) > 0:
                    # keep_all=False can return a 1D (4,) or 2D (1, 4) array
                    box_arr = boxes if boxes.ndim == 1 else boxes[0]
                    box = _square_box_from_detection(
                        tuple(float(v) for v in box_arr),
                        image.width,
                        image.height,
                        args.margin,
                    )
                    n_face += 1
                else:
                    box = _center_square_box(image.width, image.height)
                    n_fallback += 1

                cropped = _crop_and_resize(image, box, args.output_size)
                out_video_dir.mkdir(parents=True, exist_ok=True)
                # Atomic write: PIL can leave a half-decoded file on disk if
                # the process is interrupted mid-save (which then fails the
                # next dataset load with UnidentifiedImageError instead of a
                # missing-file error). Save to <name>.tmp and rename — even
                # an interrupted run leaves only a stray .tmp file, never a
                # corrupt .png that the resume check would trust.
                tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
                cropped.save(tmp_path, format="PNG")
                tmp_path.replace(out_path)
            except Exception as e:
                n_failed += 1
                print(f"  ERROR on {frame_path}: {type(e).__name__}: {e}")

            if args.log_every and n_total > 0 and n_total % args.log_every == 0:
                elapsed = time.time() - t0
                worked = max(1, n_total - n_skipped)
                rate = worked / max(elapsed, 1.0e-6)
                pct_face = 100.0 * n_face / max(1, n_face + n_fallback)
                print(
                    f"  [{n_total:6d}] face={n_face} ({pct_face:.0f}%) "
                    f"fallback={n_fallback} skipped={n_skipped} "
                    f"failed={n_failed}  {rate:.1f} img/s",
                )

    return {
        "total": n_total,
        "face_detected": n_face,
        "fallback_used": n_fallback,
        "skipped": n_skipped,
        "failed": n_failed,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument(
        "--dataset",
        choices=("face-forensics", "celeb-df"),
        default="face-forensics",
    )
    parser.add_argument(
        "--compression",
        choices=("raw", "c23", "c40"),
        default="c23",
        help="Only used for face-forensics; ignored for celeb-df.",
    )
    parser.add_argument(
        "--output-size",
        type=int,
        default=256,
        help="Edge length of the square output crop (uniform across the "
        "whole dataset). Match this to whatever resolution the training "
        "adapter resizes to so the dataset doesn't pay a second resize.",
    )
    parser.add_argument(
        "--margin",
        type=float,
        default=0.3,
        help="Fraction of the MTCNN bounding-box edge length to add as "
        "padding before the square crop. 0.0 = tight crop on the bbox; "
        "0.3 (default) = include some hair / chin context which helps "
        "the classifier on tightly-fitted deepfakes.",
    )
    parser.add_argument(
        "--max-frames-per-video",
        type=int,
        default=None,
        help="If set, stride-sample at most N frames per video. Default "
        "processes everything. Use this to bound disk usage when you "
        "know training will only see e.g. 10 frames per video.",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        choices=("cpu", "cuda"),
        help="Where to run MTCNN. cuda recommended; ~50ms per frame on "
        "RTX 3080 Ti vs ~500ms on CPU.",
    )
    parser.add_argument("--log-every", type=int, default=200)
    args = parser.parse_args()

    print(f"[extract-faces] data_root = {args.data_root}")
    print(f"[extract-faces] dataset = {args.dataset}")
    print(
        f"[extract-faces] crop = {args.output_size}x{args.output_size}, "
        f"margin = {args.margin}",
    )
    print(f"[extract-faces] device = {args.device}")
    if args.max_frames_per_video:
        print(
            f"[extract-faces] cap = {args.max_frames_per_video} frames/video "
            "(stride-sampled to match adapter)",
        )

    mtcnn = _build_mtcnn(args.device, args.output_size)

    frames_dirs = _find_frame_dirs(
        args.data_root, args.dataset, args.compression,
    )

    grand: dict[str, int] = {
        "total": 0, "face_detected": 0, "fallback_used": 0,
        "skipped": 0, "failed": 0,
    }
    for frames_dir in frames_dirs:
        if not frames_dir.exists():
            print(f"[extract-faces] skipping (not found): {frames_dir}")
            continue
        faces_dir = frames_dir.parent / "frames_faces"
        print(f"\n[extract-faces] {frames_dir} -> {faces_dir}")
        stats = _process_one_dir(frames_dir, faces_dir, mtcnn, args)
        for k, v in stats.items():
            grand[k] += v
        print(
            f"  {stats['face_detected']}/{stats['total']} faces detected, "
            f"{stats['fallback_used']} fallback, {stats['skipped']} skipped, "
            f"{stats['failed']} failed",
        )

    print("\n=== summary ===")
    for k, v in grand.items():
        print(f"  {k}: {v}")
    if grand["face_detected"] + grand["fallback_used"] > 0:
        rate = grand["face_detected"] / max(
            1, grand["face_detected"] + grand["fallback_used"],
        )
        print(f"  face detection rate: {rate:.1%}")
        if rate < 0.85:
            print(
                "  WARNING: detection rate below 85% — fallback centre crops "
                "will weaken the classifier signal. Consider lowering "
                "--margin or running on the higher-resolution source frames.",
            )
    return 0 if grand["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
