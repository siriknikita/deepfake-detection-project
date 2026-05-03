"""Extract frames from FaceForensics++ or Celeb-DF videos via ffmpeg.

For FF++, walks the canonical layout and decodes every video to PNGs:

    <root>/original_sequences/youtube/<compression>/videos/<video_id>.mp4
        ->  <root>/original_sequences/youtube/<compression>/frames/<video_id>/0000.png ...

    <root>/manipulated_sequences/<method>/<compression>/videos/<video_id>.mp4
        ->  <root>/manipulated_sequences/<method>/<compression>/frames/<video_id>/...

For Celeb-DF (v1 / v2), each subset directory holds the ``.mp4`` files
directly and we extract into a sibling ``frames/`` directory:

    <root>/Celeb-real/<video_id>.mp4       -> <root>/Celeb-real/frames/<video_id>/0000.png
    <root>/Celeb-synthesis/<video_id>.mp4  -> <root>/Celeb-synthesis/frames/<video_id>/0000.png
    <root>/YouTube-real/<video_id>.mp4     -> <root>/YouTube-real/frames/<video_id>/0000.png

ffmpeg must be installed and on PATH.

Typical use after sign-up at https://github.com/ondyari/FaceForensics::

    python scripts/extract_frames.py /path/to/FaceForensics++ \
        --compression c23 --fps 5

Celeb-DF (assumes archive is already extracted under <root>)::

    python scripts/extract_frames.py /path/to/Celeb-DF-v2 \
        --dataset celeb-df --fps 5
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


def _ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None


def _extract_one(video: Path, out_dir: Path, fps: int) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    if any(out_dir.iterdir()):
        return 0  # already extracted; skip
    cmd = [
        "ffmpeg",
        "-loglevel",
        "error",
        "-i",
        str(video),
        "-vf",
        f"fps={fps}",
        str(out_dir / "%04d.png"),
    ]
    subprocess.run(cmd, check=True)
    return sum(1 for _ in out_dir.iterdir())


def _ff_targets(root: Path, compression: str) -> list[Path]:
    targets = [root / "original_sequences" / "youtube" / compression / "videos"]
    for method in ("Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures"):
        targets.append(root / "manipulated_sequences" / method / compression / "videos")
    return targets


def _celeb_targets(root: Path) -> list[Path]:
    # Celeb-DF dumps videos directly under each subset folder; the
    # extracted frames go into a sibling `frames/` subdirectory.
    return [root / sub for sub in ("Celeb-real", "Celeb-synthesis", "YouTube-real")]


def _frames_root_for(videos_dir: Path) -> Path:
    """FF++ uses ``<dir>/videos`` + ``<dir>/frames`` as siblings; Celeb-DF
    has the ``.mp4`` files directly in the subset dir, so frames go into
    a ``frames/`` child."""
    if videos_dir.name == "videos":
        return videos_dir.parent / "frames"
    return videos_dir / "frames"


def _walk_and_extract(targets: list[Path], fps: int) -> None:
    total_videos = 0
    total_frames = 0
    for videos_dir in targets:
        if not videos_dir.exists():
            print(f"skip {videos_dir} (not present)")
            continue
        frames_root = _frames_root_for(videos_dir)
        for video in sorted(videos_dir.glob("*.mp4")):
            out_dir = frames_root / video.stem
            n = _extract_one(video, out_dir, fps)
            total_videos += 1
            total_frames += n
            print(f"  {video.name}: {n} frames -> {out_dir}")
    print(f"done: {total_videos} videos, {total_frames} new frames extracted")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "root",
        type=Path,
        help="Dataset root directory (FF++ or Celeb-DF).",
    )
    parser.add_argument(
        "--dataset",
        choices=("face-forensics", "celeb-df"),
        default="face-forensics",
        help="Which on-disk layout to walk (default: face-forensics).",
    )
    parser.add_argument(
        "--compression",
        choices=("raw", "c23", "c40"),
        default="c23",
        help="FF++ compression level to extract (default: c23). Ignored for Celeb-DF.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=5,
        help="Frames per second to extract (default: 5 — keeps datasets manageable).",
    )
    args = parser.parse_args()

    if not _ffmpeg_available():
        print("ffmpeg not found on PATH; install it first.", file=sys.stderr)
        return 2
    if not args.root.exists():
        print(f"root not found: {args.root}", file=sys.stderr)
        return 2

    targets = (
        _ff_targets(args.root, args.compression)
        if args.dataset == "face-forensics"
        else _celeb_targets(args.root)
    )
    _walk_and_extract(targets, args.fps)
    return 0


if __name__ == "__main__":
    sys.exit(main())
