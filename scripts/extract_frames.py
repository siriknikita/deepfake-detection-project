"""Extract frames from FaceForensics++ videos via ffmpeg.

Walks the canonical FF++ layout, decodes every video to PNGs, and writes
them under a sibling ``frames/<video_id>/<frame>.png`` directory:

    <root>/original_sequences/youtube/<compression>/videos/<video_id>.mp4
        ->  <root>/original_sequences/youtube/<compression>/frames/<video_id>/0000.png ...

    <root>/manipulated_sequences/<method>/<compression>/videos/<video_id>.mp4
        ->  <root>/manipulated_sequences/<method>/<compression>/frames/<video_id>/...

ffmpeg must be installed and on PATH.

Typical use after sign-up at https://github.com/ondyari/FaceForensics::

    python scripts/extract_frames.py /path/to/FaceForensics++ \
        --compression c23 --fps 5
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


def _walk_and_extract(root: Path, compression: str, fps: int) -> None:
    targets = [
        root / "original_sequences" / "youtube" / compression / "videos",
    ]
    for method in ("Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures"):
        targets.append(root / "manipulated_sequences" / method / compression / "videos")

    total_videos = 0
    total_frames = 0
    for videos_dir in targets:
        if not videos_dir.exists():
            print(f"skip {videos_dir} (not present)")
            continue
        frames_root = videos_dir.parent / "frames"
        for video in sorted(videos_dir.glob("*.mp4")):
            out_dir = frames_root / video.stem
            n = _extract_one(video, out_dir, fps)
            total_videos += 1
            total_frames += n
            print(f"  {video.name}: {n} frames -> {out_dir}")
    print(f"done: {total_videos} videos, {total_frames} new frames extracted")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("root", type=Path, help="FaceForensics++ root directory.")
    parser.add_argument(
        "--compression",
        choices=("raw", "c23", "c40"),
        default="c23",
        help="Compression level to extract (default: c23, the standard FF++ benchmark).",
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
    _walk_and_extract(args.root, args.compression, args.fps)
    return 0


if __name__ == "__main__":
    sys.exit(main())
