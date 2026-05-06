"""Stride-trim each per-video frame folder down to N frames.

Mirrors the stride-sampling the dataset adapter does at load-time, but
applied to disk so we don't ship surplus frames over the network.

Usage:
    python scripts/trim_frames.py /path/to/FaceForensics++ --keep 30
"""

from __future__ import annotations

import argparse
from pathlib import Path

_IMAGE_EXTS = {".png", ".jpg", ".jpeg"}


def _trim_dir(video_dir: Path, keep: int) -> tuple[int, int]:
    frames = sorted(p for p in video_dir.iterdir() if p.suffix.lower() in _IMAGE_EXTS)
    if len(frames) <= keep:
        return len(frames), 0
    stride = max(1, len(frames) // keep)
    keepset = set(frames[::stride][:keep])
    deleted = 0
    for f in frames:
        if f not in keepset:
            f.unlink()
            deleted += 1
    return len(keepset), deleted


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("root", type=Path)
    p.add_argument("--keep", type=int, default=30)
    args = p.parse_args()

    total_kept = 0
    total_deleted = 0
    for frames_root in args.root.rglob("frames"):
        if not frames_root.is_dir():
            continue
        for video_dir in sorted(frames_root.iterdir()):
            if not video_dir.is_dir():
                continue
            kept, deleted = _trim_dir(video_dir, args.keep)
            total_kept += kept
            total_deleted += deleted
        print(f"  {frames_root}: kept={total_kept} deleted={total_deleted}")
    print(f"done: kept {total_kept} frames, deleted {total_deleted}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
