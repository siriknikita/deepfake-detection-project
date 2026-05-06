"""Extract frames from FaceForensics++ or Celeb-DF videos via ffmpeg.

For FF++, walks the canonical layout and decodes every video to PNGs:

    <root>/original_sequences/youtube/<compression>/videos/<video_id>.mp4
        ->  <root>/original_sequences/youtube/<compression>/frames/<video_id>/0000.png ...

    <root>/manipulated_sequences/<method>/<compression>/videos/<video_id>.mp4
        ->  <root>/manipulated_sequences/<method>/<compression>/frames/<video_id>/...

With ``--include-masks``, also extracts ground-truth manipulation masks
(FF++ only). Mask source videos live one level above ``<compression>``:

    <root>/manipulated_sequences/<method>/masks/videos/<video_id>.mp4
        ->  <root>/manipulated_sequences/<method>/<compression>/masks/<video_id>/...

Masks are written as single-channel grayscale PNGs and resized with
nearest-neighbour interpolation so binary edges are not blurred.

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
import os
import shutil
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any


def _ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None


def _extract_one(video: Path, out_dir: Path, fps: int, size: int, *, grayscale: bool = False) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    if any(out_dir.iterdir()):
        return 0  # already extracted; skip
    parts = [f"fps={fps}"]
    if size > 0:
        # Nearest-neighbour scaling for masks preserves the binary edge;
        # bilinear (the default) would fabricate intermediate gray values
        # at manipulation boundaries.
        scale_flags = ":flags=neighbor" if grayscale else ""
        parts.append(f"scale={size}:{size}{scale_flags}")
    if grayscale:
        parts.append("format=gray")
    vf = ",".join(parts)
    cmd = [
        "ffmpeg",
        "-loglevel",
        "error",
        "-i",
        str(video),
        "-vf",
        vf,
        str(out_dir / "%04d.png"),
    ]
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError:
        # Wipe partial output so a retry restarts cleanly.
        for f in out_dir.iterdir():
            f.unlink(missing_ok=True)
        return -1
    return sum(1 for _ in out_dir.iterdir())


def _ff_targets(root: Path, compression: str) -> list[Path]:
    targets = [root / "original_sequences" / "youtube" / compression / "videos"]
    for method in ("Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures"):
        targets.append(root / "manipulated_sequences" / method / compression / "videos")
    return targets


def _ff_mask_targets(root: Path) -> list[Path]:
    """FF++ ships ground-truth manipulation masks as separate videos under
    ``manipulated_sequences/<method>/masks/videos/<id>.mp4``. Reals don't
    have masks (no manipulation to mark)."""
    return [
        root / "manipulated_sequences" / method / "masks" / "videos"
        for method in ("Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures")
    ]


def _mask_frames_root_for(videos_dir: Path, compression: str) -> Path:
    """Mask videos live at ``manipulated_sequences/<method>/masks/videos``
    but the dataset adapter expects extracted PNGs at
    ``manipulated_sequences/<method>/<compression>/masks/<video_id>/`` — i.e.
    parallel to ``frames/`` under the same compression bucket. Climb two
    levels (``videos`` → ``masks``) and pivot into ``<compression>/masks``."""
    method_root = videos_dir.parent.parent
    return method_root / compression / "masks"


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


def _walk_and_extract(
    targets: list[Path],
    fps: int,
    size: int,
    jobs: int,
    testing_list: set[str] | None = None,
    *,
    grayscale: bool = False,
    out_root_for: Any = None,
) -> None:
    if out_root_for is None:
        out_root_for = _frames_root_for
    tasks: list[tuple[Path, Path]] = []
    for videos_dir in targets:
        if not videos_dir.exists():
            print(f"skip {videos_dir} (not present)")
            continue
        frames_root = out_root_for(videos_dir)
        for video in sorted(videos_dir.glob("*.mp4")):
            if testing_list is not None:
                rel = f"{videos_dir.name}/{video.name}"
                if rel not in testing_list:
                    continue
            tasks.append((video, frames_root / video.stem))

    total_videos = len(tasks)
    total_frames = 0
    failed = 0
    kind = "mask videos" if grayscale else "videos"
    print(f"extracting {total_videos} {kind} with {jobs} parallel ffmpeg workers (size={size})")

    if jobs <= 1:
        for video, out_dir in tasks:
            n = _extract_one(video, out_dir, fps, size, grayscale=grayscale)
            if n < 0:
                failed += 1
            else:
                total_frames += n
            print(f"  {video.name}: {n} frames -> {out_dir}")
    else:
        with ProcessPoolExecutor(max_workers=jobs) as pool:
            futures = {
                pool.submit(_extract_one, video, out_dir, fps, size, grayscale=grayscale): (
                    video,
                    out_dir,
                )
                for video, out_dir in tasks
            }
            done = 0
            for fut in as_completed(futures):
                video, out_dir = futures[fut]
                try:
                    n = fut.result()
                except Exception as exc:  # noqa: BLE001
                    failed += 1
                    done += 1
                    print(f"  [{done}/{total_videos}] {video.name}: ERROR {exc}")
                    continue
                if n < 0:
                    failed += 1
                else:
                    total_frames += n
                done += 1
                print(f"  [{done}/{total_videos}] {video.name}: {n} frames -> {out_dir}")
    print(f"done: {total_videos} videos, {total_frames} new frames extracted, {failed} failed")


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
    parser.add_argument(
        "--jobs",
        type=int,
        default=max(1, (os.cpu_count() or 1) // 2),
        help="Parallel ffmpeg workers (default: half of available CPUs).",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=256,
        help="Resize frames to N×N during extraction (default: 256, the FF++ "
        "leaderboard convention). Pass 0 to keep native resolution.",
    )
    parser.add_argument(
        "--testing-list",
        type=Path,
        default=None,
        help="Optional path to a Celeb-DF List_of_testing_videos.txt; if set, "
        "only videos whose '<subset>/<file>.mp4' matches a list entry are extracted.",
    )
    parser.add_argument(
        "--include-masks",
        action="store_true",
        help="FF++ only: also extract per-frame manipulation masks from "
        "<root>/manipulated_sequences/<method>/masks/videos/<id>.mp4 into "
        "<root>/manipulated_sequences/<method>/<compression>/masks/<id>/<frame>.png "
        "(grayscale, nearest-neighbour-resized to preserve binary edges).",
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

    testing_list: set[str] | None = None
    if args.testing_list is not None:
        if not args.testing_list.exists():
            print(f"testing list not found: {args.testing_list}", file=sys.stderr)
            return 2
        testing_list = set()
        for line in args.testing_list.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            # Format: "<label> <subset>/<file>.mp4" — keep only the path.
            parts = line.split(maxsplit=1)
            testing_list.add(parts[1] if len(parts) == 2 else parts[0])
        print(f"testing list filter: {len(testing_list)} videos")

    _walk_and_extract(targets, args.fps, args.size, args.jobs, testing_list)

    if args.include_masks:
        if args.dataset != "face-forensics":
            print("--include-masks is only supported for FF++; skipping.", file=sys.stderr)
        else:
            mask_targets = _ff_mask_targets(args.root)
            print(f"\nextracting FF++ manipulation masks for compression={args.compression}")
            _walk_and_extract(
                mask_targets,
                args.fps,
                args.size,
                args.jobs,
                testing_list=None,  # FF++ masks aren't filtered by Celeb's list
                grayscale=True,
                out_root_for=lambda d: _mask_frames_root_for(d, args.compression),
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
