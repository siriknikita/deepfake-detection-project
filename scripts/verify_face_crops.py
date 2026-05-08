"""Find and delete corrupt face-crop PNGs.

Walks every ``frames_faces/<vid>/<frame>.png`` under ``--data-root``,
opens each with PIL, and deletes any file that fails to verify (cannot
be decoded as a PNG). After cleaning, re-running ``extract_faces.py``
fills the deleted positions back in via its existence-check resume.

Why this exists:

    PIL.UnidentifiedImageError: cannot identify image file
    '.../frames_faces/566_617/0049.png'

is what the DataLoader raises mid-training when ``extract_faces.py``
was interrupted at some point and left a partially-written PNG on
disk. The next run's ``if out_path.exists(): skip`` then *trusts* the
broken file, so the corruption survives across restarts and only
shows up when training actually tries to read it. The fix in
``extract_faces.py`` (atomic tmp+rename) prevents new corruption;
this script cleans up whatever was left over from earlier runs.

Usage::

    python scripts/verify_face_crops.py --data-root ~/data/FaceForensics++

Add ``--dry-run`` to list bad files without deleting them.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, UnidentifiedImageError


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report bad files without deleting them.",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=5000,
        help="Print a progress line every N files (0 to silence).",
    )
    args = parser.parse_args()

    if not args.data_root.is_dir():
        print(f"ERROR: {args.data_root} is not a directory")
        return 1

    n_total = 0
    n_bad = 0
    bad_paths: list[Path] = []

    print(f"[verify] scanning {args.data_root} for corrupt face crops ...")
    # Match any frames_faces tree at any depth (works for FF++ and Celeb-DF).
    for png in args.data_root.rglob("frames_faces/*/*.png"):
        n_total += 1
        is_bad = False
        try:
            # Image.open is lazy; verify() forces a parse of the image data
            # without fully decoding pixels. Catches truncated or
            # non-PNG-content files cheaply.
            with Image.open(png) as img:
                img.verify()
        except (UnidentifiedImageError, OSError, SyntaxError, ValueError) as e:
            is_bad = True
            bad_reason = f"{type(e).__name__}: {e}"
        if is_bad:
            n_bad += 1
            bad_paths.append(png)
            if args.dry_run:
                print(f"  bad : {png}  ({bad_reason})")
            else:
                try:
                    png.unlink()
                    print(f"  rm  {png}  ({bad_reason})")
                except OSError as e:
                    print(f"  ERR remove {png}: {e}")
        if args.log_every and n_total % args.log_every == 0:
            print(f"  [{n_total}] bad so far: {n_bad}")

    print()
    print(f"checked {n_total} face crop files; corrupt: {n_bad}")
    if n_bad > 0:
        if args.dry_run:
            print("(dry run; nothing deleted)")
            print(
                "Re-run without --dry-run to clean up, then re-run "
                "scripts/extract_faces.py to refill the gaps.",
            )
        else:
            print(
                "Deleted. Re-run scripts/extract_faces.py to refill the "
                "deleted positions (the existence-check resume will skip "
                "every other frame).",
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
