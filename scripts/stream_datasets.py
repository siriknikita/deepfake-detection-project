"""Stream FF++ / Celeb-DF archives from an rclone remote and keep only face crops.

Built for machines that cannot hold the archives (the FF++ frame tar is 13.8 GB,
Celeb-DF v2 is 9.9 GB of videos). Nothing but the final 256x256 face crops —
the same ``frames_faces/`` layout ``extract_faces.py`` produces — is written to
disk, so the downstream scripts (``cache_physics_maps.py``,
``train_physics_cnn.py --use-face-crops``, ``eval_celebdf.py``) work unchanged.

Two sources:

``--source ff``
    ``rclone cat`` streams ``ff-frames.tar`` once through a Python ``tarfile``
    reader. Frames of one video are buffered in memory (30 PNGs, ~3 MB), the
    video is stride-sampled to ``--frames-per-video`` frames exactly like
    ``extract_faces.py`` (so the crops match the adapter's sampling), MTCNN
    crops them, and the buffer is dropped. Resumable: the byte offset of the
    last fully processed tar member is checkpointed and ``rclone cat --offset``
    restarts from there.

``--source celeb``
    Zip archives support random access, so ``rclone serve http`` plus HTTP
    range requests fetch one ``.mp4`` member at a time (the central directory
    costs < 1 MB). ``ffmpeg`` decodes ``--frames-per-video`` uniformly spaced
    frames into a temp dir, MTCNN crops them, and the video is deleted.
    ``--testing-list-only`` restricts to the 518-video published benchmark.
    Resumable per video.

A free-space guard (``--min-free-gb``) stops the run cleanly before the disk
fills; rerunning the same command resumes.

Usage::

    python scripts/stream_datasets.py --source ff \\
        --remote gdrive:Science/Datasets/ff-frames.tar \\
        --data-root ~/Development/datasets/FaceForensics++ \\
        --frames-per-video 10 --device cpu --download-splits

    python scripts/stream_datasets.py --source celeb \\
        --remote gdrive:Science/Datasets/Celeb-DF-v2.zip \\
        --data-root ~/Development/datasets/Celeb-DF-v2 \\
        --frames-per-video 10 --device cpu --testing-list-only
"""

from __future__ import annotations

import argparse
import io
import json
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
import urllib.request
import zipfile
from collections import OrderedDict
from pathlib import Path

from PIL import Image

_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
_FF_SPLITS_BASE = "https://raw.githubusercontent.com/ondyari/FaceForensics/master/dataset/splits"


# --------------------------------------------------------------------------- utils
def _free_gb(path: Path) -> float:
    path.mkdir(parents=True, exist_ok=True)
    return shutil.disk_usage(path).free / 1e9


def _stride_sample(names: list[str], max_frames: int | None) -> list[str]:
    """Same rule as ``extract_faces._list_frames_in_video`` / the dataset adapters."""
    names = sorted(names)
    if max_frames is not None and len(names) > max_frames:
        stride = max(1, len(names) // max_frames)
        names = names[::stride][:max_frames]
    return names


class FaceCropper:
    """MTCNN highest-confidence face, square crop with margin, resize (extract_faces.py rules)."""

    def __init__(self, device: str, output_size: int, margin: float) -> None:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from extract_faces import (  # type: ignore[import-not-found]
            _build_mtcnn,
            _center_square_box,
            _crop_and_resize,
            _square_box_from_detection,
        )

        self.mtcnn = _build_mtcnn(device, output_size)
        self.output_size = output_size
        self.margin = margin
        self._square = _square_box_from_detection
        self._center = _center_square_box
        self._crop = _crop_and_resize
        self.n_face = 0
        self.n_fallback = 0

    def crop(self, image: Image.Image) -> Image.Image:
        boxes, _ = self.mtcnn.detect(image)
        if boxes is not None and len(boxes) > 0:
            box_arr = boxes if boxes.ndim == 1 else boxes[0]
            box = self._square(
                tuple(float(v) for v in box_arr), image.width, image.height, self.margin
            )
            self.n_face += 1
        else:
            box = self._center(image.width, image.height)
            self.n_fallback += 1
        return self._crop(image, box, self.output_size)


def _atomic_save(img: Image.Image, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    img.save(tmp, format="PNG")
    tmp.replace(out_path)


def _video_done(out_dir: Path, n: int) -> bool:
    return out_dir.is_dir() and sum(1 for p in out_dir.iterdir() if p.suffix == ".png") >= n


class Progress:
    def __init__(self, tag: str, log_every: int) -> None:
        self.tag, self.log_every, self.t0 = tag, log_every, time.time()
        self.videos = self.skipped = self.frames = 0

    def tick(self, cropper: FaceCropper, extra: str = "") -> None:
        if self.videos % self.log_every == 0:
            el = time.time() - self.t0
            rate = cropper.n_face / max(1, cropper.n_face + cropper.n_fallback)
            print(
                f"[{self.tag}] videos={self.videos} skipped={self.skipped} frames={self.frames} "
                f"face-rate={rate:.1%} elapsed={el / 60:.1f}min "
                f"({self.frames / max(el, 1e-6):.1f} frames/s) {extra}",
                flush=True,
            )


# --------------------------------------------------------------------------- FF++ (tar stream)
def _ff_out_dir(data_root: Path, member_name: str) -> tuple[Path, str] | None:
    """Map ``.../c23/frames/<vid>/<f>.png`` to ``(<root>/.../frames_faces/<vid>, vid)``."""
    parts = Path(member_name).parts
    # macOS tar sidecars (._name AppleDouble files, .DS_Store) are not frames.
    if any(part.startswith(".") for part in parts) or "frames" not in parts or len(parts) < 3:
        return None
    i = parts.index("frames")
    vid = parts[i + 1] if len(parts) > i + 1 else None
    if vid is None or len(parts) != i + 3:
        return None
    return data_root.joinpath(*parts[:i], "frames_faces", vid), vid


def run_ff(args: argparse.Namespace) -> int:  # noqa: PLR0912
    data_root: Path = args.data_root
    data_root.mkdir(parents=True, exist_ok=True)
    state_path = data_root / ".stream_ff_progress.json"
    offset = 0
    if state_path.exists() and not args.restart:
        offset = int(json.loads(state_path.read_text()).get("resume_offset", 0))
        print(f"[ff] resuming from tar offset {offset / 1e9:.3f} GB")
    if args.download_splits:
        sd = data_root / "splits"
        sd.mkdir(exist_ok=True)
        for n in ("train.json", "val.json", "test.json"):
            if not (sd / n).exists():
                urllib.request.urlretrieve(f"{_FF_SPLITS_BASE}/{n}", sd / n)
                print(f"[ff] downloaded splits/{n}")
    cropper = FaceCropper(args.device, args.output_size, args.margin)
    prog = Progress("ff", args.log_every)

    cmd = ["rclone", "cat", "--offset", str(offset), args.remote]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=16 << 20)
    assert proc.stdout is not None
    stream = tarfile.open(fileobj=proc.stdout, mode="r|")  # noqa: SIM115
    buffers: OrderedDict[str, tuple[Path, dict[str, bytes]]] = OrderedDict()
    last_flushed_offset = offset
    stopped_for_space = False

    def flush(vid_key: str) -> None:
        nonlocal stopped_for_space
        out_dir, frames = buffers.pop(vid_key)
        if _video_done(out_dir, min(args.frames_per_video, len(frames))):
            prog.skipped += 1
            return
        if _free_gb(data_root) < args.min_free_gb:
            print(
                f"[ff] free space below {args.min_free_gb} GB; stopping cleanly (rerun to resume)"
            )
            stopped_for_space = True
            return
        for name in _stride_sample(list(frames), args.frames_per_video):
            img = Image.open(io.BytesIO(frames[name])).convert("RGB")
            _atomic_save(cropper.crop(img), out_dir / name)
            prog.frames += 1
        prog.videos += 1
        prog.tick(cropper, f"last={out_dir.parent.parent.name}/{out_dir.name}")

    try:
        for member in stream:
            if stopped_for_space:
                break
            if not member.isfile() or Path(member.name).suffix.lower() not in _IMAGE_EXTENSIONS:
                continue
            target = _ff_out_dir(data_root, member.name)
            if target is None:
                continue
            out_dir, _vid = target
            key = str(out_dir)
            f = stream.extractfile(member)
            if f is None:
                continue
            data = f.read()
            if key not in buffers:
                # a new video started: videos older than the previous one are complete
                while len(buffers) > 1:
                    flush(next(iter(buffers)))
                    last_flushed_offset = member.offset + offset
                    state_path.write_text(json.dumps({"resume_offset": last_flushed_offset}))
                buffers[key] = (out_dir, {})
            buffers[key][1][Path(member.name).name] = data
            if args.limit_videos and prog.videos + prog.skipped >= args.limit_videos:
                break
        while buffers and not stopped_for_space:
            flush(next(iter(buffers)))
    finally:
        proc.kill()
    if not stopped_for_space and not args.limit_videos:
        state_path.write_text(json.dumps({"resume_offset": 0, "complete": True}))
        print("[ff] stream complete")
    _summary("ff", data_root, cropper, prog)
    return 2 if stopped_for_space else 0


# ------------------------------------------------------------------ Celeb-DF (zip over HTTP)
class HttpRangeFile:
    """Minimal seekable read-only file over HTTP Range requests (duck-typed for zipfile)."""

    def __init__(self, url: str, chunk: int = 8 << 20) -> None:
        import requests

        self._requests = requests
        self.url, self.pos, self.chunk = url, 0, chunk
        self.size = int(requests.head(url).headers["Content-Length"])
        self.bytes_read = 0
        self._buf, self._buf_start = b"", 0

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.pos

    def seek(self, off: int, whence: int = 0) -> int:
        self.pos = {0: off, 1: self.pos + off, 2: self.size + off}[whence]
        return self.pos

    def read(self, n: int = -1) -> bytes:
        if n is None or n < 0:
            n = self.size - self.pos
        out = b""
        while n > 0 and self.pos < self.size:
            if not (self._buf_start <= self.pos < self._buf_start + len(self._buf)):
                start = self.pos
                end = min(self.size, start + max(n, self.chunk)) - 1
                r = self._requests.get(
                    self.url, headers={"Range": f"bytes={start}-{end}"}, timeout=120
                )
                r.raise_for_status()
                self._buf, self._buf_start = r.content, start
                self.bytes_read += len(r.content)
            off = self.pos - self._buf_start
            piece = self._buf[off : off + n]
            self.pos += len(piece)
            n -= len(piece)
            out += piece
        return out


def _ffmpeg_sample_frames(video: Path, out_dir: Path, n: int) -> list[Path]:
    """Decode ``n`` uniformly spaced frames (0001.png..) using ffprobe's frame count."""
    probe = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-count_frames",
            "-show_entries",
            "stream=nb_read_frames",
            "-of",
            "csv=p=0",
            str(video),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    try:
        total = int(probe.stdout.strip().splitlines()[0])
    except (ValueError, IndexError):
        total = 0
    step = max(1, total // n) if total else 1
    out_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg",
            "-loglevel",
            "error",
            "-i",
            str(video),
            "-vf",
            f"select=not(mod(n\\,{step}))",
            "-vsync",
            "vfr",
            "-frames:v",
            str(n),
            str(out_dir / "%04d.png"),
        ],
        check=True,
    )
    return sorted(out_dir.glob("*.png"))


def run_celeb(args: argparse.Namespace) -> int:
    data_root: Path = args.data_root
    data_root.mkdir(parents=True, exist_ok=True)
    port = args.port
    remote_dir, archive = args.remote.rsplit("/", 1)
    serve = subprocess.Popen(
        ["rclone", "serve", "http", remote_dir, "--addr", f"127.0.0.1:{port}", "--read-only"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        time.sleep(3)
        f = HttpRangeFile(f"http://127.0.0.1:{port}/{archive}")
        zf = zipfile.ZipFile(f)  # type: ignore[arg-type]
        infos = [i for i in zf.infolist() if not i.is_dir()]
        print(
            f"[celeb] {archive}: {len(infos)} members, central directory fetched "
            f"({f.bytes_read / 1e6:.1f} MB)"
        )
        # testing list ships inside the archive
        lists = [i for i in infos if i.filename.endswith("List_of_testing_videos.txt")]
        keep: set[str] | None = None
        if lists:
            txt = zf.read(lists[0]).decode()
            (data_root / "List_of_testing_videos.txt").write_text(txt)
            if args.testing_list_only:
                keep = {
                    ln.split(maxsplit=1)[1].rsplit(".", 1)[0]
                    for ln in txt.splitlines()
                    if len(ln.split()) == 2
                }
                print(f"[celeb] restricting to the {len(keep)} testing-list videos")
        videos = [i for i in infos if i.filename.lower().endswith(".mp4")]
        videos = [i for i in videos if i.filename.count("/") == 1]
        if keep is not None:
            videos = [i for i in videos if i.filename.rsplit(".", 1)[0] in keep]
        cropper = FaceCropper(args.device, args.output_size, args.margin)
        prog = Progress("celeb", args.log_every)
        stopped = False
        with tempfile.TemporaryDirectory(prefix="celeb-stream-") as td:
            tmp = Path(td)
            for k, info in enumerate(videos):
                if args.limit_videos and k >= args.limit_videos:
                    break
                subset, fname = info.filename.split("/", 1)
                stem = fname.rsplit(".", 1)[0]
                out_dir = data_root / subset / "frames_faces" / stem
                if _video_done(out_dir, args.frames_per_video):
                    prog.skipped += 1
                    continue
                if _free_gb(data_root) < args.min_free_gb:
                    print(
                        f"[celeb] free space below {args.min_free_gb} GB; stopping cleanly "
                        "(rerun to resume)"
                    )
                    stopped = True
                    break
                vpath = tmp / "video.mp4"
                with zf.open(info) as src, vpath.open("wb") as dst:
                    shutil.copyfileobj(src, dst)
                fdir = tmp / "frames"
                shutil.rmtree(fdir, ignore_errors=True)
                try:
                    frames = _ffmpeg_sample_frames(vpath, fdir, args.frames_per_video)
                except subprocess.CalledProcessError as exc:
                    print(f"[celeb] ffmpeg failed on {info.filename}: {exc}")
                    continue
                for fp in frames:
                    _atomic_save(cropper.crop(Image.open(fp).convert("RGB")), out_dir / fp.name)
                    prog.frames += 1
                prog.videos += 1
                prog.tick(cropper, f"last={subset}/{stem} fetched={f.bytes_read / 1e9:.2f}GB")
        _summary("celeb", data_root, cropper, prog)
        return 2 if stopped else 0
    finally:
        serve.kill()


def _summary(tag: str, data_root: Path, cropper: FaceCropper, prog: Progress) -> None:
    size = sum(p.stat().st_size for p in data_root.rglob("*.png")) / 1e9
    rate = cropper.n_face / max(1, cropper.n_face + cropper.n_fallback)
    print(
        f"[{tag}] done: videos={prog.videos} skipped={prog.skipped} frames={prog.frames} "
        f"face-rate={rate:.1%} crops-on-disk={size:.2f} GB free={_free_gb(data_root):.1f} GB "
        f"elapsed={(time.time() - prog.t0) / 60:.1f} min",
        flush=True,
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--source", choices=("ff", "celeb"), required=True)
    ap.add_argument(
        "--remote", required=True, help="rclone path, e.g. gdrive:Science/Datasets/ff-frames.tar"
    )
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--frames-per-video", type=int, default=10)
    ap.add_argument("--output-size", type=int, default=256)
    ap.add_argument("--margin", type=float, default=0.3)
    ap.add_argument(
        "--device",
        default="cpu",
        choices=("cpu", "cuda", "mps"),
        help="MTCNN device (MPS fails in facenet-pytorch's adaptive pool; cpu is reliable)",
    )
    ap.add_argument("--min-free-gb", type=float, default=4.0)
    ap.add_argument("--limit-videos", type=int, default=0, help="stop after N videos (smoke)")
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument(
        "--download-splits", action="store_true", help="ff: fetch the official splits JSONs"
    )
    ap.add_argument("--restart", action="store_true", help="ff: ignore the resume offset")
    ap.add_argument(
        "--testing-list-only", action="store_true", help="celeb: only the 518 benchmark videos"
    )
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args()
    if shutil.which("rclone") is None:
        print("rclone not found on PATH", file=sys.stderr)
        return 1
    if args.source == "celeb" and shutil.which("ffmpeg") is None:
        print("ffmpeg not found on PATH", file=sys.stderr)
        return 1
    return run_ff(args) if args.source == "ff" else run_celeb(args)


if __name__ == "__main__":
    raise SystemExit(main())
