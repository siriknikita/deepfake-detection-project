"""Persist demo runs to disk for use as thesis/defense figures.

Every ``/api/detect`` run (image inline, or video on job completion) is
saved under ``demo-artifacts/`` (override with ``$DEMO_ARTIFACTS_DIR``):
the input face crop, each model's 6-panel manifold figure and channel
previews as real PNGs (decoded from the base64 data URIs the inference
adapter returns), the full ``result.json`` with data URIs replaced by the
saved filenames, and a human-readable ``summary.md`` table.

Saving is best-effort: any failure is swallowed and logged so it can
never break the detection response.
"""

from __future__ import annotations

import base64
import copy
import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any

log = logging.getLogger("serving.artifacts")

_DATA_URI = re.compile(r"^data:image/(?P<ext>[a-z]+);base64,(?P<b64>.+)$", re.DOTALL)


def _root() -> Path:
    import os

    env = os.environ.get("DEMO_ARTIFACTS_DIR")
    base = Path(env) if env else Path(__file__).resolve().parent.parent / "demo-artifacts"
    return base


def _slug(text: str, maxlen: int = 40) -> str:
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", (text or "").strip())
    s = s.strip("._-") or "untitled"
    return s[:maxlen]


def _decode_data_uri(value: Any, dest: Path) -> bool:
    """If ``value`` is a PNG/JPEG data URI, write it to ``dest``. Returns True if written."""
    if not isinstance(value, str):
        return False
    m = _DATA_URI.match(value)
    if not m:
        return False
    dest.write_bytes(base64.b64decode(m.group("b64")))
    return True


def _overall(models: list[dict[str, Any]]) -> tuple[str, float | None]:
    """Coarse verdict + mean P(fake) over scored models (for the folder name)."""
    scores = [m["score"] for m in models if isinstance(m.get("score"), (int, float))]
    if not scores:
        return "diagnostic", None
    mean = sum(scores) / len(scores)
    return ("fake" if mean >= 0.5 else "real"), mean


def _summary_md(result: dict[str, Any], filename: str | None, mean: float | None) -> str:
    kind = result.get("kind", "?")
    face = result.get("face", {}) or {}
    lines = [
        f"# Demo run — {filename or '(unnamed upload)'}",
        "",
        f"- **kind**: {kind}",
        f"- **face usable**: {face.get('usable')}",
    ]
    if kind == "video":
        lines.append(
            f"- **usable frames**: {face.get('usable_frames')}/{face.get('total_frames')}"
        )
    elif face.get("prob") is not None:
        lines.append(f"- **face prob**: {face.get('prob')}")
    if mean is not None:
        lines.append(f"- **mean P(fake) across scored models**: {mean * 100:.1f}%")
    lines += [
        "",
        "| Model | Verdict | P(fake) mean | P(fake) max | Confidence |",
        "|---|---|---:|---:|---:|",
    ]
    for m in result.get("models", []):
        score = m.get("score")
        if isinstance(score, (int, float)):
            conf = max(score, 1 - score)
            smax = m.get("score_max")
            smax_s = f"{smax * 100:.1f}%" if isinstance(smax, (int, float)) else "—"
            lines.append(
                f"| {m.get('label')} | {str(m.get('verdict')).upper()} "
                f"| {score * 100:.1f}% | {smax_s} | {conf * 100:.1f}% |"
            )
        else:
            lines.append(f"| {m.get('label')} | diagnostic | — | — | — |")
    lines += [
        "",
        "> Scores are uncalibrated ranker outputs; the 0.5 cut is a provisional "
        "placeholder, not a validated threshold. Only relative comparison across "
        "models and rank / AUROC are meaningful.",
        "",
    ]
    return "\n".join(lines)


# Ukrainian captions + per-phase titles so the combined figures drop
# straight into the (Ukrainian) thesis without further editing.
_UA_CHANNEL = {
    "Face crop (RGB input)": "Вхідний кадр (RGB)",
    "W_cnn (trust map)": "W_cnn — мапа довіри CNN",
    "z* (settled manifold)": "z* — осіла поверхня",
    "R (flow break)": "R — розрив потоку",
    "DCT block energy": "Енергія блоків DCT",
    "DCT high-band ratio": "Частка високих частот DCT",
    "FFT log-magnitude": "Лог-магнітуда FFT",
}
# Per-phase figure captions (підписи). These double as the file names, so
# they read like the «Рисунок …» caption the image will carry in the thesis.
_UA_PHASE = {
    "baseline_3ch": "Вхідний канал базової RGB-моделі Фази 1",
    "physics_6ch": "Вхідні канали фізичної моделі Фази 2 (6 каналів)",
    "physics_9ch": "Вхідні канали фізично-частотної моделі Фази 3 (9 каналів)",
}


def _ua_slug(text: str, maxlen: int = 90) -> str:
    """Filename slug that keeps Cyrillic letters and digits (\\w is Unicode)."""
    s = re.sub(r"[^\w-]+", "_", text.strip(), flags=re.UNICODE)
    s = re.sub(r"_+", "_", s).strip("_-")
    return (s or "figure")[:maxlen]


def _font(size: int) -> Any:
    from PIL import ImageFont

    candidates = [
        "/System/Library/Fonts/Supplemental/Arial.ttf",  # has Cyrillic
        "/Library/Fonts/Arial.ttf",
    ]
    try:  # DejaVuSans (ships with matplotlib) is a guaranteed-Cyrillic fallback
        import matplotlib

        candidates.append(
            str(Path(matplotlib.__file__).parent / "mpl-data/fonts/ttf/DejaVuSans.ttf")
        )
    except Exception:  # noqa: BLE001
        pass
    for p in candidates:
        try:
            return ImageFont.truetype(p, size)
        except Exception:  # noqa: BLE001
            continue
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def _prep(path: Path) -> Any:
    """Open an image, flatten any alpha onto white -> RGB."""
    from PIL import Image

    img = Image.open(path)
    if img.mode in ("RGBA", "LA", "P"):
        img = img.convert("RGBA")
        bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
        img = Image.alpha_composite(bg, img)
    return img.convert("RGB")


def _wrap(draw: Any, text: str, font: Any, max_w: int, max_lines: int = 2) -> list[str]:
    words = text.split()
    lines: list[str] = []
    cur = ""
    for w in words:
        trial = f"{cur} {w}".strip()
        if not cur or draw.textlength(trial, font=font) <= max_w:
            cur = trial
        else:
            lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines[:max_lines]


def _phase_grid(dest: Path, cells: list[tuple[Path, str]], ncols: int = 4) -> Path | None:
    """One figure per phase: channels wrapped into rows of ``ncols``.

    With the channel order (RGB · W_cnn · z* · R · DCT · DCT · FFT) and
    ``ncols=4``, phase-3 lands as row 1 = RGB+physics, row 2 = frequency;
    phase-2 (4 channels) is a single row. No baked-in title — the figure's
    «Рисунок …» caption is added in Typst. Per-tile captions are Ukrainian.
    """
    from PIL import Image, ImageDraw

    CELL, GAP, PAD = 360, 24, 34
    CAP_H, LINE_H = 70, 30
    f_cap = _font(24)

    imgs = [
        (_prep(p).resize((CELL, CELL), Image.LANCZOS), cap)
        for p, cap in cells
        if p.exists()
    ]
    if not imgs:
        return None

    ncols = min(ncols, len(imgs))
    nrows = (len(imgs) + ncols - 1) // ncols
    cell_h = CELL + CAP_H
    width = PAD * 2 + ncols * CELL + GAP * (ncols - 1)
    height = PAD * 2 + nrows * cell_h + GAP * (nrows - 1)

    canvas = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    y0 = PAD
    for idx, (img, cap) in enumerate(imgs):
        r, c = divmod(idx, ncols)
        x = PAD + c * (CELL + GAP)
        y = y0 + r * (cell_h + GAP)
        canvas.paste(img, (x, y))
        ty = y + CELL + 8
        for line in _wrap(draw, cap, f_cap, CELL):
            tw = draw.textlength(line, font=f_cap)
            draw.text((x + (CELL - tw) / 2, ty), line, fill=(70, 70, 70), font=f_cap)
            ty += LINE_H
    canvas.save(dest, format="PNG")
    return dest


def save_run(result: dict[str, Any], filename: str | None, kind: str) -> Path | None:
    """Save one detection run's images + metadata. Best-effort; returns the dir or None."""
    try:
        models = result.get("models", []) or []
        verdict, mean = _overall(models)
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        base = _slug(Path(filename or "upload").stem)
        run_dir = _root() / f"{stamp}__{kind}__{verdict}__{base}"
        run_dir.mkdir(parents=True, exist_ok=True)

        # Deep copy we can rewrite (data URI -> saved filename) for result.json.
        meta = copy.deepcopy(result)

        if _decode_data_uri(result.get("input"), run_dir / "input.png"):
            meta["input"] = "input.png"

        # Per phase: individual channel PNGs under <model_id>/, plus one
        # combined captioned grid <model_id>__grid.png at the run root.
        # Phase 2 (physics_6ch): 01 crop · 02 W_cnn · 03 z* · 04 R (one row).
        # Phase 3 (physics_9ch): the same four (row 1), then 05–07 the
        # frequency maps (row 2). Grid captions are Ukrainian.
        for entry, mentry in zip(models, meta.get("models", [])):
            mid = _slug(str(entry.get("id", "model")))
            phase_dir = run_dir / mid
            grid_cells: list[tuple[Path, str]] = []

            if entry.get("panel"):  # heuristic 6-panel diagnostic figure
                phase_dir.mkdir(parents=True, exist_ok=True)
                if _decode_data_uri(entry.get("panel"), phase_dir / "panel.png"):
                    mentry["panel"] = f"{mid}/panel.png"

            for i, ch in enumerate(entry.get("channels", []) or []):
                ch_label = str(ch.get("label", "ch"))
                fname = f"{i + 1:02d}_{_slug(ch_label)}.png"
                phase_dir.mkdir(parents=True, exist_ok=True)
                if _decode_data_uri(ch.get("png"), phase_dir / fname):
                    grid_cells.append(
                        (phase_dir / fname, _UA_CHANNEL.get(ch_label, ch_label))
                    )
                    rel = f"{mid}/{fname}"
                    for mch in mentry.get("channels", []):
                        if mch.get("label") == ch.get("label"):
                            mch["png"] = rel

            if len(grid_cells) > 1:  # single-channel phases don't need a grid
                caption = _UA_PHASE.get(
                    str(entry.get("id", "")), str(entry.get("label", mid))
                )
                grid_name = f"{_ua_slug(caption)}.png"
                try:
                    if _phase_grid(run_dir / grid_name, grid_cells):
                        mentry["grid"] = grid_name
                        mentry["grid_caption"] = caption
                except Exception:  # noqa: BLE001 — grid is a nicety, never fatal
                    log.exception("failed to build phase grid for %s", mid)

        (run_dir / "result.json").write_text(
            json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        (run_dir / "summary.md").write_text(
            _summary_md(result, filename, mean), encoding="utf-8"
        )

        log.info("saved demo artifacts to %s", run_dir)
        return run_dir
    except Exception:  # noqa: BLE001 — saving must never break the response
        log.exception("failed to save demo artifacts")
        return None
