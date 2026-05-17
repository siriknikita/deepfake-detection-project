"""Disclosed in-the-wild robustness probe for the trained detectors.

Motivation: qualitative testing through the web app showed the RGB and
RGB+physics models false-positiving on heavily post-processed / recompressed
real photographs, while the 9-channel frequency-augmented build corrected
them. This script turns that anecdote into a measurable, defensible result:
it batches a folder of real images (and, optionally, a folder of generated
images) through the registered CNN detectors and reports per-model
false-positive rate, detection rate, and (when both classes are present)
AUROC.

Methodology notes (read before citing results):

  * This is a *disclosed, symmetric* probe — every image in --real-dir and
    --fake-dir is scored; no frame/sample is dropped to improve the number.
  * Prefer watermark-free images. Overlay watermarks (e.g. stock-photo
    text) are themselves localised synthetic injections and confound a
    "robustness to non-generative post-processing" claim.
  * Scores come from the same single-image inference path the web app uses
    (forge_detect.serving_infer.score_image): MTCNN face crop -> physics /
    frequency channel assembly -> EfficientNet. Heuristic is diagnostic-
    only and excluded.
  * Single-seed checkpoints; report N and treat small N as illustrative,
    complementary to the §12 cross-dataset numbers, not a replacement.

Usage:

    # Real-only false-positive probe:
    python scripts/in_the_wild_eval.py \\
        --real-dir ~/itw/real \\
        --output runs/itw/itw_report.json --csv runs/itw/itw_per_image.csv

    # Real + generated (adds detection rate + AUROC):
    python scripts/in_the_wild_eval.py \\
        --real-dir ~/itw/real --fake-dir ~/itw/fake \\
        --output runs/itw/itw_report.json

    # Also export channel-preview figures for the frequency-correction
    # cases (real images RGB/6-ch call fake but 9-ch calls real):
    python scripts/in_the_wild_eval.py \\
        --real-dir ~/itw/real \\
        --save-figures runs/itw/figures --max-figures 8
"""

from __future__ import annotations

import argparse
import base64
import csv
import json
import sys
from pathlib import Path
from typing import Any

_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
_DEFAULT_MODELS = "baseline_3ch,physics_6ch,physics_9ch"


def _iter_images(root: Path, *, recursive: bool, limit: int | None) -> list[Path]:
    it = root.rglob("*") if recursive else root.iterdir()
    files = sorted(p for p in it if p.is_file() and p.suffix.lower() in _IMAGE_EXTS)
    return files[:limit] if limit is not None else files


def _score_dir(
    root: Path,
    model_ids: list[str],
    *,
    recursive: bool,
    limit: int | None,
) -> tuple[list[dict[str, Any]], int]:
    """Score every image under ``root``. Returns (rows, n_skipped)."""
    from forge_detect.serving_infer import score_image

    rows: list[dict[str, Any]] = []
    skipped = 0
    files = _iter_images(root, recursive=recursive, limit=limit)
    for i, path in enumerate(files, 1):
        try:
            out = score_image(path.read_bytes(), model_ids)
        except Exception as exc:  # one bad file shouldn't abort the run
            print(f"  [skip] {path.name}: {exc}", file=sys.stderr)
            skipped += 1
            continue
        models = out["models"]
        row: dict[str, Any] = {
            "path": str(path),
            "face_detected": out["face_detected"],
        }
        for mid in model_ids:
            row[f"{mid}__score"] = models[mid]["score"]
            row[f"{mid}__verdict"] = models[mid]["verdict"]
        rows.append(row)
        if i % 10 == 0 or i == len(files):
            print(f"  scored {i}/{len(files)} in {root.name}/")
    return rows, skipped


def _auroc(real_scores: list[float], fake_scores: list[float]) -> float | None:
    if not real_scores or not fake_scores:
        return None
    from sklearn.metrics import roc_auc_score

    y = [0] * len(real_scores) + [1] * len(fake_scores)
    s = real_scores + fake_scores
    return float(roc_auc_score(y, s))


def _summarise(
    real_rows: list[dict[str, Any]],
    fake_rows: list[dict[str, Any]],
    model_ids: list[str],
) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for mid in model_ids:
        real_s = [r[f"{mid}__score"] for r in real_rows if r[f"{mid}__score"] is not None]
        fake_s = [r[f"{mid}__score"] for r in fake_rows if r[f"{mid}__score"] is not None]
        n_real = len(real_s)
        n_fake = len(fake_s)
        fp = sum(1 for s in real_s if s >= 0.5)
        tp = sum(1 for s in fake_s if s >= 0.5)
        summary[mid] = {
            "n_real": n_real,
            "n_fake": n_fake,
            "false_positives": fp,
            "fp_rate": (fp / n_real) if n_real else None,
            "mean_p_fake_on_real": (sum(real_s) / n_real) if n_real else None,
            "detections": tp,
            "detection_rate": (tp / n_fake) if n_fake else None,
            "auroc": _auroc(real_s, fake_s),
        }
    return summary


def _correction_cases(
    real_rows: list[dict[str, Any]], model_ids: list[str],
) -> list[str]:
    """Real images RGB/6-ch flag as fake but the 9-ch build calls real.

    These are the §12-illustrative frequency-correction examples.
    """
    has_9 = "physics_9ch" in model_ids
    geo = [m for m in ("baseline_3ch", "physics_6ch") if m in model_ids]
    if not has_9 or not geo:
        return []
    cases: list[str] = []
    for r in real_rows:
        if r.get("physics_9ch__verdict") != "real":
            continue
        if any(r.get(f"{m}__verdict") == "fake" for m in geo):
            cases.append(r["path"])
    return cases


def _save_figures(paths: list[str], model_ids: list[str], out_dir: Path) -> None:
    """Export channel-preview / 6-panel figures for illustrative cases."""
    from forge_detect.serving_infer import infer_image

    out_dir.mkdir(parents=True, exist_ok=True)
    for path in paths:
        stem = Path(path).stem
        try:
            res = infer_image(Path(path).read_bytes(), model_ids)
        except Exception as exc:  # skip a single bad file, keep going
            print(f"  [fig skip] {stem}: {exc}", file=sys.stderr)
            continue
        case_dir = out_dir / stem
        case_dir.mkdir(parents=True, exist_ok=True)
        _write_data_uri(case_dir / "input.png", str(res["input"]))
        for m in res["models"]:  # type: ignore[attr-defined]
            mid = m["id"]
            if m.get("panel"):
                _write_data_uri(case_dir / f"{mid}__panel.png", m["panel"])
            for j, ch in enumerate(m.get("channels", [])):
                label = ch["label"].split(" ")[0].replace("(", "").replace("/", "")
                _write_data_uri(case_dir / f"{mid}__{j}_{label}.png", ch["png"])
        print(f"  figures -> {case_dir}/")


def _write_data_uri(path: Path, data_uri: str) -> None:
    b64 = data_uri.split(",", 1)[1] if "," in data_uri else data_uri
    path.write_bytes(base64.b64decode(b64))


def _print_table(summary: dict[str, Any]) -> None:
    print("\n=== in-the-wild summary ===")
    hdr = f"{'model':<16} {'N_real':>6} {'FP':>4} {'FP_rate':>8} "
    hdr += f"{'meanPf':>7} {'N_fake':>6} {'det_rate':>9} {'AUROC':>7}"
    print(hdr)
    print("-" * len(hdr))
    for mid, s in summary.items():
        def f(x: float | None, p: str) -> str:
            return "   -  " if x is None else format(x, p)

        print(
            f"{mid:<16} {s['n_real']:>6} {s['false_positives']:>4} "
            f"{f(s['fp_rate'], '8.3f')} {f(s['mean_p_fake_on_real'], '7.3f')} "
            f"{s['n_fake']:>6} {f(s['detection_rate'], '9.3f')} "
            f"{f(s['auroc'], '7.3f')}",
        )


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="in_the_wild_eval",
        description="Disclosed in-the-wild robustness probe for the CNN detectors.",
    )
    p.add_argument("--real-dir", type=Path, required=True, help="Folder of real images.")
    p.add_argument(
        "--fake-dir", type=Path, default=None,
        help="Optional folder of generated images (enables detection rate + AUROC).",
    )
    p.add_argument(
        "--models", default=_DEFAULT_MODELS,
        help=f"Comma-separated registry ids (default: {_DEFAULT_MODELS}).",
    )
    p.add_argument("--output", type=Path, default=None, help="Write the JSON report here.")
    p.add_argument("--csv", type=Path, default=None, help="Write per-image rows here.")
    p.add_argument(
        "--limit", type=int, default=None,
        help="Cap images per directory (quick smoke runs).",
    )
    p.add_argument(
        "--no-recursive", action="store_true",
        help="Only scan the top level of each directory.",
    )
    p.add_argument(
        "--save-figures", type=Path, default=None,
        help="Export channel-preview figures for frequency-correction cases.",
    )
    p.add_argument(
        "--max-figures", type=int, default=10,
        help="Cap exported figure cases (default: 10).",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    from forge_detect.serving_infer import InferenceError, get_model_spec

    model_ids = [m.strip() for m in args.models.split(",") if m.strip()]
    try:
        for mid in model_ids:
            get_model_spec(mid)
    except InferenceError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if not args.real_dir.is_dir():
        print(f"error: --real-dir {args.real_dir} is not a directory", file=sys.stderr)
        return 2

    recursive = not args.no_recursive
    print(f"scoring real images under {args.real_dir} ...")
    real_rows, real_skip = _score_dir(
        args.real_dir, model_ids, recursive=recursive, limit=args.limit,
    )
    fake_rows: list[dict[str, Any]] = []
    fake_skip = 0
    if args.fake_dir is not None:
        if not args.fake_dir.is_dir():
            print(f"error: --fake-dir {args.fake_dir} is not a directory", file=sys.stderr)
            return 2
        print(f"scoring generated images under {args.fake_dir} ...")
        fake_rows, fake_skip = _score_dir(
            args.fake_dir, model_ids, recursive=recursive, limit=args.limit,
        )

    summary = _summarise(real_rows, fake_rows, model_ids)
    corrections = _correction_cases(real_rows, model_ids)
    _print_table(summary)
    print(
        f"\nfrequency-correction cases (real; RGB/6-ch=fake, 9-ch=real): "
        f"{len(corrections)} / {len(real_rows)}",
    )
    if real_skip or fake_skip:
        print(f"skipped unreadable: real={real_skip} fake={fake_skip}")

    report: dict[str, Any] = {
        "models": model_ids,
        "n_real": len(real_rows),
        "n_fake": len(fake_rows),
        "summary": summary,
        "frequency_correction_cases": corrections,
        "methodology": (
            "Disclosed symmetric probe; no sample dropped. Prefer "
            "watermark-free images. Single-seed checkpoints; small N is "
            "illustrative and complements the §12 cross-dataset results."
        ),
    }
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2))
        print(f"\nwrote report -> {args.output}")
    if args.csv is not None and (real_rows or fake_rows):
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        all_rows = (
            [{**r, "label": "real"} for r in real_rows]
            + [{**r, "label": "fake"} for r in fake_rows]
        )
        with args.csv.open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(all_rows[0].keys()))
            writer.writeheader()
            writer.writerows(all_rows)
        print(f"wrote per-image csv -> {args.csv}")
    if args.save_figures is not None and corrections:
        print(f"\nexporting up to {args.max_figures} figure cases ...")
        _save_figures(corrections[: args.max_figures], model_ids, args.save_figures)

    return 0


if __name__ == "__main__":
    sys.exit(main())
