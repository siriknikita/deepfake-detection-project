"""Per-method Phase-1 refit and evaluation against the FF++ leaderboard convention.

Reuses cached `features-{train,val,test}.csv` from a prior all-methods
run of `quick_classifier.py`. For each FF++ manipulation method, filters
to (real + that method only), refits a fresh GBC, evaluates on the same
video-disjoint val/test split, and stores per-method classifier + report.
Total runtime: a few minutes.

Workflow:

    python scripts/per_method_refit.py \\
        --runs-dir /Users/foo/Downloads/diploma/quick_phase1 \\
        --output-dir /Users/foo/runs/quick_phase1_per_method

The output is a leaderboard-style table: per-method frame and video
AUROCs, written to ``<output-dir>/summary.json`` and printed at the end.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from forge_detect.classifier import (
    ClassifierMetrics,
    evaluate_classifier,
    save_classifier,
    train_classifier,
)
from forge_detect.features import FEATURE_NAMES

FF_METHODS: tuple[str, ...] = ("Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures")


def _load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        msg = f"missing CSV: {path}"
        raise SystemExit(msg)
    return pd.read_csv(path)


def _filter_real_plus_method(df: pd.DataFrame, method: str) -> pd.DataFrame:
    """Return rows that are real (label==0) OR fakes from the given method."""
    real_mask = df["label"] == 0
    fake_mask = (df["label"] == 1) & df["path"].str.contains(
        f"manipulated_sequences/{method}/", regex=False,
    )
    return df[real_mask | fake_mask].reset_index(drop=True)


def _xy(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, list[str]]:
    feats = df[list(FEATURE_NAMES)].to_numpy(dtype=np.float64)
    labels = df["label"].to_numpy(dtype=np.int64)
    video_ids = df["video_id"].astype(str).tolist() if "video_id" in df.columns else None
    if video_ids is None:
        video_ids = [Path(p).parent.name for p in df["path"].tolist()]
    return feats, labels, video_ids


def _summarize(m: ClassifierMetrics) -> dict[str, Any]:
    return {
        "frame_auroc": m.auroc,
        "frame_accuracy": m.accuracy,
        "video_auroc_mean": m.video_auroc_mean,
        "video_auroc_max": m.video_auroc_max,
        "n_videos": m.n_videos,
        "n_video_real": m.n_video_real,
        "n_video_fake": m.n_video_fake,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--runs-dir",
        type=Path,
        required=True,
        help="Directory holding features-{train,val,test}.csv from a prior run.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Where to write per-method classifiers + reports + summary.",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=list(FF_METHODS),
        choices=list(FF_METHODS),
        help="Subset of methods to refit (default: all 4).",
    )
    args = parser.parse_args()

    print(f"[refit] runs-dir   = {args.runs_dir}")
    print(f"[refit] output-dir = {args.output_dir}")
    print(f"[refit] loading features-{{train,val,test}}.csv ...")

    train_df = _load_csv(args.runs_dir / "features-train.csv")
    val_df = _load_csv(args.runs_dir / "features-val.csv")
    test_df = _load_csv(args.runs_dir / "features-test.csv")
    print(
        f"[refit] loaded: train={len(train_df)} val={len(val_df)} test={len(test_df)} rows",
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, dict[str, Any]] = {}

    for method in args.methods:
        sub_train = _filter_real_plus_method(train_df, method)
        sub_val = _filter_real_plus_method(val_df, method)
        sub_test = _filter_real_plus_method(test_df, method)

        x_train, y_train, _ = _xy(sub_train)
        x_val, y_val, vid_val = _xy(sub_val)
        x_test, y_test, vid_test = _xy(sub_test)

        n_real_train = int((y_train == 0).sum())
        n_fake_train = int((y_train == 1).sum())
        print(f"\n=== {method} ===")
        print(
            f"  train: {len(sub_train)} rows ({n_real_train} real, {n_fake_train} fake)",
        )
        print(f"  val:   {len(sub_val)} rows")
        print(f"  test:  {len(sub_test)} rows")

        if len(sub_train) == 0 or n_real_train == 0 or n_fake_train == 0:
            print(f"  [skip] {method}: missing real or fake training rows")
            continue

        clf = train_classifier(x_train, y_train)
        m_dir = args.output_dir / method
        m_dir.mkdir(parents=True, exist_ok=True)
        save_classifier(clf, m_dir / "classifier.pkl")

        val_m = evaluate_classifier(clf, x_val, y_val, video_ids=vid_val)
        test_m = evaluate_classifier(clf, x_test, y_test, video_ids=vid_test)

        report = {
            "method": method,
            "n_train_frames": len(sub_train),
            "n_val_frames": len(sub_val),
            "n_test_frames": len(sub_test),
            "val": _summarize(val_m),
            "test": _summarize(test_m),
            "test_top_features": dict(
                sorted(test_m.feature_importances.items(), key=lambda kv: -kv[1])[:10],
            ),
        }
        (m_dir / "report.json").write_text(json.dumps(report, indent=2, default=str))

        print(f"  test frame AUROC:        {test_m.auroc:.4f}")
        print(f"  test video AUROC (mean): {test_m.video_auroc_mean:.4f}")
        print(f"  test video AUROC (max):  {test_m.video_auroc_max:.4f}")

        summary[method] = {
            "test_frame_auroc": test_m.auroc,
            "test_video_auroc_mean": test_m.video_auroc_mean,
            "test_video_auroc_max": test_m.video_auroc_max,
            "n_train_frames": len(sub_train),
            "n_test_videos": test_m.n_videos,
        }

    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=str))

    print("\n=== Per-method leaderboard ===")
    print(
        f"{'Method':<18s} {'Frame AUROC':>12s} {'Video AUROC (mean)':>22s} "
        f"{'Video AUROC (max)':>22s}",
    )
    for method, s in summary.items():
        print(
            f"{method:<18s} {s['test_frame_auroc']:>12.4f} "
            f"{s['test_video_auroc_mean']:>22.4f} {s['test_video_auroc_max']:>22.4f}",
        )
    print(f"\nSummary written to {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
