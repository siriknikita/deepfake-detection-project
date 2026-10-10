"""Validation logit analysis for a trained multichannel classifier.

Answers two questions the aggregate curves cannot: (1) is a rising validation
loss a *calibration* problem (same ranking, more confident mistakes) or a
*ranking* problem; (2) does the aggregate AUROC hide divergent behaviour per
manipulation method.  Writes ``<out>.json`` (summary) and ``<out>.csv`` (per
frame: path, method, label, logit).

Usage::

    python scripts/analyze_val_logits.py --data-root <FF++ root> \\
        --weights runs_mac/<run>/last.pt --channels rgb,physics,frequency:compute \\
        --backbone mobilenet_v3_large --split val --device cpu --output <prefix>
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--weights", type=Path, required=True)
    ap.add_argument("--channels", default="rgb,physics,frequency:compute")
    ap.add_argument("--backbone", default="efficientnet_b0")
    ap.add_argument("--split", choices=("val", "test"), default="val")
    ap.add_argument("--compression", default="c23")
    ap.add_argument("--frames-per-video", type=int, default=10)
    ap.add_argument("--image-size", type=int, default=256)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--num-workers", type=int, default=1)
    ap.add_argument(
        "--confident", type=float, default=3.0, help="|logit| threshold for 'confident'"
    )
    ap.add_argument("--output", type=Path, required=True, help="output prefix (no extension)")
    args = ap.parse_args()

    import torch
    from sklearn.metrics import log_loss, roc_auc_score
    from torch.utils.data import DataLoader

    from forge_detect.baseline_cnn import _collate, build_physics_classifier
    from forge_detect.cnn import load_weights
    from forge_detect.datasets import FaceForensicsAdapter, parse_channel_spec, total_channels

    sources = parse_channel_spec(args.channels)
    ds = FaceForensicsAdapter(
        args.data_root,
        compression=args.compression,
        max_frames_per_video=args.frames_per_video,
        target_size=(args.image_size, args.image_size),
        ff_split=args.split,
        frames_subdir="frames_faces",
        channel_sources=sources,
    )
    model = build_physics_classifier(
        in_channels=total_channels(sources), pretrained=False, backbone=args.backbone
    )
    load_weights(model, args.weights)
    model = model.to(args.device).eval()
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=_collate,
    )
    logits: list[float] = []
    labels: list[int] = []
    with torch.no_grad():
        for x, y in loader:
            logits.extend(model(x.to(args.device)).squeeze(-1).float().cpu().tolist())
            labels.extend(int(v) for v in y.tolist())
    lg = np.asarray(logits)
    lb = np.asarray(labels)
    paths = [str(r.image_path) for r in ds._records]
    methods = []
    for p in paths:
        parts = Path(p).parts
        methods.append(
            parts[parts.index("manipulated_sequences") + 1]
            if "manipulated_sequences" in parts
            else "real"
        )
    methods_arr = np.asarray(methods)
    prob = 1.0 / (1.0 + np.exp(-lg))
    pred = (lg > 0).astype(int)
    wrong = pred != lb
    confident = np.abs(lg) > args.confident

    def auc(mask: np.ndarray) -> float | None:
        if mask.sum() < 2 or len(np.unique(lb[mask])) < 2:
            return None
        return float(roc_auc_score(lb[mask], lg[mask]))

    summary = {
        "weights": str(args.weights),
        "split": args.split,
        "n_frames": int(len(lg)),
        "auroc": auc(np.ones_like(lb, bool)),
        "log_loss": float(log_loss(lb, np.clip(prob, 1e-7, 1 - 1e-7))),
        "accuracy_at_0": float((pred == lb).mean()),
        "logit": {
            "mean": float(lg.mean()),
            "std": float(lg.std()),
            "abs_p50": float(np.median(np.abs(lg))),
            "abs_p90": float(np.percentile(np.abs(lg), 90)),
            "abs_max": float(np.abs(lg).max()),
        },
        "confident_fraction": float(confident.mean()),
        "confidently_wrong_fraction": float((confident & wrong).mean()),
        "wrong_fraction": float(wrong.mean()),
        "share_of_log_loss_from_confidently_wrong": float(
            (-np.log(np.clip(np.where(lb == 1, prob, 1 - prob), 1e-7, 1))[confident & wrong]).sum()
            / max((-np.log(np.clip(np.where(lb == 1, prob, 1 - prob), 1e-7, 1))).sum(), 1e-12)
        ),
        "per_label": {
            str(c): {
                "n": int((lb == c).sum()),
                "logit_mean": float(lg[lb == c].mean()),
                "logit_std": float(lg[lb == c].std()),
                "accuracy": float((pred[lb == c] == c).mean()),
            }
            for c in (0, 1)
        },
        # per method: AUROC of that method's fakes vs ALL reals (the standard per-method protocol)
        "per_method": {},
    }
    real_mask = methods_arr == "real"
    for m in sorted(set(methods) - {"real"}):
        mask = real_mask | (methods_arr == m)
        summary["per_method"][m] = {
            "n_fake_frames": int((methods_arr == m).sum()),
            "auroc_vs_reals": auc(mask),
            "fake_accuracy": float((pred[methods_arr == m] == 1).mean()),
            "fake_logit_mean": float(lg[methods_arr == m].mean()),
            "confidently_wrong_fraction": float((confident & wrong)[methods_arr == m].mean()),
        }
    summary["per_method"]["real"] = {
        "n_frames": int(real_mask.sum()),
        "real_accuracy": float((pred[real_mask] == 0).mean()),
        "real_logit_mean": float(lg[real_mask].mean()),
        "confidently_wrong_fraction": float((confident & wrong)[real_mask].mean()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.with_suffix(".json").write_text(json.dumps(summary, indent=2))
    with args.output.with_suffix(".csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["path", "method", "label", "logit"])
        for p, m, y, v in zip(paths, methods, labels, logits, strict=True):
            w.writerow([p, m, y, f"{v:.4f}"])
    print(json.dumps({k: v for k, v in summary.items() if k != "per_method"}, indent=2))
    print(json.dumps(summary["per_method"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
