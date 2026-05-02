"""Pure-CNN deepfake-classification baseline.

Trains EfficientNet-B0 end-to-end on (image, label) without the
Hyperplane-Forge math pipeline. This is the *baseline* the diploma's
empirical chapter compares against — if a learned classifier alone
matches or beats the full physics pipeline, the framework offers no
value over the standard approach and the project should pivot.

The architecture intentionally matches the public-leaderboard recipe
for FaceForensics++: torchvision EfficientNet-B0 with the
``classifier`` replaced by a 2-class linear head. Same optimizer
(AdamW), same scheduler (cosine LR), same loss (BCE on logits) as the
trust-map trainer in :mod:`forge_detect.train`, so any AUROC
difference between the two reflects the *information added by the
math pipeline*, not training-procedure noise.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


def _torch() -> Any:
    try:
        import torch
    except ImportError as e:
        msg = "baseline CNN requires PyTorch — `pip install torch torchvision`"
        raise ImportError(msg) from e
    return torch


def _select_device(prefer: str = "auto") -> str:
    torch = _torch()
    if prefer == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    return prefer


def build_baseline_classifier(*, pretrained: bool = True) -> Any:
    """Return EfficientNet-B0 with a 1-logit binary head.

    The output is a single logit per image (use BCEWithLogitsLoss for
    training, sigmoid for inference probability).
    """
    _torch()  # ensure torch is importable before pulling in torchvision
    from torch import nn
    from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0

    weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
    model = efficientnet_b0(weights=weights)
    in_features = model.classifier[-1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2, inplace=True),
        nn.Linear(in_features, 1),
    )
    return model


@dataclass
class BaselineConfig:
    """Hyperparameters for :func:`train_baseline_cnn`."""

    epochs: int = 30
    batch_size: int = 32
    learning_rate: float = 1.0e-3
    weight_decay: float = 1.0e-4
    device: str = "auto"
    mixed_precision: bool = True
    num_workers: int = 4
    val_every: int = 1
    checkpoint_dir: Path = field(default_factory=lambda: Path("runs_baseline"))
    grad_clip: float = 1.0


def _collate(batch: list[Any]) -> tuple[Any, Any]:
    torch = _torch()
    images = torch.stack([b[0] for b in batch], dim=0)
    labels = torch.tensor([b[1] for b in batch], dtype=torch.float32)
    return images, labels


def _epoch_pass(
    model: Any,
    loader: Any,
    optimizer: Any | None,
    scaler: Any | None,
    device: str,
    config: BaselineConfig,
    *,
    train: bool,
) -> dict[str, float]:
    torch = _torch()
    model.train(train)
    bce = torch.nn.BCEWithLogitsLoss()

    total_loss = 0.0
    n_batches = 0
    n_correct = 0
    n_total = 0
    proba_all: list[float] = []
    labels_all: list[float] = []

    autocast_device = "cuda" if device == "cuda" else "cpu"
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        ctx = torch.amp.autocast(autocast_device) if (scaler is not None and train) else _NullCtx()
        with ctx:
            logits = model(images).squeeze(-1)  # (B,)
            loss = bce(logits, labels)
        if train:
            assert optimizer is not None
            optimizer.zero_grad(set_to_none=True)
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
                optimizer.step()
        total_loss += float(loss.item())
        n_batches += 1
        with torch.no_grad():
            proba = torch.sigmoid(logits)
            pred = (proba >= 0.5).float()
            n_correct += int((pred == labels).sum().item())
            n_total += labels.numel()
            proba_all.extend(proba.detach().cpu().tolist())
            labels_all.extend(labels.detach().cpu().tolist())
    out = {
        "loss": total_loss / max(1, n_batches),
        "accuracy": n_correct / max(1, n_total),
    }
    # Approximate AUROC if both classes are present.
    proba_arr = np.asarray(proba_all)
    labels_arr = np.asarray(labels_all)
    if len(np.unique(labels_arr)) > 1:
        from sklearn.metrics import roc_auc_score

        out["auroc"] = float(roc_auc_score(labels_arr, proba_arr))
    return out


class _NullCtx:
    def __enter__(self) -> _NullCtx:
        return self

    def __exit__(self, *_args: object) -> None:
        return None


def _save_full_checkpoint(
    path: Path,
    *,
    model: Any,
    optimizer: Any,
    scheduler: Any,
    scaler: Any | None,
    epoch: int,
    best_val_loss: float,
    history: list[dict[str, float]],
) -> None:
    """Resumable checkpoint with optimizer + scheduler + scaler state."""
    torch = _torch()
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict() if scaler is not None else None,
        "epoch": epoch,
        "best_val_loss": best_val_loss,
        "history": history,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def _maybe_resume(
    resume_dir: Path | None,
    *,
    model: Any,
    optimizer: Any,
    scheduler: Any,
    scaler: Any | None,
) -> tuple[int, float, list[dict[str, float]]]:
    torch = _torch()
    if resume_dir is None:
        return 0, float("inf"), []
    ckpt_path = resume_dir / "checkpoint.pt"
    if not ckpt_path.exists():
        print(f"[baseline-cnn] no checkpoint at {ckpt_path} — starting fresh")
        return 0, float("inf"), []
    print(f"[baseline-cnn] resuming from {ckpt_path}")
    payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(payload["model"])
    optimizer.load_state_dict(payload["optimizer"])
    scheduler.load_state_dict(payload["scheduler"])
    if scaler is not None and payload.get("scaler") is not None:
        scaler.load_state_dict(payload["scaler"])
    return (
        int(payload["epoch"]) + 1,
        float(payload["best_val_loss"]),
        list(payload.get("history", [])),
    )


def train_baseline_cnn(
    train_dataset: Any,
    val_dataset: Any | None,
    config: BaselineConfig | None = None,
    *,
    pretrained: bool = True,
    resume_dir: Path | None = None,
) -> dict[str, list[dict[str, float]] | str]:
    """Train the pure-CNN baseline with crash-resumable checkpoints.

    See :func:`forge_detect.train.train_cnn` for the resume semantics —
    this function uses the same conventions.
    """
    torch = _torch()
    from torch.utils.data import DataLoader

    config = config or BaselineConfig()
    device = _select_device(config.device)
    print(f"[baseline-cnn] training on device={device} for {config.epochs} epochs")

    model = build_baseline_classifier(
        pretrained=(pretrained and resume_dir is None),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)
    scaler = torch.amp.GradScaler() if (config.mixed_precision and device == "cuda") else None

    start_epoch, best_val_loss, history = _maybe_resume(
        resume_dir,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        collate_fn=_collate,
        pin_memory=(device == "cuda"),
    )
    val_loader = (
        DataLoader(
            val_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            collate_fn=_collate,
        )
        if val_dataset is not None
        else None
    )

    if resume_dir is not None:
        run_dir = resume_dir
    else:
        stamp = time.strftime("%Y%m%d-%H%M%S")
        run_dir = config.checkpoint_dir / stamp
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(asdict(config), default=str, indent=2))

    if start_epoch >= config.epochs:
        print(f"[baseline-cnn] resumed run already at epoch {start_epoch} >= {config.epochs}")
        return {"history": history, "run_dir": str(run_dir)}

    for epoch in range(start_epoch, config.epochs):
        t0 = time.time()
        train_metrics = _epoch_pass(
            model,
            train_loader,
            optimizer,
            scaler,
            device,
            config,
            train=True,
        )
        log: dict[str, float] = {
            "epoch": float(epoch),
            **{f"train_{k}": v for k, v in train_metrics.items()},
        }
        if val_loader is not None and (epoch % config.val_every == 0):
            val_metrics = _epoch_pass(
                model,
                val_loader,
                None,
                None,
                device,
                config,
                train=False,
            )
            log.update({f"val_{k}": v for k, v in val_metrics.items()})
            if val_metrics["loss"] < best_val_loss:
                best_val_loss = val_metrics["loss"]
                torch.save(model.state_dict(), run_dir / "best.pt")
        torch.save(model.state_dict(), run_dir / "last.pt")
        scheduler.step()
        log["epoch_seconds"] = time.time() - t0
        history.append(log)
        print(f"[baseline-cnn] epoch {epoch}: " + " ".join(f"{k}={v:.4f}" for k, v in log.items()))
        (run_dir / "history.json").write_text(json.dumps(history, indent=2))
        _save_full_checkpoint(
            run_dir / "checkpoint.pt",
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            epoch=epoch,
            best_val_loss=best_val_loss,
            history=history,
        )

    return {"history": history, "run_dir": str(run_dir)}


def evaluate_baseline_cnn(
    model: Any,
    dataset: Any,
    *,
    device: str = "cpu",
    batch_size: int = 32,
    num_workers: int = 4,
) -> dict[str, float]:
    """Evaluate a trained baseline CNN on a (image, label) dataset → AUROC + accuracy."""
    torch = _torch()
    from torch.utils.data import DataLoader

    model = model.to(device).eval()
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=_collate,
    )
    proba_all: list[float] = []
    labels_all: list[float] = []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            logits = model(images).squeeze(-1)
            proba = torch.sigmoid(logits)
            proba_all.extend(proba.detach().cpu().tolist())
            labels_all.extend(labels.detach().cpu().tolist())
    proba_arr = np.asarray(proba_all)
    labels_arr = np.asarray(labels_all, dtype=np.int64)
    out = {
        "accuracy": float(((proba_arr >= 0.5).astype(np.int64) == labels_arr).mean()),
        "n_real": int((labels_arr == 0).sum()),
        "n_fake": int((labels_arr == 1).sum()),
    }
    if len(np.unique(labels_arr)) > 1:
        from sklearn.metrics import roc_auc_score

        out["auroc"] = float(roc_auc_score(labels_arr, proba_arr))
    else:
        out["auroc"] = float("nan")
    return out


__all__ = [
    "BaselineConfig",
    "build_baseline_classifier",
    "evaluate_baseline_cnn",
    "train_baseline_cnn",
]
