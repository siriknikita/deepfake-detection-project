"""CNN training loop for the trust-map predictor.

Trains :func:`forge_detect.cnn.build_chromatic_efficientnet` on a labeled
real / fake dataset. Two supervision regimes are supported:

- **Pixel-level** (preferred): the dataset returns a ground-truth fake
  mask alongside each image. The CNN is supervised by per-pixel BCE
  between ``W_cnn`` and ``1 − mask`` (mask = 1 means fake, target trust
  = 0). FaceForensics++ provides these masks for the four manipulation
  families.
- **Image-level** (fallback): only the image-level real / fake label is
  available. The supervision target becomes the *image mean* of
  ``W_cnn``: real images should average to 1, fake images to 0.

Mixed precision is used when CUDA is available. Checkpoints land in
``runs/<timestamp>/`` with ``best.pt`` (lowest validation loss) and
``last.pt`` (final epoch).
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    pass


def _torch() -> Any:
    try:
        import torch
    except ImportError as e:
        msg = "training requires PyTorch — `pip install torch torchvision`"
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


@dataclass
class TrainingConfig:
    """Hyperparameters for :func:`train_cnn`."""

    epochs: int = 30
    batch_size: int = 16
    learning_rate: float = 1.0e-3
    weight_decay: float = 1.0e-4
    device: str = "auto"
    mixed_precision: bool = True
    num_workers: int = 4
    val_every: int = 1
    checkpoint_dir: Path = field(default_factory=lambda: Path("runs"))
    grad_clip: float = 1.0
    log_every: int = 50


def _ensure_run_dir(base: Path) -> Path:
    stamp = time.strftime("%Y%m%d-%H%M%S")
    run_dir = base / stamp
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _supervision_loss(
    w_cnn: Any,  # (B, H, W)
    labels: Any,  # (B,) 0 = real, 1 = fake
    masks: Any | None,  # (B, H, W) or None
) -> Any:
    """Combined pixel-level + image-level BCE."""
    torch = _torch()
    eps = 1.0e-6
    w_cnn = w_cnn.clamp(eps, 1.0 - eps)
    losses: list[Any] = []
    if masks is not None:
        # Pixel-level: target = 1 - mask  (mask=1 means fake → target=0).
        target = 1.0 - masks
        pixel_loss = torch.nn.functional.binary_cross_entropy(w_cnn, target)
        losses.append(pixel_loss)
    # Image-level: mean(W_cnn) -> 1 for real, 0 for fake.
    image_pred = w_cnn.mean(dim=(1, 2))
    image_target = (1.0 - labels.float()).clamp(eps, 1.0 - eps)
    image_loss = torch.nn.functional.binary_cross_entropy(image_pred, image_target)
    losses.append(image_loss)
    return sum(losses) / len(losses)


def _collate(batch: list[Any]) -> tuple[Any, Any, Any | None]:
    """Stack `(image, label[, mask])` records into batched tensors."""
    torch = _torch()
    images = torch.stack([b[0] for b in batch], dim=0)
    labels = torch.tensor([b[1] for b in batch], dtype=torch.long)
    masks: Any | None = None
    # FaceForensicsAdapter returns (image, label, mask?); ImageFolderDataset returns (image, label).
    if (
        len(batch[0]) >= 3
        and batch[0][2] is not None
        and all(b[2] is not None for b in batch)
    ):
        masks = torch.stack([b[2] for b in batch], dim=0)
    return images, labels, masks


def _epoch_pass(
    model: Any,
    loader: Any,
    optimizer: Any | None,
    scaler: Any | None,
    device: str,
    config: TrainingConfig,
    *,
    train: bool,
) -> dict[str, float]:
    torch = _torch()
    model.train(train)
    total_loss = 0.0
    n_batches = 0
    n_correct = 0
    n_total = 0
    autocast_device = "cuda" if device == "cuda" else "cpu"
    for step, batch in enumerate(loader):
        images, labels, masks = batch
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        if masks is not None:
            masks = masks.to(device, non_blocking=True)
        ctx = torch.amp.autocast(autocast_device) if (scaler is not None and train) else _NullCtx()
        with ctx:
            w_cnn = model(images)
            loss = _supervision_loss(w_cnn, labels, masks)

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
        # Accuracy via image-level mean threshold at 0.5.
        with torch.no_grad():
            pred = (w_cnn.mean(dim=(1, 2)) < 0.5).long()  # 1 if fake (low trust)
            n_correct += int((pred == labels).sum().item())
            n_total += labels.numel()
        if train and config.log_every > 0 and step % config.log_every == 0:
            print(f"    step={step:4d} loss={loss.item():.4f}")
    return {
        "loss": total_loss / max(1, n_batches),
        "accuracy": n_correct / max(1, n_total),
    }


class _NullCtx:
    """Context manager that does nothing — stand-in for autocast disabled."""

    def __enter__(self) -> _NullCtx:
        return self

    def __exit__(self, *_args: object) -> None:
        return None


def train_cnn(
    train_dataset: Any,
    val_dataset: Any | None,
    config: TrainingConfig | None = None,
    *,
    pretrained: bool = True,
) -> dict[str, list[dict[str, float]]]:
    """Train the trust-map CNN.

    Args:
        train_dataset: A torch Dataset returning ``(image, label[, mask])``.
        val_dataset: Optional validation dataset of the same shape.
        config: Training hyperparameters.
        pretrained: Initialize the EfficientNet-B0 backbone with ImageNet
            weights (recommended unless you've already trained from scratch).

    Returns:
        ``{"history": [{"epoch": int, "train_loss": float, ...}, ...]}``.
        Best checkpoint is also written to ``<checkpoint_dir>/best.pt``.
    """
    torch = _torch()
    from torch.utils.data import DataLoader

    from forge_detect.cnn import build_chromatic_efficientnet, save_weights

    config = config or TrainingConfig()
    device = _select_device(config.device)
    print(f"training on device={device} for {config.epochs} epochs")

    model = build_chromatic_efficientnet(pretrained=pretrained).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)
    scaler = torch.amp.GradScaler() if (config.mixed_precision and device == "cuda") else None

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

    run_dir = _ensure_run_dir(config.checkpoint_dir)
    print(f"checkpoints -> {run_dir}")
    (run_dir / "config.json").write_text(json.dumps(asdict(config), default=str, indent=2))

    history: list[dict[str, float]] = []
    best_val_loss = float("inf")
    for epoch in range(config.epochs):
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
                save_weights(model, run_dir / "best.pt")
        save_weights(model, run_dir / "last.pt")
        scheduler.step()
        log["epoch_seconds"] = time.time() - t0
        history.append(log)
        print(f"epoch {epoch}: " + " ".join(f"{k}={v:.4f}" for k, v in log.items()))
        (run_dir / "history.json").write_text(json.dumps(history, indent=2))

    return {"history": history, "run_dir": str(run_dir)}  # type: ignore[dict-item]


__all__ = ["TrainingConfig", "train_cnn"]
