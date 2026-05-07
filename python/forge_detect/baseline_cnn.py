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


# ImageNet normalisation stats — torchvision's EfficientNet-B0 IMAGENET1K_V1
# weights were trained on inputs normalised by these. Skipping this step keeps
# val AUROC at chance because the pretrained features fire on the wrong
# distribution. Apply only to RGB channels; physics-map channels (W_cnn, z*, R)
# are already per-image normalised to [0, 1] and pass through untouched.
_IMAGENET_MEAN: tuple[float, float, float] = (0.485, 0.456, 0.406)
_IMAGENET_STD: tuple[float, float, float] = (0.229, 0.224, 0.225)


def _build_imagenet_normalizer() -> Any:
    """An nn.Module that subtracts ImageNet mean / divides by ImageNet std on
    the first three channels of its input, leaving any further channels
    unchanged. Used as the front layer of every classifier built here.
    """
    torch = _torch()
    from torch import nn

    class _ImageNetNormalize(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            mean = torch.tensor(_IMAGENET_MEAN).view(1, 3, 1, 1)
            std = torch.tensor(_IMAGENET_STD).view(1, 3, 1, 1)
            # persistent=False: stats are constants, not learned, and don't
            # belong in the saved state_dict — keeps best.pt clean.
            self.register_buffer("mean", mean, persistent=False)
            self.register_buffer("std", std, persistent=False)

        def forward(self, x: Any) -> Any:
            rgb = (x[:, :3] - self.mean) / self.std
            if x.shape[1] > 3:
                return torch.cat([rgb, x[:, 3:]], dim=1)
            return rgb

    return _ImageNetNormalize()


def build_baseline_classifier(*, pretrained: bool = True) -> Any:
    """Return EfficientNet-B0 with a 1-logit binary head.

    The output is a single logit per image (use BCEWithLogitsLoss for
    training, sigmoid for inference probability). The model includes an
    ImageNet-normalisation layer in front of the backbone so callers can
    feed raw [0, 1] floats from the dataset adapters directly.
    """
    _torch()  # ensure torch is importable before pulling in torchvision
    from torch import nn
    from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0

    weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
    backbone = efficientnet_b0(weights=weights)
    in_features = backbone.classifier[-1].in_features
    backbone.classifier = nn.Sequential(
        nn.Dropout(p=0.2, inplace=True),
        nn.Linear(in_features, 1),
    )
    return nn.Sequential(_build_imagenet_normalizer(), backbone)


def build_physics_classifier(*, in_channels: int = 6, pretrained: bool = True) -> Any:
    """EfficientNet-B0 with a 1-logit binary head and a stem-surgery first conv.

    The first conv is replaced with one that takes ``in_channels`` (default 6:
    RGB + W_cnn + z* + R). When ``pretrained=True``, the original RGB stem
    weights are copied into the first three input channels, and the remaining
    channels are initialised by the per-output-channel mean of those RGB
    weights. This keeps the network's epoch-0 behavior near-identical to the
    pretrained baseline while opening a path for the new channels to learn
    their own discriminative features.

    An ImageNet-normalisation layer is prepended to the model so callers can
    feed raw [0, 1] floats from the dataset adapters; only channels 0-2 (RGB)
    are normalised, and channels 3+ pass through unchanged because the
    physics maps are already per-image normalised at load time.
    """
    torch = _torch()
    from torch import nn
    from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0

    weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
    backbone = efficientnet_b0(weights=weights)
    in_features = backbone.classifier[-1].in_features
    backbone.classifier = nn.Sequential(
        nn.Dropout(p=0.2, inplace=True),
        nn.Linear(in_features, 1),
    )

    # Stem surgery. torchvision's EfficientNet-B0 stem is
    #   features[0] = ConvNormActivation(3 -> 32, 3x3, stride=2, bias=False).
    # Replace the inner Conv2d, preserving the same kernel/stride/padding/bias
    # spec so the rest of the block (BN, SiLU) keeps its expected input shape.
    old_conv = backbone.features[0][0]
    if not isinstance(old_conv, nn.Conv2d):
        msg = f"unexpected stem layer type {type(old_conv).__name__}; torchvision API changed?"
        raise RuntimeError(msg)
    # nn.Conv2d's typed signature wants tuple[int, int] (or int) for these.
    # The runtime values from `old_conv` are already 2-tuples on Conv2d, but
    # the static type is tuple[int, ...] — cast to satisfy mypy.
    from typing import cast

    new_conv = nn.Conv2d(
        in_channels=in_channels,
        out_channels=old_conv.out_channels,
        kernel_size=cast("tuple[int, int]", old_conv.kernel_size),
        stride=cast("tuple[int, int]", old_conv.stride),
        padding=cast("tuple[int, int]", old_conv.padding),
        bias=old_conv.bias is not None,
    )
    with torch.no_grad():
        old_w = old_conv.weight.data  # (out, 3, kH, kW)
        new_w = new_conv.weight.data  # (out, in_channels, kH, kW)
        if pretrained and in_channels >= 3:
            new_w[:, :3] = old_w
            if in_channels > 3:
                # Per-output-channel mean of the RGB kernels gives the new
                # channels a neutral, ImageNet-consistent starting point.
                rgb_mean = old_w.mean(dim=1, keepdim=True)  # (out, 1, kH, kW)
                new_w[:, 3:] = rgb_mean.expand(-1, in_channels - 3, -1, -1) / max(
                    1, in_channels - 3,
                )
        if old_conv.bias is not None and new_conv.bias is not None:
            new_conv.bias.data.copy_(old_conv.bias.data)
    backbone.features[0][0] = new_conv
    return nn.Sequential(_build_imagenet_normalizer(), backbone)


@dataclass
class BaselineConfig:
    """Hyperparameters for :func:`train_baseline_cnn`."""

    epochs: int = 30
    batch_size: int = 32
    learning_rate: float = 2.0e-4
    """The published Rössler FF++ EfficientNet/Xception fine-tuning recipe.

    History: this default has been wrong twice. 1e-3 was too high and
    paired with AMP caused cuDNN execution errors. 5e-4 (without AMP)
    overshot from a fresh Linear head — the per-step inline profile
    showed |grad|=1.000 for the first 400 steps (clipped) and logit std
    blowing from 0.046 to 0.821 at step 100, then collapsing to ~0.03
    by step 1500 as BCE pulled the model into the trivial z=0 minimum.
    Train AUROC plateaued at 0.503 = chance under WRS-balanced batches.

    2e-4 is the FF++-validated rate. Half the per-step magnitude of 5e-4
    keeps the optimiser inside the linear head's stable regime during
    the first hundred steps before it starts producing meaningful
    gradients."""
    weight_decay: float = 1.0e-4
    device: str = "auto"
    mixed_precision: bool = False
    """Default disabled. AMP with the default fp16 precision and tight
    gradients of fine-tuning was associated with cuDNN execution errors
    after a few epochs and never-decreasing loss before that. Re-enable
    explicitly once a working baseline is in hand."""
    num_workers: int = 4
    val_every: int = 1
    checkpoint_dir: Path = field(default_factory=lambda: Path("runs_baseline"))
    grad_clip: float = 1.0
    balance_classes: bool = True
    """Re-enable WeightedRandomSampler so train batches are ~50/50 real:fake.

    Earlier I disabled this on a "BN running stats drift" theory, after
    seeing anti-correlated val AUROC. That theory was wrong: the real
    cause was identity leakage from random splits (real "X" in train,
    fake "X_Y" in val → model memorises identity X then sees its fake in
    val). The fix is --use-ff-splits, which gives identity-disjoint
    partitions by construction.

    With official splits, balance_classes=False makes the natural ~20/80
    FF++ imbalance dominate the loss: BCE-on-logits collapses to the
    majority-class prediction (train_acc=val_acc=0.8, AUROC=0.5) before
    extracting any class signal. WRS oversamples reals so the gradient
    sees both classes equally and actual learning happens."""
    augment_hflip: bool = True
    """Random horizontal flip with p=0.5 during training. Pure geometric
    op, safe for any channel count (RGB or RGB + physics maps); doubles
    effective dataset diversity at zero compute cost."""
    log_every: int = 100
    """Print step-level metrics (loss, logit std, grad norm) every N
    training batches. 0 disables. Helps diagnose "epoch summary says
    AUROC=0.5 but is the model learning within the epoch?" — without
    this you only see the epoch average and can't tell whether loss is
    actually trending down batch-to-batch."""


def _collate(batch: list[Any]) -> tuple[Any, Any]:
    """Stack ``(image, label[, mask])`` records, dropping any optional fields.

    The 6-channel physics dataset still returns ``(image, label, mask?)`` —
    mask is unused by the binary classifier and discarded here.
    """
    torch = _torch()
    images = torch.stack([b[0] for b in batch], dim=0)
    labels = torch.tensor([b[1] for b in batch], dtype=torch.float32)
    return images, labels


def _maybe_hflip(images: Any, *, p: float = 0.5) -> Any:
    """Random horizontal flip applied independently to each image in a batch.

    Operates on ``(B, C, H, W)`` tensors with any channel count — flipping
    the W axis preserves spatial channel correspondence, which matters for
    the 6-channel physics tensor where channels 3-5 (W_cnn / z* / R) must
    remain spatially aligned with channels 0-2 (RGB) post-flip.
    """
    torch = _torch()
    flip_mask = torch.rand(images.shape[0], device=images.device) < p
    if not flip_mask.any():
        return images
    flipped = images.flip(-1)
    return torch.where(
        flip_mask.view(-1, 1, 1, 1),
        flipped,
        images,
    )


def _epoch_pass(  # noqa: PLR0912
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
    profiled_first_batch = False
    step_idx = 0
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        if train and config.augment_hflip:
            images = _maybe_hflip(images)
        ctx = torch.amp.autocast(autocast_device) if (scaler is not None and train) else _NullCtx()
        with ctx:
            logits = model(images).squeeze(-1)  # (B,)
            loss = bce(logits, labels)
        if train and not profiled_first_batch:
            # One-shot batch-0 profile — prints input / logit / grad statistics
            # so we can verify the training loop is doing what the diagnostic
            # script does. Cheap (one batch per training pass) and immediately
            # reveals "logits all near zero" / "gradients vanishing" failure
            # modes that summary metrics hide behind average-over-epoch.
            with torch.no_grad():
                lab_mean = float(labels.float().mean().item())
                img_mean = float(images.mean().item())
                img_std = float(images.std().item())
                logit_mean = float(logits.mean().item())
                logit_std = float(logits.std().item())
                logit_min = float(logits.min().item())
                logit_max = float(logits.max().item())
            print(
                f"  [profile] batch0: img(mean={img_mean:.3f} std={img_std:.3f}) "
                f"label_mean={lab_mean:.3f} (=fake fraction; ~0.5 means WRS works) "
                f"logits(mean={logit_mean:.3f} std={logit_std:.3f} "
                f"range=[{logit_min:.3f},{logit_max:.3f}])",
            )
            profiled_first_batch = True
        # Hard fail on non-finite loss — silent NaNs corrupt optimizer state and
        # produce later cuDNN execution errors that are much harder to diagnose.
        if train and not torch.isfinite(loss):
            msg = (
                f"loss became non-finite ({loss.item()}) — likely AMP gradient "
                "underflow or LR too high. Try mixed_precision=False or lower lr."
            )
            raise RuntimeError(msg)
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
                if profiled_first_batch and n_batches == 0:
                    # Profile gradient health right after backward — counts
                    # zero-gradient tensors and reports total norm. Matches the
                    # diagnostic-script output so the two are directly
                    # comparable.
                    grad_sq = 0.0
                    n_grad_params = 0
                    n_zero_grad = 0
                    for p in model.parameters():
                        if p.grad is None:
                            continue
                        n_grad_params += 1
                        g = p.grad.detach()
                        grad_sq += float((g * g).sum().item())
                        if (g.abs() < 1.0e-12).all():
                            n_zero_grad += 1
                    print(
                        f"  [profile] batch0 post-backward: |grad|={grad_sq**0.5:.4f} "
                        f"zero-grad-tensors={n_zero_grad}/{n_grad_params} "
                        f"loss={loss.item():.4f}",
                    )
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
        if (
            train
            and config.log_every > 0
            and step_idx > 0
            and step_idx % config.log_every == 0
        ):
            with torch.no_grad():
                logit_std = float(logits.std().item())
                logit_mean = float(logits.mean().item())
            grad_norm = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    grad_norm += float((p.grad * p.grad).sum().item())
            grad_norm = grad_norm**0.5
            print(
                f"    step {step_idx:4d}: loss={loss.item():.4f} "
                f"logit(mean={logit_mean:+.3f} std={logit_std:.3f}) "
                f"|grad|={grad_norm:.3f} "
                f"running_acc={n_correct/max(1,n_total):.3f}",
            )
        step_idx += 1
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
    log_tag: str = "baseline-cnn",
) -> tuple[int, float, list[dict[str, float]]]:
    torch = _torch()
    if resume_dir is None:
        return 0, float("inf"), []
    ckpt_path = resume_dir / "checkpoint.pt"
    if not ckpt_path.exists():
        print(f"[{log_tag}] no checkpoint at {ckpt_path} — starting fresh")
        return 0, float("inf"), []
    print(f"[{log_tag}] resuming from {ckpt_path}")
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
    model_factory: Any | None = None,
    log_tag: str = "baseline-cnn",
) -> dict[str, list[dict[str, float]] | str]:
    """Train a binary CNN classifier with crash-resumable checkpoints.

    See :func:`forge_detect.train.train_cnn` for the resume semantics —
    this function uses the same conventions.

    Args:
        model_factory: Callable ``(*, pretrained: bool) -> nn.Module`` that
            returns the model to train. Defaults to
            :func:`build_baseline_classifier` (3-channel RGB EfficientNet-B0).
            Pass :func:`build_physics_classifier` (or any compatible factory)
            for the 6-channel physics-tensor variant — the loss / sampler /
            optimizer / eval logic is identical, only the model + dataset
            channel count differ.
        log_tag: Prefix for every line printed by this function. Helpful when
            running baseline + physics runs side by side.
    """
    torch = _torch()
    from torch.utils.data import DataLoader

    from forge_detect.train import _make_balanced_sampler

    config = config or BaselineConfig()
    device = _select_device(config.device)
    factory = model_factory if model_factory is not None else build_baseline_classifier
    print(f"[{log_tag}] training on device={device} for {config.epochs} epochs")

    model = factory(
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
        log_tag=log_tag,
    )

    train_sampler = _make_balanced_sampler(train_dataset) if config.balance_classes else None
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
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
        print(f"[{log_tag}] resumed run already at epoch {start_epoch} >= {config.epochs}")
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
        print(f"[{log_tag}] epoch {epoch}: " + " ".join(f"{k}={v:.4f}" for k, v in log.items()))
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
    """Evaluate a trained binary CNN on a labeled dataset.

    Returns frame-level AUROC + accuracy; when the dataset's records expose
    a ``video_id`` field (FaceForensicsAdapter, CelebDFAdapter,
    ImageFolderDataset all do), also returns video-level AUROC under both
    mean-pooling and max-pooling. Mean-pool is the cross-dataset benchmark
    convention; max-pool is reported alongside as a robustness check.
    """
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
    out: dict[str, float] = {
        "accuracy": float(((proba_arr >= 0.5).astype(np.int64) == labels_arr).mean()),
        "n_real": float((labels_arr == 0).sum()),
        "n_fake": float((labels_arr == 1).sum()),
    }
    if len(np.unique(labels_arr)) > 1:
        from sklearn.metrics import roc_auc_score

        out["auroc"] = float(roc_auc_score(labels_arr, proba_arr))
    else:
        out["auroc"] = float("nan")

    video_ids = _dataset_video_ids(dataset)
    if video_ids is not None and len(video_ids) == len(labels_arr):
        from forge_detect.classifier import _video_level_metrics

        v = _video_level_metrics(proba_arr, labels_arr, np.asarray(video_ids))
        out["video_auroc_mean"] = v["auroc_mean"]
        out["video_auroc_max"] = v["auroc_max"]
        out["video_accuracy_mean"] = v["accuracy_mean"]
        out["n_videos"] = float(v["n_videos"])
        out["n_video_real"] = float(v["n_video_real"])
        out["n_video_fake"] = float(v["n_video_fake"])
    return out


def _dataset_video_ids(dataset: Any) -> list[str] | None:
    """Pull the per-record video_id list without loading any images.

    Returns ``None`` when the dataset does not expose ``_records[i].video_id``;
    in that case the caller skips video-level pooling. Recurses through
    ``torch.utils.data.Subset.dataset`` + ``indices`` for sub-sampled splits.
    """
    records = getattr(dataset, "_records", None)
    if records is not None:
        vids = [getattr(r, "video_id", None) for r in records]
        if all(v is not None for v in vids):
            return [str(v) for v in vids]
        return None
    if hasattr(dataset, "indices") and hasattr(dataset, "dataset"):
        sub = _dataset_video_ids(dataset.dataset)
        if sub is None:
            return None
        return [sub[i] for i in dataset.indices]
    return None


__all__ = [
    "BaselineConfig",
    "build_baseline_classifier",
    "build_physics_classifier",
    "evaluate_baseline_cnn",
    "train_baseline_cnn",
]
