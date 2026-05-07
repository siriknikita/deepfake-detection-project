"""Quick infrastructure check: can the baseline classifier overfit a tiny batch?

When ``train_baseline_cnn`` is not learning (loss stuck at log(2) ≈ 0.693, val
AUROC at chance), we want to disambiguate between:

  - tuning issue: LR too high / AMP underflow / scheduler eating the gradient.
    The model + dataset + loss path is fine, we just need different
    hyperparameters.

  - structural bug: the model isn't actually receiving differentiated
    inputs, the labels are wrong, or gradients aren't reaching some layer.
    Tuning won't help; we have to find and fix the bug.

This script removes every degree of freedom by overfitting a *fixed* batch of 8
frames (4 real, 4 fake) for 30 steps. If a 6M-parameter pretrained
EfficientNet-B0 cannot drive loss to ~0 on 8 images, something structural is
broken. If it can, the full-training stall is a tuning problem and the next
step is to lower LR / disable AMP.

Usage:

    uv run python scripts/diagnose_baseline.py --data-root ~/data/FaceForensics++
    uv run python scripts/diagnose_baseline.py --data-root ... --no-amp --lr 1e-4
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any


def _torch() -> Any:
    import torch

    return torch


def _build_batch(
    args: argparse.Namespace,
) -> tuple[Any, Any]:
    from forge_detect.datasets import FaceForensicsAdapter

    torch = _torch()
    ds = FaceForensicsAdapter(
        root=args.data_root,
        compression="c23",
        target_size=(args.image_size, args.image_size),
        max_frames_per_video=2,
        ff_split="train",
    )

    half = args.batch_size // 2
    real_idx = [i for i, r in enumerate(ds._records) if r.label == 0][:half]
    fake_idx = [i for i, r in enumerate(ds._records) if r.label == 1][:half]
    if not real_idx or not fake_idx:
        msg = (
            f"need at least {half} real and {half} fake frames in the train "
            f"split; found {len(real_idx)} real, {len(fake_idx)} fake. "
            "Did frame extraction succeed for both classes?"
        )
        raise RuntimeError(msg)

    indices = real_idx + fake_idx
    images, labels = [], []
    for i in indices:
        item = ds[i]
        images.append(item[0])
        labels.append(float(item[1]))
    images_t = torch.stack(images, dim=0)
    labels_t = torch.tensor(labels, dtype=torch.float32)

    print(
        f"  batch: {len(real_idx)} real + {len(fake_idx)} fake "
        f"= {len(indices)} frames, shape {tuple(images_t.shape)}",
    )
    print(f"  pixel range [{images_t.min():.3f}, {images_t.max():.3f}]")
    print(f"  labels {labels_t.tolist()}")
    return images_t, labels_t


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda", choices=("cpu", "cuda", "mps"))
    parser.add_argument("--lr", type=float, default=1.0e-4)
    parser.add_argument("--amp", action="store_true", help="enable fp16 autocast")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--image-size", type=int, default=256)
    args = parser.parse_args()

    torch = _torch()

    print("=== diagnose_baseline ===")
    print(f"  device={args.device}  lr={args.lr}  amp={args.amp}  steps={args.steps}")

    images_t, labels_t = _build_batch(args)
    images = images_t.to(args.device)
    labels = labels_t.to(args.device)

    from forge_detect.baseline_cnn import build_baseline_classifier

    model = build_baseline_classifier(pretrained=True).to(args.device)

    # Initial logits — should be small (random Linear head + ImageNet features).
    model.eval()
    with torch.no_grad():
        init_logits = model(images).squeeze(-1)
        init_probs = torch.sigmoid(init_logits)
    print(f"  initial logits = {[round(x, 3) for x in init_logits.tolist()]}")
    print(f"  initial probs  = {[round(x, 3) for x in init_probs.tolist()]}")

    # Train.
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1.0e-4)
    bce = torch.nn.BCEWithLogitsLoss()

    use_amp = args.amp and args.device == "cuda"
    scaler = torch.amp.GradScaler() if use_amp else None

    print(f"  training {args.steps} steps...")
    for step in range(args.steps):
        optimizer.zero_grad(set_to_none=True)
        if use_amp:
            with torch.amp.autocast("cuda"):
                logits = model(images).squeeze(-1)
                loss = bce(logits, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
        else:
            logits = model(images).squeeze(-1)
            loss = bce(logits, labels)
            loss.backward()

        # Gradient health metrics.
        grad_sq = 0.0
        n_params = 0
        n_zero_grad = 0
        for p in model.parameters():
            if p.grad is None:
                continue
            n_params += 1
            g = p.grad.detach()
            grad_sq += float((g * g).sum().item())
            if (g.abs() < 1.0e-12).all():
                n_zero_grad += 1
        grad_norm = grad_sq**0.5

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if use_amp:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()

        if step % 5 == 0 or step == args.steps - 1:
            print(
                f"    step {step:2d}: loss={loss.item():.4f} "
                f"|g|={grad_norm:.4f} zero-grad-tensors={n_zero_grad}/{n_params}",
            )

        if not torch.isfinite(loss):
            print(f"  !! loss became non-finite at step {step}")
            break

    # Final.
    model.eval()
    with torch.no_grad():
        final_logits = model(images).squeeze(-1)
        final_probs = torch.sigmoid(final_logits)
    print(f"  final logits = {[round(x, 3) for x in final_logits.tolist()]}")
    print(f"  final probs  = {[round(x, 3) for x in final_probs.tolist()]}")

    pred = (final_probs >= 0.5).float()
    acc = float((pred == labels).float().mean().item())
    print(f"  final overfit accuracy: {acc:.2%}")

    if acc >= 0.875:
        print("  >> PASS: model overfits a tiny batch. Training infrastructure is sound.")
        print("     The full-run loss stall is a tuning problem; check LR / AMP / sampler.")
        return 0
    print("  >> FAIL: model cannot overfit even a fixed 8-frame batch.")
    print("     Real bug somewhere in dataset / model / loss path. Investigate.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
