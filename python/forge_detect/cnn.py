"""ChromaticEfficientNet — trust-map predictor for the W_cnn input.

A torchvision EfficientNet-B0 backbone with two custom pieces:

- a **chromatic adapter** that maps a 6-channel input (R, G, B, I_w, ΔRG,
  ΔGB) into the 3-channel input EfficientNet expects;
- a **trust decoder** that upsamples the backbone's 1/32-resolution
  feature map back to input resolution and projects to a single channel
  through a 1×1 conv. The output passes through sigmoid to produce
  ``W_cnn ∈ [0, 1]^{H×W}``.

The class is importable without PyTorch installed; calling any method
raises :class:`ImportError` if torch is missing.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    import torch


def _torch() -> Any:
    try:
        import torch
    except ImportError as e:
        msg = "ChromaticEfficientNet requires PyTorch — `pip install torch torchvision`"
        raise ImportError(msg) from e
    return torch


def chromatic_inputs(rgb: Any) -> Any:
    """Build the 6-channel chromatic input from a ``(B, 3, H, W)`` RGB tensor.

    Channels: R, G, B, I_w (BT.601), ΔRG = R−G, ΔGB = G−B.
    """
    if rgb.dim() != 4 or rgb.shape[1] != 3:
        msg = f"expected (B, 3, H, W) RGB, got shape {tuple(rgb.shape)}"
        raise ValueError(msg)
    r = rgb[:, 0:1]
    g = rgb[:, 1:2]
    b = rgb[:, 2:3]
    i_w = 0.299 * r + 0.587 * g + 0.114 * b
    d_rg = r - g
    d_gb = g - b
    return _torch().cat([r, g, b, i_w, d_rg, d_gb], dim=1)


def build_chromatic_efficientnet(*, pretrained: bool = True) -> Any:
    """Construct the trust-map model as a single :class:`torch.nn.Module`.

    Args:
        pretrained: If ``True`` (default), load the IMAGENET1K_V1 weights for
            the EfficientNet-B0 backbone.

    Returns:
        An ``nn.Module`` whose ``forward(rgb_bchw_in_unit_range)`` returns
        ``W_cnn ∈ [0, 1]^{B×H×W}``.
    """
    torch = _torch()
    from torch import nn
    from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0

    weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
    backbone = efficientnet_b0(weights=weights)
    feature_channels = 1280  # EfficientNet-B0 final feature dim

    class ChromaticEfficientNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.adapter = nn.Conv2d(6, 3, kernel_size=1)
            self.features = backbone.features  # type: ignore[assignment]
            self.head = nn.Sequential(
                nn.Conv2d(feature_channels, 64, kernel_size=1),
                nn.GELU(),
                nn.Conv2d(64, 1, kernel_size=1),
            )

        def forward(self, rgb_bchw: torch.Tensor) -> torch.Tensor:
            x = chromatic_inputs(rgb_bchw)
            x = self.adapter(x)
            feat = self.features(x)  # (B, 1280, H/32, W/32)
            logits = self.head(feat)  # (B, 1, H/32, W/32)
            up = nn.functional.interpolate(
                logits,
                size=rgb_bchw.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            return torch.sigmoid(up).squeeze(1)  # (B, H, W)

    return ChromaticEfficientNet()


def load_weights(model: Any, path: str | Path) -> None:
    """Load checkpoint weights into ``model`` from ``path``."""
    torch = _torch()
    state = torch.load(path, map_location="cpu")
    model.load_state_dict(state)


def save_weights(model: Any, path: str | Path) -> None:
    """Save ``model``'s state dict to ``path``."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    _torch().save(model.state_dict(), path)


def predict_trust_map(model: Any, rgb_hw3: np.ndarray, device: str = "cpu") -> np.ndarray:
    """Run the model on a single ``(H, W, 3)`` numpy image and return ``(H, W)`` float32."""
    torch = _torch()
    if rgb_hw3.ndim != 3 or rgb_hw3.shape[2] != 3:
        msg = f"expected (H, W, 3) RGB, got {rgb_hw3.shape}"
        raise ValueError(msg)
    rgb = np.transpose(rgb_hw3.astype(np.float32, copy=False), (2, 0, 1))[None]
    x = torch.from_numpy(rgb).to(device)
    model.eval()
    with torch.no_grad():
        y = model(x)
    return y.squeeze(0).detach().cpu().numpy().astype(np.float32)


__all__ = [
    "build_chromatic_efficientnet",
    "chromatic_inputs",
    "load_weights",
    "predict_trust_map",
    "save_weights",
]
