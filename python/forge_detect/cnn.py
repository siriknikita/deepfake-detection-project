"""ChromaticEfficientNet trust-map predictor (architecture scaffold).

This module defines the trained-CNN path described in the paper. The
architecture is fully wired up but the weights are *not* shipped — training
on a labeled real/forged dataset is tracked as future work. Until then,
:func:`forge_detect.trust_map.heuristic_trust_map` is the recommended path
for end-to-end runs.

The class is importable without PyTorch present; calling any method raises
``ImportError`` if torch is missing.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import torch  # noqa: F401  (only for type hints)


def _require_torch() -> object:
    try:
        import torch
    except ImportError as e:
        msg = "ChromaticEfficientNet requires PyTorch — install via `pip install torch torchvision`"
        raise ImportError(msg) from e
    return torch


class ChromaticEfficientNet:
    """EfficientNet-B0 backbone + chromatic head + UNet-style trust decoder.

    The forward pass takes a 6-channel chromatic input (R, G, B, I_w, ΔRG,
    ΔGB) where the last three are derived from the first three; the output
    is a per-pixel trust map ``W_cnn ∈ [0, 1]`` matching the input
    resolution.

    Construction is deferred until :meth:`build` because importing torch on
    a host that does not have it must not break the rest of the package.
    """

    def __init__(self) -> None:
        self._model: object | None = None

    def build(self, *, pretrained: bool = True) -> None:
        """Materialize the underlying ``torch.nn.Module``."""
        torch = _require_torch()  # noqa: F841  (probe + import side effect)
        from torch import nn
        from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0

        weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = efficientnet_b0(weights=weights)
        backbone.classifier = nn.Identity()  # type: ignore[assignment]
        adapter = nn.Conv2d(6, 3, kernel_size=1)
        decoder = _UNetDecoder()
        self._model = nn.Sequential(adapter, backbone.features, decoder)

    def __call__(self, rgb: np.ndarray) -> np.ndarray:
        """Run the forward pass and return a numpy trust map."""
        torch_mod = _require_torch()
        if self._model is None:
            self.build(pretrained=False)
        if rgb.ndim != 3 or rgb.shape[2] != 3:
            msg = f"rgb must be (H, W, 3), got {rgb.shape}"
            raise ValueError(msg)

        x = self._chromatic_inputs(rgb)
        # Lazily import torch and convert.
        x_t = torch_mod.from_numpy(x).unsqueeze(0)  # type: ignore[attr-defined]
        with torch_mod.no_grad():  # type: ignore[attr-defined]
            y = self._model(x_t)  # type: ignore[misc]
            y = torch_mod.sigmoid(y).squeeze(0).squeeze(0)  # type: ignore[attr-defined]
        return y.cpu().numpy().astype(np.float32)

    @staticmethod
    def _chromatic_inputs(rgb: np.ndarray) -> np.ndarray:
        r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
        i_w = 0.299 * r + 0.587 * g + 0.114 * b
        d_rg = r - g
        d_gb = g - b
        stacked = np.stack([r, g, b, i_w, d_rg, d_gb], axis=0)
        return stacked.astype(np.float32, copy=False)

    def load_weights(self, path: str) -> None:
        """Load checkpoint weights from disk."""
        torch_mod = _require_torch()
        if self._model is None:
            self.build(pretrained=False)
        state = torch_mod.load(path, map_location="cpu")  # type: ignore[attr-defined]
        self._model.load_state_dict(state)  # type: ignore[union-attr]

    def save_weights(self, path: str) -> None:
        torch_mod = _require_torch()
        if self._model is None:
            msg = "model has not been built — call build() before save_weights()"
            raise RuntimeError(msg)
        torch_mod.save(self._model.state_dict(), path)  # type: ignore[union-attr,attr-defined]


class _UNetDecoder:
    """Lightweight upsampler back to input resolution.

    Lazily constructs torch layers on first use via :meth:`build` (called
    by :class:`ChromaticEfficientNet.build`). Kept module-private; not part
    of the public surface.
    """

    def __init__(self) -> None:
        self._impl: object | None = None

    def __call__(self, x: object) -> object:
        if self._impl is None:
            self._impl = self._construct()
        return self._impl(x)  # type: ignore[operator]

    @staticmethod
    def _construct() -> object:
        _require_torch()  # ensure torch is importable before pulling in nn
        from torch import nn

        return nn.Sequential(
            nn.Upsample(scale_factor=32, mode="bilinear", align_corners=False),
            nn.Conv2d(1280, 1, kernel_size=1),
        )


__all__ = ["ChromaticEfficientNet"]
