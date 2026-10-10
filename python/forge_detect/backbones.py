"""Backbone registry shared by the trust-map net and the Phase-2/3 classifiers.

Every backbone is a torchvision ImageNet model exposing the same two hooks the
rest of the code relies on:

- ``model.features[0][0]`` is the stem ``nn.Conv2d`` (3 input channels) that
  :func:`forge_detect.baseline_cnn.build_physics_classifier` widens for the
  extra physics / frequency channels;
- ``model.classifier[-1]`` is the final ``nn.Linear`` that becomes the 1-logit
  head, and the first ``nn.Linear`` in ``classifier`` reports the feature
  dimension produced by ``model.features``.

Supported names (``torchvision.models``):

==================== ========= ============= ==========================
name                 stem out  feature dim   ImageNet weights
==================== ========= ============= ==========================
``efficientnet_b0``  32        1280          ``EfficientNet_B0_Weights``
``mobilenet_v3_large`` 16      960           ``MobileNet_V3_Large_Weights``
``mobilenet_v3_small`` 16      576           ``MobileNet_V3_Small_Weights``
``mobilenet_v2``     32        1280          ``MobileNet_V2_Weights``
==================== ========= ============= ==========================

EfficientNet-B0 remains the default everywhere so existing checkpoints load
unchanged. With the 1-logit head: EfficientNet-B0 4.0M parameters / ~0.39
GFLOPs at 224²; MobileNet-V3-Large 4.2M / ~0.22 GFLOPs; MobileNet-V3-Small
1.5M / ~0.06 GFLOPs; MobileNet-V2 2.2M / ~0.30 GFLOPs (torchvision figures).
All share the same stem surgery, head, normalisation and training loop.
"""

from __future__ import annotations

from typing import Any

BACKBONES: tuple[str, ...] = (
    "efficientnet_b0",
    "mobilenet_v3_large",
    "mobilenet_v3_small",
    "mobilenet_v2",
)
DEFAULT_BACKBONE = "efficientnet_b0"


def build_backbone(name: str = DEFAULT_BACKBONE, *, pretrained: bool = True) -> Any:
    """Return the torchvision model for ``name`` (ImageNet weights when ``pretrained``)."""
    if name not in BACKBONES:
        msg = f"unknown backbone {name!r}; choose one of {BACKBONES}"
        raise ValueError(msg)
    from torchvision import models

    if name == "efficientnet_b0":
        return models.efficientnet_b0(
            weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        )
    if name == "mobilenet_v3_large":
        return models.mobilenet_v3_large(
            weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V1 if pretrained else None
        )
    if name == "mobilenet_v3_small":
        return models.mobilenet_v3_small(
            weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
        )
    return models.mobilenet_v2(
        weights=models.MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None
    )


def stem_conv(backbone: Any) -> Any:
    """The 3-channel stem convolution (``features[0][0]``) of a supported backbone."""
    from torch import nn

    conv = backbone.features[0][0]
    if not isinstance(conv, nn.Conv2d):
        msg = f"unexpected stem layer type {type(conv).__name__}; torchvision API changed?"
        raise RuntimeError(msg)
    return conv


def feature_dim(backbone: Any) -> int:
    """Channel count produced by ``backbone.features`` (input width of the classifier)."""
    from torch import nn

    for layer in backbone.classifier:
        if isinstance(layer, nn.Linear):
            return int(layer.in_features)
    msg = "backbone.classifier has no nn.Linear layer; torchvision API changed?"
    raise RuntimeError(msg)


def replace_head_with_logit(backbone: Any) -> None:
    """Swap the last ``nn.Linear`` of ``classifier`` for a 1-logit layer (in place).

    For EfficientNet-B0 and MobileNet-V2 (``[Dropout, Linear]``) this keeps the
    pretrained dropout and yields the same ``classifier.1.*`` state-dict keys the
    original EfficientNet head used; for MobileNet-V3 the hidden
    ``Linear → Hardswish → Dropout`` block is kept and only the output layer is
    replaced.
    """
    from torch import nn

    last = backbone.classifier[-1]
    if not isinstance(last, nn.Linear):
        msg = f"expected nn.Linear classifier head, got {type(last).__name__}"
        raise RuntimeError(msg)
    backbone.classifier[-1] = nn.Linear(last.in_features, 1)


__all__ = [
    "BACKBONES",
    "DEFAULT_BACKBONE",
    "build_backbone",
    "feature_dim",
    "replace_head_with_logit",
    "stem_conv",
]
