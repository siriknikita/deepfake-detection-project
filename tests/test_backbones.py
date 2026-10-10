"""Backbone registry: every supported backbone drives the three model builders."""

from __future__ import annotations

from typing import Any

import pytest

from forge_detect.backbones import BACKBONES, DEFAULT_BACKBONE


def _count_params(model: Any) -> int:
    return sum(p.numel() for p in model.parameters())


def test_default_backbone_is_efficientnet_for_checkpoint_compatibility() -> None:
    assert DEFAULT_BACKBONE == "efficientnet_b0"


@pytest.mark.parametrize("backbone", BACKBONES)
def test_baseline_and_physics_classifiers_forward(backbone: str) -> None:
    pytest.importorskip("torch")
    import torch

    from forge_detect.baseline_cnn import build_baseline_classifier, build_physics_classifier

    base = build_baseline_classifier(pretrained=False, backbone=backbone)
    with torch.no_grad():
        y = base(torch.rand(2, 3, 64, 64))
    assert y.shape == (2, 1) and torch.isfinite(y).all()
    for in_ch in (6, 9):
        phys = build_physics_classifier(in_channels=in_ch, pretrained=False, backbone=backbone)
        stem = phys[1].features[0][0]
        assert stem.in_channels == in_ch
        with torch.no_grad():
            y = phys(torch.rand(2, in_ch, 64, 64))
        assert y.shape == (2, 1) and torch.isfinite(y).all()


@pytest.mark.parametrize("backbone", BACKBONES)
def test_chromatic_trust_net_forward(backbone: str) -> None:
    pytest.importorskip("torch")
    import torch

    from forge_detect.cnn import build_chromatic_efficientnet

    model = build_chromatic_efficientnet(pretrained=False, backbone=backbone)
    with torch.no_grad():
        w = model(torch.rand(1, 3, 64, 64))
    assert w.shape == (1, 64, 64)
    assert float(w.min()) >= 0.0 and float(w.max()) <= 1.0


def test_backbone_parameter_counts_are_in_the_documented_range() -> None:
    """MobileNet-V3-Small is the light option; V3-Large has about as many parameters as
    EfficientNet-B0 (4.2M vs 4.0M with the 1-logit head) but roughly half the FLOPs."""
    pytest.importorskip("torch")
    from forge_detect.baseline_cnn import build_physics_classifier

    eff = _count_params(build_physics_classifier(in_channels=6, pretrained=False))
    mnl = _count_params(
        build_physics_classifier(in_channels=6, pretrained=False, backbone="mobilenet_v3_large")
    )
    mns = _count_params(
        build_physics_classifier(in_channels=6, pretrained=False, backbone="mobilenet_v3_small")
    )
    assert mns < 0.6 * eff
    assert 0.8 * eff < mnl < 1.2 * eff


def test_efficientnet_state_dict_keys_unchanged() -> None:
    """Existing EfficientNet checkpoints (classifier.1.*) must still load."""
    pytest.importorskip("torch")
    from forge_detect.baseline_cnn import build_baseline_classifier

    keys = set(build_baseline_classifier(pretrained=False).state_dict())
    assert "1.classifier.1.weight" in keys and "1.classifier.1.bias" in keys


def test_physics_stem_surgery_copies_rgb_kernels_when_pretrained_weights_given(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """RGB stem kernels are copied into channels 0-2 and channels 3+ get their per-output mean.

    Uses a randomly initialised MobileNet as the "pretrained" source so no download is needed.
    """
    pytest.importorskip("torch")
    import torch

    from forge_detect import baseline_cnn
    from forge_detect.backbones import build_backbone

    seeded = build_backbone("mobilenet_v3_large", pretrained=False)
    with torch.no_grad():
        seeded.features[0][0].weight.normal_()
    old_w = seeded.features[0][0].weight.detach().clone()

    def fake_build_backbone(name: str, *, pretrained: bool) -> Any:
        return seeded

    monkeypatch.setattr("forge_detect.baseline_cnn.build_backbone", fake_build_backbone)
    model = baseline_cnn.build_physics_classifier(
        in_channels=6, pretrained=True, backbone="mobilenet_v3_large"
    )
    new_w = model[1].features[0][0].weight.detach()
    assert new_w.shape[1] == 6
    assert torch.allclose(new_w[:, :3], old_w)
    expected_extra = old_w.mean(dim=1, keepdim=True) / 3
    assert torch.allclose(new_w[:, 3:], expected_extra.expand(-1, 3, -1, -1))
