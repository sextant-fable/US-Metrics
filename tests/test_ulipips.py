from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("torch")
import torch
import torch.nn as nn

import us_metrics.metrics.ulipips as ulipips
from us_metrics.data.patching import load_image_gray_u8, to_torch_nchw_224


class DummyBlock(nn.Module):
    def __init__(self, delta: float):
        super().__init__()
        self.delta = float(delta)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.delta


class DummyBackbone(nn.Module):
    def __init__(self, n_blocks: int = 12):
        super().__init__()
        self.blocks = nn.ModuleList([DummyBlock((i + 1) * 1e-3) for i in range(n_blocks)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        base = x.mean(dim=(1, 2, 3), keepdim=False).view(bsz, 1, 1)
        tokens = base.repeat(1, 17, 8)
        for block in self.blocks:
            tokens = block(tokens)
        return tokens


def _dummy_model_loader(_ckpt_path: str, device: str = "cpu") -> nn.Module:
    model = DummyBackbone().to(device).eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


def test_ulipips_same_image_is_near_zero(monkeypatch):
    monkeypatch.setattr(ulipips, "load_tinyusfm_model", _dummy_model_loader)
    img = np.full((280, 320), 127, dtype=np.uint8)
    score = ulipips.compute_ulipips(ref=img, img=img.copy(), ckpt_path="dummy.pth")
    assert abs(score) < 1e-8


def test_ulipips_layer_hooks_complete(monkeypatch):
    monkeypatch.setattr(ulipips, "load_tinyusfm_model", _dummy_model_loader)
    scorer = ulipips.TinyULIPIPS(ckpt_path="dummy.pth", layers=(3, 5, 7, 11), device="cpu")
    x = torch.zeros((1, 3, 224, 224), dtype=torch.float32)
    feats = scorer.extract_features(x)
    assert set(feats.keys()) == {"layer3", "layer5", "layer7", "layer11"}


def test_gray_and_rgb_input_to_tensor_shape():
    rgb = np.zeros((120, 90, 3), dtype=np.uint8)
    rgb[:, :, 1] = 200
    gray = load_image_gray_u8(rgb)
    assert gray.shape == (120, 90)
    t = to_torch_nchw_224(gray, device="cpu")
    assert t.shape == (1, 3, 224, 224)
