"""TinyUSFM loading helpers."""

from __future__ import annotations

from typing import Dict

import torch

from .tinyusfm import TinyUSFM, VisionTransformer


def patch_tinyusfm_forward_features() -> None:
    """Patch forward_features to accept and ignore extra kwargs (e.g. attn_mask)."""
    if not hasattr(VisionTransformer, "_original_forward_features"):
        VisionTransformer._original_forward_features = VisionTransformer.forward_features

    def patched_forward_features(self, x, **kwargs):  # pylint: disable=unused-argument
        return self._original_forward_features(x)

    VisionTransformer.forward_features = patched_forward_features


def _clean_state_dict(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {
        k.replace("module.", "").replace("backbone.", ""): v
        for k, v in state_dict.items()
    }


def load_tinyusfm_model(
    ckpt_path: str,
    device: str = "cpu",
) -> TinyUSFM:
    patch_tinyusfm_forward_features()
    model = TinyUSFM(num_classes=0).to(device).eval()

    if ckpt_path:
        ckpt = torch.load(ckpt_path, map_location="cpu")
        state_dict = ckpt.get("model", ckpt) if isinstance(ckpt, dict) else ckpt
        if isinstance(state_dict, dict):
            clean = _clean_state_dict(state_dict)
            try:
                model.load_state_dict(clean, strict=False)
            except Exception:
                if hasattr(model, "model"):
                    model.model.load_state_dict(clean, strict=False)
                else:
                    raise

    for param in model.parameters():
        param.requires_grad = False
    return model

