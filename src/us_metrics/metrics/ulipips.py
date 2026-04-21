"""TinyUSFM-uLPIPS (full-reference) implementation."""

from __future__ import annotations

import math
from typing import Dict, Iterable, Tuple

import torch
import torch.nn.functional as F

from us_metrics.backbone.loader import load_tinyusfm_model
from us_metrics.data.patching import load_image_gray_u8, to_torch_nchw_224

DEFAULT_LAYERS: Tuple[int, ...] = (3, 5, 7, 11)


def _resolve_device(device: str | None) -> str:
    if device:
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


def _layer_to_block_idx(layer_num: int) -> int:
    if layer_num < 1:
        raise ValueError(f"layer must be >=1, got {layer_num}")
    return layer_num - 1


class TinyULIPIPS:
    """Feature-hook based TinyUSFM-uLPIPS scorer."""

    def __init__(
        self,
        ckpt_path: str,
        layers: Iterable[int] = DEFAULT_LAYERS,
        device: str | None = None,
    ) -> None:
        self.device = _resolve_device(device)
        self.layers = tuple(layers)
        self.hook_features: Dict[str, torch.Tensor] = {}
        self.model = load_tinyusfm_model(ckpt_path=ckpt_path, device=self.device)
        self._register_hooks()

    def _register_hooks(self) -> None:
        real_backbone = self.model.model if hasattr(self.model, "model") else self.model
        if not hasattr(real_backbone, "blocks"):
            raise RuntimeError("TinyUSFM backbone missing `.blocks`")

        def get_activation(name):
            def hook(_module, _inp, out):
                self.hook_features[name] = out

            return hook

        for layer in self.layers:
            idx = _layer_to_block_idx(layer)
            if idx >= len(real_backbone.blocks):
                raise RuntimeError(f"Layer {layer} out of range for TinyUSFM blocks={len(real_backbone.blocks)}")
            real_backbone.blocks[idx].register_forward_hook(get_activation(f"layer{layer}"))

    def extract_features(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        self.hook_features = {}
        with torch.no_grad():
            _ = self.model(x)
        return {k: v.clone() for k, v in self.hook_features.items()}

    @staticmethod
    def gram_matrix(input_tensor: torch.Tensor) -> torch.Tensor:
        bsz, n_tokens, channels = input_tensor.size()
        features = input_tensor.permute(0, 2, 1)
        gram = torch.bmm(features, features.permute(0, 2, 1))
        return gram.div(n_tokens * channels)

    def structure_gram_distance(
        self,
        feat_a: Dict[str, torch.Tensor],
        feat_b: Dict[str, torch.Tensor],
        radius: int = 3,
        tau: float = 20.0,
    ) -> float:
        total_structure = 0.0
        total_gram = 0.0
        valid_layers = 0

        for layer in self.layers:
            key = f"layer{layer}"
            if key not in feat_a or key not in feat_b:
                continue
            fa = feat_a[key]
            fb = feat_b[key]

            if fa.dim() == 3 and fa.shape[1] > 1:
                fa = fa[:, 1:, :]
                fb = fb[:, 1:, :]

            fa_norm = F.normalize(fa, p=2, dim=-1)
            fb_norm = F.normalize(fb, p=2, dim=-1)

            bsz, n_tokens, channels = fa_norm.shape
            side = int(round(math.sqrt(n_tokens)))
            if side * side != n_tokens:
                continue

            fa_2d = fa_norm.view(bsz, side, side, channels)
            fb_2d = fb_norm.view(bsz, side, side, channels)

            dist_layer = 0.0
            count = 0
            for i in range(side):
                for j in range(side):
                    i1 = max(0, i - radius)
                    i2 = min(side, i + radius + 1)
                    j1 = max(0, j - radius)
                    j2 = min(side, j + radius + 1)

                    a_nb = fa_2d[:, i1:i2, j1:j2, :].reshape(1, -1, channels)
                    b_nb = fb_2d[:, i1:i2, j1:j2, :].reshape(1, -1, channels)

                    sim_a = torch.matmul(a_nb, a_nb.transpose(1, 2)) * tau
                    sim_b = torch.matmul(b_nb, b_nb.transpose(1, 2)) * tau
                    pa = F.softmax(sim_a, dim=-1)
                    pb = F.softmax(sim_b, dim=-1)
                    dist_layer += torch.mean(torch.abs(pa - pb)).item()
                    count += 1

            dist_layer = dist_layer / max(1, count)

            gram_a = self.gram_matrix(fa_norm)
            gram_b = self.gram_matrix(fb_norm)
            gram_dist = torch.mean(torch.abs(gram_a - gram_b)).item()

            total_structure += dist_layer
            total_gram += gram_dist
            valid_layers += 1

        if valid_layers == 0:
            return 0.0
        return float((total_structure + total_gram) / valid_layers)

    def score(
        self,
        ref_tensor: torch.Tensor,
        img_tensor: torch.Tensor,
        radius: int = 3,
        tau: float = 20.0,
    ) -> float:
        feat_ref = self.extract_features(ref_tensor.to(self.device))
        feat_img = self.extract_features(img_tensor.to(self.device))
        return self.structure_gram_distance(
            feat_ref, feat_img, radius=radius, tau=tau
        )


def compute_ulipips(
    ref,
    img,
    ckpt_path: str,
    layers: Tuple[int, ...] = DEFAULT_LAYERS,
    radius: int = 3,
    tau: float = 20.0,
    device: str | None = None,
) -> float:
    """Compute TinyUSFM-uLPIPS score between reference and test image."""
    scorer = TinyULIPIPS(ckpt_path=ckpt_path, layers=layers, device=device)
    ref_u8 = load_image_gray_u8(ref)
    img_u8 = load_image_gray_u8(img)
    ref_t = to_torch_nchw_224(ref_u8, device=scorer.device)
    img_t = to_torch_nchw_224(img_u8, device=scorer.device)
    return scorer.score(ref_t, img_t, radius=radius, tau=tau)

