"""TinyUSFM-NRQ (no-reference) implementation."""

from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture

from us_metrics.backbone.loader import load_tinyusfm_model
from us_metrics.data.patching import (
    extract_patches_u8,
    load_image_gray_u8,
    patches_to_torch,
)
from us_metrics.io.model_io import (
    build_gmm_model_from_sklearn,
    load_organ_models,
    save_organ_gmm,
)

DEFAULT_LAYERS: Tuple[int, ...] = (3, 5, 7, 11)
DEFAULT_PATCH = 224
DEFAULT_STRIDE = 112
DEFAULT_TOPK_FRAC = 0.15

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def _resolve_device(device: str | None) -> str:
    if device:
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _layer_to_block_idx(layer_num: int) -> int:
    if layer_num < 1:
        raise ValueError(f"layer must be >=1, got {layer_num}")
    return layer_num - 1


def _iter_image_files(root: Path) -> Iterable[Path]:
    for p in sorted(root.rglob("*")):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            yield p


class TinyUSFMFeatureExtractor(nn.Module):
    """Extract layer-pooled TinyUSFM descriptors z(x)."""

    def __init__(
        self,
        ckpt_path: str,
        layers: Tuple[int, ...] = DEFAULT_LAYERS,
        device: str | None = None,
    ) -> None:
        super().__init__()
        self.layers = tuple(layers)
        self.device = _resolve_device(device)
        self.model = load_tinyusfm_model(ckpt_path=ckpt_path, device=self.device)
        self.hook_features: Dict[str, torch.Tensor] = {}
        self._register_hooks()

    def _register_hooks(self) -> None:
        real = self.model.model if hasattr(self.model, "model") else self.model
        if not hasattr(real, "blocks"):
            raise RuntimeError("TinyUSFM backbone missing `.blocks`")

        def get_activation(name):
            def hook(_module, _inp, out):
                self.hook_features[name] = out

            return hook

        for layer in self.layers:
            idx = _layer_to_block_idx(layer)
            if idx >= len(real.blocks):
                raise RuntimeError(
                    f"Layer {layer} out of range for TinyUSFM blocks={len(real.blocks)}"
                )
            real.blocks[idx].register_forward_hook(get_activation(f"layer{layer}"))

    @torch.no_grad()
    def forward_z(self, x: torch.Tensor) -> torch.Tensor:
        self.hook_features = {}
        _ = self.model(x)
        vecs = []
        for layer in self.layers:
            name = f"layer{layer}"
            if name not in self.hook_features:
                raise RuntimeError(f"Missing hooked feature: {name}")
            f = self.hook_features[name]
            if f.dim() == 3 and f.shape[1] > 1:
                f = f[:, 1:, :]
            v = f.mean(dim=1)
            vecs.append(v)
        z = torch.cat(vecs, dim=1)
        z = F.normalize(z, p=2, dim=1)
        return z


def _fit_pca_gmm(
    z_features: np.ndarray,
    pca_dim: int,
    gmm_k: int,
    gmm_reg_covar: float,
    seed: int,
):
    if z_features.ndim != 2 or z_features.shape[0] < 2:
        raise ValueError("z_features must be 2D with at least 2 samples for PCA+GMM fitting")

    pca_dim = int(min(pca_dim, z_features.shape[1], z_features.shape[0]))
    gmm_k = int(min(gmm_k, z_features.shape[0]))
    if gmm_k < 1:
        raise ValueError("gmm_k must be >= 1")
    pca = PCA(n_components=pca_dim, random_state=seed)
    x = pca.fit_transform(z_features)

    gmm = GaussianMixture(
        n_components=gmm_k,
        covariance_type="diag",
        reg_covar=gmm_reg_covar,
        random_state=seed,
        max_iter=300,
        init_params="kmeans",
    )
    gmm.fit(x)

    return build_gmm_model_from_sklearn(
        pca_mean=pca.mean_,
        pca_components=pca.components_,
        gmm_weights=gmm.weights_,
        gmm_means=gmm.means_,
        gmm_covariances=gmm.covariances_,
    )


def _extract_patch_features(
    img_u8: np.ndarray,
    extractor: TinyUSFMFeatureExtractor,
    patch: int,
    stride: int,
    max_patches: int,
    gpu_chunk: int,
    seed: int,
) -> np.ndarray:
    patches = extract_patches_u8(
        img_u8=img_u8,
        patch=patch,
        stride=stride,
        max_patches=max_patches,
        seed=seed,
    )
    if not patches:
        return np.empty((0, 0), dtype=np.float64)

    z_list: List[np.ndarray] = []
    for start in range(0, len(patches), gpu_chunk):
        end = min(len(patches), start + gpu_chunk)
        batch = patches_to_torch(patches[start:end], device=extractor.device)
        with torch.no_grad():
            z = extractor.forward_z(batch).detach().cpu().numpy().astype(np.float64)
        z_list.append(z)
    return np.concatenate(z_list, axis=0)


def fit_nrq_models(
    clean_data_root,
    ckpt_path: str,
    out_dir,
    pca_dim: int = 128,
    gmm_k: int = 4,
    patch: int = DEFAULT_PATCH,
    stride: int = DEFAULT_STRIDE,
    topk_frac: float = DEFAULT_TOPK_FRAC,
    *,
    layers: Tuple[int, ...] = DEFAULT_LAYERS,
    max_patches_per_image: int = 4,
    gpu_chunk: int = 24,
    seed: int = 2026,
    device: str | None = None,
) -> Dict[str, str]:
    """Fit per-organ PCA+GMM clean manifold models and save as .npz."""
    clean_data_root = Path(clean_data_root)
    out_dir = Path(out_dir)
    models_dir = out_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    if not clean_data_root.is_dir():
        raise FileNotFoundError(str(clean_data_root))

    _set_seed(seed)
    extractor = TinyUSFMFeatureExtractor(
        ckpt_path=ckpt_path,
        layers=layers,
        device=device,
    ).eval()

    organs = [p for p in sorted(clean_data_root.iterdir()) if p.is_dir()]
    if not organs:
        raise RuntimeError(f"No organ subfolders found in: {clean_data_root}")

    model_paths: Dict[str, str] = {}
    fit_rows = []
    for organ_dir in organs:
        organ_name = organ_dir.name
        image_paths = list(_iter_image_files(organ_dir))
        if not image_paths:
            continue

        z_blocks: List[np.ndarray] = []
        for idx, image_path in enumerate(image_paths):
            img_u8 = load_image_gray_u8(image_path)
            local_seed = (seed + idx * 1315423911) & 0xFFFFFFFF
            z = _extract_patch_features(
                img_u8=img_u8,
                extractor=extractor,
                patch=patch,
                stride=stride,
                max_patches=max_patches_per_image,
                gpu_chunk=gpu_chunk,
                seed=local_seed,
            )
            if z.size == 0:
                continue
            z_blocks.append(z)

        if not z_blocks:
            continue
        z_features = np.concatenate(z_blocks, axis=0)

        model = _fit_pca_gmm(
            z_features=z_features,
            pca_dim=pca_dim,
            gmm_k=gmm_k,
            gmm_reg_covar=1e-4,
            seed=seed,
        )
        path_npz = models_dir / f"{organ_name}.npz"
        save_organ_gmm(path_npz, model)
        model_paths[organ_name] = str(path_npz)
        fit_rows.append(
            {
                "organ": organ_name,
                "n_images": len(image_paths),
                "n_features": int(z_features.shape[0]),
                "feature_dim_raw": int(z_features.shape[1]),
                "pca_dim": int(model.dim),
                "gmm_k": int(gmm_k),
            }
        )

    if not model_paths:
        raise RuntimeError("No organ models fitted. Check input data layout and image files.")

    meta = {
        "format_version": 1,
        "seed": int(seed),
        "layers": list(layers),
        "patch": int(patch),
        "stride": int(stride),
        "topk_frac": float(topk_frac),
        "pca_dim": int(pca_dim),
        "gmm_k": int(gmm_k),
        "max_patches_per_image": int(max_patches_per_image),
        "models": model_paths,
        "fit_info": fit_rows,
    }
    with (out_dir / "nrq_fit_meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    return model_paths


def _score_from_features(
    z_features: np.ndarray,
    organ_models,
    organ: Optional[str],
    topk_frac: float,
) -> float:
    if organ is not None:
        if organ not in organ_models:
            raise ValueError(f"organ '{organ}' not in models. Available: {sorted(organ_models.keys())}")
        ll = organ_models[organ].loglik_batch(z_features)
    else:
        per_organ = []
        for m in organ_models.values():
            per_organ.append(m.loglik_batch(z_features)[:, None])
        mat = np.concatenate(per_organ, axis=1)
        mmax = np.max(mat, axis=1, keepdims=True)
        ll = (mmax + np.log(np.mean(np.exp(mat - mmax), axis=1, keepdims=True)))[:, 0]

    bad = (-ll).astype(np.float64)
    n = bad.shape[0]
    k = int(max(1, round(n * topk_frac)))
    idx = np.argpartition(-bad, kth=min(k - 1, n - 1))[:k]
    return float(-np.mean(bad[idx]))


def score_nrq(
    img,
    ckpt_path: str,
    models_dir,
    organ: str | None = None,
    patch: int = DEFAULT_PATCH,
    stride: int = DEFAULT_STRIDE,
    topk_frac: float = DEFAULT_TOPK_FRAC,
    *,
    layers: Tuple[int, ...] = DEFAULT_LAYERS,
    max_patches: int = 16,
    gpu_chunk: int = 24,
    seed: int = 2026,
    device: str | None = None,
) -> float:
    """Score one image with TinyUSFM-NRQ. Higher score means better quality."""
    _set_seed(seed)
    extractor = TinyUSFMFeatureExtractor(
        ckpt_path=ckpt_path,
        layers=layers,
        device=device,
    ).eval()
    models = load_organ_models(models_dir)

    img_u8 = load_image_gray_u8(img)
    z = _extract_patch_features(
        img_u8=img_u8,
        extractor=extractor,
        patch=patch,
        stride=stride,
        max_patches=max_patches,
        gpu_chunk=gpu_chunk,
        seed=seed,
    )
    if z.size == 0:
        return float("nan")
    return _score_from_features(
        z_features=z,
        organ_models=models,
        organ=organ,
        topk_frac=topk_frac,
    )
