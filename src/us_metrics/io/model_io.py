"""Serialization for NRQ PCA+GMM organ models."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import numpy as np

MODEL_FORMAT_VERSION = 1


@dataclass
class OrganGMMModel:
    pca_mean: np.ndarray
    pca_components: np.ndarray
    gmm_weights: np.ndarray
    gmm_means: np.ndarray
    gmm_covariances: np.ndarray
    log_weights: np.ndarray
    log_det: np.ndarray
    dim: int
    format_version: int = MODEL_FORMAT_VERSION

    def transform(self, z: np.ndarray) -> np.ndarray:
        return (z.astype(np.float64) - self.pca_mean) @ self.pca_components.T

    def loglik_batch(self, z_batch: np.ndarray) -> np.ndarray:
        x = (z_batch.astype(np.float64) - self.pca_mean[None, :]) @ self.pca_components.T
        diff = x[:, None, :] - self.gmm_means[None, :, :]
        md2 = np.sum((diff * diff) / self.gmm_covariances[None, :, :], axis=2)
        a = self.log_weights[None, :] - 0.5 * (md2 + self.log_det[None, :])
        a_max = np.max(a, axis=1, keepdims=True)
        lse = a_max + np.log(np.sum(np.exp(a - a_max), axis=1, keepdims=True))
        return lse[:, 0].astype(np.float64)


def save_organ_gmm(path_npz: Path | str, model: OrganGMMModel) -> None:
    path_npz = Path(path_npz)
    path_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path_npz,
        format_version=np.int32(model.format_version),
        pca_mean=model.pca_mean,
        pca_components=model.pca_components,
        gmm_weights=model.gmm_weights,
        gmm_means=model.gmm_means,
        gmm_covariances=model.gmm_covariances,
        log_weights=model.log_weights,
        log_det=model.log_det,
        dim=np.int32(model.dim),
    )


def load_organ_gmm(path_npz: Path | str) -> OrganGMMModel:
    dat = np.load(path_npz, allow_pickle=False)
    version = int(dat["format_version"].item()) if "format_version" in dat else 0
    dim = int(dat["dim"].item() if np.ndim(dat["dim"]) == 0 else dat["dim"].reshape(-1)[0])
    model = OrganGMMModel(
        pca_mean=dat["pca_mean"],
        pca_components=dat["pca_components"],
        gmm_weights=dat["gmm_weights"],
        gmm_means=dat["gmm_means"],
        gmm_covariances=dat["gmm_covariances"],
        log_weights=dat["log_weights"],
        log_det=dat["log_det"],
        dim=dim,
        format_version=version,
    )
    return model


def load_organ_models(models_dir: Path | str) -> Dict[str, OrganGMMModel]:
    models_dir = Path(models_dir)
    if not models_dir.is_dir():
        raise FileNotFoundError(str(models_dir))
    files = sorted(models_dir.glob("*.npz"))
    if not files:
        raise RuntimeError(f"No .npz models found in: {models_dir}")
    out = {}
    for f in files:
        out[f.stem] = load_organ_gmm(f)
    return out


def build_gmm_model_from_sklearn(
    pca_mean: np.ndarray,
    pca_components: np.ndarray,
    gmm_weights: np.ndarray,
    gmm_means: np.ndarray,
    gmm_covariances: np.ndarray,
) -> OrganGMMModel:
    weights = gmm_weights.astype(np.float64)
    means = gmm_means.astype(np.float64)
    covs = gmm_covariances.astype(np.float64)
    log_w = np.log(np.clip(weights, 1e-12, 1.0))
    dim = int(means.shape[1])
    log_det = np.sum(np.log(np.clip(covs, 1e-12, None)), axis=1) + dim * math.log(2.0 * math.pi)
    return OrganGMMModel(
        pca_mean=pca_mean.astype(np.float64),
        pca_components=pca_components.astype(np.float64),
        gmm_weights=weights,
        gmm_means=means,
        gmm_covariances=covs,
        log_weights=log_w,
        log_det=log_det.astype(np.float64),
        dim=dim,
    )

