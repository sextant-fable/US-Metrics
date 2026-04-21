from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("sklearn")

from us_metrics.data.patching import extract_patches_u8
from us_metrics.metrics.nrq import _score_from_features


class FixedLogLikModel:
    def __init__(self, values):
        self.values = np.asarray(values, dtype=np.float64)

    def loglik_batch(self, z_batch: np.ndarray) -> np.ndarray:
        if len(self.values) != z_batch.shape[0]:
            raise ValueError("length mismatch")
        return self.values


class ConstantLogLikModel:
    def __init__(self, value: float):
        self.value = float(value)

    def loglik_batch(self, z_batch: np.ndarray) -> np.ndarray:
        return np.full((z_batch.shape[0],), self.value, dtype=np.float64)


def test_patch_extraction_count():
    img = np.zeros((300, 300), dtype=np.uint8)
    patches = extract_patches_u8(img, patch=224, stride=112, max_patches=0)
    assert len(patches) == 4
    assert patches[0].shape == (224, 224)


def test_topk_aggregation_behavior():
    z = np.zeros((4, 6), dtype=np.float64)
    model = FixedLogLikModel(values=[-1.0, -2.0, -3.0, -4.0])
    score = _score_from_features(
        z_features=z,
        organ_models={"A": model},
        organ="A",
        topk_frac=0.5,
    )
    assert np.isclose(score, -3.5)


def test_organ_mixture_logmeanexp_is_finite():
    z = np.zeros((8, 6), dtype=np.float64)
    models = {
        "A": ConstantLogLikModel(-1000.0),
        "B": ConstantLogLikModel(-1001.0),
    }
    score = _score_from_features(
        z_features=z,
        organ_models=models,
        organ=None,
        topk_frac=0.25,
    )
    assert np.isfinite(score)
    assert -1002.0 < score < -999.0
