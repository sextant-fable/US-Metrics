from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("sklearn")
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture

from us_metrics.io.model_io import (
    MODEL_FORMAT_VERSION,
    build_gmm_model_from_sklearn,
    load_organ_gmm,
    save_organ_gmm,
)


def test_model_io_roundtrip_consistent(tmp_path):
    rng = np.random.default_rng(2026)
    z_train = rng.normal(size=(120, 16))

    pca = PCA(n_components=8, random_state=2026)
    x = pca.fit_transform(z_train)
    gmm = GaussianMixture(
        n_components=3,
        covariance_type="diag",
        reg_covar=1e-4,
        random_state=2026,
        max_iter=300,
    )
    gmm.fit(x)

    model = build_gmm_model_from_sklearn(
        pca_mean=pca.mean_,
        pca_components=pca.components_,
        gmm_weights=gmm.weights_,
        gmm_means=gmm.means_,
        gmm_covariances=gmm.covariances_,
    )

    path = tmp_path / "thyroid.npz"
    save_organ_gmm(path, model)
    loaded = load_organ_gmm(path)

    z_probe = rng.normal(size=(25, 16))
    ll_a = model.loglik_batch(z_probe)
    ll_b = loaded.loglik_batch(z_probe)

    assert np.allclose(ll_a, ll_b, atol=1e-10)
    assert loaded.format_version == MODEL_FORMAT_VERSION
