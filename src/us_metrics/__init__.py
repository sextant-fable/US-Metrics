"""US-Metrics public API."""

from __future__ import annotations

from typing import Any

__all__ = [
    "compute_ulipips",
    "fit_nrq_models",
    "score_nrq",
]


def compute_ulipips(*args: Any, **kwargs: Any):
    from .metrics.ulipips import compute_ulipips as _impl

    return _impl(*args, **kwargs)


def fit_nrq_models(*args: Any, **kwargs: Any):
    from .metrics.nrq import fit_nrq_models as _impl

    return _impl(*args, **kwargs)


def score_nrq(*args: Any, **kwargs: Any):
    from .metrics.nrq import score_nrq as _impl

    return _impl(*args, **kwargs)
