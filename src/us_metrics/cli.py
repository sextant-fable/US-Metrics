"""Command line entrypoint for US-Metrics."""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Iterable, Tuple


def compute_ulipips(*args: Any, **kwargs: Any):
    from us_metrics.metrics.ulipips import compute_ulipips as _impl

    return _impl(*args, **kwargs)


def fit_nrq_models(*args: Any, **kwargs: Any):
    from us_metrics.metrics.nrq import fit_nrq_models as _impl

    return _impl(*args, **kwargs)


def score_nrq(*args: Any, **kwargs: Any):
    from us_metrics.metrics.nrq import score_nrq as _impl

    return _impl(*args, **kwargs)


def _parse_layers(text: str) -> Tuple[int, ...]:
    items = [x.strip() for x in text.split(",") if x.strip()]
    if not items:
        raise argparse.ArgumentTypeError("layers cannot be empty")
    try:
        layers = tuple(int(x) for x in items)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid layer list: {text}") from exc
    if any(x <= 0 for x in layers):
        raise argparse.ArgumentTypeError("layer index must be positive")
    return layers


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="us-metrics",
        description="TinyUSFM-uLPIPS and TinyUSFM-NRQ metrics",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_ul = sub.add_parser("ulipips", help="Compute TinyUSFM-uLPIPS score")
    p_ul.add_argument("--ref", required=True, help="Reference image path")
    p_ul.add_argument("--img", required=True, help="Test image path")
    p_ul.add_argument("--ckpt", required=True, help="TinyUSFM checkpoint path")
    p_ul.add_argument("--layers", type=_parse_layers, default=(3, 5, 7, 11))
    p_ul.add_argument("--radius", type=int, default=3)
    p_ul.add_argument("--tau", type=float, default=20.0)
    p_ul.add_argument("--device", default=None, help="e.g. cpu / cuda")
    p_ul.set_defaults(func=_cmd_ulipips)

    p_fit = sub.add_parser("fit-nrq", help="Fit TinyUSFM-NRQ organ models")
    p_fit.add_argument("--clean-root", required=True, help="Clean data root by organ")
    p_fit.add_argument("--ckpt", required=True, help="TinyUSFM checkpoint path")
    p_fit.add_argument("--out", required=True, help="Output directory")
    p_fit.add_argument("--layers", type=_parse_layers, default=(3, 5, 7, 11))
    p_fit.add_argument("--pca-dim", type=int, default=128)
    p_fit.add_argument("--gmm-k", type=int, default=4)
    p_fit.add_argument("--patch", type=int, default=224)
    p_fit.add_argument("--stride", type=int, default=112)
    p_fit.add_argument("--topk-frac", type=float, default=0.15)
    p_fit.add_argument("--max-patches-per-image", type=int, default=4)
    p_fit.add_argument("--gpu-chunk", type=int, default=24)
    p_fit.add_argument("--seed", type=int, default=2026)
    p_fit.add_argument("--device", default=None, help="e.g. cpu / cuda")
    p_fit.set_defaults(func=_cmd_fit_nrq)

    p_nrq = sub.add_parser("nrq", help="Score TinyUSFM-NRQ")
    p_nrq.add_argument("--img", required=True, help="Image path")
    p_nrq.add_argument("--ckpt", required=True, help="TinyUSFM checkpoint path")
    p_nrq.add_argument("--models", required=True, help="Directory with *.npz organ models")
    p_nrq.add_argument("--organ", default=None, help="Optional organ name")
    p_nrq.add_argument("--layers", type=_parse_layers, default=(3, 5, 7, 11))
    p_nrq.add_argument("--patch", type=int, default=224)
    p_nrq.add_argument("--stride", type=int, default=112)
    p_nrq.add_argument("--topk-frac", type=float, default=0.15)
    p_nrq.add_argument("--max-patches", type=int, default=16)
    p_nrq.add_argument("--gpu-chunk", type=int, default=24)
    p_nrq.add_argument("--seed", type=int, default=2026)
    p_nrq.add_argument("--device", default=None, help="e.g. cpu / cuda")
    p_nrq.set_defaults(func=_cmd_nrq)

    return parser


def _cmd_ulipips(args: argparse.Namespace) -> dict:
    score = compute_ulipips(
        ref=args.ref,
        img=args.img,
        ckpt_path=args.ckpt,
        layers=args.layers,
        radius=args.radius,
        tau=args.tau,
        device=args.device,
    )
    return {"metric": "TinyUSFM-uLPIPS", "score": float(score)}


def _cmd_fit_nrq(args: argparse.Namespace) -> dict:
    model_paths = fit_nrq_models(
        clean_data_root=args.clean_root,
        ckpt_path=args.ckpt,
        out_dir=args.out,
        pca_dim=args.pca_dim,
        gmm_k=args.gmm_k,
        patch=args.patch,
        stride=args.stride,
        topk_frac=args.topk_frac,
        layers=args.layers,
        max_patches_per_image=args.max_patches_per_image,
        gpu_chunk=args.gpu_chunk,
        seed=args.seed,
        device=args.device,
    )
    return {
        "metric": "TinyUSFM-NRQ-fit",
        "n_models": len(model_paths),
        "models": model_paths,
    }


def _cmd_nrq(args: argparse.Namespace) -> dict:
    score = score_nrq(
        img=args.img,
        ckpt_path=args.ckpt,
        models_dir=args.models,
        organ=args.organ,
        patch=args.patch,
        stride=args.stride,
        topk_frac=args.topk_frac,
        layers=args.layers,
        max_patches=args.max_patches,
        gpu_chunk=args.gpu_chunk,
        seed=args.seed,
        device=args.device,
    )
    return {"metric": "TinyUSFM-NRQ", "score": float(score), "organ": args.organ}


def main(argv: Iterable[str] | None = None) -> int:
    parser = _build_parser()
    try:
        args = parser.parse_args(list(argv) if argv is not None else None)
        result = args.func(args)
        print(json.dumps(result, ensure_ascii=False))
        return 0
    except Exception as exc:  # noqa: BLE001
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
