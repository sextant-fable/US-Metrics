## US-Metrics v0.1.0

First open-source release of our reproducible implementation for ultrasound quality metrics:

- TinyUSFM-uLPIPS (full-reference)
- TinyUSFM-NRQ (no-reference; clean manifold fitting + inference)

### Open-source scope

This release contains metric computation methods only.
The following are intentionally excluded: distortion generation, PSNR alignment, downstream task evaluation, plotting/statistics, MICCAI result artifacts, and benchmark pipelines.

### Highlights

- Standalone Python package (`src/us_metrics`)
- Unified CLI:
  - `us-metrics ulipips --ref ... --img ... --ckpt ...`
  - `us-metrics fit-nrq --clean-root ... --ckpt ... --out ...`
  - `us-metrics nrq --img ... --ckpt ... --models ... [--organ ...]`
- Paper-aligned defaults:
  - layers `{3,5,7,11}`
  - uLPIPS: `r=3`, `tau=20`
  - NRQ: `patch=224`, `stride=112`, `PCA d=128`, `GMM K=4`, `topk_frac=0.15`
- Versioned `.npz` model format

### Weights

- TinyUSFM backbone remains in an external repository / external checkpoint source.
- NRQ organ models can be open-sourced in this repo under `weights/nrq/` if file size is small.

### Citation

If you use this repository, metrics, or provided NRQ models, please cite:

`Defining Robust Ultrasound Quality Metrics via an Ultrasound Foundation Model`
