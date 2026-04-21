# US-Metrics

`US-Metrics` is a standalone package for two ultrasound-native quality metrics:

- `TinyUSFM-uLPIPS` (full-reference)
- `TinyUSFM-NRQ` (no-reference, with clean-manifold fitting + scoring)

## Features

- Independent package layout (`src/us_metrics/...`)
- TinyUSFM-compatible feature extraction with `forward_features` monkey patch for `attn_mask` compatibility
- API functions:
  - `compute_ulipips(ref, img, ckpt_path, layers=(3,5,7,11), radius=3, tau=20)`
  - `fit_nrq_models(clean_data_root, ckpt_path, out_dir, pca_dim=128, gmm_k=4, patch=224, stride=112, topk_frac=0.15)`
  - `score_nrq(img, ckpt_path, models_dir, organ=None, patch=224, stride=112, topk_frac=0.15)`
- CLI commands:
  - `us-metrics ulipips --ref ... --img ... --ckpt ...`
  - `us-metrics fit-nrq --clean-root ... --ckpt ... --out ...`
  - `us-metrics nrq --img ... --ckpt ... --models ... [--organ ...]`
- Reproducible `.npz` model format with `format_version`

## Installation

```bash
cd US-Metrics
pip install -e .
```

For development tests:

```bash
pip install -e ".[dev]"
pytest -q
```

## Pretrained Weights

Weights are not stored in this repository.

- TinyUSFM pretrained checkpoint (external link): [Google Drive](https://drive.google.com/file/d/15R3hnH0ILO39rE1gs-UgJonRqbaYTSRB/view?usp=sharing)
- TinyUSFM codebase (official repository): [MacDunno/TinyUSFM](https://github.com/MacDunno/TinyUSFM)

## Optional: Pre-fitted NRQ Models

If your NRQ organ models are small, you can open-source them directly in this repository (for example under `weights/nrq/`):

```text
weights/
  nrq/
    Thyroid.npz
    Cardiac.npz
```

Then score directly with:

```bash
us-metrics nrq \
  --img /path/to/image.png \
  --ckpt /path/to/TinyUSFM.pth \
  --models ./weights/nrq
```

## Data Layout for `fit-nrq`

`clean_data_root` is expected to be organized by organ:

```text
clean_data_root/
  Thyroid/
    xxx.png
    ...
  Cardiac/
    yyy.png
    ...
```

Images are discovered recursively under each organ folder.

## Quick Start

### 1) TinyUSFM-uLPIPS

```bash
us-metrics ulipips \
  --ref /path/to/ref.png \
  --img /path/to/test.png \
  --ckpt /path/to/TinyUSFM.pth
```

### 2) Fit TinyUSFM-NRQ Organ Models

```bash
us-metrics fit-nrq \
  --clean-root /path/to/clean_data_root \
  --ckpt /path/to/TinyUSFM.pth \
  --out /path/to/nrq_models
```

### 3) Score TinyUSFM-NRQ

Organ-mixture mode (deployable default):

```bash
us-metrics nrq \
  --img /path/to/image.png \
  --ckpt /path/to/TinyUSFM.pth \
  --models /path/to/nrq_models
```

Oracle organ mode:

```bash
us-metrics nrq \
  --img /path/to/image.png \
  --ckpt /path/to/TinyUSFM.pth \
  --models /path/to/nrq_models \
  --organ Thyroid
```

## Defaults (Paper-Aligned)

- Layers: `L = {3, 5, 7, 11}` (block idx `{2, 4, 6, 10}`)
- uLPIPS: `radius = 3`, `tau = 20`
- NRQ: `patch = 224`, `stride = 112`, `PCA d = 128`, `GMM K = 4`, `topk_frac = 0.15`

## Citation

If you use this repository, these metrics, or the provided NRQ models, please cite the paper:

```bibtex
@inproceedings{your_key_2026_usmetrics,
  title     = {Defining Robust Ultrasound Quality Metrics via an Ultrasound Foundation Model},
  author    = {<Author List>},
  booktitle = {MICCAI},
  year      = {2026}
}
```

Please replace `<Author List>` and the BibTeX key with your final publication metadata.

Please also cite the related backbone and prior works:

```bibtex
@article{tinyusfm,
  author={Ma, Chen and Jiao, Jing and Liang, Shuyu and Fu, Junhu and Wang, Qin and Li, Zeju and Wang, Yuanyuan and Guo, Yi},
  journal={IEEE Journal of Biomedical and Health Informatics},
  title={TinyUSFM: Towards Compact and Efficient Ultrasound Foundation Models},
  year={2026},
  pages={1-14},
  doi={10.1109/JBHI.2026.3678309}
}

@article{usfm,
  title={Usfm: A universal ultrasound foundation model generalized to tasks and organs towards label efficient image analysis},
  author={Jiao, Jing and Zhou, Jin and Li, Xiaokang and Xia, Menghua and Huang, Yi and Huang, Lihong and Wang, Na and Zhang, Xiaofan and Zhou, Shichong and Wang, Yuanyuan and others},
  journal={Medical image analysis},
  volume={96},
  pages={103202},
  year={2024},
  publisher={Elsevier}
}

@incollection{iugc,
  title={Unlabeled Data-Driven Fetal Landmark Detection in Intrapartum Ultrasound},
  author={Ma, Chen and Li, Yunshu and Guo, Bowen and Jiao, Jing and Huang, Yi and Wang, Yuanyuan and Guo, Yi},
  booktitle={Intrapartum Ultrasound Grand Challenge},
  pages={14--23},
  year={2025},
  publisher={Springer}
}
```

## License

Apache-2.0.
