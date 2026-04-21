<div align="center">

# US-Metrics: TinyUSFM-uLPIPS and TinyUSFM-NRQ for Ultrasound Quality Assessment

[![Framework](https://img.shields.io/badge/Framework-PyTorch-blue)](#)
[![License](https://img.shields.io/badge/License-Apache--2.0-orange)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB)](#)
[![TinyUSFM](https://img.shields.io/badge/Backbone-TinyUSFM-0A7EA4)](https://github.com/MacDunno/TinyUSFM)

Official open-source implementation of ultrasound-native quality metrics based on TinyUSFM:
**TinyUSFM-uLPIPS (FR)** and **TinyUSFM-NRQ (NR)**.

</div>

---

## Framework

> Put your paper framework figure at: `assets/us-metrics-framework.png`

<p align="center">
  <img src="assets/us-metrics-framework.png" alt="US-Metrics Framework" width="100%" />
</p>

---

## Highlights

- Ultrasound-native **full-reference** quality metric: `TinyUSFM-uLPIPS`
- Ultrasound-native **no-reference** quality metric: `TinyUSFM-NRQ`
- Shared TinyUSFM representation space for FR/NR metrics
- Standalone package layout with reproducible defaults and CLI tools
- Clean separation from private experiment pipelines and result artifacts

---

## Installation

```bash
cd US-Metrics
pip install -e .
```

Dev/test install:

```bash
pip install -e ".[dev]"
pytest -q
```

---

## Quick Start

### 1) TinyUSFM-uLPIPS (FR)

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

### 3) TinyUSFM-NRQ (NR) Scoring

Organ-mixture mode:

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

---

## Defaults (Paper-Aligned)

- Layers: `L = {3, 5, 7, 11}` (block idx `{2, 4, 6, 10}`)
- uLPIPS: `radius = 3`, `tau = 20`
- NRQ: `patch = 224`, `stride = 112`, `PCA d = 128`, `GMM K = 4`, `topk_frac = 0.15`

---

## Backbone and Weights

- TinyUSFM codebase (official): [https://github.com/MacDunno/TinyUSFM](https://github.com/MacDunno/TinyUSFM)
- TinyUSFM checkpoint (external): [Google Drive](https://drive.google.com/file/d/15R3hnH0ILO39rE1gs-UgJonRqbaYTSRB/view?usp=sharing)
- Optional pre-fitted NRQ models can be released in `weights/nrq/*.npz`

---

## Citation

If you use this repository, metrics, or released NRQ models, please cite:

```bibtex
@inproceedings{your_key_2026_usmetrics,
  title     = {Defining Robust Ultrasound Quality Metrics via an Ultrasound Foundation Model},
  author    = {<Author List>},
  booktitle = {MICCAI},
  year      = {2026}
}
```

```bibtex
@article{tinyusfm,
  author={Ma, Chen and Jiao, Jing and Liang, Shuyu and Fu, Junhu and Wang, Qin and Li, Zeju and Wang, Yuanyuan and Guo, Yi},
  journal={IEEE Journal of Biomedical and Health Informatics},
  title={TinyUSFM: Towards Compact and Efficient Ultrasound Foundation Models},
  year={2026},
  pages={1-14},
  doi={10.1109/JBHI.2026.3678309}
}
```

```bibtex
@article{usfm,
  title={Usfm: A universal ultrasound foundation model generalized to tasks and organs towards label efficient image analysis},
  author={Jiao, Jing and Zhou, Jin and Li, Xiaokang and Xia, Menghua and Huang, Yi and Huang, Lihong and Wang, Na and Zhang, Xiaofan and Zhou, Shichong and Wang, Yuanyuan and others},
  journal={Medical image analysis},
  volume={96},
  pages={103202},
  year={2024},
  publisher={Elsevier}
}
```

```bibtex
@incollection{iugc,
  title={Unlabeled Data-Driven Fetal Landmark Detection in Intrapartum Ultrasound},
  author={Ma, Chen and Li, Yunshu and Guo, Bowen and Jiao, Jing and Huang, Yi and Wang, Yuanyuan and Guo, Yi},
  booktitle={Intrapartum Ultrasound Grand Challenge},
  pages={14--23},
  year={2025},
  publisher={Springer}
}
```

---

## License

Apache-2.0. See [LICENSE](LICENSE).
