"""Image loading, normalization and patch extraction."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Tuple, Union

import cv2
import numpy as np
import torch

ArrayLikeImage = Union[str, Path, np.ndarray]


def load_image_gray_u8(img: ArrayLikeImage) -> np.ndarray:
    """Load image as grayscale uint8 from path or ndarray."""
    if isinstance(img, (str, Path)):
        arr = cv2.imread(str(img), cv2.IMREAD_UNCHANGED)
        if arr is None:
            raise FileNotFoundError(str(img))
    elif isinstance(img, np.ndarray):
        arr = img
    else:
        raise TypeError(f"Unsupported image type: {type(img)}")

    if arr.ndim == 3:
        arr = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return arr


def to_float01_3ch(gray_u8: np.ndarray) -> np.ndarray:
    x = gray_u8.astype(np.float32) / 255.0
    return np.stack([x, x, x], axis=-1)


def to_torch_nchw_224(gray_u8: np.ndarray, device: str = "cpu") -> torch.Tensor:
    """(H,W) uint8 -> (1,3,224,224) normalized to [-1,1]."""
    x3 = to_float01_3ch(gray_u8)
    x3 = cv2.resize(x3, (224, 224), interpolation=cv2.INTER_AREA)
    t = torch.from_numpy(x3).permute(2, 0, 1).unsqueeze(0).float()
    t = (t - 0.5) / 0.5
    return t.to(device)


def ensure_min_size_reflect(img_u8: np.ndarray, min_hw: int) -> np.ndarray:
    h, w = img_u8.shape
    pad_h = max(0, min_hw - h)
    pad_w = max(0, min_hw - w)
    if pad_h == 0 and pad_w == 0:
        return img_u8
    top = pad_h // 2
    bottom = pad_h - top
    left = pad_w // 2
    right = pad_w - left
    return cv2.copyMakeBorder(
        img_u8, top, bottom, left, right, borderType=cv2.BORDER_REFLECT101
    )


def patch_grid_coords(h: int, w: int, patch: int, stride: int) -> List[Tuple[int, int]]:
    ys = list(range(0, max(1, h - patch + 1), stride))
    xs = list(range(0, max(1, w - patch + 1), stride))
    if not ys:
        ys = [0]
    if not xs:
        xs = [0]

    coords = [(y, x) for y in ys for x in xs]
    if ys[-1] != h - patch:
        coords += [(h - patch, x) for x in xs]
    if xs[-1] != w - patch:
        coords += [(y, w - patch) for y in ys]
    if (h - patch, w - patch) not in coords:
        coords.append((h - patch, w - patch))

    uniq: List[Tuple[int, int]] = []
    seen = set()
    for c in coords:
        if c not in seen:
            uniq.append(c)
            seen.add(c)
    return uniq


def extract_patches_u8(
    img_u8: np.ndarray,
    patch: int = 224,
    stride: int = 112,
    max_patches: int = 0,
    seed: int = 0,
) -> List[np.ndarray]:
    """Extract overlapping patches with optional deterministic subsampling."""
    img_u8 = ensure_min_size_reflect(img_u8, patch)
    h, w = img_u8.shape
    coords = patch_grid_coords(h, w, patch, stride)
    if max_patches > 0 and len(coords) > max_patches:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(coords), size=max_patches, replace=False)
        coords = [coords[i] for i in idx]
    return [img_u8[y : y + patch, x : x + patch] for (y, x) in coords]


def patches_to_torch(
    patches_u8: Iterable[np.ndarray],
    device: str = "cpu",
) -> torch.Tensor:
    patches_u8 = list(patches_u8)
    if not patches_u8:
        return torch.empty((0, 3, 224, 224), device=device)
    arr = np.stack([p.astype(np.float32) / 255.0 for p in patches_u8], axis=0)
    arr = np.stack([arr, arr, arr], axis=1)
    t = torch.from_numpy(arr).float()
    t = (t - 0.5) / 0.5
    return t.to(device)

