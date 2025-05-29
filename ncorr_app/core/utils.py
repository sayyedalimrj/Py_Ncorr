"""
Generic helpers that do **not** belong to a specific Ncorr object.
Conversion utilities live here to keep the main classes tidy.
"""

from __future__ import annotations

import math
import os
import re
import warnings
from pathlib import Path
from typing import Iterable, Sequence

import cv2
import numpy as np
import scipy.fft as _fft

# C++ extensions – import lazily to reduce import-time cost
from importlib import import_module as _imp_mod

_ccore = _imp_mod("ncorr_app._ext._ncorr_cpp_core")
_calgs = _imp_mod("ncorr_app._ext._ncorr_cpp_algs")


# ─────────────────────────────────────────────────────────────────────────────
# Sanity-check helpers
# ─────────────────────────────────────────────────────────────────────────────
def is_real_in_bounds(val: float,
                      lo: float,
                      hi: float,
                      name: str | None = None) -> bool:
    if not (lo <= val <= hi):
        msg = f"{name or 'value'} ({val}) must be in the closed interval [{lo}, {hi}]"
        raise ValueError(msg)
    return True


def is_int_in_bounds(val: int,
                     lo: int,
                     hi: int,
                     name: str | None = None) -> bool:
    if not (isinstance(val, (int, np.integer)) and lo <= val <= hi):
        msg = f"{name or 'integer'} ({val}) must be in [{lo}, {hi}]"
        raise ValueError(msg)
    return True


# ─────────────────────────────────────────────────────────────────────────────
# Image helpers
# ─────────────────────────────────────────────────────────────────────────────
def is_proper_image_format(arr: np.ndarray, label: str = "image") -> bool:
    """
    Confirm *arr* is a single-channel (HxW) or three-channel (HxWx3)
    NumPy array of uint8/uint16/float32/float64.
    """
    if arr.ndim not in (2, 3):
        raise ValueError(f"{label} must be 2-D (gray) or 3-D (colour)")
    if arr.ndim == 3 and arr.shape[2] != 3:
        raise ValueError(f"{label}: if 3-D, last dim must be 3 channels")
    if arr.dtype not in (np.uint8, np.uint16, np.float32, np.float64):
        raise TypeError(f"{label} dtype must be uint8/16 or float32/64")
    return True


def _natural_key(path: os.PathLike | str) -> tuple:
    """
    Natural sort helper so `img_10.png` comes after `img_2.png`.
    """
    fname = Path(path).stem
    return tuple(int(tok) if tok.isdigit() else tok.lower()
                 for tok in re.split(r"(\d+)", fname))


def load_images_from_paths(paths: Sequence[str | os.PathLike],
                           lazy_load: bool = False):
    """
    Return a list of *NcorrImage* objects from arbitrary file paths.
    """
    from .image import NcorrImage  # avoid circular import

    sorted_paths = sorted(paths, key=_natural_key)
    imgs = []
    for p in sorted_paths:
        if not lazy_load:
            imgs.append(NcorrImage(p))
        else:      # placeholder for future mmap / lazy strategy
            imgs.append(NcorrImage(None, name=Path(p).name, path=str(p)))
    return imgs


def load_saved_ncorr_image(d: dict):
    """
    Re-hydrate an Ncorr image saved via Ncorr’s session-file mechanism.
    """
    from .image import NcorrImage

    required = {"gs_data", "name", "path", "type"}
    if not required.issubset(d):
        raise KeyError(f"saved image is missing keys: {required - set(d)}")

    img = NcorrImage(d["gs_data"],
                     name=d["name"],
                     path=d.get("path", ""))
    img.type = d["type"]
    if "bcoef_data" in d:
        img._bcoef_data = d["bcoef_data"]
    return img


# ─────────────────────────────────────────────────────────────────────────────
# Region helpers
# ─────────────────────────────────────────────────────────────────────────────
def form_region_constraint_func(ncorr_roi,
                                region_idx: int = 0):
    """
    Return a closure that projects incoming coordinates onto the *nearest*
    valid pixel inside *ncorr_roi.regions_py[region_idx]*.

    Port of `ncorr_util_formregionconstraint.m`.
    """
    region = ncorr_roi.regions_py[region_idx]

    # Pre-compute a fast lookup table (boolean mask) if needed
    mask = ncorr_roi._get_region_mask_np(region_idx)

    h, w = mask.shape

    def _constraint(pos_xy: tuple[int, int]) -> tuple[int, int]:
        x, y = pos_xy
        x = int(round(x)); y = int(round(y))
        x = max(0, min(w - 1, x))
        y = max(0, min(h - 1, y))
        if mask[y, x]:
            return x, y

        # naïve breadth-first search for nearest True
        from collections import deque
        q = deque([(x, y)])
        visited = set(q)
        while q:
            cx, cy = q.popleft()
            for nx, ny in ((cx + 1, cy), (cx - 1, cy), (cx, cy + 1), (cx, cy - 1)):
                if (0 <= nx < w and 0 <= ny < h and (nx, ny) not in visited):
                    visited.add((nx, ny))
                    if mask[ny, nx]:
                        return nx, ny
                    q.append((nx, ny))
        # fallback: original point
        return x, y

    return _constraint


# ─────────────────────────────────────────────────────────────────────────────
# B-spline coefficient helpers
# ─────────────────────────────────────────────────────────────────────────────
def _form_bcoef_from_gs_array(gs: np.ndarray) -> np.ndarray:
    """
    Compute 2-D cubic B-spline coefficients for *gs*.

    This is a **pure-Python/NumPy** re-implementation of
    `ncorr_class_img::form_bcoef`.  If a faster C++ path is available,
    monkey-patch this at import-time.
    """
    gs = np.asarray(gs, dtype=np.float64)

    # Step 1: normalise to [0,1] if necessary
    if gs.max() > 1.5:
        gs = gs / 255.0

    # Step 2: forward FFT → multiply by kernel → inverse FFT
    h, w = gs.shape
    # Frequency response of the cubic B-spline:
    def _H(k):          # Eqn in Unser ’93
        return (1/6) * (2 + 2*np.cos(k) + np.cos(2*k))

    ky = 2*np.pi*np.fft.fftfreq(h)
    kx = 2*np.pi*np.fft.fftfreq(w)

    Hy = _H(ky)[:, None]
    Hx = _H(kx)[None, :]

    G_hat = _fft.fft2(gs)
    B_hat = G_hat / (Hy * Hx + 1e-12)

    bcoef = np.real(_fft.ifft2(B_hat))
    return bcoef.astype(np.float64, copy=False)


# ─────────────────────────────────────────────────────────────────────────────
# C++ array ↔ NumPy conversion helpers
# ─────────────────────────────────────────────────────────────────────────────
def _numpy_to_cpp_logical_array(arr: np.ndarray,
                                tgt: "_ccore.CppClassLogicalArray | None" = None):
    arr = np.ascontiguousarray(arr, dtype=np.bool_)
    h, w = arr.shape
    tgt = tgt or _ccore.CppClassLogicalArray()
    tgt.alloc(int(h), int(w))
    tgt.set_value_numpy(arr)
    return tgt


def _cpp_logical_array_to_numpy(src: "_ccore.CppClassLogicalArray") -> np.ndarray:
    return src.get_value_numpy().astype(bool, copy=False)


def _numpy_to_cpp_double_array(arr: np.ndarray,
                               tgt: "_ccore.CppClassDoubleArray | None" = None):
    arr = np.ascontiguousarray(arr, dtype=np.float64)
    h, w = arr.shape
    tgt = tgt or _ccore.CppClassDoubleArray()
    tgt.alloc(int(h), int(w))
    tgt.set_value_numpy(arr)
    return tgt


def _cpp_double_array_to_numpy(src: "_ccore.CppClassDoubleArray") -> np.ndarray:
    return src.get_value_numpy().astype(np.float64, copy=False)


def _numpy_to_cpp_integer_array(arr: np.ndarray,
                                tgt: "_ccore.CppClassIntegerArray | None" = None):
    arr = np.ascontiguousarray(arr, dtype=np.int32)
    h, w = arr.shape
    tgt = tgt or _ccore.CppClassIntegerArray()
    tgt.alloc(int(h), int(w))
    tgt.set_value_numpy(arr)
    return tgt


def _cpp_integer_array_to_numpy(src: "_ccore.CppClassIntegerArray") -> np.ndarray:
    return src.get_value_numpy().astype(np.int32, copy=False)
