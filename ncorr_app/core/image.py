"""
`NcorrImage` – Python façade over the C++ ncorr_class_img.
"""

from __future__ import annotations

import cv2
import numpy as np

from importlib import import_module as _imp_mod
from pathlib import Path
from typing import Any

from . import utils

_ccore = _imp_mod("ncorr_app._ext._ncorr_cpp_core")


__all__ = ["NcorrImage"]


class NcorrImage:
    """
    Grayscale image container + lazy B-spline coefficient generation.

    Parameters
    ----------
    source : str | numpy.ndarray | None
        • str → path on disk (loaded via OpenCV)\n
        • ndarray → already-loaded image data (H×W[,3]) in any dtype
    name   : str
        Human-readable label (defaults to filename stem).
    path   : str
        Original disk location (if any).
    """

    # --------------------------------------------------------------------- #
    # Construction                                                          #
    # --------------------------------------------------------------------- #
    def __init__(self,
                 source: str | np.ndarray | None,
                 *,
                 name: str = "",
                 path: str = ""):

        if isinstance(source, (str, Path)):
            img_path = Path(source)
            if not img_path.is_file():
                raise FileNotFoundError(source)
            gs = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE | cv2.IMREAD_ANYDEPTH)
            if gs is None:
                raise IOError(f"failed to load image: {img_path}")
            name = name or img_path.stem
            path = path or str(img_path)
            imgtype = "file"
        elif isinstance(source, np.ndarray):
            utils.is_proper_image_format(source, "source image")
            if source.ndim == 3:                        # convert to gray
                gs = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY)
            else:
                gs = source.copy()
            name = name or "<numpy>"
            imgtype = "numpy_array"
        elif source is None:
            # Lazy placeholder – image data will be set later
            gs = np.zeros((1, 1), dtype=np.float64)
            imgtype = "placeholder"
            name = name or "<empty>"
        else:
            raise TypeError("source must be filepath or ndarray")

        # Normalise → float64 in [0,1]
        if gs.dtype == np.uint8:
            gs = gs.astype(np.float64) / 255.0
        elif gs.dtype == np.uint16:
            gs = gs.astype(np.float64) / 65535.0
        else:
            gs = gs.astype(np.float64, copy=False)
            if gs.max() > 1.5:
                gs = gs / 255.0

        self.gs_data: np.ndarray = np.ascontiguousarray(gs, dtype=np.float64)
        self.name: str  = name
        self.path: str  = str(path)
        self.height, self.width = self.gs_data.shape
        self.type: str  = imgtype

        self._bcoef_data: np.ndarray | None = None
        self.border_bcoef: int = 20

        self.max_gs: float = float(self.gs_data.max())
        self.min_gs: float = float(self.gs_data.min())

    # ------------------------------------------------------------------ API
    def get_gs(self) -> np.ndarray:
        """Return the grayscale image as float64 in [0,1]."""
        return self.gs_data

    def get_img_display(self) -> np.ndarray:
        """
        8-bit 3-channel BGR copy for GUI display.
        """
        disp = np.clip(self.gs_data * 255.0, 0, 255).astype(np.uint8)
        return cv2.cvtColor(disp, cv2.COLOR_GRAY2BGR)

    # ------------------------------------------------------------------ B-coef
    def get_bcoef(self) -> np.ndarray:
        """
        Lazy compute + cache the cubic B-spline coefficients required by
        the DIC interpolator. Uses the *same* padding strategy as MATLAB
        Ncorr (mirror padding of size `border_bcoef`).
        """
        if self._bcoef_data is None:
            pad = self.border_bcoef
            padded = np.pad(self.gs_data,
                            pad_width=((pad, pad), (pad, pad)),
                            mode="edge")
            self._bcoef_data = utils._form_bcoef_from_gs_array(padded)
        return self._bcoef_data

    # ------------------------------------------------------------------ Reduce
    def reduce(self, spacing: int) -> "NcorrImage":
        """
        Down-sample the image by `(spacing + 1)` with Gaussian pre-blur to
        avoid aliasing.  Mirrors `ncorr_class_img::reduce`.
        """
        if spacing <= 0:
            raise ValueError("spacing must be ≥1 for reduction")

        # σ chosen so that blur ≈ MATLAB imresize bicubic
        sigma = 0.4 * (spacing + 1)
        blurred = cv2.GaussianBlur(self.gs_data,
                                   ksize=(0, 0),
                                   sigmaX=sigma,
                                   borderType=cv2.BORDER_REPLICATE)
        reduced = blurred[::spacing + 1, ::spacing + 1].copy()

        new_img = NcorrImage(reduced,
                             name=f"{self.name}_r{spacing}",
                             path=self.path)
        new_img.type = "reduced"
        new_img.border_bcoef = self.border_bcoef    # copy setting
        return new_img

    # ------------------------------------------------------------------ C++ interop
    def to_cpp_repr(self):
        """
        Return a **CppNcorrClassImg** instance populated from this object.
        """
        cpp_img = _ccore.CppNcorrClassImg()
        cpp_img.type = self.type
        cpp_img.max_gs = self.max_gs
        cpp_img.border_bcoef = self.border_bcoef

        gs_cpp = utils._numpy_to_cpp_double_array(self.gs_data)
        cpp_img.gs = gs_cpp

        if self._bcoef_data is not None:
            bcoef_cpp = utils._numpy_to_cpp_double_array(self._bcoef_data)
            cpp_img.bcoef = bcoef_cpp
        else:
            # Leave default-constructed (empty) array; C++ can compute later
            pass
        return cpp_img
