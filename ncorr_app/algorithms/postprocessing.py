"""
High-level post-processing – displacement conversion, formatting and strain.

This file now includes a full Python implementation of
_determine_convert_seeds_py that mirrors ncorr_alg_convertseeds.m.
"""

from __future__ import annotations

import numpy as np
from importlib import import_module as _imp_mod
from math import isfinite

from ncorr_app.core.utils import (
    _numpy_to_cpp_logical_array,
)
from ncorr_app.core.datatypes import (
    BORDER_INTERP_DEFAULT,
    OutputState,
)

# Cut-offs for convert-seed search
DISTANCE_CUTOFF_CONVERT_SEEDS = 5.0e-3
GRAD_NORM_CUTOFF_CONVERT_SEEDS = 1.0e-5
MAX_ITER_CONVERT_SEEDS = 10

# C++ helpers
try:
    _ccore = _imp_mod("ncorr_app._ext._ncorr_cpp_core")
except ModuleNotFoundError:  # pragma: no cover
    _ccore = None

# --------------------------------------------------------------------- #
# util: interpolation with gradients (fallback finite-diff if C++ lacks)
# --------------------------------------------------------------------- #
if _ccore and hasattr(_ccore, "interp_qbs_with_gradients"):

    def _interp_val_grad(x, y, bcoef_cpp, mask_cpp, off_x, off_y, border):
        return _ccore.interp_qbs_with_gradients(
            x, y, bcoef_cpp, mask_cpp, off_x, off_y, border
        )

elif _ccore:

    def _interp_val_grad(x, y, bcoef_cpp, mask_cpp, off_x, off_y, border):
        eps = 1e-3
        val = _ccore.interp_qbs(x, y, bcoef_cpp, mask_cpp, off_x, off_y, border)[0]
        val_x_plus = _ccore.interp_qbs(
            x + eps, y, bcoef_cpp, mask_cpp, off_x, off_y, border
        )[0]
        val_x_minus = _ccore.interp_qbs(
            x - eps, y, bcoef_cpp, mask_cpp, off_x, off_y, border
        )[0]
        val_y_plus = _ccore.interp_qbs(
            x, y + eps, bcoef_cpp, mask_cpp, off_x, off_y, border
        )[0]
        val_y_minus = _ccore.interp_qbs(
            x, y - eps, bcoef_cpp, mask_cpp, off_x, off_y, border
        )[0]
        d_dx = (val_x_plus - val_x_minus) / (2 * eps)
        d_dy = (val_y_plus - val_y_minus) / (2 * eps)
        return val, d_dx, d_dy
else:

    def _interp_val_grad(x, y, bcoef_cpp, mask_cpp, off_x, off_y, border):
        raise RuntimeError("C++ core module not available")


# --------------------------------------------------------------------- #
#  MAIN: convert-seed search                                            #
# --------------------------------------------------------------------- #
def _determine_convert_seeds_py(
    lagrangian_u_pixels_np: np.ndarray,
    lagrangian_v_pixels_np: np.ndarray,
    list_of_u_bcoef_cpp_region_objs,
    list_of_v_bcoef_cpp_region_objs,
    lagrangian_ncorr_roi_obj,
    new_config_roi_reduced_obj,
    seed_window_half_width: int,
    spacing: int,
    border_interp_cpp: int,
):
    """Pure-Python analogue of ncorr_alg_convertseeds.m"""
    step = spacing + 1
    lag_mask = lagrangian_ncorr_roi_obj.get_mask_np()
    new_mask = new_config_roi_reduced_obj.get_mask_np()
    h_r, w_r = new_mask.shape

    # Pre-compute region helpers for interpolation
    region_helpers = []
    for ridx, reg in enumerate(lagrangian_ncorr_roi_obj.get_regions_py()):
        mask_cpp = _numpy_to_cpp_logical_array(
            lagrangian_ncorr_roi_obj._get_region_mask_np(ridx)
        )
        off_x = int(reg.leftbound / step)
        off_y = int(reg.upperbound / step)
        region_helpers.append((mask_cpp, off_x, off_y))

    convert_seed_info = []

    ys_new, xs_new = np.nonzero(new_mask)
    for y_new_r, x_new_r in zip(ys_new, xs_new):
        if (
            y_new_r < seed_window_half_width
            or y_new_r >= h_r - seed_window_half_width
            or x_new_r < seed_window_half_width
            or x_new_r >= w_r - seed_window_half_width
        ):
            continue

        # ---------------- coarse search ---------------------------------
        best_dist2 = np.inf
        best_pt = None  # (x_old_r, y_old_r, region_idx)
        ys_old, xs_old = np.nonzero(lag_mask)
        for y_old_r, x_old_r in zip(ys_old, xs_old):
            u_pix = lagrangian_u_pixels_np[y_old_r, x_old_r]
            v_pix = lagrangian_v_pixels_np[y_old_r, x_old_r]
            x_map = x_old_r + u_pix / step
            y_map = y_old_r + v_pix / step
            d2 = (x_new_r - x_map) ** 2 + (y_new_r - y_map) ** 2
            if d2 < best_dist2:
                ridx = lagrangian_ncorr_roi_obj._find_region_containing_point(
                    x_old_r * step, y_old_r * step
                )
                best_dist2 = d2
                best_pt = (x_old_r, y_old_r, ridx)

        if best_pt is None or not isfinite(best_dist2):
            continue

        x_old_r, y_old_r, region_idx = best_pt
        if region_idx is None:
            continue

        # ---------------- Gauss–Newton refinement -----------------------
        x_old_f, y_old_f = float(x_old_r), float(y_old_r)
        mask_cpp, off_x, off_y = region_helpers[region_idx]
        u_bcoef = list_of_u_bcoef_cpp_region_objs[region_idx]
        v_bcoef = list_of_v_bcoef_cpp_region_objs[region_idx]

        for _ in range(MAX_ITER_CONVERT_SEEDS):
            try:
                u_val, du_dx, du_dy = _interp_val_grad(
                    x_old_f, y_old_f, u_bcoef, mask_cpp, off_x, off_y, border_interp_cpp
                )
                v_val, dv_dx, dv_dy = _interp_val_grad(
                    x_old_f, y_old_f, v_bcoef, mask_cpp, off_x, off_y, border_interp_cpp
                )
            except RuntimeError:
                break

            f1 = x_new_r - (x_old_f + u_val / step)
            f2 = y_new_r - (y_old_f + v_val / step)
            J = np.array(
                [
                    [-(1 + du_dx / step), -(du_dy / step)],
                    [-(dv_dx / step), -(1 + dv_dy / step)],
                ],
                dtype=float,
            )
            try:
                delta = -np.linalg.solve(J, np.array([f1, f2]))
            except np.linalg.LinAlgError:
                break

            x_old_f += delta[0]
            y_old_f += delta[1]

            if np.linalg.norm(delta) < GRAD_NORM_CUTOFF_CONVERT_SEEDS:
                break

        # Interpolate final U,V for acceptance
        try:
            u_fin = _ccore.interp_qbs(
                x_old_f, y_old_f, u_bcoef, mask_cpp, off_x, off_y, border_interp_cpp
            )[0]
            v_fin = _ccore.interp_qbs(
                x_old_f, y_old_f, v_bcoef, mask_cpp, off_x, off_y, border_interp_cpp
            )[0]
        except RuntimeError:
            continue

        dist = np.hypot(
            x_new_r - (x_old_f + u_fin / step),
            y_new_r - (y_old_f + v_fin / step),
        )
        if dist > DISTANCE_CUTOFF_CONVERT_SEEDS:
            continue

        # ensure old point inside mask
        if (
            0 <= int(round(y_old_f)) < lag_mask.shape[0]
            and 0 <= int(round(x_old_f)) < lag_mask.shape[1]
            and lag_mask[int(round(y_old_f)), int(round(x_old_f))]
        ):
            convert_seed_info.append(
                {
                    "paramvector": [
                        x_new_r,
                        y_new_r,
                        x_old_f,
                        y_old_f,
                        u_fin,
                        v_fin,
                        dist,
                    ],
                    "num_region_new": new_config_roi_reduced_obj._find_region_containing_point(
                        x_new_r * step, y_new_r * step
                    ),
                    "num_region_old": region_idx,
                }
            )

    return convert_seed_info


def format_displacement_fields(*_args, **_kwargs):
    """Placeholder for displacement formatting."""
    raise NotImplementedError


def calculate_strain_fields(*_args, **_kwargs):
    """Placeholder for strain calculation."""
    raise NotImplementedError

