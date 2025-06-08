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
from . import CPP_LOCK

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
        with CPP_LOCK:
            return _ccore.interp_qbs_with_gradients(
                x,
                y,
                bcoef_cpp,
                mask_cpp,
                off_x,
                off_y,
                border,
            )

elif _ccore:

    def _interp_val_grad(x, y, bcoef_cpp, mask_cpp, off_x, off_y, border):
        eps = 1e-3
        with CPP_LOCK:
            val = _ccore.interp_qbs(
                x, y, bcoef_cpp, mask_cpp, off_x, off_y, border
            )[0]
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
    valid_new = (
        (ys_new >= seed_window_half_width)
        & (ys_new < h_r - seed_window_half_width)
        & (xs_new >= seed_window_half_width)
        & (xs_new < w_r - seed_window_half_width)
    )
    ys_new = ys_new[valid_new]
    xs_new = xs_new[valid_new]

    ys_old, xs_old = np.nonzero(lag_mask)
    regions_old = [
        lagrangian_ncorr_roi_obj._find_region_containing_point(x * step, y * step)
        for y, x in zip(ys_old, xs_old)
    ]
    region_arr = np.array([r if r is not None else -1 for r in regions_old], dtype=int)

    u_old = lagrangian_u_pixels_np[ys_old, xs_old]
    v_old = lagrangian_v_pixels_np[ys_old, xs_old]
    x_map = xs_old + u_old / step
    y_map = ys_old + v_old / step

    if xs_new.size == 0 or xs_old.size == 0:
        return convert_seed_info

    dx = xs_new[:, None] - x_map[None, :]
    dy = ys_new[:, None] - y_map[None, :]
    dist2 = dx * dx + dy * dy
    best_indices = np.argmin(dist2, axis=1)
    best_dists2 = dist2[np.arange(dist2.shape[0]), best_indices]

    for x_new_r, y_new_r, best_idx, best_dist2 in zip(
        xs_new, ys_new, best_indices, best_dists2
    ):
        if not isfinite(best_dist2):
            continue
        region_idx = region_arr[best_idx]
        if region_idx == -1:
            continue
        x_old_r = xs_old[best_idx]
        y_old_r = ys_old[best_idx]

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
            with CPP_LOCK:
                u_fin = _ccore.interp_qbs(
                    x_old_f,
                    y_old_f,
                    u_bcoef,
                    mask_cpp,
                    off_x,
                    off_y,
                    border_interp_cpp,
                )[0]
                v_fin = _ccore.interp_qbs(
                    x_old_f,
                    y_old_f,
                    v_bcoef,
                    mask_cpp,
                    off_x,
                    off_y,
                    border_interp_cpp,
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

