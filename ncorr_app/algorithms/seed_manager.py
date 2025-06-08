"""
Seed-related helpers for the Ncorr DIC workflow.

The original MATLAB logic scattered seed placement / propagation across
multiple m-files; here everything is consolidated so the high-level
DIC orchestrator can call a single module.
"""
from __future__ import annotations

from collections import defaultdict
from importlib import import_module as _imp_mod
from typing import Iterable, Sequence

import numpy as np

from ncorr_app.core import datatypes, utils

try:
    _ccore = _imp_mod("ncorr_app._ext._ncorr_cpp_core")
    _calgs = _imp_mod("ncorr_app._ext._ncorr_cpp_algs")
except ModuleNotFoundError:  # pragma: no cover - allow import w/o C++ ext
    _ccore = None
    _calgs = None

# --------------------------------------------------------------------------- #
#  Public helpers                                                             #
# --------------------------------------------------------------------------- #
def _determine_initial_seeds_and_diagram(ref_roi,
                                         dic_params: dict):
    """
    Choose *total_threads* well-spaced seeds inside each region of *ref_roi*
    (original pixel co-ordinates) and build the reduced-resolution
    “thread-diagram” used by Ncorr’s region-growing DIC.

    Returns
    -------
    seeds_xy  : list[(x,y)]              # **pixels in original image**
    diagramNP : ndarray[int32]           # reduced grid, region id @ seed → idx
    """
    total_threads: int = dic_params.get("total_threads", 8)
    spacing: int       = dic_params.get("spacing", 1)

    seeds_xy: list[tuple[int, int]] = []

    # ---------- select seeds per region ---------------------------------- #
    regions = [r for r in ref_roi.get_regions_py() if r.totalpoints > 0]
    if not regions:
        raise RuntimeError("ROI has no valid regions")

    per_region = max(1, int(round(total_threads / len(regions))))

    for reg in regions:
        # Build per-region mask once
        mask = ref_roi._get_region_mask_np(regions.index(reg))

        # Simple heuristic: pick points on a coarse Cartesian grid
        h, w = mask.shape
        grid = np.mgrid[0:w:(spacing + 1), 0:h:(spacing + 1)].reshape(2, -1).T
        # keep only those strictly inside the mask & not too close to border
        in_region = [tuple(pt) for pt in grid if mask[pt[1], pt[0]]]
        if not in_region:       # fallback: region centroid
            ys, xs = np.nonzero(mask)
            cx = int(xs.mean()); cy = int(ys.mean())
            in_region = [(cx, cy)]

        seeds_xy.extend(in_region[:per_region])

    # ---------- convert to reduced grid & form diagram ------------------- #
    if spacing < 0:
        raise ValueError("spacing must be ≥0")

    h_red = ref_roi.mask_np.shape[0] // (spacing + 1)
    w_red = ref_roi.mask_np.shape[1] // (spacing + 1)

    # map → reduced co-ords
    seed_reduced_lin = []
    for x, y in seeds_xy:
        xr = x // (spacing + 1)
        yr = y // (spacing + 1)
        seed_reduced_lin.append(int(yr * w_red + xr))

    seed_reduced_lin_np = np.asarray(seed_reduced_lin, dtype=np.int32)
    generators_cpp = utils._numpy_to_cpp_integer_array(
        seed_reduced_lin_np.reshape(1, -1))

    # build reduced mask
    mask_red = ref_roi.mask_np[::spacing + 1, ::spacing + 1]
    cpp_mask_red = utils._numpy_to_cpp_logical_array(mask_red)

    # allocate outputs
    diag_cpp = _ccore.CppClassDoubleArray();  diag_cpp.alloc(h_red, w_red)
    prev_cpp = _ccore.CppClassDoubleArray();  prev_cpp.alloc(h_red, w_red)

    # dummy reduced image (algorithm only needs dims / padding)
    dummy_gs = np.zeros(mask_red.shape, dtype=np.float64)
    dummy_img = _ccore.CppNcorrClassImg()
    dummy_img.gs = utils._numpy_to_cpp_double_array(dummy_gs)
    dummy_img.type = "dummy"

    # heavy kernel
    _calgs.form_threaddiagram(diag_cpp, prev_cpp,
                              generators_cpp, cpp_mask_red, dummy_img)

    diag_np = utils._cpp_double_array_to_numpy(diag_cpp).astype(np.int32)

    return seeds_xy, diag_np


def _propagate_seeds_for_step_analysis(prev_seedinfo: Sequence[dict],
                                       u_prev: np.ndarray,
                                       v_prev: np.ndarray,
                                       spacing: int):
    """
    Given the final seed table of the previous step, **move** them to the new
    “best guess” positions for the next step.

    Returns
    -------
    seeds_xy        : list[(x,y)]                # new pixel positions
    param_vecs_new  : list[list[float]]          # initial param vectors
    """
    seeds_xy: list[tuple[int, int]] = []
    param_vecs: list[list[float]] = []

    step = spacing + 1

    for seed in prev_seedinfo:
        x_old = seed["x"];  y_old = seed["y"]
        u_pix = u_prev[y_old, x_old]
        v_pix = v_prev[y_old, x_old]

        x_new = int(round(x_old + (u_pix / step))) // step * step
        y_new = int(round(y_old + (v_pix / step))) // step * step

        seeds_xy.append((x_new, y_new))
        param_vecs.append([x_new, y_new, 0, 0, 0, 0, 0, 0, 0])

    return seeds_xy, param_vecs


def calculate_all_seeds(ref_img,
                        cur_img,
                        ref_roi,
                        seed_xy_list: Sequence[tuple[int, int]],
                        dic_params: dict):
    """
    Call the heavy **calc_seeds** kernel and re-assemble the output into
    convenient Python dicts.

    Returns
    -------
    seedinfo_list_py     : list[dict]
    convergence_list_py  : list[dict]
    status               : datatypes.OutputState
    """
    # Input conversions --------------------------------------------------- #
    ref_cpp = ref_img.to_cpp_repr()
    cur_cpp = cur_img.to_cpp_repr()
    roi_cpp = ref_roi.to_cpp_repr()

    # pos_seed → linear indices on ORIGINAL pixel grid
    w = ref_img.width
    pos_seed_lin = np.asarray([y * w + x for x, y in seed_xy_list],
                              dtype=np.int32)
    pos_seed_cpp = utils._numpy_to_cpp_integer_array(
        pos_seed_lin.reshape(1, -1))

    radius = dic_params.get("radius", 15)
    diffnorm_cut = dic_params.get("cutoff_diffnorm", 1e-3)
    iter_cut     = dic_params.get("cutoff_iter", 200)
    step_enabled = dic_params.get("step_enabled", False)
    subset_trunc = dic_params.get("subset_trunc", False)

    # Heavy compute ------------------------------------------------------- #
    seed_info_cpp, conv_cpp, out_status = _calgs.calc_seeds(
        ref_cpp, cur_cpp, roi_cpp,
        0,                 # num_region (unused – handled internal)
        pos_seed_cpp,
        radius,
        diffnorm_cut,
        iter_cut,
        step_enabled,
        subset_trunc
    )

    # Parse results ------------------------------------------------------- #
    seedinfo_py = [{"x": s.x, "y": s.y, "u": s.u, "v": s.v}
                   for s in seed_info_cpp]
    convergence_py = [{"iterations": c.iterations,
                       "diffnorm_final": c.diffnorm_final}
                      for c in conv_cpp]

    status_enum = datatypes.OutputState(out_status)

    return seedinfo_py, convergence_py, status_enum
