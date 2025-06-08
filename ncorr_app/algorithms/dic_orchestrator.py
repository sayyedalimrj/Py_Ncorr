"""
High-level DIC workflow controller – loosely mirrors
`ncorr_alg_dicanalysis.m` from the original MATLAB package.
"""

from __future__ import annotations

from importlib import import_module as _imp_mod
from typing import List, Sequence

from ncorr_app.core import NcorrImage, NcorrROI, datatypes, utils
from . import seed_manager, CPP_LOCK

try:
    _ccore = _imp_mod("ncorr_app._ext._ncorr_cpp_core")
    _calgs = _imp_mod("ncorr_app._ext._ncorr_cpp_algs")
except ModuleNotFoundError:  # pragma: no cover
    _ccore = None
    _calgs = None


# --------------------------------------------------------------------------- #
def _run_rg_dic(ref_img: NcorrImage,
                cur_img: NcorrImage,
                roi: NcorrROI,
                subset_trunc: bool,
                border: int,
                spacing: int):
    """
    Thin wrapper around the `rg_dic` C++ kernel.
    """
    with CPP_LOCK:
        u_cpp, v_cpp, cc_cpp, vp_cpp, status = _calgs.rg_dic(
            ref_img.to_cpp_repr(),
            cur_img.to_cpp_repr(),
            roi.to_cpp_repr(),
            subset_trunc,
            border,
            spacing,
        )

    result = {
        "u": utils._cpp_double_array_to_numpy(u_cpp),
        "v": utils._cpp_double_array_to_numpy(v_cpp),
        "corrcoef": utils._cpp_double_array_to_numpy(cc_cpp),
        "validpoints": utils._cpp_logical_array_to_numpy(vp_cpp),
    }
    return result, datatypes.OutputState(status)


# --------------------------------------------------------------------------- #
def orchestrate_dic_analysis(ref_img: NcorrImage,
                             cur_imgs: Sequence[NcorrImage],
                             ref_roi_init: NcorrROI,
                             params: dict):
    """
    Execute the complete multi-image DIC analysis.

    Parameters
    ----------
    ref_img         : base reference image
    cur_imgs        : list of target images (ordered)
    ref_roi_init    : ROI defined on *ref_img*
    params          : dict – assorted solver settings

    Returns
    -------
    disp_results    : list[dict | None]         # one per *cur_imgs*
    final_rois      : list[NcorrROI | None]
    seedinfo_steps  : list[list[dict]]          # per image
    overall_status  : datatypes.OutputState
    """
    n_imgs = len(cur_imgs)
    disp_results: List[dict | None] = [None] * n_imgs
    final_rois:   List[NcorrROI | None] = [None] * n_imgs
    seedinfo_steps: List[list[dict]] = []

    # Step-analysis bookkeeping
    step_enabled  = params.get("stepanalysis", {}).get("enabled", False)
    spacing       = params.get("spacing", 1)
    border        = params.get("radius", 15)
    subset_trunc  = params.get("subset_trunc", False)

    # Current reference objects (may change each step)
    step_ref_img:  NcorrImage = ref_img
    step_ref_roi:  NcorrROI   = ref_roi_init

    overall_status = datatypes.OutputState.SUCCESS

    for idx, cur_img in enumerate(cur_imgs):

        # ------------------------------------------------ Seed determination
        if idx == 0 or not step_enabled:
            seeds_xy, thread_diag = seed_manager._determine_initial_seeds_and_diagram(
                step_ref_roi.reduce(spacing),
                params
            )
        else:
            prev_info = seedinfo_steps[-1]
            u_prev = disp_results[idx - 1]["u"]
            v_prev = disp_results[idx - 1]["v"]
            seeds_xy, _ = seed_manager._propagate_seeds_for_step_analysis(
                prev_info, u_prev, v_prev, spacing)
            # Thread diagram is re-used unchanged for simplicity
            # (works if seed count stays equal); recompute if you need parity.

        # ------------------------------------------------ Seed calc
        seedinfo_py, conv_py, status_seeds = seed_manager.calculate_all_seeds(
            step_ref_img, cur_img, step_ref_roi, seeds_xy, params)

        if status_seeds != datatypes.OutputState.SUCCESS:
            overall_status = status_seeds
            break

        seedinfo_steps.append(seedinfo_py)

        # ------------------------------------------------ RG-DIC solve
        dic_out, status_dic = _run_rg_dic(step_ref_img, cur_img,
                                          step_ref_roi,
                                          subset_trunc, border, spacing)
        if status_dic != datatypes.OutputState.SUCCESS:
            overall_status = status_dic
            break

        disp_results[idx] = dic_out

        # ------------------------------------------------ ROI update / union (simplified)
        # Keep same ROI for this minimal implementation
        final_rois[idx] = step_ref_roi

        # Update reference for step-analysis
        if step_enabled:
            step_ref_img = cur_img
            # naive copy – in real code you'd warp ROI via displacement
            step_ref_roi = step_ref_roi

    return disp_results, final_rois, seedinfo_steps, overall_status
