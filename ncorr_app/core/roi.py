"""
ROI handling for the Python port of Ncorr.

A region of interest (ROI) may contain multiple disjoint regions (“add”
polygons) plus interior voids (“sub” polygons).  This class wraps all
mask/region bookkeeping and exposes helpers for the C++ kernels.
"""

from __future__ import annotations
from importlib import import_module as _imp_mod

import numpy as np

from ncorr_app.core.datatypes import OutputState, BORDER_INTERP_DEFAULT

from ncorr_app.core.utils import (
    _numpy_to_cpp_logical_array,
    _numpy_to_cpp_double_array,
)

_ccore = _imp_mod("ncorr_app._ext._ncorr_cpp_core")
_calgs = _imp_mod("ncorr_app._ext._ncorr_cpp_algs")


class RegionData:  # lightweight Python proxy of CppNcorrClassRegion
    __slots__ = (
        "leftbound",
        "rightbound",
        "upperbound",
        "lowerbound",
        "totalpoints",
        "nodelist",
        "noderange",
    )

    def __init__(
        self,
        left: int,
        right: int,
        up: int,
        low: int,
        total: int,
        nodelist: np.ndarray,
        noderange: np.ndarray,
    ):
        self.leftbound = left
        self.rightbound = right
        self.upperbound = up
        self.lowerbound = low
        self.totalpoints = total
        self.nodelist = nodelist
        self.noderange = noderange


class NcorrROI:
    # ------------------------------------------------------------------ ctor
    def __init__(self, image_shape_yx: tuple[int, int] | None = None):
        self.type = ""
        self.mask_np: np.ndarray | None = (
            np.zeros(image_shape_yx, dtype=bool) if image_shape_yx else None
        )
        self.regions_py: list[RegionData] = []
        # boundaries: list[{"add": np.ndarray, "sub": [np.ndarray, ...]}]
        self.boundary_data_py: list[dict] = []
        self.draw_objects_data_py: list[dict] = []

    # ---------------------------------------------------------------- public
    def get_mask_np(self):
        return self.mask_np

    def get_regions_py(self):
        return self.regions_py

    def get_boundary_data_py(self):
        return self.boundary_data_py

    # ............................... major construction helpers omitted
    # (set_roi_from_drawings, set_roi_from_mask_internal, reduce, etc.)
    # They are assumed present from Phase-2 implementation
    # ---------------------------------------------------------------- update
    def update_roi(
        self,
        plot_u_np_pixels: np.ndarray,
        plot_v_np_pixels: np.ndarray,
        roi_plot_obj_for_disp_fields: "NcorrROI",
        new_image_shape_yx: tuple[int, int],
        spacing: int,
        radius_dic: int,
    ) -> "NcorrROI":
        """Warp the current ROI using pixel-displacement fields U/V.

        Mirrors ncorr_class_roi.m::update_roi.
        """
        step = spacing + 1
        h_new, w_new = new_image_shape_yx

        # ------ extrapolate full-field → region-wise B-spline coeffs -----
        cpp_u_full = _numpy_to_cpp_double_array(plot_u_np_pixels)
        cpp_v_full = _numpy_to_cpp_double_array(plot_v_np_pixels)

        u_extrap_cpp_list, v_extrap_cpp_list = _calgs.extrap_data(
            cpp_u_full, roi_plot_obj_for_disp_fields.to_cpp_repr(), BORDER_INTERP_DEFAULT
        )

        # B-spline coeff for every region
        u_bcoef_cpp = []
        v_bcoef_cpp = []
        region_helpers = []  # (mask_cpp, left_off, up_off)
        for ridx, reg in enumerate(roi_plot_obj_for_disp_fields.regions_py):
            u_bcoef_cpp.append(u_extrap_cpp_list[ridx])  # already coeff if extrap_data outputs coeff
            v_bcoef_cpp.append(v_extrap_cpp_list[ridx])
            mask_cpp = _numpy_to_cpp_logical_array(
                roi_plot_obj_for_disp_fields._get_region_mask_np(ridx)
            )
            region_helpers.append(
                (mask_cpp, int(reg.leftbound / step), int(reg.upperbound / step))
            )

        updated_draw_objs: list[dict] = []

        # --------------------------- iterate boundaries -------------------
        for bset in self.boundary_data_py:
            sample_pt = bset["add"][0]
            region_idx = self._find_region_containing_point(
                sample_pt[0], sample_pt[1]
            )
            if region_idx is None:
                continue
            mask_cpp, left_off, up_off = region_helpers[region_idx]
            u_bcoef = u_bcoef_cpp[region_idx]
            v_bcoef = v_bcoef_cpp[region_idx]

            # helper to warp polygon
            def _warp_poly(pts: np.ndarray) -> np.ndarray:
                new_pts = []
                for x_old, y_old in pts:
                    xr = x_old / step
                    yr = y_old / step
                    u_val, sta_u = _ccore.interp_qbs(
                        xr, yr, u_bcoef, mask_cpp, left_off, up_off, BORDER_INTERP_DEFAULT
                    )
                    v_val, sta_v = _ccore.interp_qbs(
                        xr, yr, v_bcoef, mask_cpp, left_off, up_off, BORDER_INTERP_DEFAULT
                    )
                    if (
                        OutputState(sta_u) != OutputState.SUCCESS
                        or OutputState(sta_v) != OutputState.SUCCESS
                    ):
                        continue
                    x_new = x_old + u_val
                    y_new = y_old + v_val
                    if (
                        radius_dic
                        <= x_new
                        < w_new - radius_dic
                        and radius_dic
                        <= y_new
                        < h_new - radius_dic
                    ):
                        new_pts.append([x_new, y_new])
                return np.asarray(new_pts, dtype=np.float64)

            # add poly
            new_add = _warp_poly(bset["add"])
            if new_add.shape[0] >= 3:
                updated_draw_objs.append(
                    {"pos_imroi": new_add, "type": "poly", "addorsub": "add"}
                )
            # sub polys
            for sub in bset.get("sub", []):
                new_sub = _warp_poly(sub)
                if new_sub.shape[0] >= 3:
                    updated_draw_objs.append(
                        {"pos_imroi": new_sub, "type": "poly", "addorsub": "sub"}
                    )

        # --------------------------- build new ROI ------------------------
        if not updated_draw_objs:
            return NcorrROI(image_shape_yx=new_image_shape_yx)

        new_roi = NcorrROI(image_shape_yx=new_image_shape_yx)
        new_roi.set_roi_from_drawings(updated_draw_objs, new_image_shape_yx, cutoff=0)
        return new_roi

    # ---------------------------------------------------------------------
    # private helpers – minimal versions used by update_roi
    # ---------------------------------------------------------------------
    def _get_region_mask_np(self, idx: int) -> np.ndarray:
        r = self.regions_py[idx]
        mask = np.zeros_like(self.mask_np)
        ys, xs = r.nodelist[:, 1], r.nodelist[:, 0]
        mask[ys, xs] = True
        return mask

    def _find_region_containing_point(self, x: float, y: float):
        for idx, r in enumerate(self.regions_py):
            if r.leftbound <= x <= r.rightbound and r.upperbound <= y <= r.lowerbound:
                return idx
        return None

    # to_cpp_repr() and other API methods were implemented earlier
