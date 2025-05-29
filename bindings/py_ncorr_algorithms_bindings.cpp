/**
 *  bindings/py_ncorr_algorithms_bindings.cpp
 *
 *  Thin C++ wrappers (“*_cpp_impl”) around every standalone algorithm that
 *  shipped with the original Ncorr-2D MATLAB / MEX suite.  Each wrapper
 *  recreates the former `mexFunction` orchestration in normal C++, then is
 *  exported to Python through pybind11 so the high-level orchestrator written
 *  in `ncorr_app.algorithms.*` can call them directly.
 *
 *  ▸  All computationally heavy kernels that already use OpenMP are wrapped
 *     with  `py::call_guard<py::gil_scoped_release>()` so Python threads aren’t
 *     blocked while the C++ code burns CPU.
 *  ▸  With the exception of adopting “true” C++ signatures (instead of raw
 *     `mxArray *`), the original `.cpp` sources in `cpp_src/ncorr_alg/`
 *     remain **unchanged** – only **minimal constructor overloads** were added
 *     there so the classes will accept the typed objects you see below.
 */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "standard_datatypes.h"
#include "ncorr_datatypes.h"
#include "ncorr_lib.h"

// ─── Algorithm headers (the originals) ──────────────────────────────────────
// Every original file was refactored to expose a single public class whose
// constructor now takes strongly-typed inputs plus an `analysis()` method
// that does the work.
//
// • class_formmask               (ncorr_alg_formmask.cpp)
// • class_formregions            (ncorr_alg_formregions.cpp)
// • class_formboundary           (ncorr_alg_formboundary.cpp)
// • class_formthreaddiagram      (ncorr_alg_formthreaddiagram.cpp)
// • class_formunion              (ncorr_alg_formunion.cpp)
// • class_extrapdata             (ncorr_alg_extrapdata.cpp)
// • class_adddisp                (ncorr_alg_adddisp.cpp)
// • class_convertdisp            (ncorr_alg_convert.cpp)
// • class_dispgrad               (ncorr_alg_dispgrad.cpp)
// • class_testopenmp             (ncorr_alg_testopenmp.cpp)
// • class_calcseeds              (ncorr_alg_calcseeds.cpp)
// • class_rgdic                  (ncorr_alg_rgdic.cpp)

#include "ncorr_alg_formmask.h"
#include "ncorr_alg_formregions.h"
#include "ncorr_alg_formboundary.h"
#include "ncorr_alg_formthreaddiagram.h"
#include "ncorr_alg_formunion.h"
#include "ncorr_alg_extrapdata.h"
#include "ncorr_alg_adddisp.h"
#include "ncorr_alg_convert.h"
#include "ncorr_alg_dispgrad.h"
#include "ncorr_alg_testopenmp.h"
#include "ncorr_alg_calcseeds.h"
#include "ncorr_alg_rgdic.h"

// ─── Pybind11 convenience ───────────────────────────────────────────────────
namespace py = pybind11;
using namespace pybind11::literals;

using CppClassDoubleArray       = class_double_array;
using CppClassIntegerArray      = class_integer_array;
using CppClassLogicalArray      = class_logical_array;

using CppNcorrClassImg          = ncorr_class_img;
using CppNcorrClassRoi          = ncorr_class_roi;
using CppNcorrClassRegion       = ncorr_class_region;
using CppVecStructRegion        = vec_struct_region;

/* ------------------------------------------------------------------------- */
/*  Helper “light” structs used by several kernels.  The originals live in   */
/*  their respective .cpp files, but are re-declared here for binding.       */
/* ------------------------------------------------------------------------- */
struct CppPyDrawObject {
    int          type {};            // 0 = line, 1 = polygon, …
    double       x1 {}, y1 {}, x2 {}, y2 {};
};
struct CppPySeedInfo {
    int          x {}, y {};
    double       u {}, v {};
};
struct CppPyConvergenceInfo {
    int          iterations {};
    double       diffnorm_final {};
};
struct CppPyConvertSeedInfo {
    int          x_old {}, y_old {};
    double       u_old {}, v_old {};
};

/* ------------------------------------------------------------------------- */
/*  1.  C++ → Python interface functions   (the “*_cpp_impl” layer)          */
/* ------------------------------------------------------------------------- */
static void
_form_mask_cpp_impl(const std::vector<CppPyDrawObject> &drawobjects_in,
                    CppClassLogicalArray               &mask_in_out)
{
    class_formmask alg(drawobjects_in, mask_in_out);
    alg.analysis();                        // void, modifies mask_in_out
}

static std::tuple<std::vector<CppVecStructRegion>, bool>
_form_regions_cpp_impl(const CppClassLogicalArray &mask_in,
                       int                         cutoff_in,
                       bool                        preservelength_in)
{
    class_formregions alg(mask_in, cutoff_in, preservelength_in);
    std::vector<CppVecStructRegion> regions;
    bool removed {};
    alg.analysis(regions, removed);
    return {regions, removed};
}

static std::tuple<CppClassDoubleArray, int>
_form_boundary_cpp_impl(const CppClassIntegerArray &point_init_in,
                        int                         direc_in,
                        const CppClassLogicalArray &mask_in)
{
    class_formboundary alg(point_init_in, direc_in, mask_in);
    CppClassDoubleArray boundary;
    int direc_out {};
    alg.analysis(boundary, direc_out);
    return {boundary, direc_out};
}

static void
_form_threaddiagram_cpp_impl(CppClassDoubleArray           &threaddiagram_in_out,
                             CppClassDoubleArray           &preview_threaddiagram_in_out,
                             const CppClassIntegerArray    &generators_in,
                             const CppClassLogicalArray    &regionmask_in,
                             const CppNcorrClassImg        &img_in)
{
    class_formthreaddiagram alg(threaddiagram_in_out,
                                preview_threaddiagram_in_out,
                                generators_in,
                                regionmask_in,
                                img_in);
    alg.analysis();                    // in-place
}

static std::vector<CppVecStructRegion>
_form_union_cpp_impl(const std::vector<CppNcorrClassRegion> &roi_region_in,
                     const CppClassLogicalArray             &mask_union_in)
{
    class_formunion alg(roi_region_in, mask_union_in);
    std::vector<CppVecStructRegion> region_union;
    alg.analysis(region_union);
    return region_union;
}

static std::vector<CppClassDoubleArray>
_extrap_data_cpp_impl(const CppClassDoubleArray &plot_data_in,
                      const CppNcorrClassRoi    &roi_in,
                      int                        border_extrap_in)
{
    class_extrapdata alg(plot_data_in, roi_in, border_extrap_in);
    std::vector<CppClassDoubleArray> plots_extrap;
    alg.analysis(plots_extrap);
    return plots_extrap;
}

static std::tuple<CppClassDoubleArray, CppClassDoubleArray,
                  CppClassLogicalArray, OUT>
_add_disp_cpp_impl(
        const std::vector<std::vector<CppClassDoubleArray>> &plots_u_interp_in,
        const std::vector<std::vector<CppClassDoubleArray>> &plots_v_interp_in,
        const std::vector<CppNcorrClassRoi>                 &rois_interp_in,
        int   border_interp_in,
        int   spacing_in,
        int   num_img_in,
        int   total_imgs_in)
{
    class_adddisp alg(plots_u_interp_in, plots_v_interp_in, rois_interp_in,
                      border_interp_in, spacing_in, num_img_in, total_imgs_in);
    CppClassDoubleArray u_added, v_added;
    CppClassLogicalArray validpoints;
    OUT status {};
    alg.analysis(u_added, v_added, validpoints, status);
    return {u_added, v_added, validpoints, status};
}

static std::tuple<CppClassDoubleArray, CppClassDoubleArray,
                  CppClassLogicalArray, OUT>
_convert_disp_cpp_impl(
        const std::vector<std::vector<CppClassDoubleArray>> &plots_u_interp_old_in,
        const std::vector<std::vector<CppClassDoubleArray>> &plots_v_interp_old_in,
        const std::vector<std::vector<CppClassDoubleArray>> &plots_u_interp_new_in,
        const std::vector<std::vector<CppClassDoubleArray>> &plots_v_interp_new_in,
        const std::vector<CppPyConvertSeedInfo>             &convertseedinfo_in,
        int border_interp_in,
        int spacing_in,
        int num_img_in,
        int total_imgs_in)
{
    class_convertdisp alg(plots_u_interp_old_in, plots_v_interp_old_in,
                          plots_u_interp_new_in, plots_v_interp_new_in,
                          convertseedinfo_in, border_interp_in, spacing_in,
                          num_img_in, total_imgs_in);
    CppClassDoubleArray u_new, v_new;
    CppClassLogicalArray validpoints;
    OUT status {};
    alg.analysis(u_new, v_new, validpoints, status);
    return {u_new, v_new, validpoints, status};
}

static std::tuple<CppClassDoubleArray, CppClassDoubleArray,
                  CppClassDoubleArray, CppClassDoubleArray,
                  CppClassLogicalArray, OUT>
_disp_grad_cpp_impl(const CppClassDoubleArray &plot_u_in,
                    const CppClassDoubleArray &plot_v_in,
                    const CppClassLogicalArray &roi_mask_in,
                    bool subsettrunc_in,
                    int  border_in,
                    int  spacing_in)
{
    class_dispgrad alg(plot_u_in, plot_v_in, roi_mask_in,
                       subsettrunc_in, border_in, spacing_in);
    CppClassDoubleArray dudx, dudy, dvdx, dvdy;
    CppClassLogicalArray validpoints;
    OUT status {};
    alg.analysis(dudx, dudy, dvdx, dvdy, validpoints, status);
    return {dudx, dudy, dvdx, dvdy, validpoints, status};
}

static bool
_test_openmp_cpp_impl()
{
    class_testopenmp alg;
    return alg.analysis();
}

static std::tuple<std::vector<CppPySeedInfo>,
                  std::vector<CppPyConvergenceInfo>,
                  OUT>
_calc_seeds_cpp_impl(const CppNcorrClassImg   &ref_img_in,
                     const CppNcorrClassImg   &cur_img_in,
                     const CppNcorrClassRoi   &roi_in,
                     int  num_region_in,
                     const CppClassIntegerArray &pos_seed_in,
                     int  radius_in,
                     double cutoff_diffnorm_in,
                     int  cutoff_iter_in,
                     bool step_enabled_in,
                     bool subset_trunc_in)
{
    class_calcseeds alg(ref_img_in, cur_img_in, roi_in,
                        num_region_in, pos_seed_in, radius_in,
                        cutoff_diffnorm_in, cutoff_iter_in,
                        step_enabled_in, subset_trunc_in);
    std::vector<CppPySeedInfo>        seedinfo;
    std::vector<CppPyConvergenceInfo> convinfo;
    OUT status {};
    alg.analysis(seedinfo, convinfo, status);
    return {seedinfo, convinfo, status};
}

static std::tuple<CppClassDoubleArray, CppClassDoubleArray,
                  CppClassDoubleArray, CppClassLogicalArray, OUT>
_rg_dic_cpp_impl(const CppNcorrClassImg  &ref_img_in,
                 const CppNcorrClassImg  &cur_img_in,
                 const CppNcorrClassRoi  &roi_in,
                 bool subset_trunc_in,
                 int  border_in,
                 int  spacing_in)
{
    class_rgdic alg(ref_img_in, cur_img_in, roi_in,
                    subset_trunc_in, border_in, spacing_in);
    CppClassDoubleArray u, v, corrcoef;
    CppClassLogicalArray validpoints;
    OUT status {};
    alg.analysis(u, v, corrcoef, validpoints, status);
    return {u, v, corrcoef, validpoints, status};
}

/* ------------------------------------------------------------------------- */
/*  2.  Pybind11 module                                                       */
/* ------------------------------------------------------------------------- */
PYBIND11_MODULE(_ncorr_cpp_algs, m)
{
    m.doc() = "Pybind11 bindings – Ncorr algorithm kernels";

    /* ---- Lightweight helper structs (Python-side convenience) ------------ */
    py::class_<CppPyDrawObject>(m, "CppPyDrawObject")
        .def(py::init<>())
        .def_readwrite("type", &CppPyDrawObject::type)
        .def_readwrite("x1",   &CppPyDrawObject::x1)
        .def_readwrite("y1",   &CppPyDrawObject::y1)
        .def_readwrite("x2",   &CppPyDrawObject::x2)
        .def_readwrite("y2",   &CppPyDrawObject::y2);

    py::class_<CppPySeedInfo>(m, "CppPySeedInfo")
        .def(py::init<>())
        .def_readwrite("x", &CppPySeedInfo::x)
        .def_readwrite("y", &CppPySeedInfo::y)
        .def_readwrite("u", &CppPySeedInfo::u)
        .def_readwrite("v", &CppPySeedInfo::v);

    py::class_<CppPyConvergenceInfo>(m, "CppPyConvergenceInfo")
        .def(py::init<>())
        .def_readwrite("iterations",      &CppPyConvergenceInfo::iterations)
        .def_readwrite("diffnorm_final",  &CppPyConvergenceInfo::diffnorm_final);

    py::class_<CppPyConvertSeedInfo>(m, "CppPyConvertSeedInfo")
        .def(py::init<>())
        .def_readwrite("x_old", &CppPyConvertSeedInfo::x_old)
        .def_readwrite("y_old", &CppPyConvertSeedInfo::y_old)
        .def_readwrite("u_old", &CppPyConvertSeedInfo::u_old)
        .def_readwrite("v_old", &CppPyConvertSeedInfo::v_old);

    /* ---- Algorithms ------------------------------------------------------ */
    m.def("form_mask",
          &_form_mask_cpp_impl,
          "Draw-objects → bitmap mask.");

    m.def("form_regions",
          &_form_regions_cpp_impl,
          "Find connected regions in a mask\n"
          "Returns  (regions, removed_flag).");

    m.def("form_boundary",
          &_form_boundary_cpp_impl,
          "marching-squares boundary extraction\n"
          "Returns (boundary_points, direction).");

    m.def("form_threaddiagram",
          &_form_threaddiagram_cpp_impl,
          py::call_guard<py::gil_scoped_release>(),
          "In-place generation of thread diagram preview + full diagram.");

    m.def("form_union",
          &_form_union_cpp_impl,
          "Compute union of ROI regions.");

    m.def("extrap_data",
          &_extrap_data_cpp_impl,
          py::call_guard<py::gil_scoped_release>(),
          "Extrapolate data outside ROI using inverse region.");

    m.def("add_disp",
          &_add_disp_cpp_impl,
          py::call_guard<py::gil_scoped_release>(),
          "Add displacement fields onto a common grid.");

    m.def("convert_disp",
          &_convert_disp_cpp_impl,
          py::call_guard<py::gil_scoped_release>(),
          "Convert displacement fields from old → new grid.");

    m.def("disp_grad",
          &_disp_grad_cpp_impl,
          py::call_guard<py::gil_scoped_release>(),
          "Compute displacement gradients (∂u/∂x … etc.).");

    m.def("test_openmp",
          &_test_openmp_cpp_impl,
          py::call_guard<py::gil_scoped_release>(),
          "Return *True* when OpenMP threads > 1.");

    m.def("calc_seeds",
          &_calc_seeds_cpp_impl,
          py::call_guard<py::gil_scoped_release>(),
          "Automatic seed calculation in ROI.");

    m.def("rg_dic",
          &_rg_dic_cpp_impl,
          py::call_guard<py::gil_scoped_release>(),
          "Region-growing DIC core kernel.");
}
