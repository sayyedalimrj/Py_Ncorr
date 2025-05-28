#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

// Core Ncorr C++ headers (assuming these are in include path)
#include "standard_datatypes.h"
#include "ncorr_datatypes.h"
#include "ncorr_lib.h"

// Headers for each algorithm's implementation
// These would typically be the .h files if the logic was separated from mexFunction,
// or we might need to include the .cpp files directly if refactoring is minimal
// For this exercise, we'll assume the core logic can be extracted or called.
// For a real build, you'd compile the .cpp files from cpp_src/ncorr_alg/
// and link them. Here, we'll write wrapper functions that encapsulate
// the logic derived from those .cpp files.

// Placeholder for actual includes of refactored algorithm logic
// Example: #include "ncorr_alg_formmask_impl.h"
// For now, we'll define wrapper functions that *would* call such refactored logic.

namespace py = pybind11;

// --- Helper structs for binding complex inputs/outputs ---

// For ncorr_alg_formmask
struct PyDrawObject {
    class_double_array pos_imroi;
    std::string type;
    std::string addorsub;
};

// For ncorr_alg_calcseeds and ncorr_alg_rgdic (seed information)
struct PySeedInfo {
    class_double_array paramvector; // Should be 1x9
    int num_region;
    int num_thread;
    int computepoints;
};

// For ncorr_alg_calcseeds (convergence information)
struct PyConvergenceInfo {
    int num_iterations;
    double diffnorm;
};

// For ncorr_alg_convert (convert seed information)
struct PyConvertSeedInfo {
    class_double_array paramvector; // Should be 1x7
    int num_region_new;
    int num_region_old;
};


// --- Refactored Algorithm Wrappers ---

// Placeholder: Actual implementation would require refactoring each mexFunction's
// internal class logic into these C++ functions.
// For brevity, I'm showing the signature and a pybind wrapper.
// The actual implementation would involve calling the core C++ logic.

// Based on ncorr_alg_formmask.cpp
void py_form_mask(
    const std::vector<PyDrawObject>& drawobjects_in,
    class_logical_array& mask_in_out // Modified in-place
) {
    // --- Start of adapted logic from class_formmask ---
    mask_in_out.reset();
    int length_polynode_max = 0;
    for (const auto& dobj : drawobjects_in) {
        if (dobj.type.rfind("poly", 0) == 0) {
            if (dobj.pos_imroi.height > length_polynode_max) {
                length_polynode_max = dobj.pos_imroi.height;
            }
        }
    }
    local_struct_node polynode_buffer(length_polynode_max > 0 ? length_polynode_max : 1);

    for (const auto& dobj : drawobjects_in) {
        if (dobj.type.rfind("poly", 0) == 0) {
            // ... (Full logic from ncorr_alg_formmask.cpp :: class_formmask::analysis for 'poly') ...
            // This is a simplified placeholder for the complex polygon rasterization.
            // A full port would require careful adaptation of the original rasterization logic.
            // For now, we'll assume it modifies mask_in_out.
             if (dobj.pos_imroi.height == 0 && dobj.pos_imroi.width ==0) continue; // Skip empty polygons

            int leftbound = static_cast<int>(std::max(ceil(dobj.pos_imroi.value[0]), 0.0)); // Simplified
            int rightbound = static_cast<int>(std::min(floor(dobj.pos_imroi.value[dobj.pos_imroi.height-1]), static_cast<double>(mask_in_out.width-1)));

            bool is_add = (dobj.addorsub.rfind("add",0) == 0);

            for (int j = leftbound; j <= rightbound; ++j) {
                 // Simplified: fill a box for demo for polygon
                int y_start = static_cast<int>(std::max(ceil(dobj.pos_imroi.value[dobj.pos_imroi.height]), 0.0));
                int y_end = static_cast<int>(std::min(floor(dobj.pos_imroi.value[dobj.pos_imroi.height + dobj.pos_imroi.height-1]), static_cast<double>(mask_in_out.height-1)));
                for (int l = y_start; l <= y_end; ++l) {
                     if (j >=0 && j < mask_in_out.width && l >=0 && l < mask_in_out.height)
                        mask_in_out.value[l + j * mask_in_out.height] = is_add;
                }
            }


        } else if (dobj.type.rfind("rect", 0) == 0) {
            // ... (Full logic for 'rect') ...
            int upperbound = static_cast<int>(std::max(ceil(dobj.pos_imroi.value[1]), 0.0));
            int lowerbound = static_cast<int>(std::min(floor(dobj.pos_imroi.value[1] + dobj.pos_imroi.value[3]), static_cast<double>(mask_in_out.height - 1.0)));
            int leftbound = static_cast<int>(std::max(ceil(dobj.pos_imroi.value[0]), 0.0));
            int rightbound = static_cast<int>(std::min(floor(dobj.pos_imroi.value[0] + dobj.pos_imroi.value[2]), static_cast<double>(mask_in_out.width - 1.0)));
            bool is_add = (dobj.addorsub.rfind("add",0) == 0);
            for (int j_ = leftbound; j_ <= rightbound; ++j_) {
                for (int k_ = upperbound; k_ <= lowerbound; ++k_) {
                     if (j_ >=0 && j_ < mask_in_out.width && k_ >=0 && k_ < mask_in_out.height)
                        mask_in_out.value[k_ + j_ * mask_in_out.height] = is_add;
                }
            }
        } else if (dobj.type.rfind("ellipse", 0) == 0) {
            // ... (Full logic for 'ellipse') ...
            int leftbound = static_cast<int>(std::max(ceil(dobj.pos_imroi.value[0]),0.0));
            int rightbound = static_cast<int>(std::min(floor(dobj.pos_imroi.value[0]+dobj.pos_imroi.value[2]), static_cast<double>(mask_in_out.width-1.0)));
            double a = dobj.pos_imroi.value[2]/2.0;
            double b = dobj.pos_imroi.value[3]/2.0;
            bool is_add = (dobj.addorsub.rfind("add",0) == 0);
            if (a <=0 || b <=0 ) continue;

            for (int j_ = leftbound; j_ <= rightbound; ++j_) {
                double x_norm_sq = pow(((double)j_ - dobj.pos_imroi.value[0] - a) / a, 2);
                if (x_norm_sq > 1.0) continue;
                double y_delta = b * sqrt(1.0 - x_norm_sq);
                int upperbound = static_cast<int>(std::max(ceil(dobj.pos_imroi.value[1] + b - y_delta), 0.0));
                int lowerbound = static_cast<int>(std::min(floor(dobj.pos_imroi.value[1] + b + y_delta), static_cast<double>(mask_in_out.height) - 1.0));
                for (int k_ = upperbound; k_ <= lowerbound; ++k_) {
                     if (j_ >=0 && j_ < mask_in_out.width && k_ >=0 && k_ < mask_in_out.height)
                        mask_in_out.value[k_ + j_ * mask_in_out.height] = is_add;
                }
            }
        }
    }
    // --- End of adapted logic ---
}


// Based on ncorr_alg_formregions.cpp
std::tuple<std::vector<vec_struct_region>, bool> py_form_regions(
    const class_logical_array& mask_in,
    int cutoff_in,
    bool preservelength_in
) {
    std::vector<vec_struct_region> regions_out;
    bool removed_out = false;
    form_regions(regions_out, removed_out, mask_in, cutoff_in, preservelength_in);
    return std::make_tuple(regions_out, removed_out);
}

// Based on ncorr_alg_formboundary.cpp
std::tuple<class_double_array, int> py_form_boundary(
    const class_integer_array& point_init_in, // Expecting 1x2 CppClassIntegerArray
    int direc_in,
    const class_logical_array& mask_in
) {
    if (point_init_in.width != 2 || point_init_in.height != 1) {
        throw std::runtime_error("Initial point must be 1x2.");
    }
    std::vector<std::vector<int>> vec_boundary_out;
    std::vector<int> vec_point_init_cpp(2);
    vec_point_init_cpp[0] = point_init_in.value[0];
    vec_point_init_cpp[1] = point_init_in.value[1]; // Assuming column major access or direct if 1x2
    
    int direc_in_out = direc_in; // Modifiable copy

    form_boundary(vec_boundary_out, vec_point_init_cpp, mask_in, direc_in_out);

    class_double_array boundary_out_arr;
    if (!vec_boundary_out.empty()) {
        boundary_out_arr.alloc(vec_boundary_out.size(), 2);
        for (size_t i = 0; i < vec_boundary_out.size(); ++i) {
            boundary_out_arr.value[i] = static_cast<double>(vec_boundary_out[i][0]); // x
            boundary_out_arr.value[i + boundary_out_arr.height] = static_cast<double>(vec_boundary_out[i][1]); // y
        }
    } else {
         boundary_out_arr.alloc(0,0); // Allocate empty
    }
    return std::make_tuple(boundary_out_arr, direc_in_out);
}


// Based on ncorr_alg_formthreaddiagram.cpp
void py_form_threaddiagram(
    class_double_array& threaddiagram_in_out,
    class_double_array& preview_threaddiagram_in_out,
    const class_integer_array& generators_in, // Nx2 array
    const class_logical_array& regionmask_in,
    const ncorr_class_img& img_in
) {
    // --- Start of adapted logic from class_formthreaddiagram ---
    // This requires adapting the class_formthreaddiagram internal logic into a function
    // or making the class itself more directly usable.
    // The original uses std::vector<std::list<std::vector<int>>> queue;
    // This is a complex data structure to manage directly in a simple wrapper.
    // The original C++ code directly modifies threaddiagram.value and preview_threaddiagram.value
    // A full port of this function's internals is extensive.
    // Placeholder:
    for (int i = 0; i < threaddiagram_in_out.width * threaddiagram_in_out.height; ++i) {
        threaddiagram_in_out.value[i] = -1.0;
    }
    if (generators_in.height > 0 && generators_in.width == 2) {
         for (int i = 0; i < generators_in.height; ++i) {
            int gen_x = generators_in.value[i];
            int gen_y = generators_in.value[i + generators_in.height];
            if (gen_y >=0 && gen_y < threaddiagram_in_out.height && gen_x >=0 && gen_x < threaddiagram_in_out.width)
                threaddiagram_in_out.value[gen_y + gen_x * threaddiagram_in_out.height] = static_cast<double>(i);
         }
    }
     // Simplified preview logic
    for (int i = 0; i < preview_threaddiagram_in_out.width * preview_threaddiagram_in_out.height; ++i) {
         if (i < img_in.gs.width * img_in.gs.height)
            preview_threaddiagram_in_out.value[i] = img_in.gs.value[i];
         else
             preview_threaddiagram_in_out.value[i] = 0;
    }
    // --- End of adapted logic ---
}


// Based on ncorr_alg_formunion.cpp
std::vector<vec_struct_region> py_form_union(
    const std::vector<ncorr_class_region>& roi_region_in,
    const class_logical_array& mask_union_in
) {
    std::vector<vec_struct_region> region_union_out;
    region_union_out.resize(roi_region_in.size());
    form_union(region_union_out, roi_region_in, mask_union_in, false); // inplace=false
    return region_union_out;
}

// Based on ncorr_alg_extrapdata.cpp
std::vector<class_double_array> py_extrap_data(
    const class_double_array& plot_data_in,
    const ncorr_class_roi& roi_in, // Actually std::vector<ncorr_class_roi> in mex, but uses roi[0]
    int border_extrap_in
) {
    std::vector<class_double_array> plots_extrap_out;
    plots_extrap_out.resize(roi_in.region.size());

    for (size_t i = 0; i < roi_in.region.size(); ++i) {
        if (roi_in.region[i].totalpoints > 0) {
            int height_plot_extrap = (roi_in.region[i].lowerbound - roi_in.region[i].upperbound + 1) + 2 * border_extrap_in;
            int width_plot_extrap = (roi_in.region[i].rightbound - roi_in.region[i].leftbound + 1) + 2 * border_extrap_in;
            plots_extrap_out[i].alloc(height_plot_extrap, width_plot_extrap);

            for (int j = 0; j < roi_in.region[i].noderange.height; ++j) {
                int x_plot = j + border_extrap_in;
                for (int k_ = 0; k_ < roi_in.region[i].noderange.value[j]; k_ += 2) {
                    for (int l = roi_in.region[i].nodelist.value[j + k_ * roi_in.region[i].nodelist.height]; 
                         l <= roi_in.region[i].nodelist.value[j + (k_ + 1) * roi_in.region[i].nodelist.height]; ++l) {
                        int y_plot = l - roi_in.region[i].upperbound + border_extrap_in;
                        if (y_plot >=0 && y_plot < plots_extrap_out[i].height && x_plot >=0 && x_plot < plots_extrap_out[i].width &&
                            (l + (j + roi_in.region[i].leftbound) * plot_data_in.height) < plot_data_in.height*plot_data_in.width &&
                            (l + (j + roi_in.region[i].leftbound) * plot_data_in.height) >= 0)
                           plots_extrap_out[i].value[y_plot + x_plot * plots_extrap_out[i].height] = 
                               plot_data_in.value[l + (j + roi_in.region[i].leftbound) * plot_data_in.height];
                    }
                }
            }
            ncorr_class_inverseregion inverseregion(const_cast<ncorr_class_region&>(roi_in.region[i]), border_extrap_in); // Hacky const_cast due to ncorr_class_inverseregion constructor
            expand_filt(plots_extrap_out[i], inverseregion);
        } else {
             plots_extrap_out[i].alloc(0,0); // Empty region
        }
    }
    return plots_extrap_out;
}

// Placeholder for ncorr_alg_adddisp.cpp
// Output: tuple(plot_u_added, plot_v_added, plot_validpoints, status)
std::tuple<class_double_array, class_double_array, class_logical_array, OUT>
py_add_disp(
    const std::vector<std::vector<class_double_array>>& plots_u_interp,
    const std::vector<std::vector<class_double_array>>& plots_v_interp,
    const std::vector<ncorr_class_roi>& rois_interp,
    int border_interp, int spacing, int num_img, int total_imgs
) {
    // ... Full refactoring of class_adddisp and its analysis method ...
    // This is a complex one due to nested C++ structures.
    // For now, return empty/default.
    class_double_array u_added, v_added;
    class_logical_array valid_points;
    if (!rois_interp.empty()){
        u_added.alloc(rois_interp[0].mask.height, rois_interp[0].mask.width);
        v_added.alloc(rois_interp[0].mask.height, rois_interp[0].mask.width);
        valid_points.alloc(rois_interp[0].mask.height, rois_interp[0].mask.width);
    }
    return std::make_tuple(u_added, v_added, valid_points, OUT::FAILED); // Placeholder
}


// Placeholder for ncorr_alg_convert.cpp
// Output: tuple(plot_u_new, plot_v_new, plot_validpoints, status)
std::tuple<class_double_array, class_double_array, class_logical_array, OUT>
py_convert_disp(
    const std::vector<std::vector<class_double_array>>& plots_u_interp_old,
    const std::vector<std::vector<class_double_array>>& plots_v_interp_old,
    const std::vector<ncorr_class_roi>& rois_old,
    const std::vector<ncorr_class_roi>& rois_new,
    const std::vector<PyConvertSeedInfo>& convertseedinfo_in,
    int spacing, int border_interp, int num_img, int total_imgs
) {
    // ... Full refactoring of class_convert and its analysis method ...
    class_double_array u_new, v_new;
    class_logical_array valid_points;
     if (!rois_new.empty()){
        u_new.alloc(rois_new[0].mask.height, rois_new[0].mask.width);
        v_new.alloc(rois_new[0].mask.height, rois_new[0].mask.width);
        valid_points.alloc(rois_new[0].mask.height, rois_new[0].mask.width);
    }
    return std::make_tuple(u_new, v_new, valid_points, OUT::FAILED); // Placeholder
}


// Based on ncorr_alg_dispgrad.cpp
// Output: tuple(dudx, dudy, dvdx, dvdy, validpoints, status)
std::tuple<class_double_array, class_double_array, class_double_array, class_double_array, class_logical_array, OUT>
py_disp_grad(
    const class_double_array& plot_u_in, const class_double_array& plot_v_in,
    const ncorr_class_roi& roi_in, // Actually std::vector<ncorr_class_roi> in mex, uses roi[0]
    int radius_strain_in, double pixtounits_in, int spacing_in,
    bool subsettrunc_in, int num_img_in, int total_imgs_in
) {
    // ... Full refactoring of class_dispgrad and its analysis method ...
    class_double_array dudx, dudy, dvdx, dvdy;
    class_logical_array valid_points;
    dudx.alloc(roi_in.mask.height, roi_in.mask.width);
    dudy.alloc(roi_in.mask.height, roi_in.mask.width);
    dvdx.alloc(roi_in.mask.height, roi_in.mask.width);
    dvdy.alloc(roi_in.mask.height, roi_in.mask.width);
    valid_points.alloc(roi_in.mask.height, roi_in.mask.width);
    
    // Placeholder for actual dispgrad logic
    // The original code iterates over regions in roi_in.region and points within.
    // For simplicity, this placeholder just returns empty/default.
    // A full implementation needs to adapt the class_dispgrad logic.

    return std::make_tuple(dudx, dudy, dvdx, dvdy, valid_points, OUT::FAILED); // Placeholder
}

// Based on ncorr_alg_testopenmp.cpp
bool py_test_openmp() {
    // --- Start of adapted logic from class_testopenmp ---
    bool enabled_openmp = true;
    #ifdef NCORR_OPENMP
        int total_threads_test = 4;
        std::vector<char> vec_enabled_thread(total_threads_test, 0);
        omp_set_num_threads(total_threads_test);
        #pragma omp parallel
        {
            int num_thread = omp_get_thread_num();
            if (num_thread < total_threads_test) { // Bounds check
                 vec_enabled_thread[num_thread] = 1;
            }
        }
        for (int i = 0; i < total_threads_test; ++i) {
            if (vec_enabled_thread[i] == 0) {
                enabled_openmp = false;
                break;
            }
        }
    #else
        // If NCORR_OPENMP is not defined, we might assume OpenMP is not "supported"
        // in the context of this test, or the test passes trivially for single thread.
        // The original mex returns true if all threads in test execute.
        // For a non-OpenMP build, it should probably indicate it can't "test" OpenMP.
        // Or, per original logic, if total_threads_test was 1, it would pass.
        // Let's assume if NCORR_OPENMP is not defined, the test means "single-threaded operation is fine".
        // But the spirit of the test is OpenMP functionality. So perhaps return false if not compiled with OpenMP.
        enabled_openmp = false; // Indicate OpenMP is not active/tested.
    #endif
    return enabled_openmp;
    // --- End of adapted logic ---
}

// Based on ncorr_alg_calcseeds.cpp
// Output: tuple(list_of_PySeedInfo, list_of_PyConvergenceInfo, status)
std::tuple<std::vector<PySeedInfo>, std::vector<PyConvergenceInfo>, OUT>
py_calc_seeds(
    const ncorr_class_img& reference_in, const ncorr_class_img& current_in,
    const ncorr_class_roi& roi_in, int num_region_in,
    const class_integer_array& pos_seed_in, // Expects Mx2
    int radius_in, double cutoff_diffnorm_in, int cutoff_iteration_in,
    bool enabled_stepanalysis_in, bool subsettrunc_in
) {
    // ... Full refactoring of class_calcseeds and its analysis method ...
    // This is highly complex, involves OpenMP, and many helper methods from the original class.
    // It would require careful porting of the internal logic including initialguess, ncc, iterativesearch, newton.
    std::vector<PySeedInfo> seedinfo_out;
    std::vector<PyConvergenceInfo> convergence_out;
    
    int total_threads = pos_seed_in.height;
    seedinfo_out.resize(total_threads);
    convergence_out.resize(total_threads);

    // Placeholder logic - this needs the full algorithm.
    for(int i=0; i < total_threads; ++i) {
        seedinfo_out[i].paramvector.alloc(1,9); // paramvector = [x y u v du/dx du/dy dv/dx dv/dy corrcoef]
        // fill with zeros or default values
        seedinfo_out[i].num_region = num_region_in;
        seedinfo_out[i].num_thread = i;
        seedinfo_out[i].computepoints = 0; // This would be calculated based on threaddiagram
        convergence_out[i].num_iterations = 0;
        convergence_out[i].diffnorm = 0.0;
    }

    return std::make_tuple(seedinfo_out, convergence_out, OUT::FAILED); // Placeholder
}


// Based on ncorr_alg_rgdic.cpp
// Output: tuple(plot_u, plot_v, plot_corrcoef, plot_validpoints, status)
std::tuple<class_double_array, class_double_array, class_double_array, class_logical_array, OUT>
py_rgdic(
    const ncorr_class_img& reference_in, const ncorr_class_img& current_in,
    const ncorr_class_roi& roi_in,
    const std::vector<PySeedInfo>& seedinfo_in,
    const class_integer_array& threaddiagram_in,
    int radius_in, int spacing_in, double cutoff_diffnorm_in, int cutoff_iteration_in,
    bool subsettrunc_in, int num_img_overall_in, int total_imgs_overall_in
) {
    // ... Full refactoring of class_rgdic and its analysis method ...
    // This is the most complex algorithm, involving OpenMP, priority queues, IC-GN, etc.
    class_double_array plot_u, plot_v, plot_corrcoef;
    class_logical_array plot_validpoints;

    plot_u.alloc(threaddiagram_in.height, threaddiagram_in.width);
    plot_v.alloc(threaddiagram_in.height, threaddiagram_in.width);
    plot_corrcoef.alloc(threaddiagram_in.height, threaddiagram_in.width);
    plot_validpoints.alloc(threaddiagram_in.height, threaddiagram_in.width);
    
    // Placeholder logic
    
    return std::make_tuple(plot_u, plot_v, plot_corrcoef, plot_validpoints, OUT::FAILED); // Placeholder
}


PYBIND11_MODULE(_ncorr_cpp_algs, m) {
    m.doc() = "Pybind11 bindings for Ncorr C++ algorithms";

    // Bind helper structs if they are passed to/from Python
    py::class_<PyDrawObject>(m, "PyDrawObject")
        .def(py::init<>())
        .def_readwrite("pos_imroi", &PyDrawObject::pos_imroi)
        .def_readwrite("type", &PyDrawObject::type)
        .def_readwrite("addorsub", &PyDrawObject::addorsub);

    py::class_<PySeedInfo>(m, "PySeedInfo")
        .def(py::init<>())
        .def_readwrite("paramvector", &PySeedInfo::paramvector)
        .def_readwrite("num_region", &PySeedInfo::num_region)
        .def_readwrite("num_thread", &PySeedInfo::num_thread)
        .def_readwrite("computepoints", &PySeedInfo::computepoints);

    py::class_<PyConvergenceInfo>(m, "PyConvergenceInfo")
        .def(py::init<>())
        .def_readwrite("num_iterations", &PyConvergenceInfo::num_iterations)
        .def_readwrite("diffnorm", &PyConvergenceInfo::diffnorm);
    
    py::class_<PyConvertSeedInfo>(m, "PyConvertSeedInfo")
        .def(py::init<>())
        .def_readwrite("paramvector", &PyConvertSeedInfo::paramvector)
        .def_readwrite("num_region_new", &PyConvertSeedInfo::num_region_new)
        .def_readwrite("num_region_old", &PyConvertSeedInfo::num_region_old);


    // Bind refactored algorithm functions
    m.def("form_mask", &py_form_mask, py::arg("drawobjects"), py::arg("mask_in_out").noconvert(),
          "Forms a mask from draw objects. Modifies mask_in_out in-place.");

    m.def("form_regions", &py_form_regions, py::arg("mask"), py::arg("cutoff"), py::arg("preservelength"),
          "Forms contiguous regions from a mask. Returns (list_of_regions, removed_flag).");
    
    m.def("form_boundary", &py_form_boundary, py::arg("point_init"), py::arg("direc_in"), py::arg("mask"),
          "Forms the boundary of a masked region. Returns (boundary_array, updated_direction).");

    m.def("form_threaddiagram", &py_form_threaddiagram, 
          py::arg("threaddiagram").noconvert(), py::arg("preview_threaddiagram").noconvert(), 
          py::arg("generators"), py::arg("regionmask"), py::arg("img"),
          "Forms a thread diagram for parallel processing. Modifies input arrays in-place.");

    m.def("form_union", &py_form_union, py::arg("roi_region_in"), py::arg("mask_union_in"),
          "Forms the union of ROI regions with a mask. Returns a list of new regions (CppVecStructRegion).");

    m.def("extrap_data", &py_extrap_data, py::arg("plot_data_in"), py::arg("roi_in"), py::arg("border_extrap_in"),
          "Extrapolates data for each region in the ROI. Returns a list of CppClassDoubleArray.");
    
    m.def("add_disp", &py_add_disp, 
          py::arg("plots_u_interp"), py::arg("plots_v_interp"), py::arg("rois_interp"),
          py::arg("border_interp"), py::arg("spacing"), py::arg("num_img"), py::arg("total_imgs"),
          "Adds displacement fields. Returns (plot_u_added, plot_v_added, plot_validpoints, status).");

    m.def("convert_disp", &py_convert_disp,
          py::arg("plots_u_interp_old"), py::arg("plots_v_interp_old"), py::arg("rois_old"),
          py::arg("rois_new"), py::arg("convertseedinfo_in"), py::arg("spacing"), 
          py::arg("border_interp"), py::arg("num_img"), py::arg("total_imgs"),
          "Converts displacements between configurations. Returns (plot_u_new, plot_v_new, plot_validpoints, status).");
          
    m.def("disp_grad", &py_disp_grad,
          py::arg("plot_u_in"), py::arg("plot_v_in"), py::arg("roi_in"), 
          py::arg("radius_strain_in"), py::arg("pixtounits_in"), py::arg("spacing_in"),
          py::arg("subsettrunc_in"), py::arg("num_img_in"), py::arg("total_imgs_in"),
          "Calculates displacement gradients. Returns (dudx, dudy, dvdx, dvdy, validpoints, status).");

    m.def("test_openmp", &py_test_openmp, py::call_guard<py::gil_scoped_release>(),
          "Tests if OpenMP is functional. Returns boolean.");

    m.def("calc_seeds", &py_calc_seeds,
          py::arg("reference_in"), py::arg("current_in"), py::arg("roi_in"), py::arg("num_region_in"),
          py::arg("pos_seed_in"), py::arg("radius_in"), py::arg("cutoff_diffnorm_in"), py::arg("cutoff_iteration_in"),
          py::arg("enabled_stepanalysis_in"), py::arg("subsettrunc_in"),
          py::call_guard<py::gil_scoped_release>(),
          "Calculates initial seeds for DIC. Returns (seedinfo_list, convergence_list, status).");

    m.def("rgdic", &py_rgdic,
          py::arg("reference_in"), py::arg("current_in"), py::arg("roi_in"),
          py::arg("seedinfo_in"), py::arg("threaddiagram_in"),
          py::arg("radius_in"), py::arg("spacing_in"), py::arg("cutoff_diffnorm_in"), py::arg("cutoff_iteration_in"),
          py::arg("subsettrunc_in"), py::arg("num_img_overall_in"), py::arg("total_imgs_overall_in"),
          py::call_guard<py::gil_scoped_release>(),
          "Performs Reliability-Guided DIC. Returns (plot_u, plot_v, plot_corrcoef, plot_validpoints, status).");
}