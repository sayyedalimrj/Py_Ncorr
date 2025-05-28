#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

// It's assumed that the include paths in setup.py will point to cpp_src/ncorr_lib/
// so these includes should work.
#include "standard_datatypes.h"
#include "ncorr_datatypes.h"
#include "ncorr_lib.h"

namespace py = pybind11;

// Helper macro for binding array-like classes
#define BIND_ARRAY_CLASS(Module, ClassName, BoundName, CppType, PyFormatDescriptor) \
    py::class_<ClassName>(Module, BoundName, py::buffer_protocol()) \
        .def(py::init<>()) \
        .def_readwrite("width", &ClassName::width) \
        .def_readwrite("height", &ClassName::height) \
        /* .def_readwrite("value", &ClassName::value) // Exposing raw pointer directly can be unsafe */ \
        .def("reset", &ClassName::reset) \
        .def("alloc", &ClassName::alloc, py::arg("h"), py::arg("w")) \
        .def("free", &ClassName::free) \
        .def_buffer([](ClassName &self) -> py::buffer_info { \
            if (!self.value || self.width == 0 || self.height == 0) { \
                return py::buffer_info( \
                    nullptr,                               /* Pointer to buffer */ \
                    sizeof(CppType),                       /* Size of one scalar */ \
                    PyFormatDescriptor,                    /* Python struct-style format descriptor */ \
                    2,                                     /* Number of dimensions */ \
                    {0, 0},                                /* Buffer dimensions */ \
                    {0, 0}                                 /* Strides (in bytes) for each index */ \
                ); \
            } \
            return py::buffer_info( \
                self.value,                               /* Pointer to buffer */ \
                sizeof(CppType),                       /* Size of one scalar */ \
                PyFormatDescriptor,                    /* Python struct-style format descriptor */ \
                2,                                     /* Number of dimensions */ \
                { (size_t)self.height, (size_t)self.width }, /* Buffer dimensions */ \
                { sizeof(CppType) * self.width,      /* Strides (in bytes) for each index */ \
                  sizeof(CppType) } \
            ); \
        }) \
        /* Add a method to get the value as a NumPy array for easier Python manipulation if direct buffer is tricky for users */ \
        .def("get_value_numpy", [](ClassName &self) -> py::array_t<CppType> { \
            if (!self.value || self.width == 0 || self.height == 0) return py::array_t<CppType>(); \
            /* Create a new array by copying data, or return a view if ownership is clear */ \
            /* For simplicity, returning a copy here to avoid ownership issues with raw pointers directly */ \
            /* A view would be: py::array_t<CppType>({self.height, self.width}, {sizeof(CppType) * self.width, sizeof(CppType)}, self.value, py::cast(self)); */ \
            /* Let's stick to the buffer protocol primarily, this is just an alternative getter */ \
            py::array_t<CppType> arr({self.height, self.width}); \
            CppType *ptr = static_cast<CppType *>(arr.request().ptr); \
            for (int i = 0; i < self.height; ++i) { \
                for (int j = 0; j < self.width; ++j) { \
                    ptr[i * self.width + j] = self.value[i + j * self.height]; /* Original Ncorr is column-major in MATLAB representation for mex value access */ \
                } \
            } \
            return arr; \
        }, "Returns the 'value' data as a NumPy array (column-major access like original Ncorr for 'value' was typically [j+i*height])") \
        .def("set_value_numpy", [](ClassName &self, py::array_t<CppType, py::array::c_style | py::array::forcecast> arr) { \
            py::buffer_info buf = arr.request(); \
            if (buf.ndim != 2 || buf.shape[0] != self.height || buf.shape[1] != self.width) { \
                throw std::runtime_error("Input array dimensions must match class dimensions."); \
            } \
            if (!self.value) throw std::runtime_error("'value' buffer not allocated in CppClass. Call alloc() first."); \
            CppType *ptr = static_cast<CppType *>(buf.ptr); \
            for (int i = 0; i < self.height; ++i) { \
                for (int j = 0; j < self.width; ++j) { \
                    self.value[i + j * self.height] = ptr[i * self.width + j]; /* Original Ncorr is column-major in MATLAB representation for mex value access */ \
                } \
            } \
        }, "Sets the 'value' data from a NumPy array (assumes input array is HxW).")

PYBIND11_MODULE(_ncorr_cpp_core, m) {
    m.doc() = "Pybind11 bindings for Ncorr C++ core data types and library functions";

    // Bind enums from ncorr_lib.h
    py::enum_<OUT>(m, "NcorrOutStatus")
        .value("CANCELLED", OUT::CANCELLED)
        .value("FAILED", OUT::FAILED)
        .value("SUCCESS", OUT::SUCCESS)
        .export_values();

    // Bind standard_datatypes.h classes
    BIND_ARRAY_CLASS(m, class_double_array, "CppClassDoubleArray", double, py::format_descriptor<double>::format());
    BIND_ARRAY_CLASS(m, class_integer_array, "CppClassIntegerArray", int, py::format_descriptor<int>::format());
    BIND_ARRAY_CLASS(m, class_logical_array, "CppClassLogicalArray", bool, py::format_descriptor<bool>::format());
    
    // Bind ncorr_datatypes.h classes/structs
    py::class_<ncorr_class_img>(m, "CppNcorrClassImg")
        .def(py::init<>())
        .def_readwrite("type", &ncorr_class_img::type)
        .def_readwrite("gs", &ncorr_class_img::gs) // CppClassDoubleArray
        .def_readwrite("max_gs", &ncorr_class_img::max_gs)
        .def_readwrite("bcoef", &ncorr_class_img::bcoef) // CppClassDoubleArray
        .def_readwrite("border_bcoef", &ncorr_class_img::border_bcoef);

    py::class_<ncorr_class_region>(m, "CppNcorrClassRegion")
        .def(py::init<>())
        .def_readwrite("nodelist", &ncorr_class_region::nodelist) // CppClassIntegerArray
        .def_readwrite("noderange", &ncorr_class_region::noderange) // CppClassIntegerArray
        .def_readwrite("upperbound", &ncorr_class_region::upperbound)
        .def_readwrite("lowerbound", &ncorr_class_region::lowerbound)
        .def_readwrite("leftbound", &ncorr_class_region::leftbound)
        .def_readwrite("rightbound", &ncorr_class_region::rightbound)
        .def_readwrite("totalpoints", &ncorr_class_region::totalpoints)
        .def("alloc", &ncorr_class_region::alloc, py::arg("h"), py::arg("w"))
        .def("free", &ncorr_class_region::free);
        
    // Bind vec_struct_region from ncorr_lib.h as it's used in form_regions/form_union
    py::class_<vec_struct_region>(m, "CppVecStructRegion")
        .def(py::init<>())
        .def_readwrite("nodelist", &vec_struct_region::nodelist) // std::vector<int>
        .def_readwrite("noderange", &vec_struct_region::noderange) // std::vector<int>
        .def_readwrite("height_nodelist", &vec_struct_region::height_nodelist)
        .def_readwrite("width_nodelist", &vec_struct_region::width_nodelist)
        .def_readwrite("upperbound", &vec_struct_region::upperbound)
        .def_readwrite("lowerbound", &vec_struct_region::lowerbound)
        .def_readwrite("leftbound", &vec_struct_region::leftbound)
        .def_readwrite("rightbound", &vec_struct_region::rightbound)
        .def_readwrite("totalpoints", &vec_struct_region::totalpoints);

    py::class_<struct_cirroi>(m, "CppStructCirroi")
        .def(py::init<>())
        .def_readwrite("region", &struct_cirroi::region) // CppNcorrClassRegion
        .def_readwrite("mask", &struct_cirroi::mask)     // CppClassLogicalArray
        .def_readwrite("radius", &struct_cirroi::radius)
        .def_readwrite("x", &struct_cirroi::x)
        .def_readwrite("y", &struct_cirroi::y);

    py::class_<ncorr_class_roi>(m, "CppNcorrClassRoi")
        .def(py::init<>())
        .def_readwrite("mask", &ncorr_class_roi::mask) // CppClassLogicalArray
        .def_readwrite("region", &ncorr_class_roi::region) // std::vector<CppNcorrClassRegion>
        .def_readwrite("cirroi", &ncorr_class_roi::cirroi) // std::vector<CppStructCirroi>
        .def("set_cirroi", &ncorr_class_roi::set_cirroi, py::arg("radius_i"), py::arg("thread_total"))
        .def("update_cirroi", &ncorr_class_roi::update_cirroi, py::arg("num_region"), py::arg("thread_num"))
        .def("get_cirroi", &ncorr_class_roi::get_cirroi, py::arg("x_i"), py::arg("y_i"), py::arg("num_region"), py::arg("subsettrunc"), py::arg("thread_num"))
        .def("withinregion", &ncorr_class_roi::withinregion, py::arg("x_i"), py::arg("y_i"), py::arg("num_region"));

    // Bind ncorr_class_inverseregion (inherits from ncorr_class_region)
    py::class_<ncorr_class_inverseregion, ncorr_class_region>(m, "CppNcorrClassInverseRegion")
        .def(py::init<ncorr_class_region&, int>(), py::arg("region"), py::arg("border_extrap"));


    // Bind ncorr_lib.h utility functions
    m.def("ncorr_round", &ncorr_round, py::arg("r"));
    m.def("sign", &sign, py::arg("r"));
    m.def("mod_pos", &mod_pos, py::arg("i"), py::arg("n"));

    m.def("form_boundary", [](const std::vector<int>& point_init_in, const class_logical_array& mask_in, int direc_in) {
        std::vector<std::vector<int>> vec_boundary_out;
        int direc_out = direc_in; 
        form_boundary(vec_boundary_out, point_init_in, mask_in, direc_out);
        return std::make_tuple(vec_boundary_out, direc_out);
    }, py::arg("point_init"), py::arg("mask"), py::arg("direc_in_out"), 
       "Forms the boundary of a masked region. Returns (boundary_points, updated_direction).");

    m.def("form_regions", [](const class_logical_array& mask_in, int cutoff_in, bool preservelength_in) {
        std::vector<vec_struct_region> region_out;
        bool removed_out;
        form_regions(region_out, removed_out, mask_in, cutoff_in, preservelength_in);
        return std::make_tuple(region_out, removed_out);
    }, py::arg("mask"), py::arg("cutoff"), py::arg("preservelength"),
       "Forms contiguous regions from a mask. Returns (list_of_regions, removed_flag).");
    
    m.def("form_union", [](const std::vector<ncorr_class_region>& region_in, const class_logical_array& mask_in, bool inplace_in) {
        // If inplace is true, region_in would need to be non-const. 
        // For Python, returning a new list is often cleaner unless memory is a huge concern and inplace modification is explicitly desired.
        // The C++ signature for form_union expects region_union_out (std::vector<vec_struct_region>&) and region_in (const std::vector<ncorr_class_region>&)
        // This suggests it could convert ncorr_class_region to vec_struct_region internally or that region_union_out is the primary output type.
        // For simplicity, let's assume the Python side will provide `region_in` and expect a new `region_union_out`.
        // The original C++ form_union is `void form_union(std::vector<vec_struct_region> &region_union,const std::vector<ncorr_class_region> &region,const class_logical_array &mask,const bool &inplace);`
        // If inplace is false, region_union is allocated and filled.
        // If inplace is true, it implies region_union and region are the same and it's modified. This is hard to map directly if region_in is const.
        // Given current C++ `form_union`, it's designed to output to `region_union_out`.
        if (inplace_in) {
             throw std::runtime_error("In-place form_union from Python bindings is not directly supported in this wrapper due to const correctness of input 'region'. Use inplace=False.");
        }
        std::vector<vec_struct_region> region_union_out;
        region_union_out.resize(region_in.size()); // Pre-allocate based on notes in ncorr_lib.cpp
        form_union(region_union_out, region_in, mask_in, inplace_in);
        return region_union_out;
    }, py::arg("region_input"), py::arg("mask"), py::arg("inplace"),
       "Forms the union of regions with a mask. Returns a list of CppVecStructRegion.");

    m.def("cholesky", [](std::vector<double>& mat_in_out, int size_mat_in) {
        bool positivedef_out;
        cholesky(mat_in_out, positivedef_out, size_mat_in);
        // mat_in_out is modified in-place, Python list passed by value will be copied, so return it.
        // For a NumPy array, modification could be in-place if buffer protocol is used carefully.
        // Here, std::vector<double> is passed by value from Python (copied to C++ vector), then returned.
        return std::make_tuple(mat_in_out, positivedef_out);
    }, py::arg("mat_in_out").noconvert(), py::arg("size_mat"), "Performs Cholesky decomposition. Modifies 'mat_in_out' in-place and returns (mat_out, positive_definite_flag).");

    m.def("forwardsub", [](std::vector<double>& vec_in_out, const std::vector<double>& mat_in, int size_mat_in) {
        forwardsub(vec_in_out, mat_in, size_mat_in);
        return vec_in_out; // vec_in_out modified in-place
    }, py::arg("vec_in_out").noconvert(), py::arg("mat_in"), py::arg("size_mat"), "Performs forward substitution.");

    m.def("backwardsub", [](std::vector<double>& vec_in_out, const std::vector<double>& mat_in, int size_mat_in) {
        backwardsub(vec_in_out, mat_in, size_mat_in);
        return vec_in_out; // vec_in_out modified in-place
    }, py::arg("vec_in_out").noconvert(), py::arg("mat_in"), py::arg("size_mat"), "Performs backward substitution.");

    m.def("interp_qbs", [](double x_tilda_in, double y_tilda_in, const class_double_array& plot_interp_in, const class_logical_array& mask_in, int offset_x_in, int offset_y_in, int border_bcoef_in) {
        double interp_out;
        OUT status = interp_qbs(interp_out, x_tilda_in, y_tilda_in, plot_interp_in, mask_in, offset_x_in, offset_y_in, border_bcoef_in);
        return std::make_tuple(interp_out, status);
    }, py::arg("x_tilda"), py::arg("y_tilda"), py::arg("plot_interp"), py::arg("mask"), py::arg("offset_x"), py::arg("offset_y"), py::arg("border_bcoef"),
       "Biquintic B-spline interpolation. Returns (interpolated_value, status). Warning: Uses static C++ buffers, not reentrant for multithreading from Python.");

    m.def("expand_filt", [](class_double_array& plot_extrap_in_out, const ncorr_class_inverseregion& inverseregion_in) {
        // class_double_array will be passed by reference if Python object is an instance of the bound CppClassDoubleArray
        expand_filt(plot_extrap_in_out, inverseregion_in);
        // plot_extrap_in_out is modified in-place. No explicit return needed if modifying the Python object's C++ counterpart.
    }, py::arg("plot_extrap").noconvert(), py::arg("inverseregion"), "Expands and filters data.");

}