// ──────────────────────────────────────────────────────────────────────────────
//  bindings/py_ncorr_core_bindings.cpp
//  Compatible with pybind11 ≥ 2.12   (array::set_writeable removed)
//  MSVC-safe: lambdas capture by copy via `[=]`
// ──────────────────────────────────────────────────────────────────────────────
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

#include "standard_datatypes.h"
#include "ncorr_datatypes.h"
#include "ncorr_lib.h"

namespace py  = pybind11;

// ─────────────────────────── helper: wrap C++ array into NumPy
template <typename ArrCpp>
static py::array make_numpy_from_cpp(const ArrCpp &arr_cpp,
                                     bool /*readonly*/ = true)
{
    if (!arr_cpp.value)
        throw std::runtime_error("Array has no data pointer.");

    const auto h = arr_cpp.height;
    const auto w = arr_cpp.width;

    // NOTE: owner capsule is nullptr because the C++ object owns the data.
    // NumPy will NOT try to free it.
    py::capsule owner((void *)nullptr);
    return py::array({h, w},
                     {sizeof(double) * w, sizeof(double)},
                     arr_cpp.value,
                     owner);               // writeable by default
}

// ──────────────────────────────────────────────────────────────────────────────
PYBIND11_MODULE(_ncorr_cpp_core, m)
{
    m.doc() = "Pybind11 bindings for Ncorr core C++ library";

    // ---------------------------------------------------------------- Arrays --
    py::class_<class_double_array>(m, "CppClassDoubleArray")
        .def(py::init<>())
        .def_property("width",
                      [=](class_double_array &self) { return self.width; },
                      [=](class_double_array &self, int w) { self.width = w; })
        .def_property("height",
                      [=](class_double_array &self) { return self.height; },
                      [=](class_double_array &self, int h) { self.height = h; })
        .def("get_value_numpy",
             [=](class_double_array &self) { return make_numpy_from_cpp(self); })
        .def("set_value_numpy",
             [=](class_double_array &self,
                 const py::array_t<double,
                                   py::array::c_style | py::array::forcecast> &arr)
             {
                 if (arr.ndim() != 2)
                     throw std::runtime_error("Array must be 2-D.");
                 self.free();
                 self.alloc(static_cast<int>(arr.shape(0)),
                            static_cast<int>(arr.shape(1)));
                 std::memcpy(self.value,
                             arr.data(),
                             sizeof(double) * self.width * self.height);
             })
        .def("alloc", [=](class_double_array &self, int h, int w)
             { self.alloc(h, w); })
        .def("reset", [=](class_double_array &self) { self.reset(); })
        .def("free",  [=](class_double_array &self) { self.free(); });

    py::class_<class_integer_array>(m, "CppClassIntegerArray")
        .def(py::init<>())
        .def_property("width",
                      [=](class_integer_array &self) { return self.width; },
                      [=](class_integer_array &self, int w) { self.width = w; })
        .def_property("height",
                      [=](class_integer_array &self) { return self.height; },
                      [=](class_integer_array &self, int h) { self.height = h; })
        .def("get_value_numpy",
             [=](class_integer_array &self)
             {
                 return py::array({self.height, self.width},
                                  {sizeof(int) * self.width, sizeof(int)},
                                  self.value);
             })
        .def("set_value_numpy",
             [=](class_integer_array &self,
                 const py::array_t<int,
                                   py::array::c_style | py::array::forcecast> &arr)
             {
                 if (arr.ndim() != 2)
                     throw std::runtime_error("Array must be 2-D.");
                 self.free();
                 self.alloc(static_cast<int>(arr.shape(0)),
                            static_cast<int>(arr.shape(1)));
                 std::memcpy(self.value,
                             arr.data(),
                             sizeof(int) * self.width * self.height);
             })
        .def("alloc", [=](class_integer_array &self, int h, int w)
             { self.alloc(h, w); })
        .def("reset", [=](class_integer_array &self) { self.reset(); })
        .def("free",  [=](class_integer_array &self) { self.free(); });

    py::class_<class_logical_array>(m, "CppClassLogicalArray")
        .def(py::init<>())
        .def_property("width",
                      [=](class_logical_array &self) { return self.width; },
                      [=](class_logical_array &self, int w) { self.width = w; })
        .def_property("height",
                      [=](class_logical_array &self) { return self.height; },
                      [=](class_logical_array &self, int h) { self.height = h; })
        .def("get_value_numpy",
             [=](class_logical_array &self)
             {
                 return py::array({self.height, self.width},
                                  {sizeof(bool) * self.width, sizeof(bool)},
                                  self.value);
             })
        .def("set_value_numpy",
             [=](class_logical_array &self,
                 const py::array_t<bool,
                                   py::array::c_style | py::array::forcecast> &arr)
             {
                 if (arr.ndim() != 2)
                     throw std::runtime_error("Array must be 2-D.");
                 self.free();
                 self.alloc(static_cast<int>(arr.shape(0)),
                            static_cast<int>(arr.shape(1)));
                 std::memcpy(self.value,
                             arr.data(),
                             sizeof(bool) * self.width * self.height);
             })
        .def("alloc", [=](class_logical_array &self, int h, int w)
             { self.alloc(h, w); })
        .def("reset", [=](class_logical_array &self) { self.reset(); })
        .def("free",  [=](class_logical_array &self) { self.free(); });

    // ------------------------------------------------------------ Enum OUT ---
    py::enum_<OUT>(m, "NcorrOutStatus")
        .value("CANCELLED", OUT::CANCELLED)
        .value("FAILED",    OUT::FAILED)
        .value("SUCCESS",   OUT::SUCCESS)
        .export_values();

    // ----------------------------------------------------- ncorr_class_img ---
    py::class_<ncorr_class_img>(m, "CppNcorrClassImg")
        .def(py::init<>())
        .def_readwrite("type",         &ncorr_class_img::type)
        .def_readwrite("max_gs",       &ncorr_class_img::max_gs)
        .def_readwrite("border_bcoef", &ncorr_class_img::border_bcoef)
        .def_property("gs",
                      [=](ncorr_class_img &self) {
                          return py::cast(&self.gs,
                                          py::return_value_policy::reference_internal);
                      },
                      [=](ncorr_class_img &self, class_double_array &arr) {
                          self.gs = arr;
                      })
        .def_property("bcoef",
                      [=](ncorr_class_img &self) {
                          return py::cast(&self.bcoef,
                                          py::return_value_policy::reference_internal);
                      },
                      [=](ncorr_class_img &self, class_double_array &arr) {
                          self.bcoef = arr;
                      });

    // ------------------------------------------------ ncorr_class_region -----
    py::class_<ncorr_class_region>(m, "CppNcorrClassRegion")
        .def(py::init<>())
        .def_readwrite("upperbound",  &ncorr_class_region::upperbound)
        .def_readwrite("lowerbound",  &ncorr_class_region::lowerbound)
        .def_readwrite("leftbound",   &ncorr_class_region::leftbound)
        .def_readwrite("rightbound",  &ncorr_class_region::rightbound)
        .def_readwrite("totalpoints", &ncorr_class_region::totalpoints)
        .def_property("nodelist",
                      [=](ncorr_class_region &self) {
                          return py::cast(&self.nodelist,
                                          py::return_value_policy::reference_internal);
                      },
                      [=](ncorr_class_region &self, class_integer_array &arr) {
                          self.nodelist = arr;
                      })
        .def_property("noderange",
                      [=](ncorr_class_region &self) {
                          return py::cast(&self.noderange,
                                          py::return_value_policy::reference_internal);
                      },
                      [=](ncorr_class_region &self, class_integer_array &arr) {
                          self.noderange = arr;
                      })
        .def("alloc", [=](ncorr_class_region &self, int h, int w)
             { self.alloc(h, w); })
        .def("free",  [=](ncorr_class_region &self) { self.free(); });

    // -------------------------------------------- vector<vec_struct_region> --
    py::bind_vector<std::vector<vec_struct_region>>(m, "VectorVecStructRegion");

    py::class_<vec_struct_region>(m, "CppVecStructRegion")
        .def(py::init<>())
        .def_readwrite("nodelist",        &vec_struct_region::nodelist)
        .def_readwrite("noderange",       &vec_struct_region::noderange)
        .def_readwrite("height_nodelist", &vec_struct_region::height_nodelist)
        .def_readwrite("width_nodelist",  &vec_struct_region::width_nodelist)
        .def_readwrite("upperbound",      &vec_struct_region::upperbound)
        .def_readwrite("lowerbound",      &vec_struct_region::lowerbound)
        .def_readwrite("leftbound",       &vec_struct_region::leftbound)
        .def_readwrite("rightbound",      &vec_struct_region::rightbound)
        .def_readwrite("totalpoints",     &vec_struct_region::totalpoints);

    // ------------------------------------------------- struct_cirroi ---------
    py::class_<struct_cirroi>(m, "CppStructCirroi")
        .def(py::init<>())
        .def_readwrite("radius", &struct_cirroi::radius)
        .def_readwrite("x",      &struct_cirroi::x)
        .def_readwrite("y",      &struct_cirroi::y)
        .def_readwrite("region", &struct_cirroi::region)
        .def_readwrite("mask",   &struct_cirroi::mask);

    // ------------------------------------------------ ncorr_class_roi --------
    py::class_<ncorr_class_roi>(m, "CppNcorrClassRoi")
        .def(py::init<>())
        .def_readwrite("mask",   &ncorr_class_roi::mask)
        .def_readwrite("region", &ncorr_class_roi::region)
        .def_readwrite("cirroi", &ncorr_class_roi::cirroi);

    // ---------------------------------------- ncorr_class_inverseregion ------
    py::class_<ncorr_class_inverseregion, ncorr_class_region>(
        m, "CppNcorrClassInverseRegion")
        .def(py::init<ncorr_class_region&, const int&>(),
             py::arg("region"),
             py::arg("border_extrap"));

    // ------------------------------------------------------- Free functions --
    m.def("ncorr_round", [=](double r) { return ncorr_round(r); });
    m.def("sign",        [=](double r) { return sign(r); });
    m.def("mod_pos",     [=](int i, int n) { return mod_pos(i, n); });

    m.def("form_boundary",
          [=](const std::vector<int> &point_init_in,
              const class_logical_array &mask_in,
              int direc_in_out)
          {
              std::vector<std::vector<int>> vec_bnd_out;
              auto d = direc_in_out;
              form_boundary(vec_bnd_out, point_init_in, mask_in, d);
              return py::make_tuple(vec_bnd_out, d);
          });

    m.def("form_regions",
          [=](const class_logical_array &mask_in,
              int cutoff_in,
              bool preservelength_in)
          {
              std::vector<vec_struct_region> region_out;
              bool removed_out;
              form_regions(region_out, removed_out,
                           mask_in, cutoff_in, preservelength_in);
              return py::make_tuple(region_out, removed_out);
          });

    m.def("cholesky",
          [=](std::vector<double> &mat_in_out,
              int size_mat_in)
          {
              bool pd;
              cholesky(mat_in_out, pd, size_mat_in);
              return pd;
          });
}
