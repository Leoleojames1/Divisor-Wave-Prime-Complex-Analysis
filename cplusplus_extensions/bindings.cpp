#include <pybind11/pybind11.h>
#include "cplusplus_extensions/graphics.cpp"

namespace py = pybind11;

PYBIND11_MODULE(graphics, m) {
    m.def("create_plot_2D", &create_plot_2D, "Create a 2D plot using C++ code");
}