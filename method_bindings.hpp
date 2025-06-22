#pragma once

#include <cstring>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <vector>

namespace py = pybind11;

template <typename MethodFunc>
void bind_rk_method(py::module_ &m, const std::string &name,
                    MethodFunc method) {
  m.def(
      name.c_str(),
      [method](py::function py_rhs, py::array_t<double> y0, double t0,
               double dt, size_t steps, int max_iter) {
        std::vector<double> y(y0.data(), y0.data() + y0.size());

        auto rhs =
            [py_rhs](double t,
                     const std::vector<double> &state) -> std::vector<double> {
          py::array_t<double> y_in(state.size(), state.data());
          py::object result = py_rhs(t, y_in);
          py::array_t<double> y_out = result.cast<py::array_t<double>>();
          std::vector<double> out(y_out.size());
          std::memcpy(out.data(), y_out.data(), y_out.size() * sizeof(double));
          return out;
        };

        auto result = method(rhs, y, t0, dt, steps, max_iter);
        size_t rows = result.size(), cols = result[0].size();
        py::array_t<double> out({rows, cols});
        auto buf = out.mutable_unchecked<2>();
        for (size_t i = 0; i < rows; ++i)
          for (size_t j = 0; j < cols; ++j)
            buf(i, j) = result[i][j];
        return out;
      },
      py::arg("rhs"), py::arg("y0"), py::arg("t0"), py::arg("dt"),
      py::arg("steps"), py::arg("max_iter"));
}
