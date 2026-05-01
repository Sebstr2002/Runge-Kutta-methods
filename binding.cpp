#include "src/physics.hpp"
#include "src/runge-kutta.hpp"
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

PYBIND11_MODULE(hamsolver, m) {
  m.doc() = "Hamiltonian Runge-Kutta solvers";

  // 1. Bind the ButcherTableau so Python can hold references to it
  py::class_<ButcherTableau>(m, "ButcherTableau")
      .def("is_implicit", &ButcherTableau::isImplicit)
      .def("is_valid", &ButcherTableau::isValid)
      .def("is_symplectic", &ButcherTableau::isSymplectic);

  // 2. Export the general runge_kutta function
  m.def("runge_kutta", &rungekutta::runge_kutta, "General RK solver",
        py::arg("table"), py::arg("f"), py::arg("yn"), py::arg("t0"),
        py::arg("dt"), py::arg("steps"), py::arg("max_iter") = 10);

  // 3. Export tableaus as module-level constants
  m.attr("Heun") = methods::Heun_tableau;
  m.attr("RK4") = methods::RK4_tableau;
  m.attr("Implicit_midpoint") = methods::Implicit_midpoint_tableau;
  m.attr("Gauss_Legendre") = methods::Gauss_Legendre_tableau;
  m.attr("Trapezoidal") = methods::Trapezoidal_tableau;
  m.attr("LobattoIIIA") = methods::LobattoIIIA_tableau;

  // 4. Export the C++ physics RHS
  m.def("kepler_rhs", &physics::kepler_rhs,
        "Kepler problem RHS (C++ optimized)");
}
