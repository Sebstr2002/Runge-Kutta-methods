#include "src/physics.hpp"
#include "src/runge-kutta.hpp"
#include "src/utils.hpp"
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

PYBIND11_MODULE(hamsolver, m) {
  m.doc() = "Hamiltonian Runge-Kutta solvers";

  // 1. Bind the ButcherTableau so Python can hold references to it
  py::class_<ButcherTableau>(m, "ButcherTableau")
      // Standard 3-argument constructor with kwargs
      .def(py::init<std::vector<std::vector<double>>, std::vector<double>,
                    std::vector<double>>(),
           py::arg("A"), py::arg("b"), py::arg("c"))
      // Adaptive 4-argument constructor with kwargs
      .def(py::init<std::vector<std::vector<double>>, std::vector<double>,
                    std::vector<double>, std::vector<double>>(),
           py::arg("A"), py::arg("b"), py::arg("c"), py::arg("bstar"))
      .def("is_implicit", &ButcherTableau::isImplicit)
      .def("is_valid", &ButcherTableau::isValid)
      .def("is_symplectic", &ButcherTableau::isSymplectic);

  // 2. Export the general runge_kutta function
  // Standard RK
  m.def("runge_kutta", &rungekutta::runge_kutta, "General RK solver",
        py::arg("table"), py::arg("f"), py::arg("yn"), py::arg("t0"),
        py::arg("dt"), py::arg("steps"), py::arg("max_iter") = 10);

  // Adaptive RK (Notice the fixed name and new dt_out argument!)
  m.def("adaptive_runge_kutta", &rungekutta::adaptive_runge_kutta,
        "Adaptive RK solver with Dense Output", py::arg("table"), py::arg("f"),
        py::arg("yn"), py::arg("t0"), py::arg("tf"), py::arg("initial_dt"),
        py::arg("tolerance"), py::arg("max_iter") = 10,
        py::arg("dt_out") = 0.0);

  // Expose the raw interpolator just in case the Python user ever wants it
  m.def("cubic_hermite_interpolate", &utils::cubic_hermite_interpolate,
        "Interpolate states between RK steps");
  // 3. Export tableaus as module-level constants
  m.attr("Heun") = methods::Heun_tableau;
  m.attr("RK4") = methods::RK4_tableau;
  m.attr("Implicit_midpoint") = methods::Implicit_midpoint_tableau;
  m.attr("Gauss_Legendre") = methods::Gauss_Legendre_tableau;
  m.attr("Trapezoidal") = methods::Trapezoidal_tableau;
  m.attr("LobattoIIIA") = methods::LobattoIIIA_tableau;
  m.attr("BS32") = methods::BS32_tableau;

  // 4. Export the C++ physics RHS
  m.def("kepler_rhs", &physics::kepler_rhs,
        "Kepler problem RHS (C++ optimized)");
  m.def("sun_earth_moon_rhs", &physics::sun_earth_moon_rhs,
        "2D Sun-Earth-Moon RHS (C++ optimized)");
  m.def("double_pendulum_rhs", &physics::double_pendulum_rhs,
        "Hamiltonian Double Pendulum");
  m.def("cr3bp_rhs", &physics::cr3bp_rhs,
        "Circular Restricted 3-Body Problem (Rotating Frame)");
  m.def("mercury_gr_rhs", &physics::mercury_gr_rhs,
        "Post-Newtonian General Relativity");
  m.def("damped_pendulum_rhs", &physics::damped_pendulum_rhs,
        "Damped Simple Pendulum with friction");
}
