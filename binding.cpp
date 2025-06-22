#include <pybind11/pybind11.h>

#include "method_bindings.hpp"
#include "src/runge-kutta.hpp"

// The binding module
PYBIND11_MODULE(hamsolver, m) {
  m.doc() = "Hamiltonian Runge-Kutta solvers";
  bind_rk_method(m, "Heun_method", methods::Heun_method);
  bind_rk_method(m, "RK4_method", methods::RK4_method);
  bind_rk_method(m, "Implicit_midpoint_method",
                 methods::Implicit_midpoint_method);
  bind_rk_method(m, "Gauss_Legendre_method", methods::Gauss_Legendre_method);
  bind_rk_method(m, "Trapezoidal_method", methods::Trapezoidal_method);
  bind_rk_method(m, "LobattoIIIA_method", methods::LobattoIIIA_method);
}
