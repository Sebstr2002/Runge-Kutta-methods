#pragma once
#include <cmath>
#include <functional>
#include <tuple>
#include <vector>

namespace constants {
// this tolerance will make sure tableaus are evaluated correctly
constexpr double tolerance = 1e-10;
} // namespace constants

bool checkImplicit(const std::vector<std::vector<double>> &A);

class ButcherTableau {
private:
  std::vector<std::vector<double>> A;
  std::vector<double> b;
  std::vector<double> b_star;
  std::vector<double> c;
  bool implicit;
  bool is_embedded = false;

public:
  // constructor
  ButcherTableau(std::vector<std::vector<double>> A_, std::vector<double> b_,
                 std::vector<double> c_);
  ButcherTableau(std::vector<std::vector<double>> A_, std::vector<double> b_,
                 std::vector<double> c_, std::vector<double> bstar_);

  // getters
  const std::vector<std::vector<double>> &getA() const;
  const std::vector<double> &getB() const;
  const std::vector<double> &getB_star() const;
  const std::vector<double> &getC() const;
  bool isImplicit() const;

  // properties bool isValid() const;
  bool isValid() const;
  bool isSymplectic() const;
};

// general runge-kutta methods and some famous examples
namespace rungekutta {
// Standard solver
std::tuple<std::vector<double>, std::vector<std::vector<double>>> runge_kutta(
    const ButcherTableau &table,
    const std::function<std::vector<double>(double,
                                            const std::vector<double> &)> &f,
    std::vector<double> yn, double t0, double dt, size_t steps,
    int max_iter = 10);

// Adaptive solver
std::tuple<std::vector<double>, std::vector<std::vector<double>>>
adaptive_runge_kutta(const ButcherTableau &table,
                     const std::function<std::vector<double>(
                         double, const std::vector<double> &)> &f,
                     std::vector<double> yn, double t0, double tf,
                     double initial_dt, double tolerance, int max_iter = 10,
                     double dt_out = 0.0);
}; // namespace rungekutta

namespace methods {
extern const ButcherTableau Heun_tableau;
extern const ButcherTableau Trapezoidal_tableau;
extern const ButcherTableau Implicit_midpoint_tableau;
extern const ButcherTableau RK4_tableau;
extern const ButcherTableau LobattoIIIA_tableau;
extern const ButcherTableau Gauss_Legendre_tableau;
extern const ButcherTableau BS32_tableau;
} // namespace methods
