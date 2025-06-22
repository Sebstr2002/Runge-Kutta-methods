#pragma once
#include <cmath>
#include <functional>
#include <vector>

namespace constants {
// this tolerance will make sure tableaus are evaluated correctly
constexpr double tolerance = 1e-10;
} // namespace constants

bool checkImplicit(const std::vector<std::vector<double>> A);

class ButcherTableau {
private:
  std::vector<std::vector<double>> A;
  std::vector<double> b;
  std::vector<double> c;
  bool implicit;

public:
  // constructor
  ButcherTableau(std::vector<std::vector<double>> A_, std::vector<double> b_,
                 std::vector<double> c_);

  // getters
  const std::vector<std::vector<double>> &getA() const;
  const std::vector<double> &getB() const;
  const std::vector<double> &getC() const;
  const bool &isImplicit() const;

  // properties bool isValid() const;
  bool isValid() const;
  bool isSymplectic() const;
};

// general runge-kutta methods and some famous examples
namespace rungekutta {

std::vector<std::vector<double>> runge_kutta(
    const ButcherTableau &table,
    const std::function<std::vector<double>(const std::vector<double> &)> &f,
    std::vector<double> y0, double t0, double dt, size_t steps,
    int max_iter = 10);
}

namespace methods {

// second order explicit RK methods
extern const ButcherTableau Heun_tableau;

// second order implicit RK methods
extern const ButcherTableau Trapezoidal_tableau;

extern const ButcherTableau Implicit_midpoint_tableau;

// forth order explicit RK methods
extern const ButcherTableau RK4_tableau;

// forth order implicit RK methods
extern const ButcherTableau LobattoIIIA_tableau;

extern const ButcherTableau Gauss_Legendre_tableau;

// concrete method computators
//
// second order explicit RK methods
std::vector<std::vector<double>> Heun_method(
    const std::function<std::vector<double>(const std::vector<double> &)> &f,
    std::vector<double> y0, double t0, double dt, size_t steps,
    int max_iter = 10);
// second order implicit RK methods
std::vector<std::vector<double>> Trapezoidal_method(
    const std::function<std::vector<double>(const std::vector<double> &)> &f,
    std::vector<double> y0, double t0, double dt, size_t steps,
    int max_iter = 10);

std::vector<std::vector<double>> Implicit_midpoint_method(
    const std::function<std::vector<double>(const std::vector<double> &)> &f,
    std::vector<double> y0, double t0, double dt, size_t steps,
    int max_iter = 10);

// forth order explicit RK methods
std::vector<std::vector<double>> RK4_method(
    const std::function<std::vector<double>(const std::vector<double> &)> &f,
    std::vector<double> y0, double t0, double dt, size_t steps,
    int max_iter = 10);

// forth order implicit RK methods
std::vector<std::vector<double>> LobattoIIIA_method(
    const std::function<std::vector<double>(const std::vector<double> &)> &f,
    std::vector<double> y0, double t0, double dt, size_t steps,
    int max_iter = 10);

std::vector<std::vector<double>> Gauss_Legendre_method(
    const std::function<std::vector<double>(const std::vector<double> &)> &f,
    std::vector<double> y0, double t0, double dt, size_t steps,
    int max_iter = 10);

} // namespace methods
