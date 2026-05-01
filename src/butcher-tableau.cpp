#include "runge-kutta.hpp"
#include <cmath>
#include <stdexcept>
#include <vector>

// --- Imlplicity check ---
bool checkImplicit(const std::vector<std::vector<double>> &A) {
  size_t s = A.size();
  for (size_t i = 0; i < s; ++i) {
    for (size_t j = i; j < s; ++j) {
      if (std::abs(A[i][j]) > constants::tolerance)
        return true;
    }
  }
  return false; // all A[i][j] for j ≥ i are zero → explicit
}
// --- Constructor ---
ButcherTableau::ButcherTableau(std::vector<std::vector<double>> A_,
                               std::vector<double> b_, std::vector<double> c_) {
  implicit = checkImplicit(A_);
  A = std::move(A_);
  b = std::move(b_);
  c = std::move(c_);

  if (!isValid()) {
    throw std::invalid_argument("Invalid arguments");
  }
}

// --- Getters ---
const std::vector<std::vector<double>> &ButcherTableau::getA() const {
  return A;
}
const std::vector<double> &ButcherTableau::getB() const { return b; }
const std::vector<double> &ButcherTableau::getC() const { return c; }
bool ButcherTableau::isImplicit() const { return implicit; }

// --- Validity ---
bool ButcherTableau::isValid() const {
  size_t s = A.size();
  if (b.size() != s || c.size() != s)
    return false;
  for (const auto &row : A)
    if (row.size() != s)
      return false;
  double sum = 0.0;
  for (double bi : b)
    sum += bi;
  return std::abs(sum - 1.0) < constants::tolerance;
}

// --- Symplecticity check ---
bool ButcherTableau::isSymplectic() const {
  size_t s = b.size();
  for (size_t i = 0; i < s; ++i) {
    for (size_t j = 0; j < s; ++j) {
      double lhs = b[i] * A[i][j] + b[j] * A[j][i];
      double rhs = b[i] * b[j];
      if (std::abs(lhs - rhs) > constants::tolerance)
        return false;
    }
  }
  return true;
}

namespace methods {

// second order explicit methods
const ButcherTableau Heun_tableau({{0.0, 0.0}, {1.0, 0.0}}, {0.5, 0.5},
                                  {0.0, 1.0});

// second order implicit methods
const ButcherTableau Trapezoidal_tableau({{0.0, 0.0}, {0.5, 0.5}}, {0.5, 0.5},
                                         {0.0, 1.0});

const ButcherTableau Implicit_midpoint_tableau({{0.5}}, {1.0}, {0.5});
// forth order explicit methods
const ButcherTableau RK4_tableau({{0.0, 0.0, 0.0, 0.0},
                                  {0.5, 0.0, 0.0, 0.0},
                                  {0.0, 0.5, 0.0, 0.0},
                                  {0.0, 0.0, 1.0, 0.0}},
                                 {

                                     1.0 / 6.0, 1.0 / 3.0, 1.0 / 3.0,
                                     1.0 / 6.0},
                                 {0.0, 0.5, 0.5, 1.0});

// forth order implicit methods
const ButcherTableau Gauss_Legendre_tableau(
    {{0.25, 0.25 - std::sqrt(3.0) / 6.0}, {0.25 + std::sqrt(3.0) / 6.0, 0.25}},
    {0.5, 0.5}, {0.5 - std::sqrt(3.0) / 6.0, 0.5 + std::sqrt(3.0) / 6.0});

const ButcherTableau LobattoIIIA_tableau({{0.0, 0.0, 0.0},
                                          {5.0 / 24.0, 1.0 / 3.0, -1.0 / 24.0},
                                          {1.0 / 6.0, 2.0 / 3.0, 1.0 / 6.0}},
                                         {1.0 / 6.0, 2.0 / 3.0, 1.0 / 6.0},
                                         {0.0, 0.5, 1.0});
} // namespace methods
