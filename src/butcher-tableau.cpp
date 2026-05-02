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

ButcherTableau::ButcherTableau(std::vector<std::vector<double>> A_,
                               std::vector<double> b_, std::vector<double> c_,
                               std::vector<double> bstar_) {
  implicit = checkImplicit(A_);
  A = std::move(A_);
  b = std::move(b_);
  c = std::move(c_);
  b_star = std::move(bstar_);
  is_embedded = true;

  if (!isValid()) {
    throw std::invalid_argument("Invalid arguments");
  }
}

ButcherTableau::ButcherTableau(std::vector<std::vector<double>> A_,
                               std::vector<double> b_, std::vector<double> c_,
                               std::vector<double> bstar_, int order_low_) {
  implicit = checkImplicit(A_);
  A = std::move(A_);
  b = std::move(b_);
  c = std::move(c_);
  b_star = std::move(bstar_);
  is_embedded = true;
  order_low = order_low_;

  if (!isValid()) {
    throw std::invalid_argument("Invalid arguments");
  }
}
// --- Getters ---
const std::vector<std::vector<double>> &ButcherTableau::getA() const {
  return A;
}
const std::vector<double> &ButcherTableau::getB() const { return b; }
const std::vector<double> &ButcherTableau::getB_star() const { return b_star; }
const std::vector<double> &ButcherTableau::getC() const { return c; }
bool ButcherTableau::isImplicit() const { return implicit; }
bool ButcherTableau::isEmbedded() const { return is_embedded; }
int ButcherTableau::getOrderLow() const { return order_low; }

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
const ButcherTableau BS32_tableau(
    // A matrix
    {{0.0, 0.0, 0.0, 0.0},
     {1.0 / 2.0, 0.0, 0.0, 0.0},
     {0.0, 3.0 / 4.0, 0.0, 0.0},
     {2.0 / 9.0, 1.0 / 3.0, 4.0 / 9.0, 0.0}},
    // b vector (3rd order - use this to advance the simulation)
    {2.0 / 9.0, 1.0 / 3.0, 4.0 / 9.0, 0.0},
    // c vector
    {0.0, 1.0 / 2.0, 3.0 / 4.0, 1.0},
    // b_star vector (2nd order - use this just to check the error)
    {7.0 / 24.0, 1.0 / 4.0, 1.0 / 3.0, 1.0 / 8.0},
    /*order_low=*/2);

// Classical 3/8-rule variant of RK4.  Same order, slightly better stability
// region than the standard RK4; useful as a sanity-check companion.
const ButcherTableau RK4_38_tableau({{0.0, 0.0, 0.0, 0.0},
                                     {1.0 / 3.0, 0.0, 0.0, 0.0},
                                     {-1.0 / 3.0, 1.0, 0.0, 0.0},
                                     {1.0, -1.0, 1.0, 0.0}},
                                    {1.0 / 8.0, 3.0 / 8.0, 3.0 / 8.0, 1.0 / 8.0},
                                    {0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0});

// Runge-Kutta-Fehlberg 5(4) - the original embedded pair (Fehlberg 1969).
const ButcherTableau RKF45_tableau(
    {{0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
     {1.0 / 4.0, 0.0, 0.0, 0.0, 0.0, 0.0},
     {3.0 / 32.0, 9.0 / 32.0, 0.0, 0.0, 0.0, 0.0},
     {1932.0 / 2197.0, -7200.0 / 2197.0, 7296.0 / 2197.0, 0.0, 0.0, 0.0},
     {439.0 / 216.0, -8.0, 3680.0 / 513.0, -845.0 / 4104.0, 0.0, 0.0},
     {-8.0 / 27.0, 2.0, -3544.0 / 2565.0, 1859.0 / 4104.0, -11.0 / 40.0, 0.0}},
    // 5th-order solution (use to advance)
    {16.0 / 135.0, 0.0, 6656.0 / 12825.0, 28561.0 / 56430.0, -9.0 / 50.0,
     2.0 / 55.0},
    {0.0, 1.0 / 4.0, 3.0 / 8.0, 12.0 / 13.0, 1.0, 1.0 / 2.0},
    // 4th-order embedded solution (use for error)
    {25.0 / 216.0, 0.0, 1408.0 / 2565.0, 2197.0 / 4104.0, -1.0 / 5.0, 0.0},
    /*order_low=*/4);

// Cash-Karp 5(4) - tuned for robust step control across a wider class of
// problems than RKF45.
const ButcherTableau CashKarp_tableau(
    {{0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
     {1.0 / 5.0, 0.0, 0.0, 0.0, 0.0, 0.0},
     {3.0 / 40.0, 9.0 / 40.0, 0.0, 0.0, 0.0, 0.0},
     {3.0 / 10.0, -9.0 / 10.0, 6.0 / 5.0, 0.0, 0.0, 0.0},
     {-11.0 / 54.0, 5.0 / 2.0, -70.0 / 27.0, 35.0 / 27.0, 0.0, 0.0},
     {1631.0 / 55296.0, 175.0 / 512.0, 575.0 / 13824.0, 44275.0 / 110592.0,
      253.0 / 4096.0, 0.0}},
    // 5th-order solution
    {37.0 / 378.0, 0.0, 250.0 / 621.0, 125.0 / 594.0, 0.0, 512.0 / 1771.0},
    {0.0, 1.0 / 5.0, 3.0 / 10.0, 3.0 / 5.0, 1.0, 7.0 / 8.0},
    // 4th-order embedded solution
    {2825.0 / 27648.0, 0.0, 18575.0 / 48384.0, 13525.0 / 55296.0,
     277.0 / 14336.0, 1.0 / 4.0},
    /*order_low=*/4);

// Dormand-Prince 5(4) — what MATLAB's `ode45` and SciPy's `RK45` use.
// Seven stages with the FSAL property (b == last row of A); this
// implementation does not yet exploit FSAL, so we pay for one extra RHS
// evaluation per step.
const ButcherTableau DP54_tableau(
    {{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
     {1.0 / 5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
     {3.0 / 40.0, 9.0 / 40.0, 0.0, 0.0, 0.0, 0.0, 0.0},
     {44.0 / 45.0, -56.0 / 15.0, 32.0 / 9.0, 0.0, 0.0, 0.0, 0.0},
     {19372.0 / 6561.0, -25360.0 / 2187.0, 64448.0 / 6561.0, -212.0 / 729.0,
      0.0, 0.0, 0.0},
     {9017.0 / 3168.0, -355.0 / 33.0, 46732.0 / 5247.0, 49.0 / 176.0,
      -5103.0 / 18656.0, 0.0, 0.0},
     {35.0 / 384.0, 0.0, 500.0 / 1113.0, 125.0 / 192.0, -2187.0 / 6784.0,
      11.0 / 84.0, 0.0}},
    // 5th-order solution (used to advance)
    {35.0 / 384.0, 0.0, 500.0 / 1113.0, 125.0 / 192.0, -2187.0 / 6784.0,
     11.0 / 84.0, 0.0},
    {0.0, 1.0 / 5.0, 3.0 / 10.0, 4.0 / 5.0, 8.0 / 9.0, 1.0, 1.0},
    // 4th-order embedded solution (used for error estimate)
    {5179.0 / 57600.0, 0.0, 7571.0 / 16695.0, 393.0 / 640.0,
     -92097.0 / 339200.0, 187.0 / 2100.0, 1.0 / 40.0},
    /*order_low=*/4);
} // namespace methods
