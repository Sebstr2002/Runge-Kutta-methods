#include "utils.hpp"
#include <cmath>

namespace utils {
std::vector<double> cubic_hermite_interpolate(const std::vector<double> &y0,
                                              const std::vector<double> &y1,
                                              const std::vector<double> &f0,
                                              const std::vector<double> &f1,
                                              double t0, double dt,
                                              double t_query) {

  double theta = (t_query - t0) / dt;
  double theta2 = theta * theta;
  double theta3 = theta2 * theta;

  // Hermite basis polynomials
  double h00 = 2.0 * theta3 - 3.0 * theta2 + 1.0;
  double h10 = theta3 - 2.0 * theta2 + theta;
  double h01 = -2.0 * theta3 + 3.0 * theta2;
  double h11 = theta3 - theta2;

  size_t dim = y0.size();
  std::vector<double> y_interp(dim);
  for (size_t i = 0; i < dim; ++i) {
    y_interp[i] =
        h00 * y0[i] + h10 * dt * f0[i] + h01 * y1[i] + h11 * dt * f1[i];
  }
  return y_interp;
}

std::vector<std::vector<double>> compute_jacobian(
    const std::function<std::vector<double>(double,
                                            const std::vector<double> &)> &f,
    double t, std::vector<double> y, const std::vector<double> &f0) {

  size_t dim = y.size();
  std::vector<std::vector<double>> J(dim, std::vector<double>(dim, 0.0));
  double eps = 1e-7;

  for (size_t j = 0; j < dim; ++j) {
    double yj_orig = y[j];
    y[j] += eps;
    std::vector<double> f1 = f(t, y);
    y[j] = yj_orig; // Restore

    for (size_t i = 0; i < dim; ++i) {
      J[i][j] = (f1[i] - f0[i]) / eps;
    }
  }
  return J;
}

std::vector<double> solve_linear_system(std::vector<std::vector<double>> A,
                                        std::vector<double> b) {

  size_t n = b.size();
  // Gaussian Elimination with Partial Pivoting
  for (size_t i = 0; i < n; ++i) {
    size_t max_row = i;
    for (size_t k = i + 1; k < n; ++k) {
      if (std::abs(A[k][i]) > std::abs(A[max_row][i]))
        max_row = k;
    }
    std::swap(A[i], A[max_row]);
    std::swap(b[i], b[max_row]);

    for (size_t k = i + 1; k < n; ++k) {
      double factor = A[k][i] / A[i][i];
      for (size_t j = i; j < n; ++j)
        A[k][j] -= factor * A[i][j];
      b[k] -= factor * b[i];
    }
  }

  // Back-substitution
  std::vector<double> x(n, 0.0);
  for (int i = n - 1; i >= 0; --i) {
    double sum = 0.0;
    for (size_t j = i + 1; j < n; ++j)
      sum += A[i][j] * x[j];
    x[i] = (b[i] - sum) / A[i][i];
  }
  return x;
}
} // namespace utils
