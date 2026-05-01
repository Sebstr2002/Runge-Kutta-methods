#include "runge-kutta.hpp"

namespace rungekutta {

// general runge kutta method
std::vector<std::vector<double>> runge_kutta(
    const ButcherTableau &table,
    const std::function<std::vector<double>(double,
                                            const std::vector<double> &)> &f,
    std::vector<double> yn, double t0, double dt, size_t steps, int max_iter) {
  std::vector<std::vector<double>> results;
  results.push_back(yn);

  const auto &A = table.getA();
  const auto &b = table.getB();
  const auto &c = table.getC();
  size_t s = b.size();    // number of stages so intermediate calculations
  size_t dim = yn.size(); // system dimension

  // calculates coefficients K
  for (size_t n = 0; n < steps; ++n) {
    double t = t0 + n * dt;
    std::vector<std::vector<double>> K(
        s, std::vector<double>(dim, 0.0)); // intermediates k

    if (!table.isImplicit()) {
      // Explicit RK
      for (size_t i = 0; i < s; ++i) {
        std::vector<double> yi = yn;
        for (size_t j = 0; j < i; ++j)
          for (size_t k = 0; k < dim; ++k)
            yi[k] += dt * A[i][j] * K[j][k];
        K[i] = f(t + c[i] * dt, yi);
      }
    } else {
      // Implicit RK — fixed-point iteration
      std::vector<std::vector<double>> K_prev = K;
      for (int it = 0; it < max_iter; ++it) {
        for (size_t i = 0; i < s; ++i) {
          std::vector<double> yi(dim, 0.0);
          for (size_t j = 0; j < s; ++j)
            for (size_t k = 0; k < dim; ++k)
              yi[k] += A[i][j] * K[j][k];
          for (size_t k = 0; k < dim; ++k)
            yi[k] = yn[k] + dt * yi[k];
          K[i] = f(t + c[i] * dt, yi);
        }

        // convergence check and if close enough break already
        double err = 0;
        for (size_t i = 0; i < s; ++i)
          for (size_t k = 0; k < dim; ++k)
            err += std::abs(K[i][k] - K_prev[i][k]);
        if (err < constants::tolerance)
          break;
        std::swap(K_prev, K);
      }
    }

    // Combine stages to advance solution
    std::vector<double> y_next = yn;
    for (size_t i = 0; i < s; ++i)
      for (size_t k = 0; k < dim; ++k)
        y_next[k] += dt * b[i] * K[i][k]; // calculate y n+1

    results.push_back(y_next);
    yn = y_next;
  }

  return results;
}
} // namespace rungekutta
