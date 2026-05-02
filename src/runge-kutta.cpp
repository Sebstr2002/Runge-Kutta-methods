#include "runge-kutta.hpp"
#include "utils.hpp"

namespace rungekutta {

// general runge kutta method
std::tuple<std::vector<double>, std::vector<std::vector<double>>> runge_kutta(
    const ButcherTableau &table,
    const std::function<std::vector<double>(double,
                                            const std::vector<double> &)> &f,
    std::vector<double> yn, double t0, double dt, size_t steps, int max_iter) {

  std::vector<double> times;
  std::vector<std::vector<double>> states;

  times.push_back(t0);
  states.push_back(yn);

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
              yi[k] += A[i][j] * K_prev[j][k]; // CRITICAL: Use K_prev here!

          for (size_t k = 0; k < dim; ++k)
            yi[k] = yn[k] + dt * yi[k];

          K[i] = f(t + c[i] * dt, yi); // Write new values into K
        }

        // convergence check
        double err = 0;
        for (size_t i = 0; i < s; ++i)
          for (size_t k = 0; k < dim; ++k)
            err += std::abs(K[i][k] - K_prev[i][k]);

        if (err < constants::tolerance)
          break;

        // Manually copy K to K_prev to avoid memory allocation in the hot path
        for (size_t i = 0; i < s; ++i)
          for (size_t k = 0; k < dim; ++k)
            K_prev[i][k] = K[i][k];
      }
    }

    // Combine stages to advance solution
    std::vector<double> y_next = yn;
    for (size_t i = 0; i < s; ++i)
      for (size_t k = 0; k < dim; ++k)
        y_next[k] += dt * b[i] * K[i][k]; // calculate y n+1

    times.push_back(t0 + (n + 1) * dt);
    states.push_back(y_next);
    yn = y_next;
  } // END of for loop

  return {times, states};
}

std::tuple<std::vector<double>, std::vector<std::vector<double>>>
adaptive_runge_kutta(const ButcherTableau &table,
                     const std::function<std::vector<double>(
                         double, const std::vector<double> &)> &f,
                     std::vector<double> yn, double t0, double tf,
                     double initial_dt, double tolerance, int max_iter,
                     double dt_out) {

  std::vector<double> times;
  std::vector<std::vector<double>> states;

  times.push_back(t0);
  states.push_back(yn);

  const auto &A = table.getA();
  const auto &b = table.getB();
  const auto &c = table.getC();
  const auto &bstar = table.getB_star();
  size_t s = b.size();    // number of stages so intermediate calculations
  size_t dim = yn.size(); // system dimension

  double t = t0;
  double dt = initial_dt;
  const double safety = 0.9;

  // We track the next "Animation Frame" we need to output
  double t_out_next = t0 + dt_out;
  bool use_dense_output = (dt_out > 0.0);

  while (t < tf) {
    if (t + dt > tf)
      dt = tf - t;

    // --- 1. Calculate K stages ---
    std::vector<std::vector<double>> K(s, std::vector<double>(dim, 0.0));
    if (!table.isImplicit()) {
      for (size_t i = 0; i < s; ++i) {
        std::vector<double> yi = yn;
        for (size_t j = 0; j < i; ++j)
          for (size_t k = 0; k < dim; ++k)
            yi[k] += dt * A[i][j] * K[j][k];
        K[i] = f(t + c[i] * dt, yi);
      }
    } else {
      // Implicit fixed-point iteration
      std::vector<std::vector<double>> K_prev = K;
      for (int it = 0; it < max_iter; ++it) {
        for (size_t i = 0; i < s; ++i) {
          std::vector<double> yi(dim, 0.0);
          for (size_t j = 0; j < s; ++j)
            for (size_t k = 0; k < dim; ++k)
              yi[k] += A[i][j] * K_prev[j][k];
          for (size_t k = 0; k < dim; ++k)
            yi[k] = yn[k] + dt * yi[k];
          K[i] = f(t + c[i] * dt, yi);
        }
        double err = 0;
        for (size_t i = 0; i < s; ++i)
          for (size_t k = 0; k < dim; ++k)
            err += std::abs(K[i][k] - K_prev[i][k]);
        if (err < constants::tolerance)
          break;
        K_prev = K;
      }
    }

    // --- 2. Calculate y_next and y_star ---
    std::vector<double> y_next = yn;
    std::vector<double> y_star = yn;
    for (size_t i = 0; i < s; ++i) {
      for (size_t k = 0; k < dim; ++k) {
        y_next[k] += dt * b[i] * K[i][k];
        y_star[k] += dt * bstar[i] * K[i][k]; // Uses the embedded weights!
      }
    }

    // --- 3. Error checking ---
    double max_error = 0.0;
    for (size_t k = 0; k < dim; ++k) {
      double diff = std::abs(y_next[k] - y_star[k]);
      if (diff > max_error)
        max_error = diff;
    }
    if (max_error == 0.0)
      max_error = 1e-15;

    double scale = safety * std::pow(tolerance / max_error, 1.0 / 3.0);
    if (max_error <= tolerance) {
      // --- STEP ACCEPTED ---

      if (use_dense_output) {
        // Calculate derivatives at the boundaries for the Spline
        std::vector<double> f0 = f(t, yn);
        std::vector<double> f1 = f(t + dt, y_next);

        // If the adaptive step flew past our requested output frames, catch
        // them!
        while (t_out_next <= t + dt && t_out_next <= tf) {
          std::vector<double> y_interp = utils::cubic_hermite_interpolate(
              yn, y_next, f0, f1, t, dt, t_out_next);
          times.push_back(t_out_next);
          states.push_back(y_interp);
          t_out_next += dt_out; // Line up the next frame
        }
      } else {
        // Just save every single internal step if Dense Output is off
        times.push_back(t + dt);
        states.push_back(y_next);
      }

      yn = y_next;
      t += dt;
      dt *= std::min(2.0, scale);
    } else {
      // --- STEP REJECTED ---
      dt *= std::max(0.2, scale);
    }
  }

  return {times, states};
}
} // namespace rungekutta
