#include "runge-kutta.hpp"
#include "utils.hpp"
#include <algorithm>
#include <stdexcept>

namespace rungekutta {

// --- 1. Standard Runge-Kutta ---
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
  size_t s = b.size();
  size_t dim = yn.size();

  // ==========================================
  // MEMORY WORKSPACE: Allocate exactly ONE time.
  // ==========================================
  std::vector<std::vector<double>> K(s, std::vector<double>(dim, 0.0));
  std::vector<double> yi(dim, 0.0);
  std::vector<double> y_next(dim, 0.0);

  // Newton-iteration workspace. Only touched for implicit tableaus, but
  // cheap to keep around for the explicit case too.
  const size_t sys_dim = s * dim;
  std::vector<std::vector<double>> J(dim, std::vector<double>(dim, 0.0));
  std::vector<std::vector<double>> SysJ(sys_dim,
                                        std::vector<double>(sys_dim, 0.0));
  std::vector<double> G(sys_dim, 0.0);
  std::vector<double> f_base(dim, 0.0);

  for (size_t n = 0; n < steps; ++n) {
    double t = t0 + n * dt;

    if (!table.isImplicit()) {
      for (size_t i = 0; i < s; ++i) {
        yi = yn; // Overwrites existing buffer, NO allocation!
        for (size_t j = 0; j < i; ++j) {
          for (size_t k = 0; k < dim; ++k) {
            yi[k] += dt * A[i][j] * K[j][k];
          }
        }
        K[i] = f(t + c[i] * dt, yi);
      }
    } else {
      // --- Newton iteration on the stage values K, in place. ---
      // K starts at whatever the previous step left (zero on the first
      // step) and is updated in place each Newton iteration. Reusing
      // the previous step's K gives a much better initial guess than
      // zero, especially for slowly-varying problems.
      // We "freeze" the Jacobian per time step (modified Newton): cheaper
      // than re-evaluating J each iteration and adequate for non-stiff
      // problems.
      f_base = f(t, yn);
      J = utils::compute_jacobian(f, t, yn, f_base);

      for (int it = 0; it < max_iter; ++it) {
        // Build the residual G and the system Jacobian SysJ.
        for (size_t i = 0; i < s; ++i) {
          std::fill(yi.begin(), yi.end(), 0.0);
          for (size_t j = 0; j < s; ++j) {
            for (size_t k = 0; k < dim; ++k)
              yi[k] += A[i][j] * K[j][k];
          }
          for (size_t k = 0; k < dim; ++k)
            yi[k] = yn[k] + dt * yi[k];

          std::vector<double> fi = f(t + c[i] * dt, yi);

          for (size_t k = 0; k < dim; ++k) {
            size_t row = i * dim + k;
            G[row] = K[i][k] - fi[k];

            // Kronecker product: I - dt * (A ⊗ J)
            for (size_t j = 0; j < s; ++j) {
              for (size_t l = 0; l < dim; ++l) {
                size_t col = j * dim + l;
                double kron = (row == col) ? 1.0 : 0.0;
                SysJ[row][col] = kron - dt * A[i][j] * J[k][l];
              }
            }
          }
        }

        std::vector<double> dK = utils::solve_linear_system(SysJ, G);

        // K_new = K - dK. err is the L1 size of the update step.
        double err = 0.0;
        for (size_t i = 0; i < s; ++i) {
          for (size_t k = 0; k < dim; ++k) {
            double upd = dK[i * dim + k];
            K[i][k] -= upd;
            err += std::abs(upd);
          }
        }

        if (err < constants::tolerance)
          break;
      }
    }

    y_next = yn;
    for (size_t i = 0; i < s; ++i) {
      for (size_t k = 0; k < dim; ++k) {
        y_next[k] += dt * b[i] * K[i][k];
      }
    }

    times.push_back(t0 + (n + 1) * dt);
    states.push_back(y_next);
    yn = y_next;
  }

  return {times, states};
}

// --- 2. Adaptive Runge-Kutta ---
std::tuple<std::vector<double>, std::vector<std::vector<double>>,
           std::vector<double>, std::vector<std::vector<double>>>
adaptive_runge_kutta(const ButcherTableau &table,
                     const std::function<std::vector<double>(
                         double, const std::vector<double> &)> &f,
                     std::vector<double> yn, double t0, double tf,
                     double initial_dt, double tolerance, int max_iter,
                     double dt_out, const EventFunc &event_fn,
                     bool stop_on_event) {
  std::vector<double> times;
  std::vector<std::vector<double>> states;

  times.push_back(t0);
  states.push_back(yn);

  const auto &A = table.getA();
  const auto &b = table.getB();
  const auto &c = table.getC();
  const auto &bstar = table.getB_star();

  size_t s = b.size();
  size_t dim = yn.size();

  double t = t0;
  double dt = initial_dt;
  const double safety = 0.9;
  const double dt_min = 1e-14 * std::max(std::abs(t0), std::abs(tf));
  // Step-size scaling exponent: 1 / (p_low + 1) where p_low is the order
  // of the lower-order companion in the embedded pair. BS32 → 1/3, DP54/RKF45
  // /Cash-Karp → 1/5. Read from the tableau so adding a new embedded method
  // doesn't require touching this function.
  const double exponent = 1.0 / (table.getOrderLow() + 1);

  double t_out_next = t0 + dt_out;
  bool use_dense_output = (dt_out > 0.0);

  // ==========================================
  // MEMORY WORKSPACE: Allocate exactly ONE time.
  // ==========================================
  std::vector<std::vector<double>> K(s, std::vector<double>(dim, 0.0));
  std::vector<double> yi(dim, 0.0);
  std::vector<double> y_next(dim, 0.0);
  std::vector<double> y_star(dim, 0.0);
  std::vector<double> f0(dim, 0.0);
  std::vector<double> f1(dim, 0.0);
  std::vector<double> event_times;
  std::vector<std::vector<double>> event_states;

  // Newton-iteration workspace (only used for implicit tableaus).
  const size_t sys_dim = s * dim;
  std::vector<std::vector<double>> J(dim, std::vector<double>(dim, 0.0));
  std::vector<std::vector<double>> SysJ(sys_dim,
                                        std::vector<double>(sys_dim, 0.0));
  std::vector<double> G(sys_dim, 0.0);
  std::vector<double> f_base(dim, 0.0);

  while (t < tf) {
    if (t + dt > tf)
      dt = tf - t;

    if (!table.isImplicit()) {
      for (size_t i = 0; i < s; ++i) {
        yi = yn;
        for (size_t j = 0; j < i; ++j) {
          for (size_t k = 0; k < dim; ++k) {
            yi[k] += dt * A[i][j] * K[j][k];
          }
        }
        K[i] = f(t + c[i] * dt, yi);
      }
    } else {
      // --- Newton iteration on stage values K (in place). ---
      // Same scheme as the fixed-step solver: freeze the Jacobian per
      // attempted step and update K in place each iteration.
      f_base = f(t, yn);
      J = utils::compute_jacobian(f, t, yn, f_base);

      for (int it = 0; it < max_iter; ++it) {
        for (size_t i = 0; i < s; ++i) {
          std::fill(yi.begin(), yi.end(), 0.0);
          for (size_t j = 0; j < s; ++j) {
            for (size_t k = 0; k < dim; ++k)
              yi[k] += A[i][j] * K[j][k];
          }
          for (size_t k = 0; k < dim; ++k)
            yi[k] = yn[k] + dt * yi[k];

          std::vector<double> fi = f(t + c[i] * dt, yi);

          for (size_t k = 0; k < dim; ++k) {
            size_t row = i * dim + k;
            G[row] = K[i][k] - fi[k];

            for (size_t j = 0; j < s; ++j) {
              for (size_t l = 0; l < dim; ++l) {
                size_t col = j * dim + l;
                double kron = (row == col) ? 1.0 : 0.0;
                SysJ[row][col] = kron - dt * A[i][j] * J[k][l];
              }
            }
          }
        }

        std::vector<double> dK = utils::solve_linear_system(SysJ, G);

        double err = 0.0;
        for (size_t i = 0; i < s; ++i) {
          for (size_t k = 0; k < dim; ++k) {
            double upd = dK[i * dim + k];
            K[i][k] -= upd;
            err += std::abs(upd);
          }
        }

        if (err < constants::tolerance)
          break;
      }
    }

    y_next = yn;
    y_star = yn;
    for (size_t i = 0; i < s; ++i) {
      for (size_t k = 0; k < dim; ++k) {
        y_next[k] += dt * b[i] * K[i][k];
        y_star[k] += dt * bstar[i] * K[i][k];
      }
    }

    double max_error = 0.0;
    for (size_t k = 0; k < dim; ++k) {
      double diff = std::abs(y_next[k] - y_star[k]);
      if (diff > max_error)
        max_error = diff;
    }
    if (max_error == 0.0)
      max_error = 1e-15;

    double scale = safety * std::pow(tolerance / max_error, exponent);

    if (max_error <= tolerance) {
      // --- STEP ACCEPTED ---

      // Calculate derivatives at the boundaries for the Spline
      f0 = f(t, yn);
      f1 = f(t + dt, y_next);

      // ==========================================
      // EVENT DETECTION (Root-Finding)
      // ==========================================
      bool event_triggered = false;
      if (event_fn) {
        double g0 = event_fn(t, yn);
        double g1 = event_fn(t + dt, y_next);

        // A sign change means we crossed the event threshold!
        if (g0 * g1 <= 0.0) {
          event_triggered = true;
          double t_left = t;
          double t_right = t + dt;
          double g_left = g0;
          double t_mid = t;
          std::vector<double> y_mid(dim, 0.0);

          // Bisection Root Finder using the Dense Output Spline
          for (int iter = 0; iter < 50; ++iter) {
            t_mid = t_left + 0.5 * (t_right - t_left);
            y_mid = utils::cubic_hermite_interpolate(yn, y_next, f0, f1, t, dt,
                                                     t_mid);
            double g_mid = event_fn(t_mid, y_mid);

            // Did we find the exact root?
            if (std::abs(g_mid) < 1e-12 || (t_right - t_left) < 1e-12)
              break;

            if (g_left * g_mid <= 0.0) {
              t_right = t_mid;
            } else {
              t_left = t_mid;
              g_left = g_mid;
            }
          }

          event_times.push_back(t_mid);
          event_states.push_back(y_mid);

          if (stop_on_event) {
            // Save the final frames precisely up to the event, then kill the
            // simulation
            if (use_dense_output) {
              while (t_out_next <= t_mid) {
                std::vector<double> y_interp = utils::cubic_hermite_interpolate(
                    yn, y_next, f0, f1, t, dt, t_out_next);
                times.push_back(t_out_next);
                states.push_back(y_interp);
                t_out_next += dt_out;
              }
            } else {
              times.push_back(t_mid);
              states.push_back(y_mid);
            }
            break; // EXIT THE WHILE LOOP!
          }
        }
      }

      // ==========================================
      // DENSE OUTPUT (If the event didn't stop us)
      // ==========================================
      if (!event_triggered || !stop_on_event) {
        if (use_dense_output) {
          while (t_out_next <= t + dt && t_out_next <= tf) {
            std::vector<double> y_interp = utils::cubic_hermite_interpolate(
                yn, y_next, f0, f1, t, dt, t_out_next);
            times.push_back(t_out_next);
            states.push_back(y_interp);
            t_out_next += dt_out;
          }
        } else {
          times.push_back(t + dt);
          states.push_back(y_next);
        }
      }

      yn = y_next;
      t += dt;
      dt *= std::min(2.0, scale);
    } else {
      // --- STEP REJECTED ---
      // Shrink dt without advancing t. Floor at 0.1× to avoid wild swings;
      // bail out if dt has collapsed to near round-off (degenerate problem
      // or tolerance asked for is unattainable in double precision).
      dt *= std::max(0.1, scale);
      if (dt < dt_min) {
        throw std::runtime_error(
            "adaptive_runge_kutta: step size collapsed below dt_min; "
            "the requested tolerance is unattainable for this problem.");
      }
    }
  }

  return {times, states, event_times, event_states};
}
} // namespace rungekutta
