#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <cmath>

#include "runge-kutta.hpp"

using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;

namespace {

// Linear scalar test problem: y' = -y, exact solution y(t) = exp(-t).
auto exp_decay = [](double, const std::vector<double> &y) {
  return std::vector<double>{-y[0]};
};

// 1D harmonic oscillator: q' = p, p' = -q. Energy 1/2 (p^2 + q^2) is conserved.
auto harmonic = [](double, const std::vector<double> &y) {
  return std::vector<double>{y[1], -y[0]};
};

double final_value(const std::vector<std::vector<double>> &trajectory,
                   std::size_t component) {
  return trajectory.back()[component];
}

double max_abs_drift(const std::vector<std::vector<double>> &trajectory,
                     double (*invariant)(const std::vector<double> &),
                     double reference) {
  double drift = 0.0;
  for (const auto &y : trajectory) {
    drift = std::max(drift, std::abs(invariant(y) - reference));
  }
  return drift;
}

double harmonic_energy(const std::vector<double> &y) {
  return 0.5 * (y[0] * y[0] + y[1] * y[1]);
}

} // namespace

TEST_CASE("All methods integrate y' = -y close to exp(-1)", "[integration]") {
  const std::vector<double> y0{1.0};
  const double t0 = 0.0;
  const double tf = 1.0;
  const std::size_t steps = 1000;
  const double dt = (tf - t0) / steps;
  const double expected = std::exp(-1.0);

  auto check = [&](const auto &method, double tol) {
    auto out = method(exp_decay, y0, t0, dt, steps, 50);
    CHECK_THAT(final_value(out, 0), WithinAbs(expected, tol));
  };

  check(
      [](auto f, auto y0, auto t0, auto dt, auto steps, auto max_iter) {
        return rungekutta::runge_kutta(methods::Heun_tableau, f, y0, t0, dt,
                                       steps, max_iter);
      },
      1e-4);

  check(
      [](auto f, auto y0, auto t0, auto dt, auto steps, auto max_iter) {
        return rungekutta::runge_kutta(methods::RK4_tableau, f, y0, t0, dt,
                                       steps, max_iter);
      },
      1e-8);

  check(
      [](auto f, auto y0, auto t0, auto dt, auto steps, auto max_iter) {
        return rungekutta::runge_kutta(methods::Trapezoidal_tableau, f, y0, t0,
                                       dt, steps, max_iter);
      },
      1e-4);

  check(
      [](auto f, auto y0, auto t0, auto dt, auto steps, auto max_iter) {
        return rungekutta::runge_kutta(methods::Implicit_midpoint_tableau, f,
                                       y0, t0, dt, steps, max_iter);
      },
      1e-4);

  check(
      [](auto f, auto y0, auto t0, auto dt, auto steps, auto max_iter) {
        return rungekutta::runge_kutta(methods::Gauss_Legendre_tableau, f, y0,
                                       t0, dt, steps, max_iter);
      },
      1e-8);

  check(
      [](auto f, auto y0, auto t0, auto dt, auto steps, auto max_iter) {
        return rungekutta::runge_kutta(methods::LobattoIIIA_tableau, f, y0, t0,
                                       dt, steps, max_iter);
      },
      1e-8);
}

TEST_CASE("Symplectic integrators conserve harmonic-oscillator energy",
          "[integration][symplectic]") {
  const std::vector<double> y0{1.0, 0.0};
  const double t0 = 0.0;
  const double tf = 20.0 * M_PI; // ten oscillations
  const std::size_t steps = 4000;
  const double dt = (tf - t0) / steps;
  const double E0 = harmonic_energy(y0);

  auto gauss = rungekutta::runge_kutta(methods::Gauss_Legendre_tableau,
                                       harmonic, y0, t0, dt, steps, 50);

  auto midpoint = rungekutta::runge_kutta(methods::Implicit_midpoint_tableau,
                                          harmonic, y0, t0, dt, steps, 50);

  auto rk4 = rungekutta::runge_kutta(methods::RK4_tableau, harmonic, y0, t0, dt,
                                     steps, 50);
  // Symplectic methods: drift bounded and tiny.
  CHECK(max_abs_drift(gauss, harmonic_energy, E0) < 1e-10);
  CHECK(max_abs_drift(midpoint, harmonic_energy, E0) < 1e-6);
  // RK4 still has small but strictly larger drift than the symplectic ones
  // here.
  CHECK(max_abs_drift(rk4, harmonic_energy, E0) < 1e-3);
}

TEST_CASE("RK4 exhibits 4th-order convergence on y' = -y",
          "[integration][order]") {
  const std::vector<double> y0{1.0};
  const double t0 = 0.0;
  const double tf = 1.0;
  const double expected = std::exp(-1.0);

  auto err_at = [&](std::size_t steps) {
    const double dt = (tf - t0) / steps;
    auto out = rungekutta::runge_kutta(methods::RK4_tableau, exp_decay, y0, t0,
                                       dt, steps, 50);
    return std::abs(final_value(out, 0) - expected);
  };

  const double e1 = err_at(50);
  const double e2 = err_at(100);
  const double order = std::log2(e1 / e2);
  // Expect ~4. Allow a generous window; this is asserting the right scaling,
  // not a number.
  CHECK(order > 3.5);
  CHECK(order < 4.5);
}
