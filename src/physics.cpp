#include "physics.hpp"
#include <cmath>

namespace physics {
std::vector<double> kepler_rhs(double t, const std::vector<double> &y_) {
  double x = y_[0];
  double y = y_[1];
  double px = y_[2];
  double py = y_[3];

  double r3 = std::pow(x * x + y * y, 1.5);

  return {px, py, -x / r3, -y / r3};
}
std::vector<double> sun_earth_moon_rhs(double t,
                                       const std::vector<double> &y_) {
  // y_ = [x0, y0, x1, y1, x2, y2, px0, py0, px1, py1, px2, py2]
  // Indices: 0=Sun, 1=Earth, 2=Moon
  std::vector<double> dy(12, 0.0);

  // Normalized constants
  const double G = 1.0;
  const double m[3] = {1.0, 3.0034e-6, 3.694e-8};

  // Extract for readability
  double x[3] = {y_[0], y_[2], y_[4]};
  double y[3] = {y_[1], y_[3], y_[5]};
  double px[3] = {y_[6], y_[8], y_[10]};
  double py[3] = {y_[7], y_[9], y_[11]};

  // 1. Calculate velocity:
  for (int i = 0; i < 3; ++i) {
    dy[i * 2] = px[i] / m[i];     // dx_i / dt
    dy[i * 2 + 1] = py[i] / m[i]; // dy_i / dt
  }

  // 2. Calculate force:
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      if (i == j)
        continue; // No self-interaction

      double dx = x[i] - x[j];
      double dy_pos = y[i] - y[j];
      double r2 = dx * dx + dy_pos * dy_pos;
      double r3 = std::pow(r2, 1.5);

      double force_mag = (G * m[i] * m[j]) / r3;

      dy[6 + i * 2] -= force_mag * dx;
      dy[6 + i * 2 + 1] -= force_mag * dy_pos;
    }
  }

  return dy;
}

// --- 3. Double Pendulum ---
std::vector<double> double_pendulum_rhs(double t,
                                        const std::vector<double> &y_) {
  // Phase space: angles (t1, t2) and angular momenta (p1, p2)
  double t1 = y_[0], t2 = y_[1], p1 = y_[2], p2 = y_[3];

  double delta = t1 - t2;
  double c = std::cos(delta);
  double s = std::sin(delta);

  // Denominator for m1=1, m2=1, L1=1, L2=1
  double den = 1.0 + s * s;

  // Hamilton's Equations (Velocity) -> dq/dt = dH/dp
  double dt1 = (p1 - p2 * c) / den;
  double dt2 = (2.0 * p2 - p1 * c) / den;

  // Hamilton's Equations (Force) -> dp/dt = -dH/dq
  double A1 = (p1 * p2 * s -
               (p1 * p1 + 2.0 * p2 * p2 - 2.0 * p1 * p2 * c) * s * c / den) /
              den;

  // g = 1.0
  double dp1 = -2.0 * std::sin(t1) - A1;
  double dp2 = -std::sin(t2) + A1;

  return {dt1, dt2, dp1, dp2};
}

// --- 4. Circular Restricted 3-Body Problem (CR3BP) ---
std::vector<double> cr3bp_rhs(double t, const std::vector<double> &y_) {
  // Phase space in a ROTATING frame: [x, y, px, py]
  double x = y_[0], y = y_[1], px = y_[2], py = y_[3];

  // Mass ratio (Earth-Moon system approx)
  const double mu = 0.012277471;

  // Distances to the two massive bodies
  double r1 = std::sqrt((x + mu) * (x + mu) + y * y);
  double r2 = std::sqrt((x - 1.0 + mu) * (x - 1.0 + mu) + y * y);
  double r1_3 = r1 * r1 * r1;
  double r2_3 = r2 * r2 * r2;

  // Velocity (includes Coriolis force from rotating frame)
  double dx = px + y;
  double dy = py - x;

  // Force (Gravity from both bodies + Centrifugal force)
  double dpx = py - (1.0 - mu) * (x + mu) / r1_3 - mu * (x - 1.0 + mu) / r2_3;
  double dpy = -px - (1.0 - mu) * y / r1_3 - mu * y / r2_3;

  return {dx, dy, dpx, dpy};
}

// --- 5. General Relativity ---
std::vector<double> mercury_gr_rhs(double t, const std::vector<double> &y_) {
  // Phase space: [x, y, px, py]
  double x = y_[0], y = y_[1], px = y_[2], py = y_[3];

  double r2 = x * x + y * y;
  double r = std::sqrt(r2);

  // Angular momentum L = r x p
  double L = x * py - y * px;
  double L2 = L * L;

  // General Relativity correction factor (alpha = 3 / c^2)
  const double alpha = 0.01; // note that speed of light is exagerated

  double r3 = r2 * r;
  double r5 = r3 * r2;

  // Force
  double force_mag_over_r = -1.0 / r3 - (3.0 * alpha * L2) / r5;

  double dx = px;
  double dy = py;
  double dpx = force_mag_over_r * x;
  double dpy = force_mag_over_r * y;

  return {dx, dy, dpx, dpy};
}
} // namespace physics
