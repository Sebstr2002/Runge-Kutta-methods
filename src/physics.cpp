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

  // Normalized constants (G=1, Sun=1.0)
  const double G = 1.0;
  const double m[3] = {1.0, 3.0034e-6, 3.694e-8};

  // Extract for readability
  double x[3] = {y_[0], y_[2], y_[4]};
  double y[3] = {y_[1], y_[3], y_[5]};
  double px[3] = {y_[6], y_[8], y_[10]};
  double py[3] = {y_[7], y_[9], y_[11]};

  // 1. Calculate velocity: \dot{r}_i = p_i / m_i
  for (int i = 0; i < 3; ++i) {
    dy[i * 2] = px[i] / m[i];     // dx_i / dt
    dy[i * 2 + 1] = py[i] / m[i]; // dy_i / dt
  }

  // 2. Calculate force: \dot{p}_i = -G * sum( m_i * m_j * (r_i - r_j) / r^3 )
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      if (i == j)
        continue; // No self-interaction

      double dx = x[i] - x[j];
      double dy_pos = y[i] - y[j];
      double r2 = dx * dx + dy_pos * dy_pos;
      double r3 = std::pow(r2, 1.5);

      double force_mag = (G * m[i] * m[j]) / r3;

      // Accumulate the force components
      dy[6 + i * 2] -= force_mag * dx;
      dy[6 + i * 2 + 1] -= force_mag * dy_pos;
    }
  }

  return dy;
}
} // namespace physics
