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
} // namespace utils
