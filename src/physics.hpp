#pragma once
#include <vector>

namespace physics {
// Standard 2-body Kepler problem
std::vector<double> kepler_rhs(double t, const std::vector<double> &y_);
std::vector<double> sun_earth_moon_rhs(double t, const std::vector<double> &y_);
} // namespace physics
