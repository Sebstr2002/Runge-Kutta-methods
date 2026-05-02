#pragma once
#include <vector>

namespace utils {
std::vector<double> cubic_hermite_interpolate(const std::vector<double> &y0,
                                              const std::vector<double> &y1,
                                              const std::vector<double> &f0,
                                              const std::vector<double> &f1,
                                              double t0, double dt,
                                              double t_query);
}
