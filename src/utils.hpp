#pragma once
#include <functional>
#include <vector>

namespace utils {
std::vector<double> cubic_hermite_interpolate(const std::vector<double> &y0,
                                              const std::vector<double> &y1,
                                              const std::vector<double> &f0,
                                              const std::vector<double> &f1,
                                              double t0, double dt,
                                              double t_query);
std::vector<std::vector<double>> compute_jacobian(
    const std::function<std::vector<double>(double,
                                            const std::vector<double> &)> &f,
    double t, std::vector<double> y, const std::vector<double> &f0);

std::vector<double> solve_linear_system(std::vector<std::vector<double>> A,
                                        std::vector<double> b);
} // namespace utils
