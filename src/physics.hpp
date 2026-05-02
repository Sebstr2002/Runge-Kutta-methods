#pragma once
#include <vector>

namespace physics {
// 1. Standard Kepler
std::vector<double> kepler_rhs(double t, const std::vector<double> &y_);

// 2. Sun-Earth-Moon
std::vector<double> sun_earth_moon_rhs(double t, const std::vector<double> &y_);

// 3. Hamiltonian Double Pendulum
// y = [theta1, theta2, p_theta1, p_theta2]
std::vector<double> double_pendulum_rhs(double t,
                                        const std::vector<double> &y_);

// 4. Circular Restricted 3-Body Problem
// y = [x, y, px, py] in a rotating reference frame
std::vector<double> cr3bp_rhs(double t, const std::vector<double> &y_);

// 5. Post-Newtonian General Relativity
// y = [x, y, px, py]
std::vector<double> mercury_gr_rhs(double t, const std::vector<double> &y_);

// 6. Damped Simple Pendulum
// y = [theta, omega] (angle and angular velocity)
std::vector<double> damped_pendulum_rhs(double t,
                                        const std::vector<double> &y_);
} // namespace physics
