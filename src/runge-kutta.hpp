#pragma once
#include <cmath>
#include <functional>
#include <vector>

namespace constants {
// this tolerance will make sure tableaus are evaluated correctly
constexpr double tolerance = 1e-10;
} // namespace constants

bool checkImplicit(const std::vector<std::vector<double>> A);

class ButcherTableau {
private:
  std::vector<std::vector<double>> A;
  std::vector<double> b;
  std::vector<double> c;
  bool implicit;

public:
  // constructor
  ButcherTableau(std::vector<std::vector<double>> A_, std::vector<double> b_,
                 std::vector<double> c_);

  // getters
  const std::vector<std::vector<double>> &getA() const;
  const std::vector<double> &getB() const;
  const std::vector<double> &getC() const;
  const bool &isImplicit() const;

  // properties bool isValid() const;
  bool isValid() const;
  bool isSymplectic() const;
};

// general runge-kutta methods and some famous examples
namespace rungekutta {

std::vector<std::vector<double>> runge_kutta(
    const ButcherTableau &table,
    const std::function<std::vector<double>(const std::vector<double> &)> &f,
    std::vector<double> y0, double t0, int max_iter = 10);
}
