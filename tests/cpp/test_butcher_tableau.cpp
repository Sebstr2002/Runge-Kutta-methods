#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "runge-kutta.hpp"

using Catch::Matchers::WithinAbs;

TEST_CASE("Builtin tableaus pass shape and consistency checks", "[tableau]") {
    REQUIRE(methods::Heun_tableau.isValid());
    REQUIRE(methods::RK4_tableau.isValid());
    REQUIRE(methods::Trapezoidal_tableau.isValid());
    REQUIRE(methods::Implicit_midpoint_tableau.isValid());
    REQUIRE(methods::Gauss_Legendre_tableau.isValid());
    REQUIRE(methods::LobattoIIIA_tableau.isValid());
}

TEST_CASE("Implicit/explicit classification matches expectation", "[tableau]") {
    CHECK_FALSE(methods::Heun_tableau.isImplicit());
    CHECK_FALSE(methods::RK4_tableau.isImplicit());
    CHECK(methods::Trapezoidal_tableau.isImplicit());
    CHECK(methods::Implicit_midpoint_tableau.isImplicit());
    CHECK(methods::Gauss_Legendre_tableau.isImplicit());
    CHECK(methods::LobattoIIIA_tableau.isImplicit());
}

TEST_CASE("Symplecticity check labels methods correctly", "[tableau][symplectic]") {
    // Implicit midpoint and Gauss-Legendre are symplectic.
    CHECK(methods::Implicit_midpoint_tableau.isSymplectic());
    CHECK(methods::Gauss_Legendre_tableau.isSymplectic());
    // Classical RK4 is not symplectic.
    CHECK_FALSE(methods::RK4_tableau.isSymplectic());
    // Heun (explicit trapezoidal) is not symplectic either.
    CHECK_FALSE(methods::Heun_tableau.isSymplectic());
}

TEST_CASE("Hand-built malformed tableau is rejected at construction",
          "[tableau]") {
    // b does not sum to 1 -> constructor throws std::invalid_argument.
    CHECK_THROWS_AS(
        ButcherTableau({{0.0, 0.0}, {1.0, 0.0}}, {0.5, 0.4}, {0.0, 1.0}),
        std::invalid_argument);
}

TEST_CASE("New embedded tableaus are valid and report order_low", "[tableau]") {
    REQUIRE(methods::RKF45_tableau.isValid());
    REQUIRE(methods::CashKarp_tableau.isValid());
    REQUIRE(methods::DP54_tableau.isValid());
    REQUIRE(methods::RK4_38_tableau.isValid());
    CHECK(methods::BS32_tableau.getOrderLow() == 2);
    CHECK(methods::RKF45_tableau.getOrderLow() == 4);
    CHECK(methods::CashKarp_tableau.getOrderLow() == 4);
    CHECK(methods::DP54_tableau.getOrderLow() == 4);
    CHECK_FALSE(methods::DP54_tableau.isImplicit());
    CHECK_FALSE(methods::RKF45_tableau.isImplicit());
}
