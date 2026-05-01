"""End-to-end tests for the bound Runge-Kutta methods.

These tests call the real Python module exactly the way the example scripts
do — that's the surface a user touches, so it's the right place to pin
behaviour.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

import hamsolver


# ---------------------------------------------------------------------------
# Test problems
# ---------------------------------------------------------------------------

def exp_decay(_t, y):
    """y' = -y, exact solution y(t) = exp(-t)."""
    return np.array([-y[0]])


def harmonic(_t, y):
    """1D harmonic oscillator: q' = p, p' = -q."""
    return np.array([y[1], -y[0]])


def kepler(_t, y):
    """Planar Kepler problem: q' = p, p' = -q / |q|^3."""
    x, y_, px, py = y
    r3 = (x * x + y_ * y_) ** 1.5
    return np.array([px, py, -x / r3, -y_ / r3])


def kepler_initial(eps: float) -> np.ndarray:
    return np.array([1.0 - eps, 0.0, 0.0, math.sqrt((1.0 + eps) / (1.0 - eps))])


def kepler_energy(traj: np.ndarray) -> np.ndarray:
    x, y, px, py = traj.T
    r = np.sqrt(x * x + y * y)
    return 0.5 * (px * px + py * py) - 1.0 / r


def kepler_angular_momentum(traj: np.ndarray) -> np.ndarray:
    x, y, px, py = traj.T
    return x * py - y * px


# ---------------------------------------------------------------------------
# Method registry — keep in sync with the example scripts.
# ---------------------------------------------------------------------------

ALL_METHODS = [
    ("Heun", hamsolver.Heun_method, 2),
    ("RK4", hamsolver.RK4_method, 4),
    ("Trapezoidal", hamsolver.Trapezoidal_method, 2),
    ("Implicit_midpoint", hamsolver.Implicit_midpoint_method, 2),
    ("Gauss_Legendre", hamsolver.Gauss_Legendre_method, 4),
    ("LobattoIIIA", hamsolver.LobattoIIIA_method, 4),
]

SYMPLECTIC_METHODS = [
    ("Implicit_midpoint", hamsolver.Implicit_midpoint_method),
    ("Gauss_Legendre", hamsolver.Gauss_Legendre_method),
]


# ---------------------------------------------------------------------------
# Smoke / sanity
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name,method,_order", ALL_METHODS)
def test_output_shape(name, method, _order):
    y0 = np.array([1.0])
    out = method(exp_decay, y0, 0.0, 0.01, 100, max_iter=50)
    assert out.shape == (101, 1), f"{name} returned shape {out.shape}"


@pytest.mark.parametrize("name,method,_order", ALL_METHODS)
def test_initial_condition_preserved(name, method, _order):
    y0 = np.array([1.0, 0.0])
    out = method(harmonic, y0, 0.0, 0.01, 10, max_iter=50)
    np.testing.assert_allclose(out[0], y0, atol=1e-14, err_msg=name)


# ---------------------------------------------------------------------------
# Accuracy
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name,method,order", ALL_METHODS)
def test_accuracy_on_exp_decay(name, method, order):
    """Final-time error should match a tolerance scaled to the method order."""
    y0 = np.array([1.0])
    steps = 1000
    out = method(exp_decay, y0, 0.0, 1.0 / steps, steps, max_iter=50)
    err = abs(out[-1, 0] - math.exp(-1.0))
    tol = {2: 1e-4, 4: 1e-8}[order]
    assert err < tol, f"{name}: err={err:.2e} not < {tol:.0e}"


@pytest.mark.parametrize(
    "name,method", [(n, m) for n, m, o in ALL_METHODS if o == 4]
)
def test_fourth_order_convergence(name, method):
    """Halving dt should reduce error by ~16 for an order-4 method."""
    y0 = np.array([1.0])

    def err_at(steps: int) -> float:
        out = method(exp_decay, y0, 0.0, 1.0 / steps, steps, max_iter=50)
        return abs(out[-1, 0] - math.exp(-1.0))

    e1 = err_at(50)
    e2 = err_at(100)
    order = math.log2(e1 / e2)
    assert 3.5 < order < 4.5, f"{name}: measured order {order:.2f}"


# ---------------------------------------------------------------------------
# Geometric (Hamiltonian) properties
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name,method", SYMPLECTIC_METHODS)
def test_symplectic_methods_conserve_energy_kepler(name, method):
    """Symplectic integrators have *bounded* (non-secular) energy drift."""
    eps = 0.3
    y0 = kepler_initial(eps)
    steps = 2000
    tf = 4 * math.pi  # two periods
    dt = tf / steps
    out = method(kepler, y0, 0.0, dt, steps, max_iter=50)
    E = kepler_energy(out)
    drift = np.max(np.abs(E - E[0]))
    assert drift < 1e-4, f"{name}: drift {drift:.2e}"


@pytest.mark.parametrize("name,method,_order", ALL_METHODS)
def test_angular_momentum_conserved_kepler(name, method, _order):
    """L is exactly conserved by the flow; integrators differ in how well they preserve it."""
    eps = 0.3
    y0 = kepler_initial(eps)
    steps = 2000
    tf = 2 * math.pi
    dt = tf / steps
    out = method(kepler, y0, 0.0, dt, steps, max_iter=50)
    L = kepler_angular_momentum(out)
    assert np.max(np.abs(L - L[0])) < 1e-3, name
