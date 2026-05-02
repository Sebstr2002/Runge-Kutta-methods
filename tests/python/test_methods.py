"""End-to-end tests for the bound Runge-Kutta methods.

These tests call the real Python module exactly the way the example scripts
do — that's the surface a user touches, so it's the right place to pin
behaviour. Driven through the generic ``hamsolver.runge_kutta(table, ...)``
entry point so the whole tableau registry stays in one list.
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


def integrate(table, f, y0, t0, dt, steps, max_iter=50):
    """Convenience wrapper around the generic driver — returns states only."""
    _, states = hamsolver.runge_kutta(
        table=table, f=f, yn=y0, t0=t0, dt=dt, steps=steps, max_iter=max_iter
    )
    return np.asarray(states)


# ---------------------------------------------------------------------------
# Method registries — keep in sync with the example scripts.
# ---------------------------------------------------------------------------
# Each entry is (name, tableau, order). Symplectic methods are split out so
# the energy-drift test only runs against them.

ALL_METHODS = [
    ("Heun",              hamsolver.Heun,              2),
    ("RK4",               hamsolver.RK4,               4),
    ("RK4_38",            hamsolver.RK4_38,            4),
    ("Trapezoidal",       hamsolver.Trapezoidal,       2),
    ("Implicit_midpoint", hamsolver.Implicit_midpoint, 2),
    ("Gauss_Legendre",    hamsolver.Gauss_Legendre,    4),
    ("LobattoIIIA",       hamsolver.LobattoIIIA,       4),
]

SYMPLECTIC_METHODS = [
    ("Implicit_midpoint", hamsolver.Implicit_midpoint),
    ("Gauss_Legendre",    hamsolver.Gauss_Legendre),
]

# Embedded pairs that drive the adaptive solver.
EMBEDDED_METHODS = [
    ("BS32",     hamsolver.BS32,     3),  # main order 3
    ("RKF45",    hamsolver.RKF45,    5),
    ("CashKarp", hamsolver.CashKarp, 5),
    ("DP54",     hamsolver.DP54,     5),
]


# ---------------------------------------------------------------------------
# Smoke / sanity
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name,table,_order", ALL_METHODS)
def test_output_shape(name, table, _order):
    y0 = np.array([1.0])
    out = integrate(table, exp_decay, y0, 0.0, 0.01, 100)
    assert out.shape == (101, 1), f"{name} returned shape {out.shape}"


@pytest.mark.parametrize("name,table,_order", ALL_METHODS)
def test_initial_condition_preserved(name, table, _order):
    y0 = np.array([1.0, 0.0])
    out = integrate(table, harmonic, y0, 0.0, 0.01, 10)
    np.testing.assert_allclose(out[0], y0, atol=1e-14, err_msg=name)


def test_tableau_metadata_exposed():
    """The Python-side ButcherTableau should expose the new introspection API."""
    assert hamsolver.RK4.is_valid()
    assert not hamsolver.RK4.is_implicit()
    assert hamsolver.Gauss_Legendre.is_implicit()
    assert hamsolver.Gauss_Legendre.is_symplectic()
    assert hamsolver.BS32.is_embedded()
    assert hamsolver.DP54.order_low() == 4
    assert hamsolver.BS32.order_low() == 2


# ---------------------------------------------------------------------------
# Accuracy
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name,table,order", ALL_METHODS)
def test_accuracy_on_exp_decay(name, table, order):
    """Final-time error should match a tolerance scaled to the method order."""
    y0 = np.array([1.0])
    steps = 1000
    out = integrate(table, exp_decay, y0, 0.0, 1.0 / steps, steps)
    err = abs(out[-1, 0] - math.exp(-1.0))
    tol = {2: 1e-4, 4: 1e-8}[order]
    assert err < tol, f"{name}: err={err:.2e} not < {tol:.0e}"


@pytest.mark.parametrize(
    "name,table", [(n, t) for n, t, o in ALL_METHODS if o == 4]
)
def test_fourth_order_convergence(name, table):
    """Halving dt should reduce error by ~16 for an order-4 method."""
    y0 = np.array([1.0])

    def err_at(steps: int) -> float:
        out = integrate(table, exp_decay, y0, 0.0, 1.0 / steps, steps)
        return abs(out[-1, 0] - math.exp(-1.0))

    e1 = err_at(50)
    e2 = err_at(100)
    order = math.log2(e1 / e2)
    assert 3.5 < order < 4.5, f"{name}: measured order {order:.2f}"


# ---------------------------------------------------------------------------
# Geometric (Hamiltonian) properties
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name,table", SYMPLECTIC_METHODS)
def test_symplectic_methods_conserve_energy_kepler(name, table):
    """Symplectic integrators have *bounded* (non-secular) energy drift."""
    eps = 0.3
    y0 = kepler_initial(eps)
    steps = 2000
    tf = 4 * math.pi  # two periods
    dt = tf / steps
    out = integrate(table, kepler, y0, 0.0, dt, steps)
    E = kepler_energy(out)
    drift = np.max(np.abs(E - E[0]))
    assert drift < 1e-4, f"{name}: drift {drift:.2e}"


@pytest.mark.parametrize("name,table,_order", ALL_METHODS)
def test_angular_momentum_conserved_kepler(name, table, _order):
    """L is exactly conserved by the flow; integrators differ in how well they preserve it."""
    eps = 0.3
    y0 = kepler_initial(eps)
    steps = 2000
    tf = 2 * math.pi
    dt = tf / steps
    out = integrate(table, kepler, y0, 0.0, dt, steps)
    L = kepler_angular_momentum(out)
    assert np.max(np.abs(L - L[0])) < 1e-3, name


# ---------------------------------------------------------------------------
# Adaptive solver — smoke + the bug we just fixed
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name,table,_order", EMBEDDED_METHODS)
def test_adaptive_lands_on_tf(name, table, _order):
    """Each embedded pair should integrate y' = -y to t=1 and finish there."""
    y0 = np.array([1.0])
    times, states, _, _ = hamsolver.adaptive_runge_kutta(
        table=table, f=exp_decay, yn=y0, t0=0.0, tf=1.0,
        initial_dt=0.1, tolerance=1e-6,
    )
    assert times[-1] == pytest.approx(1.0, abs=1e-12), name
    assert states[-1][0] == pytest.approx(math.exp(-1.0), abs=1e-4), name


def test_adaptive_terminates_when_first_step_too_coarse():
    """Regression test: previously a too-large initial_dt looped forever
    because there was no rejection branch in adaptive_runge_kutta."""
    y0 = np.array([1.0])
    # initial_dt=1.0 with tolerance=1e-9 will be rejected on the first try;
    # the controller must shrink dt and keep going.
    times, states, _, _ = hamsolver.adaptive_runge_kutta(
        table=hamsolver.DP54, f=exp_decay, yn=y0, t0=0.0, tf=1.0,
        initial_dt=1.0, tolerance=1e-9,
    )
    assert times[-1] == pytest.approx(1.0, abs=1e-12)
    assert states[-1][0] == pytest.approx(math.exp(-1.0), abs=1e-7)


def test_adaptive_event_detection_stops_on_zero_crossing():
    """The damped pendulum should hit theta=0 well before tf=10."""
    y0 = np.array([1.5, 0.0])
    times, _, ev_times, ev_states = hamsolver.adaptive_runge_kutta(
        table=hamsolver.BS32, f=hamsolver.damped_pendulum_rhs, yn=y0,
        t0=0.0, tf=10.0, initial_dt=0.01, tolerance=1e-6, dt_out=0.05,
        event_fn=lambda _t, y: y[0], stop_on_event=True,
    )
    assert len(ev_times) == 1, "expected exactly one zero-crossing event"
    assert times[-1] < 10.0, "simulation should have stopped on the event"
    assert abs(ev_states[0][0]) < 1e-6, "theta at impact should be ~0"
