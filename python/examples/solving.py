"""Numerical comparison of every fixed-step method against the planar Kepler problem.

Reports position drift, momentum drift, and the mean/std of energy and
angular momentum over one full orbital period. The Hamiltonian RHS is the
built-in C++ ``hamsolver.kepler_rhs`` so the integrator never crosses back
into Python during a step.
"""
import numpy as np
import hamsolver

# ---------------------------------------------------------------------------
# Initial conditions: highly-eccentric Kepler orbit parameterised by eps.
# ---------------------------------------------------------------------------
eps = 0.9
y0 = np.array([
    1 - eps,
    0.0,
    0.0,
    np.sqrt((1 + eps) / (1 - eps)),
])
t0, tf, steps = 0.0, 2 * np.pi, 10000
dt = (tf - t0) / steps


def compute_energy(traj):
    x, y, px, py = traj.T
    r = np.sqrt(x * x + y * y)
    return 0.5 * (px * px + py * py) - 1.0 / r


def compute_L(traj):
    x, y, px, py = traj.T
    return x * py - y * px


# (display name, tableau)
methods = [
    ("RK4",               hamsolver.RK4),
    ("RK4 (3/8-rule)",    hamsolver.RK4_38),
    ("Heun",              hamsolver.Heun),
    ("Implicit Midpoint", hamsolver.Implicit_midpoint),
    ("Trapezoidal",       hamsolver.Trapezoidal),
    ("Gauss-Legendre",    hamsolver.Gauss_Legendre),
    ("LobattoIIIA",       hamsolver.LobattoIIIA),
]

for name, table in methods:
    print(f"\n--- {name} ---")
    _times, states = hamsolver.runge_kutta(
        table=table, f=hamsolver.kepler_rhs,
        yn=y0, t0=t0, dt=dt, steps=steps, max_iter=10,
    )
    traj = np.asarray(states)

    pos_err = np.linalg.norm(traj[-1, :2] - traj[0, :2])
    mom_err = np.linalg.norm(traj[-1, 2:] - traj[0, 2:])
    E = compute_energy(traj)
    L = compute_L(traj)

    print(f"Position error after one period: {pos_err:.1e}")
    print(f"Momentum error after one period: {mom_err:.1e}")
    print(f"Energy    avg: {E.mean():.5e}, std: {E.std():.1e}")
    print(f"Angular L avg: {L.mean():.5e}, std: {L.std():.1e}")
