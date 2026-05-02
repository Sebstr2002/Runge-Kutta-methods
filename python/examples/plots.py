"""Plot energy and angular-momentum drift for every fixed-step method on Kepler.

Writes ``plots/energy_methods.pdf`` and ``plots/angular_momentum_methods.pdf``.
Each method is integrated exactly once per run; both invariants are computed
from the same trajectory.
"""
import os

import matplotlib.pyplot as plt
import numpy as np

import hamsolver

os.makedirs("plots", exist_ok=True)

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

eps = 0.3
y0 = np.array([1 - eps, 0.0, 0.0, np.sqrt((1 + eps) / (1 - eps))])
t0, tf, steps = 0.0, 2 * np.pi, 10000
dt = (tf - t0) / steps
ts = np.linspace(t0, tf, steps + 1)
H_exact = -0.5
L_exact = np.sqrt(1 - eps**2)

methods = [
    ("RK4",               hamsolver.RK4),
    ("RK4 (3/8-rule)",    hamsolver.RK4_38),
    ("Heun",              hamsolver.Heun),
    ("Implicit Midpoint", hamsolver.Implicit_midpoint),
    ("Trapezoidal",       hamsolver.Trapezoidal),
    ("Gauss-Legendre",    hamsolver.Gauss_Legendre),
    ("LobattoIIIA",       hamsolver.LobattoIIIA),
]

# ---------------------------------------------------------------------------
# Integrate each method ONCE; reuse the trajectory for both diagnostics.
# ---------------------------------------------------------------------------

trajectories = {}
for name, table in methods:
    _times, states = hamsolver.runge_kutta(
        table=table, f=hamsolver.kepler_rhs,
        yn=y0, t0=t0, dt=dt, steps=steps, max_iter=10,
    )
    trajectories[name] = np.asarray(states)

# ---------------------------------------------------------------------------
# Energy
# ---------------------------------------------------------------------------

plt.figure()
for name, traj in trajectories.items():
    x, y_, px, py = traj.T
    r = np.sqrt(x * x + y_ * y_)
    H = 0.5 * (px * px + py * py) - 1.0 / r
    plt.plot(ts, H, label=name)
plt.axhline(H_exact, linestyle="--", color="k", label="Exact -0.5")
plt.title("Energy over time")
plt.xlabel("t")
plt.ylabel("H(t)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("plots/energy_methods.pdf")

# ---------------------------------------------------------------------------
# Angular momentum
# ---------------------------------------------------------------------------

plt.figure()
for name, traj in trajectories.items():
    x, y_, px, py = traj.T
    L = x * py - y_ * px
    plt.plot(ts, L, label=name)
plt.axhline(L_exact, linestyle="--", color="k", label=f"Exact {L_exact:.4f}")
plt.title("Angular Momentum over time")
plt.xlabel("t")
plt.ylabel("L(t)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("plots/angular_momentum_methods.pdf")

print("Saved: plots/energy_methods.pdf and plots/angular_momentum_methods.pdf")
