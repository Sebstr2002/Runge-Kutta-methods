"""Plot Kepler orbits for several eccentricities using Gauss-Legendre.

Writes ``plots/orbit_comparison_gauss-legendre.pdf``.
"""
import os

import matplotlib.pyplot as plt
import numpy as np

import hamsolver

os.makedirs("plots", exist_ok=True)

method_name = "Gauss-Legendre"
table = hamsolver.Gauss_Legendre

t0, tf, steps = 0.0, 2 * np.pi, 10000
dt = (tf - t0) / steps

epsilons = [0.1, 0.3, 0.9]
colors = ["blue", "green", "red"]

plt.figure(figsize=(8, 6))

for eps, color in zip(epsilons, colors):
    y0 = np.array([1 - eps, 0.0, 0.0, np.sqrt((1 + eps) / (1 - eps))])
    _times, states = hamsolver.runge_kutta(
        table=table, f=hamsolver.kepler_rhs,
        yn=y0, t0=t0, dt=dt, steps=steps, max_iter=10,
    )
    traj = np.asarray(states)
    x, y = traj[:, 0], traj[:, 1]

    plt.plot(x, y, label=fr"$\epsilon = {eps}$", color=color)
    plt.plot(x[0], y[0], "o", color=color, markersize=4)

plt.axis("equal")
plt.xlabel("x")
plt.ylabel("y")
plt.title(f"Kepler Orbits for Different Eccentricities ({method_name})")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(f"plots/orbit_comparison_{method_name.lower().replace(' ', '_')}.pdf")
plt.show()
