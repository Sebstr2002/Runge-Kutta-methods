import numpy as np
import matplotlib.pyplot as plt
import hamsolver
import os

# Create output folder
os.makedirs("plots", exist_ok=True)

# Hamiltonian RHS
def rhs(t, y):
    x, y_, px, py = y
    r3 = (x**2 + y_**2)**1.5
    return np.array([px, py, -x / r3, -y_ / r3])

# Method (choose any from your bindings)
method_name = "Gauss-Legendre"
method = hamsolver.Gauss_Legendre_method

# Time settings
t0, tf, steps = 0.0, 2 * np.pi, 10000
dt = (tf-t0) /steps

# Eccentricities to simulate
epsilons = [0.1, 0.3, 0.9]
colors = ['blue', 'green', 'red']

# Create the figure
plt.figure(figsize=(8, 6))

for eps, color in zip(epsilons, colors):
    # Initial conditions for given epsilon
    y0 = np.array([
        1 - eps,
        0.0,
        0.0,
        np.sqrt((1 + eps) / (1 - eps))
    ])

    result = method(rhs, y0, t0, dt, steps, max_iter=10)
    x, y_ = result[:, 0], result[:, 1]

    label = fr"$\epsilon = {eps}$"
    plt.plot(x, y_, label=label, color=color)
    plt.plot(x[0], y_[0], 'o', color=color, markersize=4)

# Plot settings
plt.axis('equal')
plt.xlabel("x")
plt.ylabel("y")
plt.title(f"Kepler Orbits for Different Eccentricities ({method_name})")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(f"plots/orbit_comparison_{method_name.lower().replace(' ', '_')}.pdf")
plt.show()
