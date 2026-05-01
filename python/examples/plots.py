import numpy as np
import matplotlib.pyplot as plt
import hamsolver
import os

# Create output folder
os.makedirs("plots", exist_ok=True)

# RHS of Kepler Hamiltonian system
def hamiltonian_rhs(t, y):
    x, y_, px, py = y
    r3 = (x**2 + y_**2)**1.5
    return np.array([px, py, -x / r3, -y_ / r3])

def compute_energy(x, y, px, py):
    r = np.sqrt(x**2 + y**2)
    return 0.5 * (px**2 + py**2) - 1.0 / r

def compute_L(x, y, px, py):
    return x * py - y * px

# Initial conditions
eps = 0.3
y0 = np.array([1 - eps, 0.0, 0.0, np.sqrt((1 + eps) / (1 - eps))])
t0, tf, steps = 0.0, 2 * np.pi, 10000
dt = (tf-t0) /steps
ts = np.linspace(t0, tf, steps + 1)
H_exact = -0.5

methods = [
    ("RK4", hamsolver.RK4_method),
    ("Heun", hamsolver.Heun_method),
    ("Implicit Midpoint", hamsolver.Implicit_midpoint_method),
    ("Trapezoidal", hamsolver.Trapezoidal_method),
    ("Gauss-Legendre", hamsolver.Gauss_Legendre_method),
    ("LobattoIIIA", hamsolver.LobattoIIIA_method)
]

# Plot energy
plt.figure()
for name, method in methods:
    result = method(hamiltonian_rhs, y0, t0, dt, steps, 10)
    x, y_, px, py = result.T
    H = compute_energy(x, y_, px, py)
    plt.plot(ts, H, label=name)
plt.axhline(H_exact, linestyle='--', color='k', label='Exact -0.5')
plt.title("Energy over time")
plt.xlabel("t")
plt.ylabel("H(t)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("plots/energy_methods.pdf")

# Plot angular momentum
plt.figure()
for name, method in methods:
    result = method(hamiltonian_rhs, y0, t0, dt, steps, 10)
    x, y_, px, py = result.T
    L = compute_L(x, y_, px, py)
    plt.plot(ts, L, label=name)
plt.axhline(np.sqrt(1 - eps**2), linestyle='--', color='k', label='Exact L')
plt.title("Angular Momentum over time")
plt.xlabel("t")
plt.ylabel("L(t)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("plots/angular_momentum_methods.pdf")

print("Saved: plots/energy_methods.pdf and angular_momentum_methods.pdf")
