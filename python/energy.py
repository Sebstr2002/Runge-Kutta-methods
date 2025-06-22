import numpy as np
import matplotlib.pyplot as plt
import hamsolver

# --- Hamiltonian RHS ---
def hamiltonian_rhs(t, y):
    x, y_, px, py = y
    r3 = (x**2 + y_**2)**1.5
    return np.array([
        px,
        py,
        -x / r3,
        -y_ / r3
    ])

# --- Energy function ---
def compute_energy(x, y, px, py):
    r = np.sqrt(x**2 + y**2)
    return 0.5 * (px**2 + py**2) - 1.0 / r

# --- Initial setup ---
eps = 0.3
y0 = np.array([1 - eps, 0.0, 0.0, np.sqrt((1 + eps) / (1 - eps))])
t0, tf, dt = 0.0, 2 * np.pi, 0.001
steps = int((tf - t0) / dt)
ts = np.linspace(t0, tf, steps + 1)
H_exact = -0.5

# --- Method registry ---
methods = [
    ("RK4", hamsolver.RK4_method),
    ("Heun", hamsolver.Heun_method),
    ("Implicit Midpoint", hamsolver.Implicit_midpoint_method),
    ("Trapezoidal", hamsolver.Trapezoidal_method),
    ("Gauss-Legendre", hamsolver.Gauss_Legendre_method),
    ("LobattoIIIA", hamsolver.LobattoIIIA_method)
]

# --- Plot Energy ---
plt.figure(figsize=(10, 6))

result = hamsolver.Gauss_Legendre_method(hamiltonian_rhs, y0, t0, dt, steps, max_iter=10)
x, y_, px, py = result.T
H = compute_energy(x, y_, px, py)
plt.plot(ts, H, label="implicit")

# Plot exact energy
plt.axhline(H_exact, color='k', linestyle='--', label='Exact H = -0.5')

plt.xlabel("t")
plt.ylabel("Energy H(t)")
plt.title("Energy over Time for Different Methods")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("plots/energy_plot.pdf")
