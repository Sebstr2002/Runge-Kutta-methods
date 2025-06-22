import numpy as np
import hamsolver
import matplotlib.pyplot as plt

# hamiltonian right hand side:
def rhs(t, y_): # t is a cpp double so python float and y_ is an np array or cpp vector
    x, y, px, py = y_
    r3 = (x**2 + y**2)**1.5 #derivative denominator
    return np.array([
        px,
        py,
        -x / r3,
        -y / r3
    ]) # hamiltonian equations of motion

#initial conditions
eps = 0.3
y0 = np.array([
    1-eps,
    0.0,
    0.0,
    np.sqrt((1 + eps) / (1 - eps))
])
t0, tf, dt = 0.0, 2 * np.pi, 0.001
steps = int((tf-t0) / dt)
ts = np.linspace(t0, tf, steps + 1)

# energy
def compute_Energy(x, y, px, py):
    r = np.sqrt(x**2 + y**2)
    return 0.5 * (px**2 + py**2) - 1.0 / r

# angular momentum
def compute_L(x, y, px, py):
    return x * py - y * px


# Method registry 
methods = [
    ("RK4", hamsolver.RK4_method),
    ("Heun", hamsolver.Heun_method),
    ("Implicit Midpoint", hamsolver.Implicit_midpoint_method),
    ("Trapezoidal", hamsolver.Trapezoidal_method),
    ("Gauss-Legendre", hamsolver.Gauss_Legendre_method),
    ("LobattoIIIA", hamsolver.LobattoIIIA_method)
]

# testing all methods:
for name, method in methods:
    print(f"\n--- {name} ---")

    result = method(rhs, y0, t0, dt, steps, max_iter=10)

    x, y, px, py = result.T
    E = compute_Energy(x, y, px, py)
    L = compute_L(x, y, px, py)

    start = result[0, :2].astype(np.longdouble)
    end = result[-1, :2].astype(np.longdouble)
    error_ld = np.linalg.norm(end - start)
    E_mean = np.mean(E)
    E_std = np.std(E)
    L_mean = np.mean(L)
    L_std = np.std(L)

    print(f"Position error after one period: {error_ld:.5e}")
    print(f"Energy    avg: {E_mean:.5e}, std: {E_std:.5e}")
    print(f"Angular L avg: {L_mean:.5e}, std: {L_std:.5e}")

