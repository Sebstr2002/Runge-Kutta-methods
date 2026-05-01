import numpy as np
import time
import hamsolver
from nbody import make_nbody_rhs
from initial import earth_sun_moon_initial
from constants import G

y0, masses = earth_sun_moon_initial()

python_rhs = make_nbody_rhs(masses, G=G)

cpp_rhs = hamsolver.sun_earth_moon_rhs

t0, tf, steps = 0.0, 2*np.pi, 5000
dt = (tf - t0) / steps

# Python version
print("\nRunning pure python rhs...")
start_time = time.time()
result_python = hamsolver.runge_kutta(
    table=hamsolver.Gauss_Legendre,
    f=python_rhs,
    yn=y0,
    t0=t0,
    dt=dt,
    steps=steps,
    max_iter=20
)
py_time = time.time() - start_time
print(f"Python RHS completed in {py_time:.3f} seconds.")

print("\nRunning Cpp native rhs...")
start_time = time.time()
result_cpp = hamsolver.runge_kutta(
    table=hamsolver.Gauss_Legendre,
    f=cpp_rhs,
    yn=y0,
    t0=t0,
    dt=dt,
    steps=steps,
    max_iter=20
)
cpp_time = time.time() - start_time
print(f"C++ RHS completed in {cpp_time:.3f} seconds.")
print(f"Speedup: C++ is {py_time / cpp_time:.1f}x faster!")

def total_energy(y, masses, G=1.0):
    N = masses.size
    q = y[:2*N].reshape(N, 2)
    p = y[2*N:].reshape(N, 2)
    
    # Kinetic Energy: p^2 / 2m
    KE = 0.5 * np.sum(np.sum(p*p, axis=1) / masses)
    
    # Potential Energy: -G * m1 * m2 / r
    PE = 0.0
    for i in range(N):
        for j in range(i+1, N):
            dist = np.linalg.norm(q[i] - q[j])
            PE -= G * masses[i] * masses[j] / dist
            
    return KE + PE

# Convert result to numpy for slicing
final_result = np.array(result_cpp)

E0 = total_energy(final_result[0], masses)
Ef = total_energy(final_result[-1], masses)

print(f"\n|ΔE/E0| over one year: {abs(Ef - E0) / abs(E0):.2e}")
