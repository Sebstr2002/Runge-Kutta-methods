<h1 align="center">hamsolver</h1>

<p align="center">
  <em>A small, fast Runge–Kutta toolbox for Hamiltonian and general ODE problems —<br>
  written in C++17, driven from Python via pybind11.</em>
</p>

<p align="center">
  <img alt="C++" src="https://img.shields.io/badge/C%2B%2B-17-00599C?logo=c%2B%2B&logoColor=white">
  <img alt="Python" src="https://img.shields.io/badge/Python-3.9%2B-3776AB?logo=python&logoColor=white">
  <img alt="CMake" src="https://img.shields.io/badge/CMake-%E2%89%A53.14-064F8C?logo=cmake&logoColor=white">
  <img alt="pybind11" src="https://img.shields.io/badge/pybind11-binding-2A2A2A">
  <img alt="License" src="https://img.shields.io/badge/license-MIT-green">
</p>

---

## What it is

`hamsolver` is a Runge–Kutta integrator built around a generic Butcher-tableau driver. One C++ function (`runge_kutta`) handles every named method — explicit, implicit, symplectic, embedded — and a thin pybind11 layer makes the whole thing feel like a regular Python module.

It was written to study the long-time behaviour of Hamiltonian systems (Kepler orbits, double pendulums, the post-Newtonian Mercury problem, …) where *which* integrator you pick matters at least as much as the step size you give it.

```python
import hamsolver

# Generic entry point: pass any tableau as the first argument.
times, states = hamsolver.runge_kutta(
    table=hamsolver.Gauss_Legendre,        # 4th-order, symplectic, implicit
    f=hamsolver.kepler_rhs,                # built-in C++ RHS — no Python callback in the hot loop
    yn=[0.7, 0.0, 0.0, 1.5],
    t0=0.0, dt=2 * 3.14159 / 10000,
    steps=10000, max_iter=20,
)
```

## Highlights

- 🧮 **Six built-in tableaus** — Heun, RK4, Trapezoidal, Implicit Midpoint, 4th-order Gauss–Legendre, LobattoIIIA, and BS3(2) for adaptive work.
- ⚙️ **Generic driver** — the same `runge_kutta(table, f, …)` works for every method; explicit tableaus take the standard staged update, implicit tableaus solve `(I − Δt·A⊗J) δK = G` with real Newton iteration.
- 📈 **Adaptive solver** with **dense output** (cubic-Hermite spline between accepted steps) and **event detection** (bisection on the spline).
- ⚡ **Built-in C++ Hamiltonians** — Kepler, Sun–Earth–Moon, double pendulum, CR3BP, post-Newtonian Mercury, damped pendulum — so the ODE right-hand side stays in C++ and the Python interpreter is out of the inner loop.
- 🧪 **Catch2 + pytest tests** in one CTest run, including empirical convergence-order checks and symplecticity-aware energy-drift bounds.
- 🐍 **Bring your own tableau** — `hamsolver.ButcherTableau(A, b, c[, bstar])` validates shape and Σbᵢ = 1, classifies implicit/symplectic for you, and plugs straight into the solvers.

## Build

Requires **C++17**, **CMake ≥ 3.14**, **pybind11**, and **Python 3.9+** (with NumPy and matplotlib for the example scripts).

```bash
cmake -S . -B build
cmake --build build -j
cp build/hamsolver.so python/examples/      # so the example scripts can import it
```

Or just:

```bash
make install
```

## Run the tests

```bash
ctest --test-dir build --output-on-failure
```

Catch2 v3 is fetched automatically the first time you configure. The test executable links against the same `hamsolver_core` static library that the Python module wraps, so what's tested is what runs.

## A whirlwind tour of the examples

Run from `python/examples/` (so `import hamsolver` finds the `.so` next to the script):

| Script                 | What it shows |
|------------------------|---------------|
| `double_pendulum.py`   | Adaptive BS32 + cubic-Hermite dense output animating the chaotic double pendulum at exactly 60 fps. |
| `mercury.py`           | Post-Newtonian Mercury orbit — the perihelion *precesses*, drawing a flower in phase space. |
| `text.py`              | Damped pendulum with **event detection** (`stop_on_event=True`) — solver halts the moment the bob crosses zero. |

```python
# text.py — event detection in five lines
import hamsolver

times, states, ev_times, ev_states = hamsolver.adaptive_runge_kutta(
    table=hamsolver.BS32, f=hamsolver.damped_pendulum_rhs,
    yn=[1.5, 0.0], t0=0.0, tf=10.0,
    initial_dt=0.01, tolerance=1e-6, dt_out=0.05,
    event_fn=lambda t, y: y[0],   # zero-crossing of the angle
    stop_on_event=True,
)
print(f"Crossed zero at t = {ev_times[0]:.6f}s")
```

## Project layout

```
.
├── src/
│   ├── runge-kutta.{hpp,cpp}      generic driver + adaptive solver
│   ├── butcher-tableau.cpp        ButcherTableau class + named tableaus
│   ├── physics.{hpp,cpp}          built-in C++ Hamiltonian RHS functions
│   └── utils.{hpp,cpp}            Hermite spline, Jacobian (FD), linear solver
├── binding.cpp                    pybind11 module — exposes everything to Python
├── python/
│   ├── examples/                  runnable scripts (animations, plots)
│   └── sem/                       Sun–Earth–Moon n-body experiment
├── tests/
│   ├── cpp/                       Catch2 unit tests
│   └── python/                    pytest end-to-end tests
└── CMakeLists.txt
```

## Roadmap

The integrator is functional but not finished. The current rough edges and the planned next steps are tracked in [`INFO.md`](./INFO.md) — most notably the missing step-rejection branch in `adaptive_runge_kutta`, plus wishlist items like Yoshida composition methods, Eigen-backed linear algebra, and a `solve()` convenience layer that auto-detects fixed vs. adaptive from the tableau.

## License

MIT.
