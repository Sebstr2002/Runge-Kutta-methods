# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build

The project compiles a C++ pybind11 module named `hamsolver` that the Python scripts in `python/` import.

```bash
cmake -S . -B build
cmake --build build
cp build/hamsolver.so python/      # python scripts import `hamsolver` from cwd
```

Requires `pybind11` (CMake `find_package(pybind11 REQUIRED)`) and a C++17 compiler. Module naming is forced to `hamsolver.so` (no `lib` prefix, no Python ABI suffix) by `set_target_properties` in `CMakeLists.txt` — keep this when editing the CMake file or imports break.

## Run

The Python entry points are scripts (no test framework, no CLI), executed from `python/` so `hamsolver.so` is on the import path:

```bash
cd python
python solving.py    # numerical comparison: position/momentum drift, E and L stats
python plots.py      # writes plots/energy_methods.pdf, plots/angular_momentum_methods.pdf
python orbits.py     # writes plots/orbit_comparison_<method>.pdf
```

## Architecture

The library is a generic Runge–Kutta integrator parameterised by a Butcher tableau, with a fixed set of named tableaus exposed to Python.

- `src/runge-kutta.hpp` / `src/butcher-tableau.cpp`: `ButcherTableau` holds `A`, `b`, `c`. The constructor calls `checkImplicit(A)` (any non-zero entry on or above the diagonal ⇒ implicit) and caches the result. `isValid()` checks shape consistency and `Σbᵢ = 1`; `isSymplectic()` checks the standard `bᵢAᵢⱼ + bⱼAⱼᵢ = bᵢbⱼ` condition. The tolerance for all of these is `constants::tolerance = 1e-10`.
- `src/runge-kutta.cpp`: one `rungekutta::runge_kutta` function drives every method. For explicit tableaus it uses the standard staged update; for implicit tableaus it does **fixed-point iteration** on the stage values `K` (not Newton, despite the comment), bounded by `max_iter` and short-circuited when the L1 change drops below `constants::tolerance`. `max_iter` therefore matters for implicit methods only.
- `methods::*_tableau` constants and matching `methods::*_method` wrappers (Heun, RK4, Trapezoidal, Implicit_midpoint, Gauss_Legendre, LobattoIIIA) live in the same files. Adding a method = define a tableau constant + a thin wrapper that calls `rungekutta::runge_kutta` + a `bind_rk_method` line in `binding.cpp` + (optionally) an entry in the `methods` registry list in `python/solving.py` and `python/plots.py`.
- `binding.cpp` + `method_bindings.hpp`: `bind_rk_method` is a template that wraps each C++ method with a uniform Python signature `(rhs, y0, t0, dt, steps, max_iter)`. It converts between `numpy` arrays and `std::vector<double>` per RHS call, so the Python `rhs` callback is invoked once per stage per step — this is the hot path.
- Python side: every script defines its own Hamiltonian RHS (currently the 2D Kepler problem) and a `methods` list pairing display names with `hamsolver.*_method` callables. Initial conditions parameterise the orbit by eccentricity `eps` via `y0 = [1-eps, 0, 0, sqrt((1+eps)/(1-eps))]`, integrated over one period `tf = 2π`. The conserved quantities used for diagnostics are energy `H = ½(px²+py²) − 1/r` and angular momentum `L = x·py − y·px`.
