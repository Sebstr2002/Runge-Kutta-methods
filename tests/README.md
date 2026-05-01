# Tests

## Layout

- `tests/cpp/` — C++ unit tests using **Catch2 v3** (fetched automatically by
  CMake). They link against the same static `hamsolver_core` library that the
  Python module wraps, so what you test is what runs in production.
- `tests/python/` — End-to-end tests of the bound module via **pytest**.

## Running

From the project root:

```bash
# 1. Configure & build (downloads Catch2 the first time).
cmake -S . -B build
cmake --build build -j

# 2. Run everything via CTest.
ctest --test-dir build --output-on-failure

# Or run each side individually:
./build/tests/cpp/hamsolver_tests          # C++ tests directly
PYTHONPATH=build pytest tests/python -q    # Python tests directly
```

`ctest` runs the C++ binary and a `python_tests` job that launches `pytest`.
The Python job is skipped if `python3` isn't found at configure time. If you
don't have pytest yet:

```bash
pip install pytest
```

## What's covered

C++ tests:

- All builtin Butcher tableaus pass `isValid()`.
- `isImplicit()` / `isSymplectic()` give the correct answer on each tableau.
- Every method integrates `y' = -y` close to `exp(-1)` (tolerance scaled to
  method order).
- Symplectic methods bound the energy drift on a harmonic oscillator far
  more tightly than RK4.
- RK4 exhibits empirical 4th-order convergence under step-size halving.

Python tests:

- Output shape and initial-condition preservation for every binding.
- Final-time accuracy on `y' = -y`, parametrised by method order.
- Fourth-order methods empirically halve the step ⇒ ~16× error reduction.
- Symplectic methods bound the energy drift on the planar Kepler problem.
- All methods conserve angular momentum on Kepler within tolerance.

## Adding a new method

When you add a tableau (see `CLAUDE.md` for the C++/binding side):

1. Add a `CHECK` line in `tests/cpp/test_butcher_tableau.cpp` for `isValid`,
   `isImplicit`, `isSymplectic`.
2. Add an entry to `ALL_METHODS` (and `SYMPLECTIC_METHODS` if appropriate)
   in `tests/python/test_methods.py`. The parametrised tests will then exercise
   it automatically.
