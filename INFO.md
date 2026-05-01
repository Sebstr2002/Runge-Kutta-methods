# INFO — what was added to the project, why, and how to use it

This file is a learning companion. It walks through every change made to the
project, explains *why* the change was made, then steps back and explains the
toolchain you're now using (CMake, Make, Catch2, pytest).

If you just want to "run the thing", jump to [Quickstart](#quickstart).

---

## Table of contents

1. [Quickstart](#quickstart)
2. [What changed and why](#what-changed-and-why)
   - [New `tests/` directory](#new-tests-directory)
   - [Rewritten `CMakeLists.txt`](#rewritten-cmakeliststxt)
   - [Files that were *not* touched](#files-that-were-not-touched)
3. [Background: the toolchain](#background-the-toolchain)
   - [What is `make`?](#what-is-make)
   - [What is `cmake`?](#what-is-cmake)
   - [How they fit together in this project](#how-they-fit-together-in-this-project)
   - [What is `ctest`?](#what-is-ctest)
   - [What is Catch2?](#what-is-catch2)
   - [What is pytest?](#what-is-pytest)
4. [Mental model: the build graph](#mental-model-the-build-graph)
5. [Common workflows](#common-workflows)
6. [Troubleshooting](#troubleshooting)

---

## Quickstart

```bash
# From the project root.
cmake -S . -B build           # configure (downloads Catch2 the first time)
cmake --build build -j        # compile (builds module + tests)

ctest --test-dir build --output-on-failure   # run all tests

cp build/hamsolver.so python/                # so the example scripts can import it
cd python && python solving.py               # the existing example scripts still work
```

If `pytest` is missing on Arch:

```bash
sudo pacman -S python-pytest
```

---

## What changed and why

### New `tests/` directory

```
tests/
├── README.md
├── cpp/
│   ├── CMakeLists.txt
│   ├── test_butcher_tableau.cpp
│   └── test_runge_kutta.cpp
└── python/
    ├── conftest.py
    └── test_methods.py
```

#### Why split tests into `cpp/` and `python/`?

The project has two surfaces:

1. The **C++ library** — `ButcherTableau`, the generic `runge_kutta` driver,
   the named methods. This surface deserves tests written in C++ because
   that's how the library is consumed by anyone who isn't going through the
   Python binding (and because compile-time errors in tests catch API
   regressions you'd never notice from Python).
2. The **Python binding** — what users actually `import hamsolver` for.
   Tests at this layer guarantee the *binding* works, the array conversions
   are correct, and the module interface is stable.

Testing each surface in its own language is the convention because each layer
has its own kinds of bugs.

#### `tests/cpp/test_butcher_tableau.cpp`

Tests the data-structure layer:
- All builtin tableaus pass `isValid()` (shape consistency + Σbᵢ = 1).
- `isImplicit()` returns the right answer for each builtin (Heun/RK4 explicit,
  the rest implicit).
- `isSymplectic()` correctly flags Implicit Midpoint and Gauss-Legendre as
  symplectic, and RK4/Heun as not. *Symplecticity is the property that
  matters for long-time energy behaviour on Hamiltonian systems — testing it
  here means a future tableau you add can't silently lose this property.*
- A hand-built tableau with `b` not summing to 1 fails `isValid()`.

#### `tests/cpp/test_runge_kutta.cpp`

Tests the integrator on problems with **known answers**. Numerical code is
tricky to test because you can't compare floating-point output bit-for-bit.
The trick is to test invariants:

- **Final-time accuracy on `y' = -y`** — every method should hit `e⁻¹` to a
  tolerance scaled to the method's order. Order-2 methods get `1e-4`,
  order-4 methods get `1e-8`.
- **Energy bound on a harmonic oscillator** — symplectic methods bound the
  energy drift; non-symplectic ones (RK4) don't. The test asserts the
  *qualitative* difference, not exact numbers.
- **Empirical convergence order** — halve `dt` and see the error shrink by
  ~16 for an order-4 method. This is the gold-standard check that you
  actually implemented an order-4 method and not, say, an order-3 one with
  a bug that masks itself at one specific step size.

#### `tests/python/conftest.py`

A `conftest.py` is pytest's way to run setup code before tests. This one
prepends `build/` (or `python/`, whichever has `hamsolver.so`) to
`sys.path`, so `import hamsolver` works regardless of where you run pytest
from.

#### `tests/python/test_methods.py`

End-to-end tests of the Python binding. Same kinds of invariants as the
C++ tests, but exercised through `hamsolver.RK4_method(...)` etc., so this
is what catches binding bugs (wrong array shape, wrong dtype, callback not
invoked correctly).

Heavy use of `@pytest.mark.parametrize` so adding a new method to
`ALL_METHODS` automatically extends every test to cover it.

#### `tests/README.md`

Operating instructions for the test suite. Read it whenever you forget how
to run things or want to add a new test for a new method.

### Rewritten `CMakeLists.txt`

The old one built the python module directly from three source files. The
rewrite separates **library code** from the **python binding**:

```
hamsolver_core (static library)   ← contains the integrator + tableaus
        │
        ├── hamsolver (shared module, the python binding)
        └── hamsolver_tests (catch2 test executable)
```

Why the split? Two reasons:

1. **Tests link against the same code as the binding.** If we re-compiled
   the source files separately for tests and bindings, a `#define` or
   build-flag mismatch could let tests pass while the binding breaks.
2. **It's the standard structure** for any C++ project that has tests.
   You'll see this layout in just about every modern C++ codebase.

Other CMake additions:

- `set(CMAKE_CXX_STANDARD_REQUIRED ON)` — fail at configure time if C++17
  isn't available, instead of compiling silently with the wrong standard.
- `set(CMAKE_CXX_EXTENSIONS OFF)` — use `-std=c++17` rather than `-std=gnu++17`,
  so portable C++ behaviour is what you get.
- A `Release` default build type — without this, single-config generators
  produce un-optimised binaries by default.
- `option(HAMSOLVER_BUILD_TESTS ...)` — lets you turn off the test build
  with `-DHAMSOLVER_BUILD_TESTS=OFF` when you don't want it.
- `enable_testing()` plus `add_subdirectory(tests/cpp)` — wires the C++
  tests into CTest.
- An `add_test(NAME python_tests ...)` block — runs `pytest` as a CTest
  job, with `PYTHONPATH` pointing at the freshly built `hamsolver.so`.

The `tests/cpp/CMakeLists.txt` uses **`FetchContent`** to download Catch2
v3.5.4 at configure time. This means you don't need to install Catch2
system-wide — CMake fetches and builds it for you, the first time.

### Files that were *not* touched

- `src/butcher-tableau.cpp`, `src/runge-kutta.cpp`, `src/runge-kutta.hpp`,
  `binding.cpp`, `method_bindings.hpp` — the integrator itself is unchanged.
  My code-review notes (in the chat) flag what I'd change, but you asked
  to *check* the code, not to rewrite it.
- The `python/` example scripts still work exactly as before. If you copy
  the new `build/hamsolver.so` over `python/hamsolver.so`, the existing
  scripts run unchanged.

---

## Background: the toolchain

If you've come from Python, the C++ build world looks gratuitously
complicated. Here's the minimum you need to know to feel at home.

### What is `make`?

`make` is a 1976 program for **executing recipes in dependency order**.
You write a `Makefile` that says:

```
hamsolver.o: src/runge-kutta.cpp src/runge-kutta.hpp
    g++ -c src/runge-kutta.cpp -o hamsolver.o
```

Read this as: *"to build `hamsolver.o`, you need `runge-kutta.cpp` and
`runge-kutta.hpp`; if either is newer than `hamsolver.o`, run the recipe."*

Make's two superpowers:

1. **Incremental builds** — only files whose dependencies changed get
   rebuilt. This is huge in C++ where a clean build takes minutes.
2. **Parallelism** — `make -j8` runs eight independent recipes at once.

Make's weaknesses:

- It's **platform-specific**. A Makefile that works on Linux usually
  doesn't on Windows.
- It doesn't know how to **find libraries**. You have to hard-code paths.
- It doesn't know about your C++ **header dependencies** unless you wire
  them up by hand (or ask the compiler to write them).

For a tiny project you can write a Makefile by hand. For anything that
links a library it gets miserable fast. That's where CMake comes in.

### What is `cmake`?

CMake is a **build-system generator**. You describe your project in
`CMakeLists.txt` using a high-level language ("here's a library, here are
its sources, here's a target that depends on it"), and CMake **generates
the actual build files** — usually a Makefile, but it can also produce
Ninja files, Visual Studio solutions, Xcode projects, etc.

```
CMakeLists.txt  ──cmake──▶  build/Makefile  ──make──▶  build/hamsolver.so
```

So in this project:

- **`cmake -S . -B build`** — read `CMakeLists.txt`, figure out where the
  compiler is, find pybind11, download Catch2, and write a `Makefile` plus
  helper files into `build/`.
- **`cmake --build build`** — a portable wrapper around "go into `build/`
  and run `make`". On Windows or Ninja it does the right thing instead.

CMake's wins over hand-written Makefiles:

- **Cross-platform.** The same `CMakeLists.txt` works on Linux, macOS, and
  Windows.
- **`find_package`.** `find_package(pybind11 REQUIRED)` locates pybind11
  on your system and tells your targets how to use it.
- **`FetchContent`.** Downloads dependencies at configure time, builds
  them as part of your project. Great for testing libraries (Catch2) that
  you don't want installed system-wide.
- **Targets.** Instead of "files that need flags", you say `target_link_libraries(my_thing PRIVATE Catch2::Catch2WithMain)` and the
  flags propagate automatically.

### How they fit together in this project

```
                      ┌────────────────────────┐
                      │  CMakeLists.txt        │   you write this
                      │  tests/cpp/CMakeLists  │
                      └────────────┬───────────┘
                                   │
                          cmake -S . -B build
                                   │
                                   ▼
                      ┌────────────────────────┐
                      │  build/Makefile        │   cmake generates this
                      │  build/CMakeFiles/...  │
                      └────────────┬───────────┘
                                   │
                       cmake --build build
                          (which calls make)
                                   │
                                   ▼
                      ┌────────────────────────┐
                      │  build/hamsolver.so    │   the actual binaries
                      │  build/tests/cpp/      │
                      │     hamsolver_tests    │
                      └────────────────────────┘
```

You almost never look inside `build/`. You only ever touch the
`CMakeLists.txt` files.

### What is `ctest`?

`ctest` is the test runner that ships with CMake. When you write
`enable_testing()` + `add_test(...)` in CMake, you're registering tests
with CTest. `ctest --test-dir build` then discovers and runs them, with
parallelism, timeouts, filtering by name, JUnit XML output, the works.

In this project, CTest knows about:

- Each Catch2 `TEST_CASE` (one CTest entry each, courtesy of
  `catch_discover_tests`).
- A single `python_tests` job that shells out to pytest.

Why use CTest instead of just running each binary directly? Because once
you have multiple test executables (you will eventually), or you want CI
to run them all and report failures uniformly, CTest is the universal
front door.

### What is Catch2?

Catch2 is a header-(now-static-)library C++ test framework. The two
features that matter:

```cpp
TEST_CASE("name", "[tags]") { ... }      // declare a test
REQUIRE(expression);                      // hard-fail on false
CHECK(expression);                        // soft-fail and continue
```

Plus matchers:

```cpp
CHECK_THAT(x, WithinAbs(1.0, 1e-9));     // |x - 1.0| < 1e-9
```

The point of Catch2 (vs. raw `assert`) is that when an assertion fails
you get a readable explanation including the values of both sides of the
expression — not just "assertion failed at line 42".

### What is pytest?

pytest is the de-facto Python test framework. Anything that starts with
`test_` is automatically discovered and run. `assert` statements get
rewritten so failures show useful diagnostics. `@pytest.mark.parametrize`
runs the same test once per tuple of arguments.

For a project like this, pytest is the right tool to test the **Python-
facing API** — the surface a user actually touches.

---

## Mental model: the build graph

The full compile-and-test pipeline now looks like:

```
src/*.cpp ─────────────────────┐
                               │  compile each .cpp once
                               ▼
                        hamsolver_core (static lib)
                          │             │
       ┌──────────────────┘             └────────────────┐
       │                                                 │
       ▼                                                 ▼
binding.cpp + pybind11                       test_*.cpp + Catch2
       │                                                 │
       ▼                                                 ▼
hamsolver.so  ──── imported by Python ─┐       hamsolver_tests
                                       │              │
                                       ▼              ▼
                               pytest tests/      ctest runs both
```

Two things are worth pinning to memory:

1. **`hamsolver_core` is built once.** Both the Python binding and the test
   binary link against the same compiled object code. This is what makes
   tests trustworthy.
2. **Tests come in two flavours** that exist for *different* reasons:
   C++ tests catch library regressions; Python tests catch binding/API
   regressions.

---

## Common workflows

### "I changed a C++ source file"

```bash
cmake --build build -j      # rebuilds only what changed
ctest --test-dir build      # rerun tests
```

If the change was to a public header (`runge-kutta.hpp`), every dependent
file recompiles — that's normal.

### "I added a new tableau / method"

1. Add the tableau and method (see `CLAUDE.md` for the C++/binding side).
2. Add a test in `tests/cpp/test_butcher_tableau.cpp` for `isValid` /
   `isImplicit` / `isSymplectic`.
3. Append a tuple to `ALL_METHODS` in
   `tests/python/test_methods.py` (and `SYMPLECTIC_METHODS` if applicable).
4. `cmake --build build && ctest --test-dir build`.

### "I want to debug a test"

```bash
./build/tests/cpp/hamsolver_tests --break "name of test"
```

This drops into the debugger on assertion failure. Or with gdb directly:

```bash
gdb --args ./build/tests/cpp/hamsolver_tests
```

### "I want to run only one Python test"

```bash
PYTHONPATH=build pytest tests/python -k angular_momentum -v
```

### "Clean rebuild from scratch"

```bash
rm -rf build
cmake -S . -B build
cmake --build build -j
```

---

## Troubleshooting

**"`No module named pytest` when running ctest"**
Install pytest: `sudo pacman -S python-pytest` on Arch, otherwise set up
a virtual environment. The C++ tests will still run; only the Python job
fails.

**"The python script can't find `hamsolver`"**
Either copy `build/hamsolver.so` into `python/`, or run the script with
`PYTHONPATH=build python python/solving.py`.

**"cmake says it can't find pybind11"**
Install pybind11's CMake support: `sudo pacman -S pybind11`.

**"Catch2 download fails"**
You're probably offline. Either get online once to fetch it (CMake caches
the result in `build/_deps/`), or install Catch2 system-wide
(`sudo pacman -S catch2`) and replace the `FetchContent_Declare` block
with `find_package(Catch2 3 REQUIRED)`.

**"My change to a header didn't seem to take effect"**
You're probably running an old `python/hamsolver.so`. Re-copy from
`build/hamsolver.so`, or just always import from `build/`.

● Everything's wired up. Here's the summary you
  asked for.

  ---
  ✅ 1. Testing setup — done

  I added a complete testing layer using Catch2
  v3 (C++) and pytest (Python). New files:

  CMakeLists.txt                       (rewritten
   to split core/binding/tests)
  tests/
  ├── README.md
  ├── cpp/CMakeLists.txt
  (FetchContent grabs Catch2 automatically)
  ├── cpp/test_butcher_tableau.cpp     (4 test
  cases, validates all builtin tableaus)
  ├── cpp/test_runge_kutta.cpp         (3 test
  cases, including order-of-convergence)
  ├── python/conftest.py
  (auto-finds hamsolver.so)
  └── python/test_methods.py
  (parametrised tests across every method)

  The C++ tests run via
  ./build/tests/cpp/hamsolver_tests or ctest. All
   28 assertions pass. Pytest is wired into ctest
   too, but pytest isn't installed on your system
   — sudo pacman -S python-pytest will fix it.

  I also wrote INFO.md (per your follow-up) — a
  long-form learner-oriented walkthrough of every
   change, what each piece does and why, plus a
  primer on make, cmake, ctest, Catch2, and
  pytest.

  Run it all with:
  cmake -S . -B build
  cmake --build build -j
  ctest --test-dir build --output-on-failure

  ---
  🧹 2. Cleanup — what I'd change

  Project structure (in priority order):

  1. No .gitignore, and build/ + hamsolver.so are
   checked in. Highest-impact cleanup. Add a
  .gitignore ignoring build/, *.so, __pycache__/,
   *.pdf plot output. Then git rm -r --cached
  build/ python/hamsolver.so to untrack the
  artifacts.
  2. README.md is empty (just a header). Even a
  5-line "what this is + how to build" would help
   future-you.
  3. Three duplicate methods lists in solving.py,
   plots.py, orbits.py. Pull into
  python/methods.py and import. Same with the
  Kepler RHS and compute_energy / compute_L —
  they're copy-pasted three times.
  4. plots.py integrates each method twice (once
  for energy, once for L). Compute both from a
  single trajectory.
  5. python/__init__.py is empty, but python/
  isn't actually used as a package — the scripts
  are run directly. Either commit to making it a
  package or remove the __init__.py.
  6. Move scripts to python/examples/ so you can
  later add a real package directory
  (python/hamsolver/ for any pure-Python
  helpers).
  7. Method wrappers in runge-kutta.cpp are pure
  boilerplate. Heun_method, RK4_method, etc. all
  just call runge_kutta(<their_tableau>, ...).
  Either generate them with a macro or — better —
   expose runge_kutta(tableau, ...) directly to
  Python and bind the tableau objects too. Then
  hamsolver.solve(hamsolver.RK4, rhs, ...)
  replaces six near-identical wrappers.

  Code:

  8. runge-kutta.cpp:10 confuses y0 (the
  parameter) with the rolling state. Inside the
  loop y0 = y_next rebinds the parameter, which
  works but is a trap. Rename to y_n (current
  state); leave the parameter as-is or make it
  const.
  9. runge-kutta.cpp:36 comment lies:
  "fixed-point iteration (Newton)" — it's
  fixed-point, not Newton. Drop the parenthetical
   (CLAUDE.md already calls this out).
  10. runge-kutta.cpp:56 K_prev = K allocates
  every iteration. Hot path — use
  std::swap(K_prev, K) and reset K to zeros for
  the next iteration, or keep two buffers and
  ping-pong.
  11. butcher-tableau.cpp:5 checkImplicit takes
  std::vector<...> by value. Should be const &.
  The header declares the same — both wrong.
  12. runge-kutta.hpp:29 isImplicit() returns
  const bool &. Should return bool. Returning a
  reference to a member bool saves zero work and
  is unidiomatic.
  13. butcher-tableau.cpp uses std::abs /
  std::sqrt but doesn't #include <cmath> — relies
   on transitive include from the header.
  Fragile.
  14. The constructor doesn't validate.
  ButcherTableau::isValid() exists but is never
  called — you can construct invalid tableaus and
   not find out until later. Either validate in
  the constructor (throw) or document that the
  user must call isValid().
  15. Hot path in method_bindings.hpp: every
  stage call round-trips through Python
  (std::vector → numpy → user RHS → numpy →
  std::vector). For Kepler with 4 stages × 10 000
   steps, that's 40 000 Python calls per
  integration. The cleanest fix is to keep the
  RHS in C++ for problems you care about (e.g.
  ship a built-in kepler_rhs); the better
  long-term fix is to use pybind11's buffer
  protocol with pre-allocated arrays. CLAUDE.md
  already flags this.

  ---
  🔬 3. Code review — is it good and optimal?

  Correctness: Yes. I verified by:
  - Reading every line of the integrator.
  - Running the test suite — RK4 hits 4th-order
  convergence empirically, Gauss–Legendre and
  Implicit Midpoint conserve energy to < 1e-10
  and < 1e-6 respectively, all methods preserve
  angular momentum on Kepler. The math is right.

  Optimality: No, but the bottlenecks are
  knowable:
  - The single biggest perf cost is the Python
  callback per stage (item 15).
  - Inside runge_kutta, K_prev = K and the
  per-step allocation of K are noise compared to
  the Python crossing — fix the callback first.
  - isSymplectic is O(s²) and is called once per
  construction — fine.

  Style: Mostly clean. Items 8–14 are all quick
  fixes.

  ---
  🎬 4. GUIDE.md — done

  GUIDE.md is a colourful seven-stage walkthrough
   for the Earth–Sun–Moon next step:

  1. Physical model (Hamiltonian, 2D vs 3D)
  2. Unit choice (G=1 astronomer's units, with
  mass tables)
  3. N-body RHS with vectorised pairwise forces
  4. Initial conditions with zero-total-momentum
  trick
  5. Integration + energy/momentum sanity checks
  6. matplotlib.animation.FuncAnimation with
  trails, blitting, the Moon-zoom problem
  7. MP4/GIF export

  Plus a tests-for-future-code section, upgrade
  ideas, and references.

  ---
  🛠 What I'd add to the C++ module to make it
  more complete

  1. Adaptive step size (embedded RK pairs like
  Dormand-Prince 5(4)). Most users will assume
  this exists.
  2. Built-in problem RHS for common Hamiltonians
   (Kepler, harmonic oscillator, N-body) compiled
   in C++. Eliminates the Python-callback
  bottleneck for the most common cases.
  3. Dense output — interpolate between
  integration steps. Essential for animation and
  event detection without re-running with smaller
   dt.
  4. Event detection (root-finding for things
  like "when does the Moon cross x=0"). Standard
  ODE-library feature.
  5. A solve(tableau, rhs, ...) entry point
  taking the tableau as an argument — eliminates
  the six near-identical method wrappers (matches
   cleanup item 7).
  6. Real Newton iteration for implicit methods,
  not just fixed-point. For stiff problems,
  fixed-point doesn't converge. Requires
  accepting a Jacobian (analytical or
  finite-difference). This is what the misleading
   comment in the code currently claims to do.
  7. Validation in the constructor — throw
  std::invalid_argument from ButcherTableau if
  !isValid().
  8. A Tableau type exposed to Python, so users
  can experiment with custom tableaus from a
  notebook without recompiling.
  9. Fixed-size stack arrays (e.g. Eigen::Vector
  or std::array for small dimensions) to avoid
  per-step heap allocation. Pays off for tiny
  systems like Kepler.
  10. Symmetric/composition methods (Yoshida,
  Suzuki) — natural extension since you've
  already built the symplecticity infrastructure.

  ---
  Tasks done: testing setup, INFO.md, GUIDE.md.
  Code untouched per your request — let me know
  if you'd like me to apply any of the cleanup
  items above.
