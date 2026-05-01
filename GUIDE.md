# 🌍🌑☀️  GUIDE — Earth–Sun–Moon and animating it in Python

> A colourful, opinionated walk-through for taking your `hamsolver`
> integrator from the *one-body Kepler problem* to a real
> **three-body Earth–Sun–Moon system**, then animating the result with
> matplotlib.
>
> *Goal: by the end of this you have a `python/three_body.py` that produces
> a smooth, physically-believable animation of the Earth and Moon orbiting
> the Sun.*

---

## 🗺️  Roadmap

| Stage | What you do | Deliverable |
|------:|:------------|:------------|
| 1 | 🔭 Pick a physical model | scribble on paper |
| 2 | 📐 Choose units that don't blow up | `constants.py` |
| 3 | 🧮 Write the N-body Hamiltonian RHS | `nbody.py` |
| 4 | 🚀 Pick initial conditions | `initial_conditions.py` |
| 5 | 🎯 Integrate and sanity-check | `three_body.py` |
| 6 | 🎬 Animate with `matplotlib.animation` | `animate.py` |
| 7 | 💾 Export to `.gif` / `.mp4` | a file you can show off |

---

## 🌟 Stage 1 — Physical model

The Sun, Earth, and Moon attract each other through Newtonian gravity.
For three point masses with positions $\vec{r}_i$ and momenta $\vec{p}_i$,
the Hamiltonian is:

$$
H = \sum_{i=1}^{3} \frac{|\vec{p}_i|^2}{2 m_i}
     - G \sum_{i<j} \frac{m_i m_j}{|\vec{r}_i - \vec{r}_j|}
$$

The equations of motion are:

$$
\dot{\vec{r}}_i = \frac{\vec{p}_i}{m_i}, \qquad
\dot{\vec{p}}_i = -G \sum_{j \neq i} \frac{m_i m_j (\vec{r}_i - \vec{r}_j)}{|\vec{r}_i - \vec{r}_j|^3}
$$

Each body has 3 position components + 3 momentum components → **18-dim
state vector** for three bodies in 3D. *(Want to keep it simpler at first?
Restrict to a plane → 12-dim state. Recommended for your first go.)*

> 🟢 **Decision point.** Start in 2D. 12 components is plenty to debug;
> add the third spatial dimension later if you actually want orbital
> inclination of the Moon.

---

## 📐 Stage 2 — Units

The naive choice (SI: kg, m, s) gives you `G ≈ 6.67e-11` and orbital
distances of `1.5e11`. Multiply those together in a numerical integrator
and floating-point hates you. **Pick units where the numbers are O(1).**

A clean choice:

| Quantity | Unit | Numerical value |
|---|---|---|
| Length | 1 AU (Earth–Sun distance) | 1 |
| Mass | M☉ (solar mass) | 1 |
| Time | year / (2π) | 1 |

In these units **G = 1**. (This is the standard "astronomer's unit
system".) Earth's orbital period is then $2\pi$ — exactly what your
existing Kepler examples already use.

Earth and Moon masses become:

| Body | Mass (in M☉) |
|---|---|
| Sun | `1.0` |
| Earth | `3.0034e-6` |
| Moon | `3.694e-8` |

The Moon's distance from Earth is `~0.00257 AU`. **This matters** — see
Stage 5.

Put these in `python/constants.py`:

```python
import numpy as np

G = 1.0

M_SUN   = 1.0
M_EARTH = 3.0034e-6
M_MOON  = 3.694e-8

AU                  = 1.0
EARTH_MOON_DISTANCE = 2.57e-3  # AU

YEAR = 2 * np.pi
```

---

## 🧮 Stage 3 — Writing the N-body RHS in Python

State layout (recommended): pack positions first, then momenta, body by
body. For 3 bodies in 2D:

```
y = [x1, y1, x2, y2, x3, y3,  px1, py1, px2, py2, px3, py3]
```

```python
# python/nbody.py
import numpy as np

def make_nbody_rhs(masses, G=1.0, softening=0.0):
    """Return rhs(t, y) for an N-body system.

    State vector layout:
        y[0 : 2*N]       = positions  (x,y) per body, packed
        y[2*N : 4*N]     = momenta    (px,py) per body, packed
    """
    masses = np.asarray(masses, dtype=np.float64)
    N = masses.size
    eps2 = softening * softening

    def rhs(_t, y):
        q = y[:2*N].reshape(N, 2)         # positions
        p = y[2*N:].reshape(N, 2)         # momenta

        # dq/dt = p / m
        dq = p / masses[:, None]

        # dp/dt = -G sum_{j!=i} m_i m_j (q_i - q_j) / |q_i - q_j|^3
        diff = q[:, None, :] - q[None, :, :]              # (N, N, 2)
        r2   = np.einsum('ijk,ijk->ij', diff, diff) + eps2
        np.fill_diagonal(r2, 1.0)                          # avoid /0 on diag
        inv_r3 = r2 ** -1.5
        np.fill_diagonal(inv_r3, 0.0)
        accel  = -G * np.einsum('j,ij,ijk->ik', masses, inv_r3, diff)
        dp     = masses[:, None] * accel

        return np.concatenate([dq.ravel(), dp.ravel()])

    return rhs
```

> 💡 **Why the softening parameter?** When two bodies get close,
> $|\vec{r}_i - \vec{r}_j|^{-3}$ explodes. For Earth–Sun–Moon this
> doesn't happen, but it's a one-character safety net for future
> simulations. Leave it 0 for this guide.

---

## 🚀 Stage 4 — Initial conditions

The trick: place each body on a **circular** orbit *around the
centre of mass of everything inside it*.

1. **Sun at the origin.** Momentum chosen at the very end so the system's
   total momentum is zero (otherwise the whole thing drifts off-screen).
2. **Earth at 1 AU.** Velocity perpendicular to the Sun–Earth line, with
   magnitude $\sqrt{G M_\odot / r}$.
3. **Moon at Earth + 0.00257 AU.** Velocity = Earth's velocity + the
   Moon's orbital velocity around Earth.
4. **Zero total momentum.** Set Sun's momentum to cancel Earth's + Moon's.

```python
# python/initial_conditions.py
import numpy as np
from constants import G, M_SUN, M_EARTH, M_MOON, AU, EARTH_MOON_DISTANCE

def earth_sun_moon_initial():
    # Positions
    x_sun   = np.array([0.0, 0.0])
    x_earth = np.array([AU,  0.0])
    x_moon  = x_earth + np.array([EARTH_MOON_DISTANCE, 0.0])

    # Earth: circular orbit around Sun
    v_earth = np.array([0.0, np.sqrt(G * M_SUN / AU)])

    # Moon: Earth's velocity + circular orbit around Earth
    v_moon_rel_earth = np.array([0.0, np.sqrt(G * M_EARTH / EARTH_MOON_DISTANCE)])
    v_moon = v_earth + v_moon_rel_earth

    # Sun: momentum cancels (so the centre of mass stays put)
    p_earth = M_EARTH * v_earth
    p_moon  = M_MOON  * v_moon
    p_sun   = -(p_earth + p_moon)
    v_sun   = p_sun / M_SUN

    masses = np.array([M_SUN, M_EARTH, M_MOON])

    q = np.stack([x_sun, x_earth, x_moon])     # (3, 2)
    v = np.stack([v_sun, v_earth, v_moon])     # (3, 2)
    p = masses[:, None] * v                    # (3, 2)

    y0 = np.concatenate([q.ravel(), p.ravel()])
    return y0, masses
```

---

## 🎯 Stage 5 — Integrate, then sanity-check

> ⚠️ **The timescale problem.** Earth's orbit takes $2\pi$ in our units;
> the Moon's orbit takes about $2\pi / \sqrt{(M_\oplus / M_\odot) / r_\text{em}^3} \approx 0.485$.
> So the Moon orbits the Earth about **13× per Earth-year**, and your
> integrator step needs to be small enough to resolve a *Moon orbit*, not
> just an Earth orbit.

A reasonable starting point:

- Total time: 1 year → `tf = 2*pi`.
- Steps: `~5000` — gives `dt ≈ 1.26e-3`, which resolves a Moon orbit
  in about 380 steps. Plenty.

```python
# python/three_body.py
import numpy as np
import hamsolver
from nbody import make_nbody_rhs
from initial_conditions import earth_sun_moon_initial
from constants import G

y0, masses = earth_sun_moon_initial()
rhs = make_nbody_rhs(masses, G=G)

t0, tf, steps = 0.0, 2 * np.pi, 5000
dt = (tf - t0) / steps

# Use a symplectic integrator — drift in energy on long runs is the killer here.
result = hamsolver.Gauss_Legendre_method(rhs, y0, t0, dt, steps, max_iter=20)

# Energy check
def total_energy(y, masses, G=1.0):
    N = masses.size
    q = y[:2*N].reshape(N, 2)
    p = y[2*N:].reshape(N, 2)
    KE = 0.5 * np.sum(np.sum(p*p, axis=1) / masses)
    PE = 0.0
    for i in range(N):
        for j in range(i+1, N):
            PE -= G * masses[i] * masses[j] / np.linalg.norm(q[i] - q[j])
    return KE + PE

E0 = total_energy(result[0], masses)
Ef = total_energy(result[-1], masses)
print(f"|ΔE/E0| over one year: {abs(Ef-E0)/abs(E0):.2e}")
```

Expect something like `1e-9` from Gauss–Legendre and `1e-5` from RK4. If
you get `1e-2`, your step is too big or your initial conditions are off.

> 🟢 **Why symplectic?** Three-body simulations run for many orbits.
> A non-symplectic method drifts in energy linearly with time, so the
> Moon will eventually spiral away. Symplectic methods bound the drift.

---

## 🎬 Stage 6 — Animation in matplotlib

The trick is `matplotlib.animation.FuncAnimation`: you give it an
initialization function and an update function. Every frame, the update
function tweaks the artists (line objects, scatter dots) to reflect the
current state.

```python
# python/animate.py
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from constants import M_SUN, M_EARTH, M_MOON

# `result` is your (steps+1, 12) trajectory from Stage 5.

N = 3
positions = result[:, :2*N].reshape(-1, N, 2)   # (frames, 3 bodies, xy)

# Sub-sample for animation (5000 frames is overkill for the eye)
stride = 10
positions = positions[::stride]

fig, ax = plt.subplots(figsize=(8, 8))
ax.set_xlim(-1.3, 1.3)
ax.set_ylim(-1.3, 1.3)
ax.set_aspect('equal')
ax.set_facecolor('black')
ax.grid(alpha=0.2, color='gray')

# Trails
sun_trail,   = ax.plot([], [], '-', color='gold',    lw=0.5, alpha=0.6)
earth_trail, = ax.plot([], [], '-', color='deepskyblue', lw=0.8, alpha=0.7)
moon_trail,  = ax.plot([], [], '-', color='lightgray', lw=0.5, alpha=0.5)

# Body markers
sun_dot,   = ax.plot([], [], 'o', color='gold',         markersize=18, zorder=3)
earth_dot, = ax.plot([], [], 'o', color='deepskyblue',  markersize=8,  zorder=3)
moon_dot,  = ax.plot([], [], 'o', color='lightgray',    markersize=4,  zorder=3)

trail_len = 200  # frames

def init():
    return sun_trail, earth_trail, moon_trail, sun_dot, earth_dot, moon_dot

def update(frame):
    lo = max(0, frame - trail_len)
    sun_trail.set_data(positions[lo:frame+1, 0, 0], positions[lo:frame+1, 0, 1])
    earth_trail.set_data(positions[lo:frame+1, 1, 0], positions[lo:frame+1, 1, 1])
    moon_trail.set_data(positions[lo:frame+1, 2, 0], positions[lo:frame+1, 2, 1])

    sun_dot.set_data([positions[frame, 0, 0]], [positions[frame, 0, 1]])
    earth_dot.set_data([positions[frame, 1, 0]], [positions[frame, 1, 1]])
    moon_dot.set_data([positions[frame, 2, 0]], [positions[frame, 2, 1]])

    return sun_trail, earth_trail, moon_trail, sun_dot, earth_dot, moon_dot

anim = animation.FuncAnimation(
    fig, update,
    frames=len(positions),
    init_func=init,
    interval=20,        # ms between frames
    blit=True,
)

plt.show()
```

> 🎯 **`blit=True`** redraws only the changed artists each frame.
> Massive speed-up; turn it off if your background ever needs to update
> (e.g. zooming).

> 🪐 **The Moon problem.** At `1 AU` zoom level the Moon's orbit
> (`0.00257 AU`) is invisible. To see it, zoom in on Earth in a second
> subplot, or animate in an Earth-centred frame:
>
> ```python
> earth_pos = positions[:, 1, :]
> moon_relative = positions[:, 2, :] - earth_pos
> # Then animate moon_relative in a 0.005 × 0.005 AU window centred on Earth.
> ```

---

## 💾 Stage 7 — Export

```python
# Save as MP4 (needs ffmpeg)
anim.save("earth_sun_moon.mp4", fps=30, dpi=150)

# Save as GIF (needs ImageMagick or pillow)
anim.save("earth_sun_moon.gif", writer="pillow", fps=24)
```

On Arch:

```bash
sudo pacman -S ffmpeg imagemagick
```

For posting on social media: 1080p MP4 at 30 fps is the safe default.

---

## 🧪 Suggested tests for the new code

When you're happy with the simulation, lock it in with tests:

```python
# tests/python/test_nbody.py
def test_nbody_conserves_energy():
    y0, masses = earth_sun_moon_initial()
    rhs = make_nbody_rhs(masses)
    out = hamsolver.Gauss_Legendre_method(rhs, y0, 0.0, 1e-3, 1000, 20)
    E0 = total_energy(out[0], masses)
    Ef = total_energy(out[-1], masses)
    assert abs(Ef - E0) / abs(E0) < 1e-6

def test_nbody_conserves_total_momentum():
    y0, masses = earth_sun_moon_initial()
    rhs = make_nbody_rhs(masses)
    out = hamsolver.Gauss_Legendre_method(rhs, y0, 0.0, 1e-3, 1000, 20)
    p0 = out[0,  6:].reshape(3, 2).sum(axis=0)
    pf = out[-1, 6:].reshape(3, 2).sum(axis=0)
    np.testing.assert_allclose(pf, p0, atol=1e-12)
```

---

## 🚧 Possible upgrades (after the basic simulation works)

- 🌌 **3D**: bump 2→3 in the state-vector layout.
- 🪐 **More bodies**: feed `make_nbody_rhs` a longer mass vector.
  Inner-solar-system simulation is `make_nbody_rhs([M_sun, M_mercury, ...])`.
- ⏱ **Adaptive step size**: out of scope for the current `hamsolver` (see
  cleanup notes), but a future feature would help here.
- 🎨 **Lighting / glow**: add a multi-layer scatter for each body
  (small bright core + larger faint halo).
- 🌠 **Camera follow**: have the view smoothly recentre on Earth.

---

## 📚 References to read alongside

- Hairer, Lubich, Wanner — *Geometric Numerical Integration*. The bible
  for symplectic methods, written for people doing exactly this.
- The matplotlib `animation` tutorial:
  <https://matplotlib.org/stable/api/animation_api.html>.
- Rein & Spiegel — REBOUND. A serious open-source N-body code; great for
  comparing your results against.

---

🌍🌑☀️ *Have fun. Three-body chaos is an aesthetic.*
