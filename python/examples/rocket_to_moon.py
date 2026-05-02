"""Hohmann-like transfer from Earth to the Moon, with impact detected via the
adaptive solver's event detection.

Units are normalised so that the Earth-Moon distance, Earth's mass, and the
Moon's orbital angular frequency are all 1. One time unit is therefore
≈ 4.34 days (= sidereal month / 2π).

Setup:
    - Earth fixed at the origin, mass 1.
    - Moon on a prescribed circular orbit at radius 1, mass μ_M ≈ 0.0123 M⊕.
    - Rocket: a point particle launched tangentially from Earth's surface
      with the Hohmann-transfer periapsis velocity, lead-angled so the Moon
      is at the right place when the rocket arrives.

The event function returns ``|rocket - moon| - R_moon``; the moment it
crosses zero the C++ adaptive solver bisects on the cubic-Hermite spline,
records the exact impact, and (with ``stop_on_event=True``) terminates.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

import hamsolver

# ---------------------------------------------------------------------------
# Constants (G = M_Earth = D_Earth-Moon = 1, ω_Moon = 1)
# ---------------------------------------------------------------------------
G       = 1.0
M_E     = 1.0
MU_M    = 0.0123              # Moon mass / Earth mass
OMEGA_M = 1.0                 # circular orbit ω = sqrt(G·M_E/D³) = 1

# Real radii rescaled by the Earth-Moon distance (~384 400 km).
R_E = 6371.0 / 384400.0
R_M = 1737.0 / 384400.0

ONE_TIME_UNIT_DAYS  = 27.32 / (2 * np.pi)                       # ≈ 4.348 days
ONE_SPEED_UNIT_KMS  = 384_400.0 / (ONE_TIME_UNIT_DAYS * 86400)  # ≈ 1.024 km/s


def moon_position(t: float, theta0: float) -> np.ndarray:
    """Moon's analytical circular orbit around the Earth."""
    angle = OMEGA_M * t + theta0
    return np.array([np.cos(angle), np.sin(angle)])


def make_rhs(theta0: float):
    """Closes over θ₀ so the Moon's phase is baked into the RHS."""
    def rhs(t, y):
        x, yy, vx, vy = y

        # Earth's pull
        r_E = np.hypot(x, yy)
        a_earth = -G * M_E * np.array([x, yy]) / r_E**3

        # Moon's pull (Moon position is a function of t, not a state variable)
        mx, my = moon_position(t, theta0)
        rel = np.array([x - mx, yy - my])
        r_M = np.hypot(rel[0], rel[1])
        a_moon = -G * MU_M * rel / r_M**3

        a = a_earth + a_moon
        return np.array([vx, vy, a[0], a[1]])
    return rhs


def make_event(theta0: float):
    """g(t, y) = distance(rocket, moon) - R_moon. Sign change ⇒ impact."""
    def hit_moon(t, y):
        mx, my = moon_position(t, theta0)
        return np.hypot(y[0] - mx, y[1] - my) - R_M
    return hit_moon


# ---------------------------------------------------------------------------
# Hohmann transfer from low-Earth periapsis (r = R_E) to the Moon's orbit
# ---------------------------------------------------------------------------
a_transfer    = 0.5 * (R_E + 1.0)                                   # semi-major axis
v_periapsis   = np.sqrt(G * M_E * (2.0 / R_E - 1.0 / a_transfer))   # vis-viva at R_E
transfer_time = np.pi * np.sqrt(a_transfer**3 / (G * M_E))          # half period

# Lead angle: Moon should be at angle π (the apoapsis side) when we arrive.
theta_M0 = np.pi - OMEGA_M * transfer_time

print(f"Hohmann periapsis velocity: v_p = {v_periapsis:.4f}  "
      f"(~ {v_periapsis * ONE_SPEED_UNIT_KMS:.2f} km/s)")
print(f"Transfer time:              T   = {transfer_time:.4f}  "
      f"(~ {transfer_time*ONE_TIME_UNIT_DAYS:.2f} days)")
print(f"Required Moon lead angle:   θ₀  = {theta_M0:.4f} rad "
      f"= {np.degrees(theta_M0):.1f}°")

# Launch from (R_E, 0) tangentially in +y.
y0 = np.array([R_E, 0.0, 0.0, v_periapsis])
t0 = 0.0
tf = transfer_time * 1.6   # generous upper bound — the event will stop us early

# ---------------------------------------------------------------------------
# Integrate
# ---------------------------------------------------------------------------
times, states, ev_times, ev_states = hamsolver.adaptive_runge_kutta(
    table        = hamsolver.DP54,
    f            = make_rhs(theta_M0),
    yn           = y0,
    t0           = t0,
    tf           = tf,
    initial_dt   = 1e-4,    # start tiny: gravitational acceleration is huge near R_E
    tolerance    = 1e-9,
    max_iter     = 10,
    dt_out       = 0.005,   # ≈ 200 frames per time unit
    event_fn     = make_event(theta_M0),
    stop_on_event= True,
)
states = np.asarray(states)
print(f"\nIntegration: {len(times)} frames, ended at t = {times[-1]:.4f}")

if ev_times:
    impact_t = ev_times[0]
    impact_y = np.asarray(ev_states[0])
    moon_at_impact = moon_position(impact_t, theta_M0)
    final_dist = np.hypot(impact_y[0] - moon_at_impact[0],
                          impact_y[1] - moon_at_impact[1])
    print(f"\n🚀  IMPACT at t = {impact_t:.4f} "
          f"({impact_t * ONE_TIME_UNIT_DAYS:.2f} days)")
    print(f"     |rocket − moon|  =  {final_dist:.6e}  "
          f"(R_moon = {R_M:.6e})")
    v_imp = np.hypot(impact_y[2], impact_y[3])
    print(f"     Approach speed   =  {v_imp:.4f}  "
          f"(~ {v_imp * ONE_SPEED_UNIT_KMS:.2f} km/s)")
else:
    print("\nNo impact detected — the rocket missed.")

# ---------------------------------------------------------------------------
# Animation
# ---------------------------------------------------------------------------
xs, ys = states[:, 0], states[:, 1]

# Bodies are tiny in true-to-scale units; scale them up just for visibility.
EARTH_SCALE = 8.0
MOON_SCALE  = 8.0

fig, ax = plt.subplots(figsize=(8, 8))
fig.canvas.manager.set_window_title("Rocket to the Moon")
fig.patch.set_facecolor("black")
ax.set_facecolor("black")
ax.set_xlim(-1.25, 1.25)
ax.set_ylim(-1.25, 1.25)
ax.set_aspect("equal")
ax.grid(True, alpha=0.2, color="gray")
ax.tick_params(colors="white")
for spine in ax.spines.values():
    spine.set_color("white")
ax.set_title("Rocket from Earth to Moon — DP54 adaptive + event detection",
             color="white")

# Reference Moon orbit
theta_ref = np.linspace(0, 2 * np.pi, 200)
ax.plot(np.cos(theta_ref), np.sin(theta_ref),
        "--", color="gray", alpha=0.4, lw=0.6)

# Earth (visually scaled)
earth_patch = plt.Circle((0, 0), R_E * EARTH_SCALE,
                         color="deepskyblue", zorder=5)
ax.add_patch(earth_patch)
ax.plot(0, 0, "x", color="white", markersize=4)  # Earth's actual centre

# Moon (will be repositioned per frame)
moon_patch = plt.Circle((0, 0), R_M * MOON_SCALE,
                        color="lightgray", zorder=5)
ax.add_patch(moon_patch)

# Rocket trail and marker
trail,  = ax.plot([], [], "-", color="red",    lw=1.2, alpha=0.8)
rocket, = ax.plot([], [], "o", color="yellow", markersize=5, zorder=6)
boom,   = ax.plot([], [], "*", color="orange", markersize=18, zorder=7,
                  alpha=0.0)
status_text = ax.text(0.02, 0.98, "", transform=ax.transAxes,
                      ha="left", va="top", color="white", fontsize=10,
                      family="monospace")

speedup  = max(1, len(times) // 600)
n_frames = len(times) // speedup


def init():
    trail.set_data([], [])
    rocket.set_data([], [])
    boom.set_data([], [])
    boom.set_alpha(0.0)
    status_text.set_text("")
    return [trail, rocket, moon_patch, boom, status_text]


def update(frame):
    idx = min(frame * speedup, len(times) - 1)
    trail.set_data(xs[:idx + 1], ys[:idx + 1])
    rocket.set_data([xs[idx]], [ys[idx]])

    mx, my = moon_position(times[idx], theta_M0)
    moon_patch.center = (mx, my)

    status_text.set_text(
        f" t = {times[idx]:5.3f}  "
        f"({times[idx]*ONE_TIME_UNIT_DAYS:5.2f} d)\n"
        f" r = {np.hypot(xs[idx], ys[idx]):.4f}"
    )

    # Show a flash at the moment of impact (the last frame, since we
    # told the solver to stop on event).
    if ev_times and idx == len(times) - 1:
        boom.set_data([xs[idx]], [ys[idx]])
        boom.set_alpha(1.0)
    return [trail, rocket, moon_patch, boom, status_text]


anim = animation.FuncAnimation(
    fig, update, frames=n_frames, init_func=init,
    interval=20, blit=True, repeat=False,
)
plt.show()
