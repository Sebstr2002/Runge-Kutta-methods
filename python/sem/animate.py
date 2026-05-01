import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import hamsolver
from initial import earth_sun_moon_initial


# 1. Run the C++ Simulation
print("Running C++ integration...")
y0, masses = earth_sun_moon_initial()
t0, tf, steps = 0.0, 2 * np.pi, 5000
dt = (tf - t0) / steps

result_cpp = hamsolver.runge_kutta(
    table=hamsolver.Gauss_Legendre,
    f=hamsolver.sun_earth_moon_rhs,
    yn=y0, t0=t0, dt=dt, steps=steps, max_iter=20
)
result = np.array(result_cpp)
print("Integration complete. Preparing animation...")

N = 3
positions = result[:, :2*N].reshape(-1, N, 2)

stride = 10
positions = positions[::stride]

# figure
fig, (ax_main, ax_zoom) = plt.subplots(1, 2, figsize=(14, 7))
fig.canvas.manager.set_window_title('N-Body problem')
fig.patch.set_facecolor('black')

ax_main.set_xlim(-1.2, 1.2)
ax_main.set_ylim(-1.2, 1.2)
ax_main.set_aspect('equal')
ax_main.set_facecolor('black')
ax_main.set_title("Global View (1.2 AU)", color='white')
ax_main.grid(alpha=0.2, color='gray')

# Zoom style (Earth in centre)
zoom_window = 0.005 # AU
ax_zoom.set_xlim(-zoom_window, zoom_window)
ax_zoom.set_ylim(-zoom_window, zoom_window)
ax_zoom.set_aspect('equal')
ax_zoom.set_facecolor('black')
ax_zoom.set_title("Earth-Centric View (0.005 AU)", color='white')
ax_zoom.grid(alpha=0.2, color='gray')

# Main view objects
main_trails = [
    ax_main.plot([], [], '-', color='gold', lw=0.5, alpha=0.6)[0],        # Sun
    ax_main.plot([], [], '-', color='deepskyblue', lw=0.8, alpha=0.7)[0], # Earth
    ax_main.plot([], [], '-', color='lightgray', lw=0.5, alpha=0.5)[0]    # Moon
]
main_dots = [
    ax_main.plot([], [], 'o', color='gold', markersize=16, zorder=3)[0],
    ax_main.plot([], [], 'o', color='deepskyblue', markersize=6, zorder=3)[0],
    ax_main.plot([], [], 'o', color='lightgray', markersize=2, zorder=3)[0]
]

# Zoom view objects (Only Earth and Moon)
zoom_trails = [
    ax_zoom.plot([], [], '-', color='deepskyblue', lw=1.0, alpha=0.7)[0], # Earth
    ax_zoom.plot([], [], '-', color='lightgray', lw=0.8, alpha=0.5)[0]    # Moon
]
zoom_dots = [
    ax_zoom.plot([], [], 'o', color='deepskyblue', markersize=12, zorder=3)[0],
    ax_zoom.plot([], [], 'o', color='lightgray', markersize=5, zorder=3)[0]
]

trail_len = 150 # frames

def init():
    return main_trails + main_dots + zoom_trails + zoom_dots

def update(frame):
    lo = max(0, frame - trail_len)
    
    # --- Update Main View ---
    for i in range(3):
        main_trails[i].set_data(positions[lo:frame+1, i, 0], positions[lo:frame+1, i, 1])
        main_dots[i].set_data([positions[frame, i, 0]], [positions[frame, i, 1]])

    # --- Update Zoom View ---
    # Center everything on Earth's current position
    earth_pos = positions[frame, 1, :]
    
    # Earth stays at (0,0) in the zoom window
    zoom_dots[0].set_data([0], [0]) 
    zoom_trails[0].set_data([0], [0]) 
    
    # Calculate Moon relative to Earth for the trail and dot
    moon_rel_trail_x = positions[lo:frame+1, 2, 0] - positions[lo:frame+1, 1, 0]
    moon_rel_trail_y = positions[lo:frame+1, 2, 1] - positions[lo:frame+1, 1, 1]
    zoom_trails[1].set_data(moon_rel_trail_x, moon_rel_trail_y)
    zoom_dots[1].set_data([positions[frame, 2, 0] - earth_pos[0]], 
                          [positions[frame, 2, 1] - earth_pos[1]])

    return main_trails + main_dots + zoom_trails + zoom_dots

anim = animation.FuncAnimation(
    fig, update,
    frames=len(positions),
    init_func=init,
    interval=20,
    blit=True,
)

plt.tight_layout()
plt.show()
