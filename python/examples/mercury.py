import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import hamsolver

# --- 1. Initial Conditions ---
# y = [x, y, px, py]
# We start it at an extreme eccentric orbit to make the precession obvious
y0 = [0.47, 0.0, 0.0, 1.2]

# We need a long simulation time to see multiple orbits complete!
t0 = 0.0
tf = 150.0  
fps = 60
dt_out = 0.005  # Dense output frame step

print(f"Calculating Relativistic Orbit to t={tf}s using Adaptive BS32...")

# --- 2. Run the Engine ---
# Notice we unpack 4 variables because of your Event Detection upgrade!
times, states, ev_times, ev_states = hamsolver.adaptive_runge_kutta(
    table=hamsolver.BS32,
    f=hamsolver.mercury_gr_rhs,
    yn=y0,
    t0=t0,
    tf=tf,
    initial_dt=0.01,
    tolerance=1e-7,
    dt_out=dt_out
)

states = np.array(states)
print(f"Simulation complete. Generated {len(times)} perfectly spaced frames.")

# Extract Cartesian coordinates
x = states[:, 0]
y = states[:, 1]

# --- 3. Animate the Spirograph ---
fig, ax = plt.subplots(figsize=(7, 7))
fig.canvas.manager.set_window_title('General Relativity: Mercury Precession')

# Set the view limits slightly larger than our apoapsis
ax.set_xlim(-1.5, 1.5)
ax.set_ylim(-1.5, 1.5)
ax.set_aspect('equal')
ax.grid(True, alpha=0.2)
ax.set_facecolor('black') # Space is black!
ax.set_title("Relativistic Perihelion Precession", color='white')
fig.patch.set_facecolor('black')

# Draw the Sun
sun = plt.Circle((0, 0), 0.05, color='yellow', zorder=5)
ax.add_patch(sun)

# We want the trail to stay permanently so we can see the flower pattern!
trail, = ax.plot([], [], '-', lw=1.2, color='cyan', alpha=0.6)
planet, = ax.plot([], [], 'o', color='white', markersize=6, zorder=6)

def init():
    trail.set_data([], [])
    planet.set_data([], [])
    return trail, planet

# To make it render faster, we can skip frames or draw multiple steps per frame.
# Let's draw 3 steps per frame so the animation flies around the sun!
speedup = 3

def update(frame):
    # Calculate the actual index based on speedup
    idx = frame * speedup
    if idx >= len(times):
        idx = len(times) - 1
        
    # Draw the entire trail up to the current frame
    trail.set_data(x[:idx], y[:idx])
    
    # Update the planet's current position
    planet.set_data([x[idx]], [y[idx]])
    
    return trail, planet

# Calculate total animation frames
total_frames = len(times) // speedup

anim = animation.FuncAnimation(fig, update, frames=total_frames, init_func=init, 
                               interval=1000/fps, blit=True)

plt.show()
