import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from functools import partial
import hamsolver

custom_g = 9.81
custom_L = 4.0
custom_c = 0.0

configured_rhs = partial(hamsolver.damped_pendulum_rhs, g=custom_g, L=custom_L, c=custom_c)

eps = 0.001
y0 = [np.pi+eps, 0.0]

t0 = 0.0
tf = 30.0  # Simulate 20 seconds of chaos
fps = 60
dt_out = 1.0 / fps  # Exactly 60 frames per second!

print(f"\nRunning Adaptive BS32 Solver to {tf}s...")
times, states, _, _ = hamsolver.adaptive_runge_kutta(
    table=hamsolver.BS32,
    f=configured_rhs,
    yn=y0,
    t0=t0,
    tf=tf,
    initial_dt=0.01,
    tolerance=1e-6,
    dt_out=dt_out # <-- This triggers your C++ Cubic Hermite Spline!
)

states = np.array(states)
print(f"Engine took adaptive steps but returned exactly {len(times)} frames!")

# --- 3. Convert Angles to Cartesian Coordinates (L1=1, L2=1) ---
theta1 = states[:, 0]

x1 = np.sin(theta1)
y1 = -np.cos(theta1)

# --- 4. Animate ---
fig, ax = plt.subplots(figsize=(6, 6))
fig.canvas.manager.set_window_title('Adaptive Pendulum')
ax.set_xlim(-2.5, 2.5)
ax.set_ylim(-2.5, 2.5)
ax.set_aspect('equal')
ax.grid(True, alpha=0.3)
ax.set_title("Pendulum (Adaptive BS32)")

line, = ax.plot([], [], 'o-', lw=2, color='blue', markersize=8)
trail, = ax.plot([], [], '-', lw=1, color='red', alpha=0.5)

trail_len = 100

def init():
    line.set_data([], [])
    trail.set_data([], [])
    return line, trail

def update(frame):
    # The pendulum rods: Origin -> Mass 1 -> Mass 2
    line.set_data([0, x1[frame]], [0, y1[frame]])
    
    # The trail follows the second mass
    start = max(0, frame - trail_len)
    trail.set_data(x1[start:frame], y1[start:frame])
    return line, trail

anim = animation.FuncAnimation(fig, update, frames=len(times), init_func=init, 
                               interval=1000/fps, blit=True)
plt.show()
