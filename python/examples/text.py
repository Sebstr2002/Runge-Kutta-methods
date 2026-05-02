import numpy as np
import hamsolver

# Initial State: Dropped from 1.5 radians (almost 90 degrees)
y0 = [1.5, 0.0] 

# The Event Function: "Return 0 when the angle (y[0]) crosses 0"
# Pybind11 automatically passes this Python lambda into the C++ Bisection algorithm!
def crosses_zero(t, y):
    return y[0] 

times, states, ev_times, ev_states = hamsolver.adaptive_runge_kutta(
    table=hamsolver.BS32,
    f=hamsolver.damped_pendulum_rhs,
    yn=y0,
    t0=0.0,
    tf=10.0,            # Max time is 10s
    initial_dt=0.01,
    tolerance=1e-6,
    dt_out=0.05,
    event_fn=crosses_zero,  # Pass the event tracker!
    stop_on_event=True      # Kill the simulation on impact!
)

print(f"Simulation requested to run for 10.0s.")
print(f"Simulation actually ended at t = {times[-1]:.6f}s")

if len(ev_times) > 0:
    print(f"\nEVENT DETECTED!")
    print(f"Impact Time: {ev_times[0]:.6f}s")
    print(f"Impact Angle (Should be 0): {ev_states[0][0]:.12f} radians")
