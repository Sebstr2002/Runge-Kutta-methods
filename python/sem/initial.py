import numpy as np
from constants import (
    G,
    M_SUN, M_EARTH, M_MOON,
    AU, EARTH_MOON_DISTANCE
)


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

