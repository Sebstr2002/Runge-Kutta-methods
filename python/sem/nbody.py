import numpy as np

def make_nbody_rhs(masses, G=1.0, softening=0.0):
    masses = np.array(masses, dtype=np.float64)
    N = masses.size
    eps2 = softening * softening

    def rhs(_t, y):
        y = np.asarray(y)

        q = y[:2*N].reshape(N,2) #pos
        p = y[2*N:].reshape(N,2) #momenta

        dq = p / masses[:, None]

        # standart Newton gravity
        diff = q[:, None, :] - q[None, :, :]
        r2 = np.einsum('ijk,ijk->ij', diff, diff) + eps2
        np.fill_diagonal(r2, 1.0)
        inv_r3 = r2 ** -1.5
        np.fill_diagonal(inv_r3, 0.0)
        accel = - G * np.einsum('j, ij, ijk->ik', masses, inv_r3, diff)
        dp = masses[:, None] * accel

        return np.concatenate([dp.ravel(), dp.ravel()])

    return rhs
