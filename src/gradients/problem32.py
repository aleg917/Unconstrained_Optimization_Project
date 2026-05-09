"""Gradiente esatto del Problem 32."""
import numpy as np


def grad_f32(x):
    """Gradiente esatto del Generalized Broyden Tridiagonal.

    dF/dx_j = f_j*(3 - 4 x_j) - f_{j-1} - f_{j+1},   con f_0 = f_{n+1} = 0.
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    x_ext = np.concatenate(([0.0], x, [0.0]))
    fk = (3.0 - 2.0 * x) * x + 1.0 - x_ext[0:n] - x_ext[2:n + 2]
    f_ext = np.concatenate(([0.0], fk, [0.0]))
    return fk * (3.0 - 4.0 * x) - f_ext[0:n] - f_ext[2:n + 2]
