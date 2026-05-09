"""Gradiente esatto del Problem 16."""
import numpy as np


def grad_f16(x):
    """Gradiente esatto del Banded Trigonometric.

    Interno (e bordo j=1): dF/dx_j = j*sin(x_j) + 2*cos(x_j)
    Bordo j=n:             dF/dx_n = n*sin(x_n) - (n-1)*cos(x_n)
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    j = np.arange(1, n + 1)
    g = j * np.sin(x) + 2.0 * np.cos(x)
    g[-1] = n * np.sin(x[-1]) - (n - 1) * np.cos(x[-1])
    return g
