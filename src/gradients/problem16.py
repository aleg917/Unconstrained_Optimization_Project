"""Exact gradient of problem 16"""
import numpy as np


def grad_f16(x):
    """
    Exact gradient of the banded trigonometric problem.

    Closed form:
        dF/dx_j = j * sin x_j + 2 * cos x_j, for 1 <= j <= n-1
        dF/dx_n = n * sin x_n - (n-1) * cos x_n.
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    j = np.arange(1, n + 1)
    g = j * np.sin(x) + 2.0 * np.cos(x)
    g[-1] = n * np.sin(x[-1]) - (n - 1) * np.cos(x[-1])
    return g
