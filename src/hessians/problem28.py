"""Exact Hessian of problem 28"""
import numpy as np


def hess_f28(x):
    """
    Exact Hessian of the variably dimensioned function (dense n x n).

    H_{ij} = delta_{ij} + i * j * (1 + 6 S^2),   S = sum_i i (x_i - 1).
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    i = np.arange(1, n + 1, dtype=float)         # indices 1, 2, ..., n
    S = float(np.dot(i, x - 1.0))                # S = sum_i i (x_i - 1)
    c = 1.0 + 6.0 * S * S                        # common factor (1 + 6 S^2)
    H = c * (i[:, None] * i[None, :])            # entry (i, j) = i * j * (1 + 6 S^2)
    H += np.eye(n)                               # add delta_{ij} on the diagonal
    return H
