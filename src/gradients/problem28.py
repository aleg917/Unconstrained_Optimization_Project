"""Exact gradient of problem 28"""
import numpy as np


def grad_f28(x):
    """
    Exact gradient of the variably dimensioned function.

    dF/dx_j = (x_j - 1) + j S + 2 j S^3 = (x_j - 1) + j S (1 + 2 S^2).

    The computation requires a single pass to evaluate S in O(n), then
    another O(n) pass to form the gradient vector.
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    j = np.arange(1, n + 1)
    d = x - 1.0
    S = float(np.dot(j, d))
    return d + j * (S * (1.0 + 2.0 * S * S))
