"""Gradiente esatto del Problem 28."""
import numpy as np


def grad_f28(x):
    """Gradiente esatto del Variably Dimensioned function.

    Con S(x) = sum_{i=1..n} i (x_i - 1):
        dF/dx_j = (x_j - 1) + j * S * (1 + 2 S^2),   j = 1, ..., n.

    Computabile in O(n).
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    j = np.arange(1, n + 1)
    d = x - 1.0
    S = float(np.dot(j, d))
    return d + j * (S * (1.0 + 2.0 * S * S))
