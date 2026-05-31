"""Exact Hessian of problem 28"""
import numpy as np


def hess_f28(x):
    """
    Exact Hessian of the variably dimensioned function (dense n x n).

    H_{ij} = delta_{ij} + i * j * (1 + 6 S^2),   S = sum_i i (x_i - 1),
    i.e. the rank-1 form H = I + (1 + 6 S^2) * j j^T.

    WARNING: O(n^2) memory; for large n use the matrix-free Hv path instead.
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    j = np.arange(1, n + 1, dtype=float)
    S = float(np.dot(j, x - 1.0))            # weighted sum S, one O(n) pass
    c = 1.0 + 6.0 * S * S                    # common scalar factor of the rank-1 term
    return np.eye(n) + c * np.outer(j, j)    # identity plus rank-1 outer product
