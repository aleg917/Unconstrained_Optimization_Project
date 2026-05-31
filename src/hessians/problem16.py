"""Exact Hessian of problem 16"""
import numpy as np
import scipy.sparse as sp


def hess_f16(x):
    """
    Exact Hessian of the banded trigonometric problem (diagonal, sparse CSC).

    Each gradient component g_j depends only on x_j, so the Hessian is diagonal:
        d2F/dx_j^2 = j * cos x_j - 2 * sin x_j, for 1 <= j <= n-1
        d2F/dx_n^2 = n * cos x_n + (n-1) * sin x_n.
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    j = np.arange(1, n + 1)
    diag_vals = j * np.cos(x) - 2.0 * np.sin(x)                   # second derivative, j = 1..n-1
    diag_vals[-1] = n * np.cos(x[-1]) + (n - 1) * np.sin(x[-1])   # last component differs
    return sp.diags(diag_vals, 0, format='csc')                  # store only the diagonal -> O(n) memory
