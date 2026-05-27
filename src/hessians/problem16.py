"""Exact Hessian of Problem 16 — Banded Trigonometric.

Starting from the gradient (derived in src/gradients/problem16.py):
    dF/dx_j = j * sin x_j + 2 * cos x_j,        1 <= j <= n-1
    dF/dx_n = n * sin x_n - (n-1) * cos x_n.

Key observation for the matrix structure: gradient component g_j depends
ONLY on x_j (no other variable appears in the closed-form expression).
Therefore all mixed partial derivatives are zero and the Hessian is diagonal.

Differentiating dF/dx_j a second time with respect to x_j:
    d2F/dx_j^2 = j * cos x_j - 2 * sin x_j,     1 <= j <= n-1
    d2F/dx_n^2 = n * cos x_n + (n-1) * sin x_n
    d2F/dx_i dx_j = 0,                           i != j

Implementation note: the Hessian is returned as a sparse CSC matrix
(O(n) storage) to enable Modified Newton even at large n.
"""
import numpy as np
import scipy.sparse as sp


def hess_f16(x):
    """Exact Hessian of the Banded Trigonometric (sparse CSC diagonal matrix)."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    j = np.arange(1, n + 1)
    diag_vals = j * np.cos(x) - 2.0 * np.sin(x)
    diag_vals[-1] = n * np.cos(x[-1]) + (n - 1) * np.sin(x[-1])
    return sp.diags(diag_vals, 0, format='csc')
