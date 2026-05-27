"""Exact Hessian of Problem 28 — Variably Dimensioned.

Starting from the gradient (derived in src/gradients/problem28.py):
    dF/dx_j = (x_j - 1) + j * S * (1 + 2 S^2),
    S(x)   = sum_{i=1..n} i (x_i - 1).

To obtain element H_{ij} we differentiate the j-th gradient component
with respect to x_i, term by term:
    d/dx_i (x_j - 1)            = delta_{ij}
    d/dx_i [ j S (1 + 2 S^2) ]  = j * [ (dS/dx_i)(1 + 2 S^2)
                                         + S * 4 S * (dS/dx_i) ]
                                = j * i * (1 + 2 S^2 + 4 S^2)
                                = i * j * (1 + 6 S^2).
Summing:
    H_{ij} = delta_{ij} + i * j * (1 + 6 S^2).

WARNING: O(n^2) memory.  For large n (typically n >= 1e4) materializing
the full n x n matrix is infeasible; in those regimes use the matrix-free
Hv path of Truncated Newton, which computes Hessian-vector products via
finite differences of the exact gradient.
"""
import numpy as np


def hess_f28(x):
    """Exact Hessian of the Variably Dimensioned function (dense n x n matrix).

    The rank-1 structure H = I + c * j j^T (where j = (1,...,n) and
    c = 1 + 6 S^2) is built via an outer product.
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    j = np.arange(1, n + 1, dtype=float)
    S = float(np.dot(j, x - 1.0))
    c = 1.0 + 6.0 * S * S
    return np.eye(n) + c * np.outer(j, j)
