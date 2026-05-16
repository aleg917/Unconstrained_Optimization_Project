"""Hessiana esatta del Problem 28 — Variably Dimensioned.

Con S(x) = sum_{i=1..n} i (x_i - 1):
    F(x) = 1/2 [ sum_k (x_k - 1)^2 + S^2 + S^4 ]

Gradiente:
    dF/dx_j = (x_j - 1) + j * S * (1 + 2 S^2)

Hessiana (rank-1 update di I):
    d2F/dx_i dx_j = delta_{ij} + i j (1 + 6 S^2)
    H = I + (1 + 6 S^2) * jvec * jvec^T,  jvec = (1, ..., n)^T

WARNING: O(n^2) memoria. Non praticabile per n >= 1e5 in baseline.
La forma fattorizzata (1 + 6 S^2, jvec) con Sherman-Morrison e' estensione futura.
"""
import numpy as np


def hess_f28(x):
    """Hessiana esatta del Variably Dimensioned function (matrice n x n densa)."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    j = np.arange(1, n + 1, dtype=float)
    S = float(np.dot(j, x - 1.0))
    c = 1.0 + 6.0 * S * S
    return np.eye(n) + c * np.outer(j, j)
