"""Hessiana esatta del Problem 16 — Banded Trigonometric.

Derivazione:
    F(x) = sum_{i=1..n} i * [(1 - cos x_i) + sin x_{i-1} - sin x_{i+1}]
con x_0 = x_{n+1} = 0.

Il gradiente e':
    dF/dx_j = j sin(x_j) + 2 cos(x_j),       1 <= j < n
    dF/dx_n = n sin(x_n) - (n-1) cos(x_n)

Poiche' dF/dx_j dipende solo da x_j, le derivate miste sono nulle e
l'Hessiana e' diagonale:
    d2F/dx_j^2 = j cos(x_j) - 2 sin(x_j),     1 <= j < n
    d2F/dx_n^2 = n cos(x_n) + (n-1) sin(x_n)
    d2F/dx_j dx_k = 0,                        j != k

Nota: il path sparso/1D (storage O(n)) e' un'estensione futura per n grande.
"""
import numpy as np


def hess_f16(x):
    """Hessiana esatta del Banded Trigonometric (matrice n x n densa, diagonale)."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    j = np.arange(1, n + 1)
    diag_vals = j * np.cos(x) - 2.0 * np.sin(x)
    diag_vals[-1] = n * np.cos(x[-1]) + (n - 1) * np.sin(x[-1])
    return np.diag(diag_vals)
