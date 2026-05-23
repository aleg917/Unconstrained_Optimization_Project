"""Hessiana esatta del Problem 16 — Banded Trigonometric.

Punto di partenza (gia' ricavato nel gradiente):
    dF/dx_j = j * sin x_j + 2 * cos x_j,        1 <= j <= n-1
    dF/dx_n = n * sin x_n - (n-1) * cos x_n.

Osservazione chiave per la struttura della matrice: la componente
j-esima del gradiente dipende SOLO da x_j (nessun'altra variabile
compare nella formula chiusa). Di conseguenza tutte le derivate
miste sono nulle e l'Hessiana e' diagonale.

Derivando dF/dx_j una seconda volta rispetto a x_j si ottiene:
    d2F/dx_j^2 = j * cos x_j - 2 * sin x_j,     1 <= j <= n-1
    d2F/dx_n^2 = n * cos x_n + (n-1) * sin x_n
    d2F/dx_i dx_j = 0,                          i != j

Nota implementativa: il path sparso/1D (storage O(n)) e' un'estensione
futura per n grande; qui si materializza la matrice diagonale n x n.
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
