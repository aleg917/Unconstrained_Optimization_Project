"""Problem 32 — Generalized Broyden Tridiagonal (Luksan-Vlcek V897-03).

F(x) = 1/2 sum_{k=1..n} f_k(x)^2,
f_k(x) = (3 - 2 x_k) x_k + 1 - x_{k-1} - x_{k+1},  con x_0 = x_{n+1} = 0.
"""
import numpy as np


def f32(x):
    """Valuta F(x) (vettorizzato, O(n))."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    x_ext = np.concatenate(([0.0], x, [0.0]))
    fk = (3.0 - 2.0 * x) * x + 1.0 - x_ext[0:n] - x_ext[2:n + 2]
    return 0.5 * float(np.dot(fk, fk))


def x_bar_32(n):
    """Punto iniziale suggerito: x_bar_l = -1 per ogni l."""
    return -np.ones(n)
