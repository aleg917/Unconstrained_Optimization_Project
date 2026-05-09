"""Problem 16 — Banded Trigonometric (Luksan-Vlcek V897-03).

F(x) = sum_{i=1..n} i * [(1 - cos x_i) + sin x_{i-1} - sin x_{i+1}]
con x_0 = x_{n+1} = 0.
"""
import numpy as np


def f16(x):
    """Valuta F(x) (vettorizzato, O(n))."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    x_ext = np.concatenate(([0.0], x, [0.0]))
    i = np.arange(1, n + 1)
    terms = i * ((1.0 - np.cos(x_ext[1:n + 1]))
                 + np.sin(x_ext[0:n])
                 - np.sin(x_ext[2:n + 2]))
    return float(terms.sum())


def x_bar_16(n):
    """Punto iniziale suggerito: x_bar_i = 1 per ogni i."""
    return np.ones(n)
