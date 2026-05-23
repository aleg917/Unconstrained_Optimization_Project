"""Problem 28 — Variably Dimensioned (Luksan-Vlcek V897-03; Moré-Garbow-Hillström).

F(x) = 1/2 sum_{k=1..n+2} f_k(x)^2,
f_k(x) = x_k - 1,                          1 <= k <= n
f_{n+1}(x) = sum_{i=1..n} i (x_i - 1)
f_{n+2}(x) = ( sum_{i=1..n} i (x_i - 1) )^2

Per non riscrivere tre volte la stessa somma pesata, introduciamo
l'abbreviazione di scrittura
    S(x) := sum_{i=1..n} i (x_i - 1),
così che
    F(x) = 1/2 [ sum_k (x_k - 1)^2 + S(x)^2 + S(x)^4 ].

Minimo globale: x* = (1, ..., 1) con F* = 0 (in x* si annullano sia
i residui x_k - 1 sia la combinazione S).
"""
import numpy as np


def f28(x):
    """Valuta F(x) (vettorizzato, O(n))."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    j = np.arange(1, n + 1)
    d = x - 1.0
    S = float(np.dot(j, d))
    return 0.5 * (float(np.dot(d, d)) + S * S + S ** 4)


def x_bar_28(n):
    """Punto iniziale suggerito: x_bar_l = 1 - l/n, l = 1, ..., n."""
    l = np.arange(1, n + 1)
    return 1.0 - l / n
