"""Problem 28 — Variably Dimensioned (Luksan-Vlcek V897-03; More-Garbow-Hillstrom).

F(x) = 1/2 sum_{k=1..n+2} f_k(x)^2,
f_k(x) = x_k - 1,                          1 <= k <= n
f_{n+1}(x) = sum_{i=1..n} i (x_i - 1)
f_{n+2}(x) = ( sum_{i=1..n} i (x_i - 1) )^2

Introducing the shorthand
    S(x) := sum_{i=1..n} i (x_i - 1),
we get
    F(x) = 1/2 [ sum_k (x_k - 1)^2 + S(x)^2 + S(x)^4 ].

Global minimum: x* = (1, ..., 1) with F* = 0 (both the residuals x_k - 1
and the weighted sum S vanish at x*).
"""
import numpy as np


def f28(x):
    """Evaluate F(x) (vectorized, O(n))."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    j = np.arange(1, n + 1)
    d = x - 1.0
    S = float(np.dot(j, d))
    return 0.5 * (float(np.dot(d, d)) + S * S + S ** 4)


def x_bar_28(n):
    """Suggested starting point: x_bar_l = 1 - l/n, l = 1, ..., n."""
    l = np.arange(1, n + 1)
    return 1.0 - l / n
