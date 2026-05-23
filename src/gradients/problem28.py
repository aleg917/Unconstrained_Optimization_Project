"""Gradiente esatto del Problem 28 (Variably Dimensioned)."""
import numpy as np


def grad_f28(x):
    """Gradiente esatto del Variably Dimensioned function.

    Partendo da
        F(x) = 1/2 [ sum_k (x_k - 1)^2 + S(x)^2 + S(x)^4 ],
        S(x) = sum_{i=1..n} i (x_i - 1),
    deriviamo rispetto a x_j (1 <= j <= n) un addendo per volta:
        d/dx_j [ 1/2 sum_k (x_k - 1)^2 ] = (x_j - 1)
        d/dx_j [ 1/2 S^2 ]               = S * (dS/dx_j) = j * S
        d/dx_j [ 1/2 S^4 ]               = 2 S^3 * (dS/dx_j) = 2 j S^3
    sommando:
        dF/dx_j = (x_j - 1) + j S + 2 j S^3
                = (x_j - 1) + j S (1 + 2 S^2).

    Il calcolo richiede una sola scansione del vettore: si valuta S in
    O(n) e poi si forma il gradiente con un'altra passata O(n).
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    j = np.arange(1, n + 1)
    d = x - 1.0
    S = float(np.dot(j, d))
    return d + j * (S * (1.0 + 2.0 * S * S))
