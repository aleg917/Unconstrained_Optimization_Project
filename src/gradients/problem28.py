"""Exact gradient of Problem 28 (Variably Dimensioned)."""
import numpy as np


def grad_f28(x):
    """Exact gradient of the Variably Dimensioned function.

    Starting from
        F(x) = 1/2 [ sum_k (x_k - 1)^2 + S(x)^2 + S(x)^4 ],
        S(x) = sum_{i=1..n} i (x_i - 1),
    we differentiate with respect to x_j (1 <= j <= n) term by term:
        d/dx_j [ 1/2 sum_k (x_k - 1)^2 ] = (x_j - 1)
        d/dx_j [ 1/2 S^2 ]               = S * (dS/dx_j) = j * S
        d/dx_j [ 1/2 S^4 ]               = 2 S^3 * (dS/dx_j) = 2 j S^3
    summing up:
        dF/dx_j = (x_j - 1) + j S + 2 j S^3
                = (x_j - 1) + j S (1 + 2 S^2).

    The computation requires a single pass to evaluate S in O(n), then
    another O(n) pass to form the gradient vector.
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    j = np.arange(1, n + 1)
    d = x - 1.0
    S = float(np.dot(j, d))
    return d + j * (S * (1.0 + 2.0 * S * S))
