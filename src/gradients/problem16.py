"""Exact gradient of Problem 16 (Banded Trigonometric)."""
import numpy as np


def grad_f16(x):
    """Exact gradient of the Banded Trigonometric function.

    Starting from
        F(x) = sum_{i=1..n} i * [ (1 - cos x_i) + sin x_{i-1} - sin x_{i+1} ],
        x_0 = x_{n+1} = 0,
    we rewrite the three sums in terms of x_j (j = 1, ..., n), shifting
    the index in the sin terms:

        sum_i i * sin x_{i-1}   -->  sum_j (j+1) * sin x_j,   j = 1..n-1
        sum_i (-i) * sin x_{i+1} --> sum_j -(j-1) * sin x_j,  j = 2..n

    (the two boundary terms that fall outside the grid, j = n+1 and j = 0,
    vanish because x_0 = x_{n+1} = 0).  Differentiating with respect to x_j
    yields three contributions: j * sin x_j from the diagonal term
    j * (1 - cos x_j), plus a coefficient on cos x_j that depends on the
    boundary.

    For j = 1:        cos coefficient = (1+1) = 2        (left shift only)
    For 2 <= j < n:   cos coefficient = (j+1) - (j-1) = 2
    For j = n:        cos coefficient = 0 - (n-1) = -(n-1)
                       (left shift absent because sin x_{n+1} = 0)

    Closed form:
        dF/dx_j =   j * sin x_j + 2 * cos x_j,         1 <= j <= n-1
        dF/dx_n =   n * sin x_n - (n-1) * cos x_n.
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    j = np.arange(1, n + 1)
    g = j * np.sin(x) + 2.0 * np.cos(x)
    g[-1] = n * np.sin(x[-1]) - (n - 1) * np.cos(x[-1])
    return g
