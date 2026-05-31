"""Problem 28: Variably Dimensioned Function"""
import numpy as np


def f28(x):
    x = np.asarray(x, dtype=float)
    n = len(x)
    j = np.arange(1, n + 1)
    d = x - 1.0
    S = float(np.dot(j, d))
    return 0.5 * (float(np.dot(d, d)) + S * S + S ** 4)


def x_bar_28(n):
    l = np.arange(1, n + 1)
    return 1.0 - l / n
