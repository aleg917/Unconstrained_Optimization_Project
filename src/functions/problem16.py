"""Problem 16: Banded trigonometric problem"""
import numpy as np


def f16(x):
    x = np.asarray(x, dtype=float)
    n = len(x)
    x_ext = np.concatenate(([0.0], x, [0.0]))
    i = np.arange(1, n + 1)
    terms = i * ((1.0 - np.cos(x_ext[1:n + 1]))
                 + np.sin(x_ext[0:n])
                 - np.sin(x_ext[2:n + 2]))
    return float(terms.sum())


def x_bar_16(n):
    return np.ones(n)
