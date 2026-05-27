"""Armijo backtracking line search.

Given a descent direction d and a current point x, finds a step size alpha
such that the sufficient decrease (Armijo) condition is satisfied:

    f(x + alpha * d) <= f(x) + c1 * alpha * <grad f(x), d>

The step is initialized at alpha0 (typically 1 for Newton-type methods)
and halved (multiplied by rho) until the condition holds or max_iter
backtracking steps are exhausted.
"""
import time

import numpy as np


def armijo_backtracking(f, x, fx, g, d, alpha0=1.0, c1=1e-4, rho=0.5,
                        max_iter=50, t_start=None, time_limit=None):
    """Backtracking line search with Armijo sufficient-decrease condition.

    Finds alpha such that  f(x + alpha*d) <= fx + c1*alpha*<g, d>.

    Parameters
    ----------
    f         : callable, F(x) -> scalar
    x         : current point, ndarray (n,)
    fx        : cached value F(x)
    g         : gradient at x, ndarray (n,)
    d         : descent direction, ndarray (n,)
    alpha0    : initial step size (default 1, the "full Newton step")
    c1        : Armijo constant, in (0, 1).  Standard references like
                Nocedal-Wright recommend (0, 1/2); the course slides
                state (0, 1).
    rho       : reduction factor per backtrack, in (0, 1)
    max_iter  : maximum number of backtracking reductions
    t_start   : reference time from time.perf_counter() (for time limit)
    time_limit: wall-clock budget in seconds (None = unlimited)

    Returns
    -------
    alpha     : accepted step size
    n_back    : number of reductions performed
    """
    g_dot_d = float(g @ d)
    alpha = alpha0
    for j in range(max_iter):
        if time_limit is not None and (time.perf_counter() - t_start) > time_limit:
            return alpha, j
        if f(x + alpha * d) <= fx + c1 * alpha * g_dot_d:
            return alpha, j
        alpha *= rho
    return alpha, max_iter
