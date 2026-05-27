"""Gradient approximation via centered finite differences (generic).

Two step-size variants:
    - "fixed":  h_i = 10^{-k}
    - "scaled": h_i = 10^{-k} * |x_i|   (falls back to fixed when x_i = 0)

Typically k in {4, 8, 12} (as required by the assignment).

Uses centered differences for O(h^2) accuracy at the cost of 2n function
evaluations per gradient (vs n+1 for forward differences).
"""
import time

import numpy as np


class TimeLimitExceeded(Exception):
    """Raised when a wall-clock time limit is exceeded inside an FD loop."""
    pass


# Module-level time budget.  Optimization methods call set_time_budget()
# on entry so that ALL FD routines (even those created as lambdas outside
# the method) respect the wall-clock limit.  Uses a mutable list so that
# imports in other modules see mutations.
_time_budget = [None, None]   # [t_start, time_limit]


def set_time_budget(t_start, time_limit):
    _time_budget[0] = t_start
    _time_budget[1] = time_limit


def clear_time_budget():
    _time_budget[0] = None
    _time_budget[1] = None


def grad_fd(f, x, k=8, scaled=False, t_start=None, time_limit=None):
    """Centered finite-difference gradient approximation, O(h^2).

        df/dx_i ~ [f(x + h_i e_i) - f(x - h_i e_i)] / (2 h_i)

    Parameters
    ----------
    f          : callable, F(x) -> scalar
    x          : ndarray (n,)
    k          : step-size exponent (h = 10^{-k})
    scaled     : if True, h_i = 10^{-k} * |x_i| (falls back to 10^{-k} when x_i = 0)
    t_start    : float or None, reference time (time.perf_counter())
    time_limit : float or None, wall-clock budget in seconds

    Returns
    -------
    g_fd   : ndarray (n,), approximation of grad F(x)

    Raises
    ------
    TimeLimitExceeded if the wall-clock budget is exceeded.
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    g = np.zeros(n)
    h_base = 10.0 ** (-k)
    _ts = t_start if t_start is not None else _time_budget[0]
    _tl = time_limit if time_limit is not None else _time_budget[1]
    _check_time = (_ts is not None and _tl is not None)
    for i in range(n):
        if _check_time and i % 200 == 0 and time.perf_counter() - _ts > _tl:
            raise TimeLimitExceeded()
        h = h_base * abs(x[i]) if (scaled and x[i] != 0.0) else h_base
        x_fwd = x.copy()
        x_bwd = x.copy()
        x_fwd[i] += h
        x_bwd[i] -= h
        g[i] = (f(x_fwd) - f(x_bwd)) / (2.0 * h)
    return g
