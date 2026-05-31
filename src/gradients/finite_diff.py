"""
Centered finite-difference gradients.
Step size h_i = 10^{-k}, or 10^{-k}*|x_i| when scaled=True (k in [4, 8, 12]).
"""
import time

import numpy as np

from ..time_budget import TimeLimitExceeded, _time_budget


def grad_fd(f, x, k=8, scaled=False, t_start=None, time_limit=None):
    """Approximate the gradient of f at x by centered finite differences.

    The form is fixed to the centered (symmetric) scheme, accurate to O(h^2):

        dF/dx_i ~ [f(x + h e_i) - f(x - h e_i)] / (2 h)

    where e_i is the i-th unit vector, i.e. only coordinate i is perturbed.

    Parameters
    ----------
    f : callable
        Objective F(x) -> float, taking an array of length n.
    x : array_like, shape (n,)
        Point at which the gradient is approximated.
    k : int, default 8
        Step-size exponent: the base step is h = 10**(-k).
    scaled : bool, default False
        If True, scale the step per component as h_i = 10**(-k) * |x_i|
        (falls back to 10**(-k) when x_i == 0).
    t_start : float or None, default None
        Start time (from time.perf_counter()) of the wall-clock budget.
        If None, the shared module-level budget is used instead.
    time_limit : float or None, default None
        Wall-clock budget in seconds. If None, the module-level budget is
        used. Time is checked only when both t_start and time_limit are set.

    Returns
    -------
    g : ndarray, shape (n,)
        Approximated gradient of F at x.

    Raises
    ------
    TimeLimitExceeded
        If the wall-clock budget is exceeded while looping over components.
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    g = np.zeros(n) # gradient vector to fill in (one entry per variable)

    # Base step h = 10^(-k)
    h_base = 10.0 ** (-k)

    # Read the start time and time limit; if not passed as arguments, fall back
    # to the ones shared at module level. Time is only checked when both are
    # available.
    _ts = t_start if t_start is not None else _time_budget[0]
    _tl = time_limit if time_limit is not None else _time_budget[1]
    _check_time = (_ts is not None and _tl is not None)

    # Compute one partial derivative at a time, along each variable i.
    for i in range(n):
        # Every 200 variables, check the clock: if time is up, stop the computation by raising an exception.
        if _check_time and i % 200 == 0 and time.perf_counter() - _ts > _tl:
            raise TimeLimitExceeded()

        # Step for variable i: if scaled=True it is adapted to the magnitude
        # of x[i]; if x[i] is 0 we fall back to the base step.
        h = h_base * abs(x[i]) if (scaled and x[i] != 0.0) else h_base

        # Two copies of the point: in one we move coordinate i forward by +h, in the other backward by -h.
        x_fwd = x.copy()
        x_bwd = x.copy()
        x_fwd[i] += h
        x_bwd[i] -= h

        # Centered difference: change in f between the two points, divided by 2h.
        g[i] = (f(x_fwd) - f(x_bwd)) / (2.0 * h)

    return g
