"""Hessian approximations via centered finite differences of the exact gradient.

Step h = 10^{-k}, or 10^{-k}*|x_i| when scaled=True (k typically 4/8/12).
Centered stencils give O(h^2) accuracy; symmetry is enforced as (H + H^T)/2.
"""
import time

import numpy as np

from ..time_budget import TimeLimitExceeded, _time_budget


def _step(x, k, scaled):
    """
    Build the per-component finite-difference step vector h.

    Returns one step h_i per coordinate, used by the coordinate stencils in
    hess_fd and hess_fd_diag:

        unscaled : h_i = 10**(-k)              (same step for every component)
        scaled   : h_i = 10**(-k) * |x_i|      (step scaled by |x_i|; where
                                                x_i == 0 it falls back to 10**(-k))

    Parameters
    ----------
    x : array_like, shape (n,)
        Point at which the step is computed (used only by the scaled rule).
    k : int
        Step-size exponent: the base step is h = 10**(-k).
    scaled : bool
        If True use h_i = 10**(-k) * |x_i|; otherwise h_i = 10**(-k).

    Returns
    -------
    h : ndarray, shape (n,)
        Per-component step vector.
    """
    x = np.asarray(x, dtype=float)
    h_base = 10.0 ** (-k)                          # base step h = 10^(-k)
    if not scaled:
        return np.full(x.shape, h_base, dtype=float)   # same step for every component
    h = h_base * np.abs(x)                         # scale each step by the size of x_i
    h[h == 0.0] = h_base                           # where x_i == 0, fall back to the base step
    return h


def hess_fd(grad_f, x, k=8, *, scaled=False, symmetrize=True,
            t_start=None, time_limit=None):
    """
    Approximate the Hessian H(x) column by column (centered).

    Each column j is the centered difference of the gradient when only
    coordinate j is perturbed:

        H[:, j] ~ (grad f(x + h e_j) - grad f(x - h e_j)) / (2 h).

    Cost: 2n gradient evaluations.

    When to use
    -----------
    The general case: H is dense or its sparsity is unknown (e.g. Problem 28).
    This is the FD Hessian wired into Modified Newton (call with hess_f=None),
    which needs the full matrix to factorize it (Cholesky). Costs 2n gradient
    evaluations and O(n^2) storage, so at large n prefer hess_fd_diag when H is
    known to be diagonal, or the matrix-free hv_fd used by Truncated Newton.

    Parameters
    ----------
    grad_f : callable
        Gradient map x -> grad f(x), returning an array of length n.
    x : array_like, shape (n,)
        Point at which the Hessian is approximated.
    k : int, default 8
        Step-size exponent: the base step is h = 10**(-k).
    scaled : bool, default False
        If True, h_i = 10**(-k) * |x_i| (falls back to 10**(-k) when x_i == 0).
    symmetrize : bool, default True
        If True, return the symmetrized matrix (H + H^T) / 2.
    t_start : float or None, default None
        Start time of the wall-clock budget; falls back to the module budget.
    time_limit : float or None, default None
        Wall-clock budget in seconds; falls back to the module budget.

    Returns
    -------
    H : ndarray, shape (n, n)
        Approximated Hessian of f at x.

    Raises
    ------
    TimeLimitExceeded
        If the wall-clock budget is exceeded while looping over columns.
    """
    x = np.asarray(x, dtype=float)
    n = x.size
    h = _step(x, k, scaled)               # per-component step vector
    H = np.empty((n, n), dtype=float)     # matrix to fill column by column

    # Read the time budget from the arguments, else from the shared module one;
    # time is only checked when both values are available.
    _ts = t_start if t_start is not None else _time_budget[0]
    _tl = time_limit if time_limit is not None else _time_budget[1]
    _check_time = (_ts is not None and _tl is not None)

    # Build one column at a time by perturbing only coordinate j.
    for j in range(n):
        # Every 100 columns, stop the computation if the allotted time is up.
        if _check_time and j % 100 == 0 and time.perf_counter() - _ts > _tl:
            raise TimeLimitExceeded()
        xp = x.copy(); xp[j] += h[j]      # point moved forward along coordinate j
        xm = x.copy(); xm[j] -= h[j]      # point moved backward along coordinate j
        H[:, j] = (grad_f(xp) - grad_f(xm)) / (2.0 * h[j])   # centered difference

    if symmetrize:
        H = 0.5 * (H + H.T)               # force exact symmetry
    return H


def hess_fd_diag(grad_f, x, k=8, *, scaled=False):
    """
    Diagonal Hessian via a single global perturbation (centered).

    Valid when H is diagonal (each gradient component g_j depends only on x_j).
    Then the whole diagonal follows from ONE pair of gradient calls, regardless
    of n:

        diag(H)[j] ~ (g_j(x + h) - g_j(x - h)) / (2 h_j).

    When to use
    -----------
    Only when the Hessian is known to be diagonal — each gradient component
    g_j depends solely on x_j (e.g. Problem 16). One pair of gradient calls
    then recovers the whole diagonal regardless of n, vs the 2n calls and
    O(n^2) storage of hess_fd. Do NOT use it when off-diagonal terms exist:
    the result is wrong.

    Parameters
    ----------
    grad_f : callable
        Gradient map x -> grad f(x), returning an array of length n.
    x : array_like, shape (n,)
        Point at which the diagonal is approximated.
    k : int, default 8
        Step-size exponent: the base step is h = 10**(-k).
    scaled : bool, default False
        If True, h_i = 10**(-k) * |x_i| (falls back to 10**(-k) when x_i == 0).

    Returns
    -------
    diag : ndarray, shape (n,)
        The diagonal of H (compact representation of the full matrix).
    """
    x = np.asarray(x, dtype=float)
    h = _step(x, k, scaled)               # per-component step vector
    gp = grad_f(x + h)                    # gradient with all coordinates moved forward
    gm = grad_f(x - h)                    # gradient with all coordinates moved backward
    return (gp - gm) / (2.0 * h)          # component-wise centered difference


def hv_fd(grad_f, x, v, k=8, *, scaled=False):
    """
    Matrix-free Hessian-vector product H(x) v (centered).

    Approximates the product without ever assembling H, using two gradient
    calls along the direction v:

        H(x) v ~ (grad f(x + h v) - grad f(x - h v)) / (2 h).

    When to use
    -----------
    When you need only the action of H on a vector, not the matrix itself —
    the inner CG solver of Truncated Newton (its default hv_func). Two gradient
    calls per product and no O(n^2) assembly, so it scales to large n. When the
    direction v can have extreme magnitude, prefer hv_fd_normalized_v.

    Parameters
    ----------
    grad_f : callable
        Gradient map x -> grad f(x).
    x : array_like, shape (n,)
        Point at which the product is evaluated.
    v : array_like, shape (n,)
        Direction the Hessian is multiplied by.
    k : int, default 8
        Step-size exponent: the base step is h = 10**(-k).
    scaled : bool, default False
        If True, rescale the (scalar) step by max(||x||_inf, 1).

    Returns
    -------
    Hv : ndarray, shape (n,)
        Approximated product H(x) v.
    """
    x = np.asarray(x, dtype=float)
    v = np.asarray(v, dtype=float)
    h_base = 10.0 ** (-k)                 # base step h = 10^(-k)
    # Step is a scalar here because we perturb along a generic direction v.
    h = h_base * max(np.max(np.abs(x)), 1.0) if scaled else h_base
    return (grad_f(x + h * v) - grad_f(x - h * v)) / (2.0 * h)   # centered difference


def hv_fd_normalized_v(grad_f, x, v, k=8, *, scaled=False):
    """Like hv_fd, but normalize the direction to unit length first.

    Since the Hessian is linear in the direction, H(x) v = ||v|| * H(x) (v/||v||):
    we evaluate the product along v/||v|| and multiply the result back by ||v||.
    Normalizing keeps the actual perturbation on a unit scale, avoiding overshoot
    or underflow when ||v|| is huge or tiny (e.g. Problem 28, where
    ||grad(x_bar)|| = O(n^7)). Returns the zero vector when ||v|| == 0.

    When to use
    -----------
    Same role as hv_fd (Truncated Newton's hv_func), but when ||v|| can be huge
    or tiny so that the raw step h*v would overshoot or underflow — e.g.
    Problem 28. Pass it to Truncated Newton as hv_func in place of hv_fd.

    Parameters
    ----------
    grad_f : callable
        Gradient map x -> grad f(x).
    x : array_like, shape (n,)
        Point at which the product is evaluated.
    v : array_like, shape (n,)
        Direction the Hessian is multiplied by.
    k : int, default 8
        Step-size exponent: the base step is h = 10**(-k).
    scaled : bool, default False
        Forwarded to hv_fd.

    Returns
    -------
    Hv : ndarray, shape (n,)
        Approximated product H(x) v.
    """
    v = np.asarray(v, dtype=float)
    nv = float(np.linalg.norm(v))         # length of the direction
    if nv == 0.0:
        return np.zeros_like(np.asarray(x, dtype=float))   # H(x) * 0 = 0
    v_hat = v / nv                        # unit-length direction
    # Delegate to hv_fd (single source of truth for the step rule), then undo
    # the normalization with the linear rescale by ||v||.
    return nv * hv_fd(grad_f, x, v_hat, k=k, scaled=scaled)
