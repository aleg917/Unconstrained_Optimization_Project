"""Hessian approximation via centered finite differences of the exact gradient.

Implements the idea from the NO4LSP course slides:
    H_f(x) = J_{grad f}(x)
Given the exact gradient grad f, we apply centered FD formulas to compute
the Jacobian of grad f.  Centered stencil gives O(h^2) accuracy.

Symmetry is enforced via  H <- (H + H^T) / 2  as indicated in the slides.

Three routines:

    hess_fd       — column-by-column assembly (2n gradient evaluations).
                    General-purpose baseline, used as unit-test reference.

    hess_fd_diag  — CPR single-color case for diagonal Hessians (Problem 16).
                    A single global perturbation along x +/- h*1 extracts the
                    entire diagonal, independent of n.

    hv_fd         — matrix-free Hessian-vector product, used by the inner CG
                    solver of Truncated Newton.  Two gradient calls per Hv.

For each routine the step h has two variants (as required by the assignment):
    fixed:   h    = 10^{-k}
    scaled:  h_i  = 10^{-k} * |x_i|  (falls back to 10^{-k} when x_i = 0)
with k in {4, 8, 12}.
"""
import time

import numpy as np

from ..gradients.finite_diff import TimeLimitExceeded, _time_budget


def _step(x, k, scaled):
    """Per-component step vector h_i for each direction e_i.

    fixed   -> h_i = 10^{-k}
    scaled  -> h_i = 10^{-k} * |x_i|  if x_i != 0, else 10^{-k}
    """
    x = np.asarray(x, dtype=float)
    h_base = 10.0 ** (-k)
    if not scaled:
        return np.full(x.shape, h_base, dtype=float)
    h = h_base * np.abs(x)
    h[h == 0.0] = h_base
    return h


def hess_fd(grad_f, x, k=8, *, scaled=False, symmetrize=True,
            t_start=None, time_limit=None):
    """Approximate H(x) ~ J_{grad f}(x) column by column (centered, O(h^2)).

    Cost: 2n gradient evaluations.

    Parameters
    ----------
    grad_f     : callable, x -> grad f(x), ndarray (n,)
    x          : ndarray (n,)
    k          : step-size exponent (h = 10^{-k})
    scaled     : if True, h_i = 10^{-k} * |x_i| (falls back when x_i = 0)
    symmetrize : if True, returns (H + H^T) / 2
    t_start    : float or None, reference time (time.perf_counter())
    time_limit : float or None, wall-clock budget in seconds

    Returns
    -------
    H : ndarray (n, n)

    Raises
    ------
    TimeLimitExceeded if the wall-clock budget is exceeded.
    """
    x = np.asarray(x, dtype=float)
    n = x.size
    h = _step(x, k, scaled)
    H = np.empty((n, n), dtype=float)
    _ts = t_start if t_start is not None else _time_budget[0]
    _tl = time_limit if time_limit is not None else _time_budget[1]
    _check_time = (_ts is not None and _tl is not None)

    for j in range(n):
        if _check_time and j % 100 == 0 and time.perf_counter() - _ts > _tl:
            raise TimeLimitExceeded()
        xp = x.copy(); xp[j] += h[j]
        xm = x.copy(); xm[j] -= h[j]
        H[:, j] = (grad_f(xp) - grad_f(xm)) / (2.0 * h[j])

    if symmetrize:
        H = 0.5 * (H + H.T)
    return H


def hess_fd_diag(grad_f, x, k=8, *, scaled=False):
    """CPR single-color: diagonal Hessian via ONE global perturbation.

    Valid under the assumption (verified for Problem 16) that H is diagonal,
    i.e. each gradient component g_j depends only on x_j.  In that case:

        diag(H)[j] = (g_j(x + h) - g_j(x - h)) / (2 h_j)

    Cost: 2 gradient evaluations, INDEPENDENT of n.

    Returns
    -------
    diag : ndarray (n,) — the diagonal of H (compact representation)
    """
    x = np.asarray(x, dtype=float)
    h = _step(x, k, scaled)
    gp = grad_f(x + h)
    gm = grad_f(x - h)
    return (gp - gm) / (2.0 * h)


def hv_fd(grad_f, x, v, k=8, *, scaled=False):
    """Matrix-free Hessian-vector product (centered, O(h^2)).

        H(x) v  ~  (grad f(x + h v) - grad f(x - h v)) / (2 h)

    No matrix is ever assembled.  Two gradient calls per Hv application.
    Used by the inner CG solver of Truncated Newton.

    Step h (scalar, not per-component, because the perturbation is along
    a generic direction v):
        fixed   -> h = 10^{-k}
        scaled  -> h = 10^{-k} * max(||x||_inf, 1)   (global rescaling)

    Parameters
    ----------
    grad_f  : callable, x -> grad f(x)
    x       : ndarray (n,)
    v       : ndarray (n,), direction
    k       : step-size exponent
    scaled  : if True, rescale h by max(||x||_inf, 1)

    Returns
    -------
    Hv : ndarray (n,)
    """
    x = np.asarray(x, dtype=float)
    v = np.asarray(v, dtype=float)
    h_base = 10.0 ** (-k)
    h = h_base * max(np.max(np.abs(x)), 1.0) if scaled else h_base
    return (grad_f(x + h * v) - grad_f(x - h * v)) / (2.0 * h)
