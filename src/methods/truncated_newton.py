"""Truncated Newton (Linesearch Newton-CG) + Armijo backtracking.

Forcing sequence eta_k:
    'linear'      -> eta_k = eta_fixed (in (0,1))         => linear rate
    'superlinear' -> eta_k = min(0.5, sqrt(||g_k||))      => superlinear rate
    'quadratic'   -> eta_k = min(0.5, ||g_k||)            => quadratic rate
    explicit float -> eta_k = constant = float(forcing).

The external stopping criterion is a plug-in strategy (see
src.stopping_criteria); the Armijo line search is reused from
armijo_backtracking.

Three usage modes (determined by which callables are provided):
    1) No FD    : pass both grad_f and hess_f (exact derivatives)
    2) FD Hess  : pass grad_f, leave hess_f=None -> Hv via hv_fd (matrix-free)
    3) FD both  : leave grad_f=None -> centered FD gradient; hess_f=None -> hv_fd
"""
import time

import numpy as np
import scipy.sparse as sp

from ..gradients.finite_diff import grad_fd
from ..hessians.finite_diff import hv_fd
from ..time_budget import (TimeLimitExceeded, set_time_budget,
                           clear_time_budget)
from ..stopping_criteria.base import StoppingCriterion
from .armijo_backtracking import armijo_backtracking


def _compute_eta(forcing, eta_fixed, g_norm):
    """Compute eta_k (relative tolerance for the inner CG solver).
        i)   eta_k = constant in (0,1)         -> linear convergence
        ii)  eta_k -> 0                        -> superlinear convergence
        iii) eta_k = O(||grad f(x_k)||)        -> quadratic convergence
    """
    if isinstance(forcing, (int, float)) and not isinstance(forcing, bool):
        return float(forcing)
    if forcing == "linear":
        return float(eta_fixed)
    if forcing == "superlinear":
        return min(0.5, float(np.sqrt(g_norm)))
    if forcing == "quadratic":
        return min(0.5, float(g_norm))
    raise ValueError(
        f"forcing must be 'linear'|'superlinear'|'quadratic' or float, got {forcing!r}"
    )


def _cg_truncated(H, b, eta_k, cg_max_iter, t_start=None, time_limit=None):
    """Inner CG for the system  H z = b,  with early exit on three cases
    described in the slides:

        case i)   curvature always positive, exit by inexact-Newton tolerance
                  ||H z - b|| <= eta_k ||b||                  -> "tol"
        case ii)  non-positive curvature at j == 0,
                  returns z = b = -grad f(x_k)                -> "negcurv_j0"
        case iii) non-positive curvature at j > 0,
                  returns the last valid z_j                   -> "negcurv_jpos"
        case iv)  reached the inner iteration cap              -> "maxiter"
        case v)   wall-clock time limit exceeded               -> "time_limit"

    In all cases the returned direction is a descent direction.

    Parameters
    ----------
    H            : ndarray (n, n) OR callable d -> Hd.  The callable form
                   enables the matrix-free path (FD of the gradient) without
                   ever materializing the Hessian — essential for n >> 10^4.
    b            : ndarray (n,),  = -grad f(x_k)
    eta_k        : float in (0,1), relative tolerance for the CG residual
    cg_max_iter  : int, maximum number of inner iterations
    t_start      : float or None, reference time from time.perf_counter()
    time_limit   : float or None, wall-clock budget in seconds

    Returns
    -------
    z            : ndarray (n,),  step p_TN^(k)
    j            : int,           number of CG iterations performed
    termination  : str,           one of "tol"|"negcurv_j0"|"negcurv_jpos"|"maxiter"|"time_limit"
    """
    # H may be a matrix (use its matmul) or a callable d -> H d (matrix-free).
    apply_H = H if callable(H) else H.__matmul__
    n = b.size
    # Standard CG started from z = 0, so the initial residual and search
    # direction both equal b (= -grad f): r = b - H*0 = b, d = r.
    z = np.zeros(n)
    r = b.copy()
    d = b.copy()
    norm_b = float(np.linalg.norm(b))
    rTr = float(r @ r)

    if cg_max_iter <= 0 or norm_b == 0.0:   # nothing to solve (already at a stationary point)
        return z, 0, "tol"

    for j in range(cg_max_iter):
        # case v) time budget exhausted: return a descent direction either way --
        # the steepest-descent step b = -g if we have no iterate yet, else the
        # partial CG iterate z_j accumulated so far.
        if time_limit is not None and (time.perf_counter() - t_start) > time_limit:
            if j == 0:
                return b.copy(), 0, "time_limit"
            return z, j, "time_limit"

        Hd = apply_H(d)
        dHd = float(d @ Hd)            # curvature of H along the search direction d
        # cases ii)/iii) non-positive curvature: CG cannot continue safely.
        # Fall back to steepest descent (-g) on the first step, else keep the
        # last valid iterate; both are guaranteed descent directions.
        if dHd <= 0.0:
            if j == 0:
                return b.copy(), 0, "negcurv_j0"
            return z, j, "negcurv_jpos"

        # standard CG recurrence: step along d, then update the residual
        alpha = rTr / dHd
        z = z + alpha * d
        r_new = r - alpha * Hd

        # case i) inexact-Newton stop: residual shrunk below the forcing tol eta_k
        if float(np.linalg.norm(r_new)) <= eta_k * norm_b:
            return z, j + 1, "tol"

        rTr_new = float(r_new @ r_new)
        beta = rTr_new / rTr          # Fletcher-Reeves coefficient
        d = r_new + beta * d          # next conjugate direction
        r, rTr = r_new, rTr_new

    return z, cg_max_iter, "maxiter"   # case iv) hit the inner iteration cap


def truncated_newton(f, x0, stopping,
                     grad_f=None, hess_f=None,
                     k=8, scaled=False, hv_func=hv_fd,
                     alpha0=1.0, c1=1e-4, rho=0.5,
                     forcing="superlinear", eta_fixed=0.5,
                     cg_max_iter=None,
                     max_iter=1000, time_limit=None, max_iter_backtrack=50,
                     return_history=False):
    """Truncated Newton / Linesearch Newton-CG + Armijo.

    The loop ordering matches the course implementation: steps (2.1-2.3) then check (2.4).
    The stopping criterion is an instance of StoppingCriterion.

    Three usage modes:
        1) No FD:      pass grad_f and hess_f       (both exact)
        2) FD Hessian:  pass grad_f, hess_f=None     -> Hv via hv_fd
        3) FD both:    grad_f=None, hess_f=None      -> centered FD gradient + hv_fd

    When hess_f is provided it is called as hess_f(x) and the return type
    determines the CG operator:
        - 1D ndarray        -> diagonal Hessian (element-wise multiplication)
        - sparse matrix     -> diagonal extracted, same as above
        - 2D ndarray        -> full matrix (matmul)

    When hess_f is None the method uses hv_fd (matrix-free, never assembles H).

    Parameters
    ----------
    f                 : objective function, callable x -> scalar
    x0                : starting point, ndarray (n,)
    stopping          : instance of StoppingCriterion
    grad_f            : gradient callable x -> ndarray (n,).  If None,
                        computed via centered FD of f using (k, scaled).
    hess_f            : Hessian callable x -> matrix or diagonal.  If None,
                        Hv products computed via hv_fd (matrix-free).
    k                 : FD step-size exponent (h = 10^{-k}), used when
                        grad_f or hess_f is None
    scaled            : if True, FD step is rescaled by |x_i|
    hv_func           : callable (grad_f, x, v, k=..., scaled=...) -> Hv.
                        Used ONLY on the matrix-free path (hess_f=None); inert
                        in exact mode (a real Hessian is supplied). Defaults to
                        hv_fd; pass hv_fd_normalized_v to normalize the direction.
    alpha0, c1, rho   : Armijo line-search parameters
    forcing           : eta_k strategy: 'linear'|'superlinear'|'quadratic'
                        or a float in (0,1)
    eta_fixed         : constant value for forcing='linear' (default 0.5)
    cg_max_iter       : max inner CG iterations.  None -> n (problem dimension)
    max_iter          : max outer iterations (safety)
    time_limit        : wall-clock budget in seconds (None = unlimited)
    max_iter_backtrack: max Armijo backtracking steps
    return_history    : if True, stores the iteration trajectory

    Returns
    -------
    result : dict with keys
        x_star, f_star, grad_norm, n_iter, success, stop_reason,
        cg_iters_total, neg_curvature_count, elapsed_s,
        [history]: list of dicts per iteration.
    """
    if x0 is None or stopping is None:
        raise ValueError("truncated_newton requires x0 and stopping")

    t_start = time.perf_counter()
    # Publish (t_start, time_limit) to a module-level slot so the finite-
    # difference routines (grad_fd / hv_fd, which receive no timing arguments)
    # can read it and raise TimeLimitExceeded mid-evaluation.
    set_time_budget(t_start, time_limit)

    # --- resolve FD callables ----
    # If no exact gradient was supplied, replace it with a centered FD gradient,
    # freezing (k, scaled) into the closure.  hess_f is NOT resolved here: when
    # it is None the loop below uses the matrix-free Hessian-vector path (hv_func)
    # instead of ever assembling a Hessian.
    if grad_f is None:
        _k, _scaled = k, scaled
        grad_f = lambda x: grad_fd(f, x, k=_k, scaled=_scaled)

    x = np.asarray(x0, dtype=float).copy()
    n = x.size
    if cg_max_iter is None:
        cg_max_iter = n

    history = []
    cg_iters_total = 0
    neg_curv_count = 0
    n_backtrack_total = 0
    success = False
    stop_reason = "max_iter"
    k_iter = 0
    fx = float('inf')
    g_norm = float('inf')

    try:
        fx = f(x)
        g = grad_f(x)
        g_norm = float(np.linalg.norm(g))

        if time_limit is not None and (time.perf_counter() - t_start) > time_limit:
            stop_reason = "time_limit"
            raise TimeLimitExceeded()

        # Hand the criterion the starting state once, before the loop, so that
        # relative tests can store their reference values (e.g. the initial
        # gradient norm ||g_0|| used to scale a relative tolerance).
        stopping.initialize(x, fx, g)

        if return_history:
            history.append({'x': x.copy(), 'f': fx, 'grad_norm': g_norm,
                            'alpha': None, 'p_norm': None,
                            'cg_iters': None, 'nu_k': None,
                            'cg_termination': None})

        for k_iter in range(max_iter):
            if time_limit is not None and (time.perf_counter() - t_start) > time_limit:
                stop_reason = "time_limit"
                break

            # 2.1: build the operator apply_H : d -> H d that the inner CG needs.
            # CG only ever uses the Hessian through this product, so we never
            # need the full matrix.  Four cases by what hess_f returns:
            if hess_f is not None:
                H_result = hess_f(x)
                if isinstance(H_result, np.ndarray) and H_result.ndim == 1:
                    # diagonal Hessian given as a 1D vector (e.g. Problem 16):
                    # the product is just element-wise multiplication
                    diag = H_result
                    apply_H = lambda d, _diag=diag: _diag * d
                elif sp.issparse(H_result):
                    # sparse (diagonal) Hessian: pull out the diagonal, same as above
                    diag = np.asarray(H_result.diagonal(), dtype=float)
                    apply_H = lambda d, _diag=diag: _diag * d
                else:
                    # full dense Hessian: CG calls its matmul (apply_H(d) = H @ d)
                    apply_H = H_result
            else:
                # matrix-free: no Hessian at all, approximate H*d by finite
                # differences of the gradient at the frozen point x_k (hv_func).
                x_k = x
                _k, _scaled, _hv = k, scaled, hv_func
                apply_H = lambda d, _x=x_k: _hv(grad_f, _x, d, k=_k, scaled=_scaled)

            # 2.2: solve  H p = -g  inexactly with CG.  eta_k is the forcing term
            # (how accurately to solve this iteration): smaller eta_k => tighter
            # inner solve => faster outer convergence but more CG work.
            b = -g
            eta_k = _compute_eta(forcing, eta_fixed, g_norm)
            p, cg_j, cg_term = _cg_truncated(apply_H, b, eta_k, cg_max_iter,
                                              t_start=t_start, time_limit=time_limit)
            cg_iters_total += cg_j
            if cg_term.startswith("negcurv"):   # CG hit non-positive curvature
                neg_curv_count += 1
            if cg_term == "time_limit":         # CG itself ran out of budget: stop now
                stop_reason = "time_limit"
                break

            # 2.3: Armijo backtracking picks a step length alpha along p, then update
            alpha, n_bt = armijo_backtracking(f, x, fx, g, p,
                                              alpha0=alpha0, c1=c1, rho=rho,
                                              max_iter=max_iter_backtrack,
                                              t_start=t_start, time_limit=time_limit)
            n_backtrack_total += n_bt
            # Time check BEFORE committing the step: if the budget ran out during
            # the line search, armijo_backtracking may have returned an untested
            # alpha, so we stop and keep the previous (good) iterate rather than
            # take a possibly non-decreasing step.
            if time_limit is not None and (time.perf_counter() - t_start) > time_limit:
                stop_reason = "time_limit"
                break

            x_prev, fx_prev = x.copy(), fx     # remember previous iterate for the stopping test
            x = x + alpha * p                  # x^(k+1) = x^(k) + alpha * p
            fx = f(x)                          # refresh f, gradient and gradient norm at new x
            g = grad_f(x)
            g_norm = float(np.linalg.norm(g))

            if return_history:
                history.append({'x': x.copy(), 'f': fx, 'grad_norm': g_norm,
                                'alpha': alpha,
                                'p_norm': float(np.linalg.norm(p)),
                                'cg_iters': cg_j, 'nu_k': float(eta_k),
                                'cg_termination': cg_term})

            # 2.4: check the convergence criterion on the new iterate.  It
            #      returns True once its test passes (e.g. ||g|| below tolerance,
            #      or a small change in x or f between iterations).  A True here
            #      is the ONLY genuine-convergence exit: success=True, and
            #      stop_reason records which criterion fired (its .name), as
            #      opposed to the "time_limit"/"max_iter" exits which leave
            #      success=False.
            if stopping.should_stop(k_iter + 1, x, fx, g, x_prev, fx_prev):
                success = True
                stop_reason = stopping.name
                break

    # Two ways the budget can end the run: the inline `if ... break` checks above
    # (which set stop_reason and leave the loop normally), and TimeLimitExceeded
    # raised deep inside an FD gradient / Hessian-vector evaluation, which unwinds
    # to here.  Both leave success=False and the last completed iterate in x.
    except TimeLimitExceeded:
        stop_reason = "time_limit"
    finally:
        clear_time_budget()

    result = {
        'x_star': x,
        'f_star': fx,
        'grad_norm': g_norm,
        'n_iter': k_iter + 1 if k_iter > 0 or g_norm != float('inf') else 0,
        'success': success,
        'stop_reason': stop_reason,
        'cg_iters_total': cg_iters_total,
        'neg_curvature_count': neg_curv_count,
        'n_backtrack_total': n_backtrack_total,
        'elapsed_s': time.perf_counter() - t_start,
    }
    if return_history:
        result['history'] = history
    return result
