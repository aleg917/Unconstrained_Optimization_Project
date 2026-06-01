"""Preconditioned Truncated Newton (Linesearch Newton-CG with preconditioning).
"""
import time
import warnings

import numpy as np
import scipy.linalg as sla
import scipy.sparse as sp

from ..gradients.finite_diff import grad_fd
from ..hessians.finite_diff import hv_fd
from ..time_budget import (TimeLimitExceeded, set_time_budget,
                           clear_time_budget)
from .armijo_backtracking import armijo_backtracking
from .truncated_newton import _compute_eta, _cg_truncated


def preconditioned_truncated_newton(f, x0, stopping,
                                    grad_f=None, hess_f=None,
                                    k=8, scaled=False, hv_func=hv_fd,
                                    alpha0=1.0, c1=1e-4, rho=0.5,
                                    forcing="superlinear", eta_fixed=0.5,
                                    cg_max_iter=None,
                                    max_iter=1000, time_limit=None,
                                    max_iter_backtrack=50,
                                    return_history=False):
    """Preconditioned Truncated Newton / Linesearch Newton-CG + Armijo.

    Same signature and same usage as ``truncated_newton``.  The only difference
    is the preconditioning of the inner CG solver (see the module docstring).

    Preconditioner construction (only if an ASSEMBLED Hessian is available, i.e.
    hess_f is provided and returns a matrix):
        - Diagonal/sparse Hessian (Problem 16): exact and trivial IC,
          L = diag(sqrt(diag(H))).  Requires diag(H) > 0.
        - Dense Hessian (Problem 28): full Cholesky
          ``scipy.linalg.cholesky(H, lower=True)`` (H28 is SPD, so the factor
          is exact; chosen instead of spilu because it is simpler).
    Fallback (warning + unpreconditioned CG):
        - hess_f is None  -> matrix-free path, no matrix to factorize;
        - factorization failed (Hessian not positive definite).

    Parameters
    ----------
    f                 : objective function, callable x -> scalar
    x0                : starting point, ndarray (n,)
    stopping          : instance of StoppingCriterion
    grad_f            : gradient callable x -> ndarray (n,).  If None,
                        computed via centered FD of f using (k, scaled).
    hess_f            : Hessian callable x -> matrix or diagonal.  If None,
                        Hv products computed via hv_func (matrix-free) and
                        preconditioning is disabled (fallback to TN).
    k                 : FD step-size exponent (h = 10^{-k}), used when
                        grad_f or hess_f is None
    scaled            : if True, FD step is rescaled by |x_i|
    hv_func           : callable (grad_f, x, v, k=..., scaled=...) -> Hv.
                        Used ONLY on the matrix-free path (hess_f=None); inert
                        in exact mode. Defaults to hv_fd.
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
        cg_iters_total, neg_curvature_count, n_backtrack_total, elapsed_s,
        [history]: list of dicts per iteration.
    """
    if x0 is None or stopping is None:
        raise ValueError("preconditioned_truncated_newton requires x0 and stopping")

    t_start = time.perf_counter()
    # Publish (t_start, time_limit) in a module-level slot so that the finite-
    # difference routines (grad_fd / hv_fd, which take no time arguments) can
    # read it and raise TimeLimitExceeded mid-evaluation.
    set_time_budget(t_start, time_limit)

    # --- resolve FD callables ---
    # If an exact gradient is not provided, it is replaced with a centered FD
    # gradient, freezing (k, scaled) into the closure.  hess_f is NOT resolved
    # here: when it is None the loop below uses the matrix-free path (hv_func)
    # and preconditioning stays disabled.
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
    _pc_warned = False   # emit the fallback warning ONLY once per run

    def _warn_once(msg):
        # Avoid repeating the same warning on every outer iteration.
        nonlocal _pc_warned
        if not _pc_warned:
            warnings.warn(msg, stacklevel=2)
            _pc_warned = True

    try:
        fx = f(x)
        g = grad_f(x)
        g_norm = float(np.linalg.norm(g))

        if time_limit is not None and (time.perf_counter() - t_start) > time_limit:
            stop_reason = "time_limit"
            raise TimeLimitExceeded()

        # Pass the initial state to the criterion only once, before the loop, so
        # that relative tests can store their reference values (e.g. the initial
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

            # 2.1-2.2: build the operator apply_H (H(x) d) and, if possible, the
            # two triangular "solves" of the preconditioner  Linv = L^-1,
            # LinvT = L^-T  (with M = L L^T ~ H). The factorization happens ONCE
            # per outer iteration (H changes with x): same cost structure as
            # Modified Newton.
            preconditioned = False
            Linv = LinvT = None

            if hess_f is not None:
                H_result = hess_f(x)
                if isinstance(H_result, np.ndarray) and H_result.ndim == 1:
                    # Diagonal Hessian provided as a 1D vector.
                    diag = H_result
                    apply_H = lambda d, _diag=diag: _diag * d
                    if np.all(diag > 0.0):
                        # Exact IC: L = diag(sqrt(diag)); since L is diagonal and
                        # symmetric, L^-1 = L^-T = division by sqrt(diag).
                        sd = np.sqrt(diag)
                        Linv = LinvT = lambda w, _sd=sd: w / _sd
                        preconditioned = True
                    else:
                        _warn_once("PTN: Hessian diagonal not positive; "
                                   "falling back to unpreconditioned CG.")
                elif sp.issparse(H_result):
                    # Sparse diagonal Hessian (Problem 16): we extract its
                    # diagonal, as the basic TN does.
                    diag = np.asarray(H_result.diagonal(), dtype=float)
                    apply_H = lambda d, _diag=diag: _diag * d
                    if np.all(diag > 0.0):
                        sd = np.sqrt(diag)
                        Linv = LinvT = lambda w, _sd=sd: w / _sd
                        preconditioned = True
                    else:
                        _warn_once("PTN: Hessian diagonal not positive; "
                                   "falling back to unpreconditioned CG.")
                else:
                    # Dense Hessian (Problem 28). apply_H as a matmul (wrapped in
                    # a lambda so it is also callable inside A_tilde; the product
                    # is numerically identical to that of the basic TN).
                    apply_H = lambda d, _H=H_result: _H @ d
                    try:
                        # H28 = I + c j j^T is SPD (c >= 1) -> exact Cholesky.
                        L = sla.cholesky(H_result, lower=True)
                        Linv = lambda w, _L=L: sla.solve_triangular(_L, w, lower=True)
                        LinvT = lambda w, _L=L: sla.solve_triangular(_L.T, w, lower=False)
                        preconditioned = True
                    except sla.LinAlgError:
                        _warn_once("PTN: Cholesky failed (Hessian not positive "
                                   "definite); falling back to unpreconditioned CG.")
            else:
                # Matrix-free path (hess_f=None): no matrix to factorize ->
                # preconditioning impossible, behave like the basic Truncated
                # Newton.
                x_k = x
                _k, _scaled, _hv = k, scaled, hv_func
                apply_H = lambda d, _x=x_k: _hv(grad_f, _x, d, k=_k, scaled=_scaled)
                _warn_once("PTN: hess_f=None (matrix-free mode), no matrix to "
                           "precondition; falling back to unpreconditioned CG.")

            b = -g
            eta_k = _compute_eta(forcing, eta_fixed, g_norm)

            if preconditioned:
                # Preconditioned operator A~(w) = L^-1 ( H ( L^-T w ) ) and right-
                # hand side b~ = L^-1 (-g).  We reuse _cg_truncated UNCHANGED on
                # (A~, b~) and then map the direction back to the original space:
                # d = L^-T d~.
                A_tilde = lambda w: Linv(apply_H(LinvT(w)))
                b_tilde = Linv(b)
                d_tilde, cg_j, cg_term = _cg_truncated(A_tilde, b_tilde, eta_k,
                                                       cg_max_iter,
                                                       t_start=t_start,
                                                       time_limit=time_limit)
                # On negative curvature at j==0, _cg_truncated returns b~; then
                # p = L^-T b~ = M^-1(-g), still a descent direction
                # (g^T p = -g^T M^-1 g < 0 because M = L L^T is SPD).
                p = LinvT(d_tilde)
            else:
                p, cg_j, cg_term = _cg_truncated(apply_H, b, eta_k, cg_max_iter,
                                                 t_start=t_start, time_limit=time_limit)

            cg_iters_total += cg_j
            if cg_term.startswith("negcurv"):   # CG encountered non-positive curvature
                neg_curv_count += 1
            if cg_term == "time_limit":          # CG itself ran out of budget: stop
                stop_reason = "time_limit"
                break

            # 2.3: Armijo backtracking picks the step alpha along p, then updates
            alpha, n_bt = armijo_backtracking(f, x, fx, g, p,
                                              alpha0=alpha0, c1=c1, rho=rho,
                                              max_iter=max_iter_backtrack,
                                              t_start=t_start, time_limit=time_limit)
            n_backtrack_total += n_bt
            # Time check BEFORE applying the step: if the budget ran out during
            # the line search, armijo_backtracking may have returned an
            # unverified alpha, so we stop keeping the previous (good) iterate
            # instead of taking a step that might not decrease f.
            if time_limit is not None and (time.perf_counter() - t_start) > time_limit:
                stop_reason = "time_limit"
                break

            x_prev, fx_prev = x.copy(), fx     # remember the previous iterate for the stopping test
            x = x + alpha * p                  # x^(k+1) = x^(k) + alpha * p
            fx = f(x)                          # recompute f, gradient and gradient norm at x
            g = grad_f(x)
            g_norm = float(np.linalg.norm(g))

            if return_history:
                history.append({'x': x.copy(), 'f': fx, 'grad_norm': g_norm,
                                'alpha': alpha,
                                'p_norm': float(np.linalg.norm(p)),
                                'cg_iters': cg_j, 'nu_k': float(eta_k),
                                'cg_termination': cg_term})

            # 2.4: check the convergence criterion on the new iterate.  Returns
            #      True when its test is satisfied (e.g. ||g|| below tolerance,
            #      or a small change in x or f between iterations).  A True here
            #      is the ONLY true-convergence exit: success=True and stop_reason
            #      records which criterion fired (its .name), unlike the
            #      "time_limit"/"max_iter" exits which leave success=False.
            if stopping.should_stop(k_iter + 1, x, fx, g, x_prev, fx_prev):
                success = True
                stop_reason = stopping.name
                break

    # Two ways the budget can terminate the run: the inline `if ... break` checks
    # above (which set stop_reason and exit the loop normally) and
    # TimeLimitExceeded raised deep inside an FD gradient evaluation /
    # Hessian-vector product, which propagates up here.  In both cases
    # success=False and x holds the last complete iterate.
    except TimeLimitExceeded:
        stop_reason = "time_limit"
    finally:
        clear_time_budget()   # always release the shared module-level budget

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
