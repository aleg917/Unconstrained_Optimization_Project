"""Modified Newton + Armijo Backtracking line search.

The stopping criterion is a plug-in strategy (see src.stopping_criteria);
the line search is Armijo backtracking (see armijo_backtracking module).

Three usage modes (determined by which callables are provided):
    1) No FD    : pass both grad_f and hess_f       (both exact)
    2) FD Hess  : pass grad_f, leave hess_f=None    -> hess_fd of grad_f
    3) FD both  : leave grad_f=None, hess_f=None    -> FD gradient + FD Hessian
"""
import time

import numpy as np
import scipy.linalg as sla
import scipy.sparse as sp

from ..gradients.finite_diff import grad_fd
from ..hessians.finite_diff import hess_fd
from ..time_budget import (TimeLimitExceeded, set_time_budget,
                           clear_time_budget)
from ..stopping_criteria.base import StoppingCriterion
from .armijo_backtracking import armijo_backtracking


def _modify_hessian_cholesky(H, beta=1e-3, max_iter=100,
                             t_start=None, time_limit=None):
    """Build B_k via Cholesky with added multiple of identity.

    Given H = grad^2 f(x^(k)), finds the smallest tau_k >= 0 such that
    H + tau_k I is (sufficiently) positive definite, using Cholesky failure
    as an oracle for non-positive-definiteness.

    Initialization of tau (from the slides):
        if all diagonal entries > 0:  tau_k^(0) = 0
        otherwise:                    tau_k^(0) = beta - min_i [H]_ii

    Iteration:
        try Cholesky  R^T R = H + tau_k^(j) I
        if success:  B_k = H + tau_k^(j) I,  done
        if fail:     tau_k^(j+1) = max(2 * tau_k^(j), beta)

    Parameters
    ----------
    H        : ndarray (n, n) or scipy.sparse matrix (Hessian)
    beta     : heuristic parameter, typical value 1e-3 (see slides)
    max_iter : maximum number of tau doublings
    t_start    : reference time from time.perf_counter()
    time_limit : wall-clock budget in seconds (None = unlimited)

    Returns
    -------
    R_or_diag : ndarray (n, n) upper triangular (dense case)
                or ndarray (n,) modified diagonal (sparse case)
    tau       : multiple of identity actually added
    n_adj     : number of tau doublings (0 if H was already pos. def.)

    Raises
    ------
    TimeLimitExceeded if the wall-clock budget is exceeded.
    RuntimeError if no tau within max_iter makes H + tau*I pos. def.
    """
    # Wall-clock budget is enforced only when BOTH a start time and a limit were
    # passed in (the dense loop below uses this flag); otherwise time is ignored.
    _check = (t_start is not None and time_limit is not None)

    # --- fast path: sparse (diagonal) Hessian, e.g. Problem 16 ---
    # A diagonal matrix is positive definite iff every diagonal entry is > 0,
    # so no factorization is needed: we test min(diagonal) directly and return
    # the (shifted) diagonal itself in place of a Cholesky factor.
    if sp.issparse(H):
        diag_h = np.asarray(H.diagonal(), dtype=float)
        diag_min = float(diag_h.min())
        tau = 0.0 if diag_min > 0.0 else beta - diag_min   # same tau init as dense path

        # No time check here: each pass is O(n) and the doublings are few, so
        # this loop is effectively instantaneous even for large n.
        for j in range(max_iter):
            shifted = diag_h + tau                  # diagonal of H + tau*I
            if float(shifted.min()) > 0.0:          # all entries > 0  =>  pos. def.
                return shifted, tau, j
            tau = max(2.0 * tau, beta)              # not yet PD: grow tau and retry

        raise RuntimeError(
            f"_modify_hessian_cholesky: no tau within {max_iter} iterations "
            f"makes H + tau*I positive definite (last tau={tau:.3e})."
        )

    # --- dense path: full Hessian, e.g. Problem 28 ---
    # No cheap positive-definiteness test exists for a full matrix, so Cholesky
    # itself is the oracle: it succeeds iff H + tau*I is positive definite.
    diag_min = float(np.min(np.diag(H)))
    tau = 0.0 if diag_min > 0.0 else beta - diag_min   # same tau init as sparse path

    n = H.shape[0]
    I = np.eye(n)
    for j in range(max_iter):
        # Each Cholesky attempt is O(n^3): if the budget is already spent, stop
        # before starting another one.  The exception unwinds up to the caller
        # (modified_newton), which catches it and reports stop_reason="time_limit".
        if _check and (time.perf_counter() - t_start) > time_limit:
            raise TimeLimitExceeded()
        try:
            R = sla.cholesky(H + tau * I, lower=False)   # success => PD; R is the factor
            return R, tau, j
        except sla.LinAlgError:                          # failure => not PD yet
            tau = max(2.0 * tau, beta)                   # grow tau and retry

    raise RuntimeError(
        f"_modify_hessian_cholesky: no tau within {max_iter} iterations "
        f"makes H + tau*I positive definite (last tau={tau:.3e})."
    )


def modified_newton(f, x0, stopping,
                    grad_f=None, hess_f=None,
                    k=8, scaled=False,
                    alpha0=1.0, c1=1e-4, rho=0.5,
                    beta=1e-3, max_tau_iter=100,
                    max_iter=1000, time_limit=None, max_iter_backtrack=50,
                    return_history=False):
    """Modified Newton + Armijo backtracking.

    The loop ordering matches the course implementation: steps (2.1-2.3) then check (2.4).
    The stopping criterion is passed as an instance of StoppingCriterion.

    Three usage modes:
        1) No FD:      pass grad_f and hess_f       (both exact)
        2) FD Hessian:  pass grad_f, hess_f=None     -> hess_fd(grad_f, ...)
        3) FD both:    grad_f=None, hess_f=None      -> FD gradient + FD Hessian

    When hess_f returns a sparse matrix, the solver exploits the diagonal
    structure (element-wise division instead of Cholesky factorization).

    Parameters
    ----------
    f                 : objective function, callable x -> scalar
    x0                : starting point, ndarray (n,)
    stopping          : instance of StoppingCriterion
    grad_f            : gradient callable x -> ndarray (n,).  If None,
                        computed via centered FD of f using (k, scaled).
    hess_f            : Hessian callable x -> matrix.  May return a dense
                        ndarray (n, n) or a scipy.sparse matrix; a sparse
                        (diagonal) return selects the element-wise solve path
                        instead of the dense Cholesky one.  If None, computed
                        via hess_fd(grad_f, ...) using (k, scaled).
    k                 : FD step-size exponent (h = 10^{-k}), used when
                        grad_f or hess_f is None
    scaled            : if True, FD step is rescaled by |x_i|
    alpha0, c1, rho   : Armijo line-search parameters
    beta              : heuristic parameter for Cholesky modification
    max_tau_iter      : max number of tau doublings per iteration
    max_iter          : max number of outer iterations (safety)
    time_limit        : wall-clock budget in seconds (None = unlimited)
    max_iter_backtrack: max Armijo backtracking steps
    return_history    : if True, stores the iteration trajectory

    Returns
    -------
    result : dict with keys
        x_star, f_star, grad_norm, n_iter, success, stop_reason,
        n_chol_adjustments_total, elapsed_s,
        [history]: list of dicts per iteration.
    """
    t_start = time.perf_counter()
    # Publish (t_start, time_limit) to a module-level slot so the finite-
    # difference routines (which run deep inside hess_f / grad_f and receive no
    # timing arguments) can read it and raise TimeLimitExceeded mid-evaluation.
    # Cleared in the finally block below.
    set_time_budget(t_start, time_limit)

    # --- resolve FD callables ---
    # Whatever derivative the caller did NOT supply is replaced by a finite-
    # difference version, with the FD settings (k, scaled) frozen into the
    # closure.  This is what realizes the three usage modes:
    #   grad_f given, hess_f given  ->  nothing replaced        (exact / exact)
    #   grad_f given, hess_f None   ->  hess_f := FD of grad_f  (FD Hessian)
    #   grad_f None,  hess_f None   ->  grad_f := FD of f, then hess_f := FD of it
    if grad_f is None:
        _k, _scaled = k, scaled  # bind by value so the closure is frozen
        grad_f = lambda x: grad_fd(f, x, k=_k, scaled=_scaled)
    if hess_f is None:
        _k, _scaled = k, scaled
        # built from grad_f, which may itself be the FD gradient defined just above
        hess_f = lambda x: hess_fd(grad_f, x, k=_k, scaled=_scaled)

    x = np.asarray(x0, dtype=float).copy()
    history = []
    n_chol_total = 0
    n_backtrack_total = 0
    success = False
    stop_reason = "max_iter"
    k_iter = 0
    fx = float('inf')
    g_norm = float('inf')

    try:
        # Evaluate the objective and gradient at the starting point x0.
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
                            'alpha': None, 'tau': None, 'p_norm': None})

        for k_iter in range(max_iter):
            if time_limit is not None and (time.perf_counter() - t_start) > time_limit:
                stop_reason = "time_limit"
                break

            # 2.1: build B_k = H + tau*I, a positive-definite modification of the
            #      Hessian.  R_or_diag is the dense Cholesky factor R (dense H), or
            #      the modified diagonal vector (sparse/diagonal H) - see step 2.2.
            H = hess_f(x)
            R_or_diag, tau, n_adj = _modify_hessian_cholesky(H, beta=beta,
                                                              max_iter=max_tau_iter,
                                                              t_start=t_start,
                                                              time_limit=time_limit)
            n_chol_total += n_adj

            # 2.2: solve  B_k p = -g  for the Newton step p (two cases, matching 2.1)
            if sp.issparse(H):
                p = -g / R_or_diag        # diagonal B_k: one division per entry (B_ii p_i = -g_i)
            else:
                p = sla.cho_solve((R_or_diag, False), -g)   # dense B_k: reuse factor R (R^T R p = -g)

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
                                'alpha': alpha, 'tau': tau,
                                'p_norm': float(np.linalg.norm(p))})

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
    # raised deep inside a Cholesky/FD evaluation, which unwinds to here.  Both
    # leave success=False and the last completed iterate in x.
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
        'n_chol_adjustments_total': n_chol_total,
        'n_backtrack_total': n_backtrack_total,
        'elapsed_s': time.perf_counter() - t_start,
    }
    if return_history:
        result['history'] = history
    return result
