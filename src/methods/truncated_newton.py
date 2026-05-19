"""Truncated Newton (Linesearch Newton-CG) + Armijo backtracking.

Implementazione delle slides del corso NO4LSP (Politecnico di Torino, A.Y.
2025/2026), capitolo "Truncated Newton method / Linesearch Newton-CG".
La procedura coincide matematicamente con il metodo di Newton inesatto
risolto via Conjugate Gradient (Dembo-Steihaug; cfr. Nocedal-Wright cap. 7
"Inexact Newton Methods" e teoremi 6.1-6.2 in [SW]).

Schema (slides):
    Given x^(0). For k = 0, 1, 2, ...:
        Solve  B_k z = c_k     con  B_k = grad^2 f(x^(k)),  c_k = -grad f(x^(k))
        2.1  z^(0) = 0,  d^(0) = -grad f(x^(k)) = c_k,  r^(0) = c_k.
        2.2  For j = 0, 1, 2, ...   % inner CG loop
            2.2.1  Se d_j^T B_k d_j > 0:
                       proseguire con CG (update z, r, beta, d).
                   Altrimenti (curvatura non positiva):
                       se j == 0  -> z = -grad f(x^(k))   (caso ii: steepest)
                       altrimenti -> z = z_j              (caso iii)
                       STOP inner.
            2.2.2  Se ||B_k z - c_k|| <= eta_k ||c_k||  STOP (Newton inesatto).
        2.3  p_TN^(k) := z;  alpha^(k) via Armijo + backtracking;
             x^(k+1) = x^(k) + alpha^(k) p_TN^(k).
        2.4  Stop se il criterio di arresto e' soddisfatto.

Forcing sequence eta_k (Thm 6.2 in [SW]):
    'linear'      -> eta_k = eta_fixed (in (0,1))         => rate lineare
    'superlinear' -> eta_k = min(0.5, sqrt(||g_k||))      => rate superlineare
    'quadratic'   -> eta_k = min(0.5, ||g_k||)            => rate quadratico
    float esplicito -> eta_k costante = float(forcing).

Il criterio di arresto esterno e' una strategia plug-in (vedi
src.stopping_criteria), identica a modified_newton; la line search Armijo
e' riusata dal modulo omonimo.
"""
import numpy as np

from ..stopping_criteria.base import StoppingCriterion
from .armijo_backtracking import armijo_backtracking


def _compute_eta(forcing, eta_fixed, g_norm):
    """Calcola eta_k (tolleranza relativa per il solver CG interno).

    Cfr. teorema 6.2 in [SW] / ultima slide della professoressa:
        i)   eta_k = costante in (0,1)         -> convergenza lineare
        ii)  eta_k -> 0                        -> superlineare
        iii) eta_k = O(||grad f(x_k)||)        -> quadratica
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


def _cg_truncated(H, b, eta_k, cg_max_iter):
    """CG interno per il sistema  H z = b,  con uscita anticipata sui tre
    casi descritti nelle slides:

        case i)   curvatura sempre positiva, esce per tolleranza Newton-inesatto
                  ||H z - b|| <= eta_k ||b||                  -> "tol"
        case ii)  curvatura non positiva gia' a j == 0,
                  restituisce z = b = -grad f(x_k)            -> "negcurv_j0"
        case iii) curvatura non positiva a j > 0,
                  restituisce l'ultimo z_j valido             -> "negcurv_jpos"
        (case iv) raggiunto il cap interno                    -> "maxiter"

    In tutti i casi la direzione restituita e' di discesa (vedi slides).

    Parametri
    ---------
    H            : ndarray (n, n), Hessiana corrente (simmetrica)
    b            : ndarray (n,),  = -grad f(x_k) = c_k
    eta_k        : float in (0,1), tolleranza relativa per il residuo CG
    cg_max_iter  : int, numero massimo di iterazioni interne

    Ritorna
    -------
    z            : ndarray (n,),  passo p_TN^(k)
    j            : int,           numero di iterazioni CG effettuate
    termination  : str,           uno tra "tol" | "negcurv_j0" | "negcurv_jpos" | "maxiter"
    """
    n = b.size
    z = np.zeros(n)
    r = b.copy()              # r_0 = b - H z_0 = b
    d = b.copy()              # d_0 = -grad f = b
    norm_b = float(np.linalg.norm(b))
    rTr = float(r @ r)

    if cg_max_iter <= 0 or norm_b == 0.0:
        return z, 0, "tol"   # gradiente nullo: gia' al minimo

    for j in range(cg_max_iter):
        Hd = H @ d
        dHd = float(d @ Hd)
        if dHd <= 0.0:
            if j == 0:
                # case ii: usiamo la direzione di steepest descent
                return b.copy(), 0, "negcurv_j0"
            # case iii: accettiamo l'ultimo z_j calcolato (descent direction)
            return z, j, "negcurv_jpos"

        alpha = rTr / dHd
        z = z + alpha * d
        r_new = r - alpha * Hd   # r_{j+1} = r_j - alpha_j H d_j

        if float(np.linalg.norm(r_new)) <= eta_k * norm_b:
            return z, j + 1, "tol"

        rTr_new = float(r_new @ r_new)
        beta = rTr_new / rTr
        d = r_new + beta * d
        r, rTr = r_new, rTr_new

    return z, cg_max_iter, "maxiter"


def truncated_newton(f, grad_f, hess_f, x0, stopping: StoppingCriterion,
                     alpha0=1.0, c1=1e-4, rho=0.5,
                     forcing="superlinear", eta_fixed=0.5,
                     cg_max_iter=None,
                     max_iter=1000, return_history=False):
    """Truncated Newton / Linesearch Newton-CG (slides NO4LSP) + Armijo.

    Loop ordering identico alle slides e a modified_newton: step (2.1-2.3)
    poi check (2.4). Il criterio di arresto e' un'istanza di
    StoppingCriterion (pattern plug-in).

    Parametri
    ---------
    f, grad_f, hess_f : F, gradiente, Hessiana (callables x -> ...)
    x0                : punto iniziale, ndarray (n,)
    stopping          : istanza di StoppingCriterion
    alpha0, c1, rho   : parametri della line search di Armijo (outer step)
    forcing           : modalita' di scelta di eta_k. Valori:
                          'linear'      -> eta_k = eta_fixed
                          'superlinear' -> eta_k = min(0.5, sqrt(||g_k||))
                          'quadratic'   -> eta_k = min(0.5, ||g_k||)
                          oppure un float in (0,1) (costante esplicita).
    eta_fixed         : valore costante per forcing='linear' (default 0.5).
    cg_max_iter       : numero massimo di iterazioni CG interne. None -> n
                        (dimensione del problema; classico in aritmetica esatta).
    max_iter          : numero massimo di iterazioni esterne.
    return_history    : se True, memorizza la traiettoria (utile per n=2).

    Ritorna
    -------
    result : dict con chiavi
        x_star, f_star, grad_norm, n_iter, success, stop_reason,
        cg_iters_total, neg_curvature_count,
        [history]: lista di dict {x, f, grad_norm, alpha, p_norm,
                                  cg_iters, nu_k, cg_termination}.
    """
    x = np.asarray(x0, dtype=float).copy()
    n = x.size
    if cg_max_iter is None:
        cg_max_iter = n

    history = []

    fx = f(x)
    g = grad_f(x)
    g_norm = float(np.linalg.norm(g))

    stopping.initialize(x, fx, g)

    if return_history:
        history.append({'x': x.copy(), 'f': fx, 'grad_norm': g_norm,
                        'alpha': None, 'p_norm': None,
                        'cg_iters': None, 'nu_k': None,
                        'cg_termination': None})

    cg_iters_total = 0
    neg_curv_count = 0
    success = False
    stop_reason = "max_iter"
    k = 0

    for k in range(max_iter):
        # 2.1-2.2: setup e inner CG per  B_k z = c_k
        H = hess_f(x)
        b = -g
        eta_k = _compute_eta(forcing, eta_fixed, g_norm)
        p, cg_j, cg_term = _cg_truncated(H, b, eta_k, cg_max_iter)
        cg_iters_total += cg_j
        if cg_term.startswith("negcurv"):
            neg_curv_count += 1

        # 2.3: backtracking + Armijo, poi update
        alpha, _ = armijo_backtracking(f, x, fx, g, p,
                                       alpha0=alpha0, c1=c1, rho=rho)
        x_prev, fx_prev = x.copy(), fx
        x = x + alpha * p
        fx = f(x)
        g = grad_f(x)
        g_norm = float(np.linalg.norm(g))

        if return_history:
            history.append({'x': x.copy(), 'f': fx, 'grad_norm': g_norm,
                            'alpha': alpha,
                            'p_norm': float(np.linalg.norm(p)),
                            'cg_iters': cg_j, 'nu_k': float(eta_k),
                            'cg_termination': cg_term})

        # 2.4: stop dopo l'update (k+1 per i criteri 'change')
        if stopping.should_stop(k + 1, x, fx, g, x_prev, fx_prev):
            success = True
            stop_reason = stopping.name
            break

    result = {
        'x_star': x,
        'f_star': fx,
        'grad_norm': g_norm,
        'n_iter': k + 1,
        'success': success,
        'stop_reason': stop_reason,
        'cg_iters_total': cg_iters_total,
        'neg_curvature_count': neg_curv_count,
    }
    if return_history:
        result['history'] = history
    return result
