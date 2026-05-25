"""Modified Newton + Armijo Backtracking line search.

Implementazione delle slides del corso NO4LSP (Politecnico di Torino, A.Y.
2025/2026). Le slides citano esplicitamente "Alg. 6.3 in 1st ed. of [NW],
Alg. 3.3 in the 2nd ed." come riferimento, quindi la procedura coincide
matematicamente con quella di Nocedal-Wright; qui seguiamo la notazione e
l'ordering della professoressa.

Schema (slides):
    Given x^(0). For k = 0, 1, 2, ...:
        2.1  Build  B_k = grad^2 f(x^(k)) + E_k
                    con E_k = 0 se grad^2 f e' "sufficientemente" pos. def.,
                    altrimenti E_k = tau_k I scelto per garantirlo.
        2.2  Solve  B_k p_MN^(k) = -grad f(x^(k))
        2.3  x^(k+1) = x^(k) + alpha^(k) p_MN^(k),
             alpha^(k) da backtracking + Armijo.
        2.4  Stop if a stopping criterion is satisfied.

Il criterio di arresto e' una strategia plug-in (vedi src.stopping_criteria)
identica a steepest_descent; la line search e' Armijo backtracking, riusata
direttamente dal modulo steepest_descent.
"""
import numpy as np
import scipy.linalg as sla

from ..stopping_criteria.base import StoppingCriterion
from .armijo_backtracking import armijo_backtracking


def _modify_hessian_cholesky(H, beta=1e-3, max_iter=100):
    """Algoritmo dalle slides per costruire B_k (Cholesky with added multiple
    of identity; cfr. Nocedal-Wright Alg. 6.3 / 3.3).

    Sia H = grad^2 f(x^(k)). L'algoritmo cerca tau_k >= 0 minimo tale che
    H + tau_k I sia (sufficientemente) positiva definita, sfruttando il
    fallimento della Cholesky come oracle di non-definita-positivita'.

    Inizializzazione di tau (slides):
        se [H]_ii > 0 per ogni i:        tau_k^(0) = 0
        altrimenti:                       tau_k^(0) = beta - min_i [H]_ii

    Iterazione (slides):
        prova Cholesky R^T R = H + tau_k^(j) I
        se ok:    B_k = H + tau_k^(j) I,  fine
        se fail:  tau_k^(j+1) = max(2 tau_k^(j), beta)

    Parametri
    ---------
    H        : ndarray (n, n), Hessiana corrente (assunta simmetrica)
    beta     : parametro euristico, valore tipico 1e-3 (cfr. slides)
    max_iter : numero massimo di raddoppi di tau

    Ritorna
    -------
    R     : ndarray (n, n) triangolare SUPERIORE tale che R^T R = H + tau I
    tau   : multiplo di identita' effettivamente aggiunto
    n_adj : numero di raddoppi di tau effettuati (0 se H gia' definita positiva)

    Solleva
    -------
    RuntimeError se nessun tau entro max_iter rende H + tau*I definita positiva.
    """
    diag_min = float(np.min(np.diag(H)))
    if diag_min > 0.0:
        tau = 0.0
    else:
        # tau_k^(0) = beta - min_i [H]_ii  (con min <= 0, equivalente a |min| + beta)
        tau = beta - diag_min

    n = H.shape[0]
    I = np.eye(n)
    for j in range(max_iter):
        try:
            R = sla.cholesky(H + tau * I, lower=False)  # R^T R = H + tau I
            return R, tau, j
        except sla.LinAlgError:
            tau = max(2.0 * tau, beta)

    raise RuntimeError(
        f"_modify_hessian_cholesky: nessun tau entro {max_iter} iterazioni "
        f"rende H + tau*I definita positiva (ultimo tau={tau:.3e})."
    )


def modified_newton(f, grad_f, hess_f, x0, stopping: StoppingCriterion,
                    alpha0=1.0, c1=1e-4, rho=0.5,
                    beta=1e-3, max_tau_iter=100,
                    max_iter=1000, max_iter_backtrack=50,
                    return_history=False):
    """Modified Newton (slides NO4LSP) + Armijo backtracking.

    Loop ordering identico alle slides: step (2.1-2.3) -> check (2.4).
    Il criterio di arresto e' passato come istanza di StoppingCriterion; il
    metodo non sa quale tipo sia (pattern plug-in).

    Parametri
    ---------
    f, grad_f, hess_f : F, gradiente, Hessiana (callables x -> ...)
    x0                : punto iniziale, ndarray (n,)
    stopping          : istanza di StoppingCriterion
    alpha0,c1,rho     : parametri della line search di Armijo
    beta              : parametro euristico di _modify_hessian_cholesky
    max_tau_iter      : numero massimo di raddoppi di tau per ogni iterazione
    max_iter          : numero massimo di iterazioni esterne (safety)
    return_history    : se True, memorizza la traiettoria (utile per n=2)

    Ritorna
    -------
    result : dict con chiavi
        x_star, f_star, grad_norm, n_iter, success, stop_reason,
        n_chol_adjustments_total,
        [history]: lista di dict {x, f, grad_norm, alpha, tau, p_norm}.
    """
    x = np.asarray(x0, dtype=float).copy()
    history = []

    fx = f(x)
    g = grad_f(x)
    g_norm = float(np.linalg.norm(g))

    stopping.initialize(x, fx, g)

    if return_history:
        history.append({'x': x.copy(), 'f': fx, 'grad_norm': g_norm,
                        'alpha': None, 'tau': None, 'p_norm': None})

    n_chol_total = 0
    success = False
    stop_reason = "max_iter"
    k = 0

    for k in range(max_iter):
        # 2.1: build B_k via Cholesky con regolarizzazione
        H = hess_f(x)
        R, tau, n_adj = _modify_hessian_cholesky(H, beta=beta,
                                                 max_iter=max_tau_iter)
        n_chol_total += n_adj

        # 2.2: solve  R^T R p_MN = -g  via due solve triangolari
        p = sla.cho_solve((R, False), -g)

        # 2.3: backtracking + Armijo, poi update
        alpha, _ = armijo_backtracking(f, x, fx, g, p,
                                       alpha0=alpha0, c1=c1, rho=rho,
                                       max_iter=max_iter_backtrack)
        x_prev, fx_prev = x.copy(), fx
        x = x + alpha * p
        fx = f(x)
        g = grad_f(x)
        g_norm = float(np.linalg.norm(g))

        if return_history:
            history.append({'x': x.copy(), 'f': fx, 'grad_norm': g_norm,
                            'alpha': alpha, 'tau': tau,
                            'p_norm': float(np.linalg.norm(p))})

        # 2.4: stop dopo l'update (passiamo k+1 cosi' i criteri 'change'
        # vedono k>=1 e i guard "k==0" restano dormienti)
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
        'n_chol_adjustments_total': n_chol_total,
    }
    if return_history:
        result['history'] = history
    return result
