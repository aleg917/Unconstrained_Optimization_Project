"""Steepest Descent + Armijo Backtracking line search.

Il criterio di arresto e' una *strategia plug-in*: si passa un'istanza di
`StoppingCriterion` (vedi `src.stopping_criteria`) e il metodo non sa ne'
si interessa di quale criterio sta usando. Questo permette di confrontare
sperimentalmente i sei criteri (3 assoluti + 3 relativi) sullo stesso run.
`max_iter` resta come safety net universale.
"""
import numpy as np

from ..stopping_criteria.base import StoppingCriterion


def armijo_backtracking(f, x, fx, g, d, alpha0=1.0, c1=1e-4, rho=0.5,
                        max_iter=50):
    """Backtracking line search (Armijo).

    Trova alpha tale che  f(x + alpha*d) <= fx + c1*alpha*<g, d>.

    Parametri
    ---------
    f         : callable, F(x) -> scalare
    x         : punto corrente, ndarray (n,)
    fx        : valore F(x) gia' calcolato
    g         : gradiente in x, ndarray (n,)
    d         : direzione di discesa, ndarray (n,)  (di solito -g)
    alpha0    : passo iniziale (default 1, "Newton step")
    c1        : costante di Armijo, in (0, 1/2)
    rho       : fattore di riduzione, in (0, 1)
    max_iter  : numero massimo di backtrack

    Ritorna
    -------
    alpha     : passo accettato
    n_back    : numero di riduzioni effettuate
    """
    g_dot_d = float(g @ d)   # tipicamente negativo per d = -g
    alpha = alpha0
    for k in range(max_iter):
        if f(x + alpha * d) <= fx + c1 * alpha * g_dot_d:
            return alpha, k
        alpha *= rho
    return alpha, max_iter


def steepest_descent(f, grad_f, x0, stopping: StoppingCriterion,
                     alpha0=1.0, c1=1e-4, rho=0.5,
                     max_iter=1000, return_history=False):
    """Discesa piu' ripida con backtracking di Armijo e stopping plug-in.

    Parametri
    ---------
    f, grad_f      : F e gradiente esatto/approssimato
    x0             : punto iniziale, ndarray (n,)
    stopping       : istanza di StoppingCriterion (es. GradNormAbsolute(1e-6))
    alpha0,c1,rho  : parametri della line search
    max_iter       : numero massimo di iterazioni (safety)
    return_history : se True, memorizza la traiettoria (utile per n=2)

    Ritorna
    -------
    result : dict con chiavi
        x_star, f_star, grad_norm, n_iter, success, stop_reason,
        [history]: lista di dict {x, f, grad_norm, alpha} per ogni passo.
    """
    x = np.asarray(x0, dtype=float).copy()
    history = []

    fx = f(x)
    g = grad_f(x)
    g_norm = float(np.linalg.norm(g))

    stopping.initialize(x, fx, g)

    if return_history:
        history.append({'x': x.copy(), 'f': fx,
                        'grad_norm': g_norm, 'alpha': None})

    x_prev, fx_prev = x.copy(), fx
    success = False
    stop_reason = "max_iter"
    k = 0

    for k in range(max_iter):
        if stopping.should_stop(k, x, fx, g, x_prev, fx_prev):
            success = True
            stop_reason = stopping.name
            break

        d = -g
        alpha, _ = armijo_backtracking(f, x, fx, g, d,
                                       alpha0=alpha0, c1=c1, rho=rho)

        x_prev, fx_prev = x.copy(), fx
        x = x + alpha * d
        fx = f(x)
        g = grad_f(x)
        g_norm = float(np.linalg.norm(g))

        if return_history:
            history.append({'x': x.copy(), 'f': fx,
                            'grad_norm': g_norm, 'alpha': alpha})

    result = {
        'x_star': x,
        'f_star': fx,
        'grad_norm': g_norm,
        'n_iter': k + 1,
        'success': success,
        'stop_reason': stop_reason,
    }
    if return_history:
        result['history'] = history
    return result
