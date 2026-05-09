"""Steepest Descent + Armijo Backtracking line search.

NB: scheletro/riferimento — i loop sono completi ma puoi sostituire i
parametri o le condizioni di arresto a seconda di cosa vuoi discutere
nella relazione (vedi PLAN.md).
"""
import numpy as np


def armijo_backtracking(f, x, fx, g, d, alpha0=1.0, c1=1e-4, rho=0.5,
                        max_iter=50):
    """Backtracking line search (Armijo).

    Trova alpha tale che  f(x + alpha*d) <= fx + c1*alpha*<g, d>.

    Parametri
    ---------
    f         : callable, F(x) -> scalare
    x         : punto corrente, ndarray (n,)
    fx        : valore F(x) gia calcolato
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


def steepest_descent(f, grad_f, x0, alpha0=1.0, c1=1e-4, rho=0.5,
                     tol=1e-6, tol_rel=None, max_iter=1000,
                     return_history=False):
    """Discesa più ripida con backtracking di Armijo.

    Parametri
    ---------
    f, grad_f      : F e gradiente esatto/approssimato
    x0             : punto iniziale, ndarray (n,)
    alpha0,c1,rho  : parametri della line search
    tol            : tolleranza assoluta su ||grad||
    tol_rel        : tolleranza relativa su ||grad|| / ||grad(x0)||
                     (None => disabilitata)
    max_iter       : numero massimo di iterazioni
    return_history : se True, memorizza la traiettoria (utile per n=2)

    Ritorna
    -------
    result : dict con chiavi
        x_star, f_star, grad_norm, n_iter, success, message,
        [history]: lista di dict {x, f, grad_norm, alpha} per ogni passo.
    """
    x = np.asarray(x0, dtype=float).copy()
    history = []

    fx = f(x)
    g = grad_f(x)
    g_norm0 = float(np.linalg.norm(g))
    g_norm = g_norm0

    if return_history:
        history.append({'x': x.copy(), 'f': fx,
                        'grad_norm': g_norm, 'alpha': None})

    success = False
    message = "max_iter raggiunto"
    k = 0

    for k in range(max_iter):
        # criteri di arresto sul gradiente
        if g_norm <= tol:
            success = True
            message = f"||grad|| <= tol={tol:g}"
            break
        if tol_rel is not None and g_norm <= tol_rel * g_norm0:
            success = True
            message = f"||grad||/||grad0|| <= tol_rel={tol_rel:g}"
            break

        d = -g
        alpha, _ = armijo_backtracking(f, x, fx, g, d,
                                       alpha0=alpha0, c1=c1, rho=rho)
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
        'message': message,
    }
    if return_history:
        result['history'] = history
    return result
