import numpy as np


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