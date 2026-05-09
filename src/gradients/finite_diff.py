"""Approssimazione del gradiente via differenze finite forward (generico).

Due varianti del passo:
    - "fixed":  h_i = 10^{-k}
    - "scaled": h_i = 10^{-k} * |x_i|   (con fallback al fixed se x_i=0)

Tipicamente k in {4, 8, 12} (richiesto dall'assignment).

Versione generica O(n) chiamate a f. Per n grande conviene una versione
"sparse" che sfrutta il fatto che, per Prob.16 e Prob.32, ogni componente
x_j compare in al massimo 3 termini consecutivi della somma — vedi
PLAN.md sezione "Implementazione efficiente".
"""
import numpy as np


def grad_fd_forward(f, x, k=8, scaled=False):
    """Gradiente forward-difference (O(n) chiamate a f).

        df/dx_i ≈ [f(x + h_i e_i) - f(x)] / h_i

    Parametri
    ---------
    f      : callable, F(x) -> scalare
    x      : ndarray (n,)
    k      : esponente del passo (h = 10^{-k})
    scaled : se True usa h_i = 10^{-k} * |x_i| (con fallback se x_i=0)

    Ritorna
    -------
    g_fd   : ndarray (n,) approssimazione di grad F(x)
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    fx = f(x)
    g = np.zeros(n)
    h_base = 10.0 ** (-k)
    for i in range(n):
        h = h_base * abs(x[i]) if (scaled and x[i] != 0.0) else h_base
        x_pert = x.copy()
        x_pert[i] += h
        g[i] = (f(x_pert) - fx) / h
    return g
