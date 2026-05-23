"""Approssimazione della Hessiana via differenze finite del gradiente esatto.

Implementa l'idea della slide del corso (NO4LSP):
    H_f(x) = J_{grad f}(x)
quindi, avendo il gradiente esatto grad f, si applicano le formule di FD
per la jacobiana di grad f. Centrato (O(h^2)) o forward (O(h)).

Si impone simmetria  H <- (H + H^T) / 2  come indicato in slide.

Tre routine:

    hess_fd       — assemblaggio colonna-per-colonna (n perturbazioni
                    forward, 2n centrato). Baseline "general purpose"
                    usato come riferimento di unit-test.

    hess_fd_diag  — caso CPR a 1 colore per Hessiana diagonale (Problema
                    16). Una sola perturbazione lungo x ± h*1 estrae
                    l'intera diagonale, indipendentemente da n.

    hv_fd         — Hv matrix-free, per la CG interna del Truncated
                    Newton. Una chiamata al gradiente per applicazione.

Per ciascuna routine il passo h ha due varianti (richieste dall'assignment):
    fixed:   h    = 10^{-k}
    scaled:  h_i  = 10^{-k} * |x_i|  (con fallback se x_i = 0)
con k in {4, 8, 12}.
"""
import numpy as np


def _step(x, k, scaled):
    """Vettore dei passi h_i per ciascuna direzione e_i.

    fixed   -> h_i = 10^{-k}
    scaled  -> h_i = 10^{-k} * |x_i|  se x_i != 0, altrimenti 10^{-k}
               (stessa convenzione di src/gradients/finite_diff.py).
    """
    x = np.asarray(x, dtype=float)
    h_base = 10.0 ** (-k)
    if not scaled:
        return np.full(x.shape, h_base, dtype=float)
    h = h_base * np.abs(x)
    h[h == 0.0] = h_base
    return h


def hess_fd(grad_f, x, k=8, *, scaled=False, stencil="centered",
            symmetrize=True):
    """Approssima H(x) ~ J_{grad f}(x) colonna per colonna.

    Costo (chiamate a grad_f):
        forward  : n + 1
        centered : 2n

    Parametri
    ---------
    grad_f     : callable, x -> grad f(x), ndarray (n,)
    x          : ndarray (n,)
    k          : esponente del passo (h = 10^{-k})
    scaled     : se True usa h_i = 10^{-k} * |x_i| (fallback se x_i=0)
    stencil    : 'centered' (default, O(h^2)) o 'forward' (O(h))
    symmetrize : se True restituisce (H + H^T) / 2

    Ritorna
    -------
    H : ndarray (n, n)
    """
    x = np.asarray(x, dtype=float)
    n = x.size
    h = _step(x, k, scaled)
    H = np.empty((n, n), dtype=float)

    if stencil == "forward":
        g0 = grad_f(x)
        for j in range(n):
            xp = x.copy()
            xp[j] += h[j]
            H[:, j] = (grad_f(xp) - g0) / h[j]
    elif stencil == "centered":
        for j in range(n):
            xp = x.copy(); xp[j] += h[j]
            xm = x.copy(); xm[j] -= h[j]
            H[:, j] = (grad_f(xp) - grad_f(xm)) / (2.0 * h[j])
    else:
        raise ValueError(f"stencil must be 'forward' or 'centered', got {stencil!r}")

    if symmetrize:
        H = 0.5 * (H + H.T)
    return H


def hess_fd_diag(grad_f, x, k=8, *, scaled=False, stencil="centered"):
    """CPR a 1 colore: Hessiana diagonale via UNA perturbazione globale.

    Valido sotto l'assunzione (verificata in P16) che H sia diagonale,
    cioe' che ogni componente g_j di grad f dipenda solo da x_j. In tale
    caso, dalla espansione di Taylor:

        grad f(x + h_vec) - grad f(x - h_vec) ~ 2 * H * h_vec
        => diag(H)[j] = (g_j(x + h_vec) - g_j(x - h_vec)) / (2 * h_vec[j])

    Costo (chiamate a grad_f):
        forward  : 2  (ovvero 1 + 1 baseline)
        centered : 2
    INDIPENDENTEMENTE DA n.

    Ritorna
    -------
    diag : ndarray (n,)  — diagonale di H (rappresentazione compatta)
    """
    x = np.asarray(x, dtype=float)
    h = _step(x, k, scaled)

    if stencil == "forward":
        g0 = grad_f(x)
        gp = grad_f(x + h)
        return (gp - g0) / h
    if stencil == "centered":
        gp = grad_f(x + h)
        gm = grad_f(x - h)
        return (gp - gm) / (2.0 * h)
    raise ValueError(f"stencil must be 'forward' or 'centered', got {stencil!r}")


def hv_fd(grad_f, x, v, k=8, *, scaled=False, stencil="centered"):
    """Prodotto Hessiana-vettore matrix-free.

        H(x) v  ~  (grad f(x + h v) - grad f(x - h v)) / (2 h)   (centered)
        H(x) v  ~  (grad f(x + h v) - grad f(x))       / h        (forward)

    Niente assemblaggio di H. Una/due chiamate a grad_f per Hv.
    Usato dalla CG interna del Truncated Newton.

    Sul passo h (scalare, non vettoriale, perche' la perturbazione e'
    lungo una direzione v generica):
        fixed   -> h = 10^{-k}
        scaled  -> h = 10^{-k} * max(||x||_inf, 1)   (riscalamento globale,
                   evita problemi quando v ha norma molto diversa da 1)

    Parametri
    ---------
    grad_f  : callable, x -> grad f(x)
    x       : ndarray (n,)
    v       : ndarray (n,) direzione
    k       : esponente del passo
    scaled  : variante riscalata sul punto x
    stencil : 'centered' o 'forward'

    Ritorna
    -------
    Hv : ndarray (n,)
    """
    x = np.asarray(x, dtype=float)
    v = np.asarray(v, dtype=float)
    h_base = 10.0 ** (-k)
    h = h_base * max(np.max(np.abs(x)), 1.0) if scaled else h_base

    if stencil == "forward":
        return (grad_f(x + h * v) - grad_f(x)) / h
    if stencil == "centered":
        return (grad_f(x + h * v) - grad_f(x - h * v)) / (2.0 * h)
    raise ValueError(f"stencil must be 'forward' or 'centered', got {stencil!r}")
