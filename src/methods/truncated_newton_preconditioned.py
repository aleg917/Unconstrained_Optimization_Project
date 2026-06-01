"""Preconditioned Truncated Newton (Linesearch Newton-CG con precondizionamento).

Questo file e' una copia di ``truncated_newton.py`` a cui e' stato aggiunto un
PRECONDIZIONAMENTO del sistema interno risolto dal CG.  Tutto il resto
(forcing eta_k, Armijo, history, criterio di arresto, gestione del time limit,
dizionario dei risultati) e' identico al Truncated Newton di base: per questo
riusiamo direttamente ``_compute_eta`` e ``_cg_truncated`` importandoli da
``truncated_newton`` (il solutore CG resta cosi' provatamente lo stesso).

Idea del precondizionamento "split" (two-sided) con Cholesky (incompleto).
Invece di risolvere

    H d = -g

con il CG (numero di condizionamento k(H), potenzialmente enorme: per il
Problema 28 H = I + c j j^T e' molto mal condizionata), risolviamo il sistema
TRASFORMATO

    (L^-1 H L^-T) d~ = L^-1 (-g),      poi     d = L^-T d~,

dove L e' il fattore (incompleto) di Cholesky di H, cioe' M = L L^T ~ H.
L'operatore L^-1 H L^-T ha numero di condizionamento molto piu' piccolo (= 1
quando L L^T = H esattamente), quindi il CG converge in pochissime iterazioni.

Perche' possiamo riusare ``_cg_truncated`` SENZA modificarlo: la trasformazione
e' una CONGRUENZA,

    w^T (L^-1 H L^-T) w = (L^-T w)^T H (L^-T w),

quindi preserva il SEGNO della curvatura.  La logica di curvatura negativa del
CG (che decide se proseguire o fermarsi) resta percio' valida tale e quale.

Quando il precondizionatore NON e' costruibile (percorso matrix-free hess_f=None:
non esiste alcuna matrice da fattorizzare; oppure Hessiana non definita positiva)
si emette un warning e si ricade sul CG NON precondizionato, esattamente come il
Truncated Newton di base.
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

    Stessa firma e stessi modi d'uso di ``truncated_newton``.  L'unica
    differenza e' il precondizionamento del solutore CG interno (vedi il
    docstring del modulo).

    Costruzione del precondizionatore (solo se e' disponibile una Hessiana
    ASSEMBLATA, cioe' hess_f e' fornita e restituisce una matrice):
        - Hessiana diagonale/sparsa (Problema 16): IC esatto e banale,
          L = diag(sqrt(diag(H))).  Richiede diag(H) > 0.
        - Hessiana densa (Problema 28): Cholesky completo
          ``scipy.linalg.cholesky(H, lower=True)`` (H28 e' SPD, quindi il
          fattore e' esatto; scelto al posto di spilu perche' piu' semplice).
    Fallback (warning + CG non precondizionato):
        - hess_f is None  -> percorso matrix-free, nessuna matrice da
          fattorizzare;
        - fattorizzazione fallita (Hessiana non definita positiva).

    Parameters
    ----------
    f                 : objective function, callable x -> scalar
    x0                : starting point, ndarray (n,)
    stopping          : instance of StoppingCriterion
    grad_f            : gradient callable x -> ndarray (n,).  If None,
                        computed via centered FD of f using (k, scaled).
    hess_f            : Hessian callable x -> matrix or diagonal.  If None,
                        Hv products computed via hv_func (matrix-free) e il
                        precondizionamento e' disattivato (fallback a TN).
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
    # Pubblica (t_start, time_limit) in uno slot a livello di modulo cosi' che le
    # routine alle differenze finite (grad_fd / hv_fd, prive di argomenti di tempo)
    # possano leggerlo e sollevare TimeLimitExceeded a meta' valutazione.  Viene
    # azzerato nel blocco finally piu' sotto.
    set_time_budget(t_start, time_limit)

    # --- resolve FD callables ------------------------------------------------
    # Se non e' fornito un gradiente esatto lo si sostituisce con un gradiente FD
    # centrato, congelando (k, scaled) nella closure.  hess_f NON viene risolta
    # qui: quando e' None il ciclo sotto usa il percorso matrix-free (hv_func) e
    # il precondizionamento resta disattivato.
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
    _pc_warned = False   # emette il warning di fallback UNA sola volta per run

    def _warn_once(msg):
        # Evita di ripetere lo stesso warning a ogni iterazione esterna.
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

        # Passa al criterio lo stato iniziale una sola volta, prima del ciclo,
        # cosi' i test relativi possono memorizzare i valori di riferimento (es.
        # la norma iniziale del gradiente ||g_0|| usata per scalare una
        # tolleranza relativa).
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

            # 2.1-2.2: costruisci l'operatore apply_H (H(x) d) e, se possibile,
            # i due "solve" triangolari del precondizionatore  Linv = L^-1,
            # LinvT = L^-T  (con M = L L^T ~ H). La fattorizzazione avviene UNA
            # volta per iterazione esterna (H cambia con x): stessa struttura di
            # costo del Modified Newton.
            preconditioned = False
            Linv = LinvT = None

            if hess_f is not None:
                H_result = hess_f(x)
                if isinstance(H_result, np.ndarray) and H_result.ndim == 1:
                    # Hessiana diagonale fornita come vettore 1D.
                    diag = H_result
                    apply_H = lambda d, _diag=diag: _diag * d
                    if np.all(diag > 0.0):
                        # IC esatto: L = diag(sqrt(diag)); essendo L diagonale e
                        # simmetrica, L^-1 = L^-T = divisione per sqrt(diag).
                        sd = np.sqrt(diag)
                        Linv = LinvT = lambda w, _sd=sd: w / _sd
                        preconditioned = True
                    else:
                        _warn_once("PTN: diagonale della Hessiana non positiva; "
                                   "fallback a CG non precondizionato.")
                elif sp.issparse(H_result):
                    # Hessiana sparsa diagonale (Problema 16): ne estraiamo la
                    # diagonale, come fa il TN di base.
                    diag = np.asarray(H_result.diagonal(), dtype=float)
                    apply_H = lambda d, _diag=diag: _diag * d
                    if np.all(diag > 0.0):
                        sd = np.sqrt(diag)
                        Linv = LinvT = lambda w, _sd=sd: w / _sd
                        preconditioned = True
                    else:
                        _warn_once("PTN: diagonale della Hessiana non positiva; "
                                   "fallback a CG non precondizionato.")
                else:
                    # Hessiana densa (Problema 28). apply_H come matmul (avvolto in
                    # una lambda cosi' e' richiamabile anche dentro A_tilde; il
                    # prodotto e' numericamente identico a quello del TN base).
                    apply_H = lambda d, _H=H_result: _H @ d
                    try:
                        # H28 = I + c j j^T e' SPD (c >= 1) -> Cholesky esatto.
                        L = sla.cholesky(H_result, lower=True)
                        Linv = lambda w, _L=L: sla.solve_triangular(_L, w, lower=True)
                        LinvT = lambda w, _L=L: sla.solve_triangular(_L.T, w, lower=False)
                        preconditioned = True
                    except sla.LinAlgError:
                        _warn_once("PTN: Cholesky fallita (Hessiana non definita "
                                   "positiva); fallback a CG non precondizionato.")
            else:
                # Percorso matrix-free (hess_f=None): nessuna matrice da
                # fattorizzare -> impossibile precondizionare, ci si comporta
                # come il Truncated Newton di base.
                x_k = x
                _k, _scaled, _hv = k, scaled, hv_func
                apply_H = lambda d, _x=x_k: _hv(grad_f, _x, d, k=_k, scaled=_scaled)
                _warn_once("PTN: hess_f=None (modalita' matrix-free), nessuna "
                           "matrice da precondizionare; fallback a CG non "
                           "precondizionato.")

            b = -g
            eta_k = _compute_eta(forcing, eta_fixed, g_norm)

            if preconditioned:
                # Operatore precondizionato A~(w) = L^-1 ( H ( L^-T w ) ) e termine
                # noto b~ = L^-1 (-g).  Riusiamo _cg_truncated INVARIATO su (A~, b~)
                # e poi riportiamo la direzione nello spazio originale: d = L^-T d~.
                A_tilde = lambda w: Linv(apply_H(LinvT(w)))
                b_tilde = Linv(b)
                d_tilde, cg_j, cg_term = _cg_truncated(A_tilde, b_tilde, eta_k,
                                                       cg_max_iter,
                                                       t_start=t_start,
                                                       time_limit=time_limit)
                # Su curvatura negativa a j==0, _cg_truncated ritorna b~; allora
                # p = L^-T b~ = M^-1(-g), ancora direzione di discesa
                # (g^T p = -g^T M^-1 g < 0 perche' M = L L^T e' SPD).
                p = LinvT(d_tilde)
            else:
                p, cg_j, cg_term = _cg_truncated(apply_H, b, eta_k, cg_max_iter,
                                                 t_start=t_start, time_limit=time_limit)

            cg_iters_total += cg_j
            if cg_term.startswith("negcurv"):   # il CG ha incontrato curvatura non positiva
                neg_curv_count += 1
            if cg_term == "time_limit":          # il CG stesso ha esaurito il budget: stop
                stop_reason = "time_limit"
                break

            # 2.3: l'Armijo backtracking sceglie il passo alpha lungo p, poi aggiorna
            alpha, n_bt = armijo_backtracking(f, x, fx, g, p,
                                              alpha0=alpha0, c1=c1, rho=rho,
                                              max_iter=max_iter_backtrack,
                                              t_start=t_start, time_limit=time_limit)
            n_backtrack_total += n_bt
            # Controllo del tempo PRIMA di applicare il passo: se il budget e'
            # finito durante la line search, armijo_backtracking puo' aver
            # restituito un alpha non verificato, quindi ci fermiamo mantenendo
            # l'iterato precedente (buono) invece di compiere un passo che
            # potrebbe non diminuire f.  Il tempo speso dal refresh di f/gradiente
            # qui sotto e' intercettato dal controllo in cima all'iterazione
            # successiva, percio' non serve un secondo controllo qui.
            if time_limit is not None and (time.perf_counter() - t_start) > time_limit:
                stop_reason = "time_limit"
                break

            x_prev, fx_prev = x.copy(), fx     # ricorda l'iterato precedente per il test di arresto
            x = x + alpha * p                  # x^(k+1) = x^(k) + alpha * p
            fx = f(x)                          # ricalcola f, gradiente e norma del gradiente in x
            g = grad_f(x)
            g_norm = float(np.linalg.norm(g))

            if return_history:
                history.append({'x': x.copy(), 'f': fx, 'grad_norm': g_norm,
                                'alpha': alpha,
                                'p_norm': float(np.linalg.norm(p)),
                                'cg_iters': cg_j, 'nu_k': float(eta_k),
                                'cg_termination': cg_term})

            # 2.4: verifica il criterio di convergenza sul nuovo iterato.  Restituisce
            #      True quando il suo test e' soddisfatto (es. ||g|| sotto tolleranza,
            #      o piccola variazione di x o di f tra iterazioni).  Un True qui e'
            #      l'UNICA uscita di vera convergenza: success=True e stop_reason
            #      registra quale criterio e' scattato (il suo .name), a differenza
            #      delle uscite "time_limit"/"max_iter" che lasciano success=False.
            if stopping.should_stop(k_iter + 1, x, fx, g, x_prev, fx_prev):
                success = True
                stop_reason = stopping.name
                break

    # Due modi in cui il budget puo' terminare la run: i controlli inline
    # `if ... break` qui sopra (che impostano stop_reason e escono dal ciclo
    # normalmente) e TimeLimitExceeded sollevata in profondita' dentro una
    # valutazione FD del gradiente / prodotto Hessiana-vettore, che risale fin
    # qui.  In entrambi i casi success=False e in x resta l'ultimo iterato completo.
    except TimeLimitExceeded:
        stop_reason = "time_limit"
    finally:
        clear_time_budget()   # rilascia sempre il budget condiviso a livello di modulo

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
