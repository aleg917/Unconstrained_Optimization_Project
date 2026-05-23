"""Sweep di sensibilita' al parametro k della FD per la Hessiana / Hv.

Per ciascun (problema, n, metodo, stencil, scaled, k) lanciamo il solver
con criterio di arresto ||grad||<1e-6 e tetto di iterazioni alto, poi
registriamo iterazioni, successo, norma finale, tempo.

Lo scopo e' identificare:
  - se compare la "U-curve" (k=4 dominato dalla troncatura, k=8 ottimo,
    k=12 dominato dall'arrotondamento)
  - se scaled h_i = 10^{-k}|x_i| aiuta su P28 (gradiente a magnitudo
    eterogenea)
  - se esiste una combinazione che permetta a Truncated Newton matrix-free
    di convergere a n >= 1000 su P28

Run:
    python scripts/sweep_k_sensitivity.py
"""
import os
import sys
import time

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.functions import f16, f28, x_bar_16, x_bar_28
from src.gradients import grad_f16, grad_f28
from src.hessians import hess_fd, hess_fd_diag, hv_fd
from src.methods.modified_newton import modified_newton
from src.methods.truncated_newton import truncated_newton
from src.stopping_criteria import GradNormAbsolute


PROBLEMS = {
    'P16': {
        'f': f16, 'grad': grad_f16,
        'x0': lambda n: x_bar_16(n),
        'sizes': [2, 100, 1000],
        'description': 'Banded Trigonometric (Hessiana diagonale, 1 colore CPR)',
    },
    'P28': {
        'f': f28, 'grad': grad_f28,
        # Per P28 x_bar_28 e' divergente a n grande; perturbazione piccola
        # vicino al minimo x*=1 (cfr. bench_tn_matrix_free.py)
        'x0': lambda n: np.ones(n) + 0.001 * np.random.default_rng(1234).standard_normal(n),
        'sizes': [10, 100, 1000],
        'description': 'Variably Dimensioned (Hessiana densa, no CPR)',
    },
}

K_VALUES = [4, 8, 12]
STENCILS = ['forward', 'centered']
SCALED = [False, True]
TOL = 1e-6
MAX_ITER = 300


def fmt_row(method, prob, n, k, scaled, stencil, res, t):
    return (f"  {method:10s} {prob:>4s} n={n:<5d} k={k:>2d} "
            f"{stencil:8s} scaled={str(scaled):5s} | "
            f"iter={res['n_iter']:>3d} ok={str(res['success']):>5s} "
            f"||g||={res['grad_norm']:>10.2e}  t={t:>6.2f}s")


def sweep_modified_newton(prob_id):
    pinfo = PROBLEMS[prob_id]
    print(f"\n--- Modified Newton + hess_fd  on {prob_id} ({pinfo['description']}) ---")
    for n in pinfo['sizes']:
        x0 = pinfo['x0'](n)
        stop = GradNormAbsolute(tol=TOL)
        for stencil in STENCILS:
            for scaled in SCALED:
                for k in K_VALUES:
                    hess_cb = lambda x, _k=k, _s=scaled, _st=stencil: \
                        hess_fd(pinfo['grad'], x, k=_k, scaled=_s, stencil=_st)
                    t0 = time.perf_counter()
                    try:
                        res = modified_newton(
                            pinfo['f'], pinfo['grad'], hess_cb, x0,
                            stopping=stop, max_iter=MAX_ITER)
                        t = time.perf_counter() - t0
                        print(fmt_row('MOD-NEWT', prob_id, n, k, scaled, stencil, res, t))
                    except Exception as e:
                        t = time.perf_counter() - t0
                        print(f"  MOD-NEWT   {prob_id:>4s} n={n:<5d} k={k:>2d} "
                              f"{stencil:8s} scaled={str(scaled):5s} | "
                              f"EXCEPTION {type(e).__name__}: {e!s:.60s}  t={t:.2f}s")


def sweep_truncated_newton(prob_id):
    pinfo = PROBLEMS[prob_id]
    print(f"\n--- Truncated Newton + hv_fd (matrix-free)  on {prob_id} ---")
    for n in pinfo['sizes']:
        x0 = pinfo['x0'](n)
        stop = GradNormAbsolute(tol=TOL)
        for stencil in STENCILS:
            for scaled in SCALED:
                for k in K_VALUES:
                    hv_cb = lambda x, v, _k=k, _s=scaled, _st=stencil: \
                        hv_fd(pinfo['grad'], x, v, k=_k, scaled=_s, stencil=_st)
                    t0 = time.perf_counter()
                    try:
                        res = truncated_newton(
                            pinfo['f'], pinfo['grad'], x0=x0,
                            stopping=stop, max_iter=MAX_ITER,
                            hv_func=hv_cb)
                        t = time.perf_counter() - t0
                        print(fmt_row('TRUNC-N', prob_id, n, k, scaled, stencil, res, t))
                    except Exception as e:
                        t = time.perf_counter() - t0
                        print(f"  TRUNC-N    {prob_id:>4s} n={n:<5d} k={k:>2d} "
                              f"{stencil:8s} scaled={str(scaled):5s} | "
                              f"EXCEPTION {type(e).__name__}: {e!s:.60s}  t={t:.2f}s")


if __name__ == '__main__':
    print("=" * 110)
    print("K-SENSITIVITY SWEEP")
    print(f"tol={TOL:.0e}  max_iter={MAX_ITER}")
    print("=" * 110)

    for prob_id in ('P16', 'P28'):
        sweep_modified_newton(prob_id)

    for prob_id in ('P16', 'P28'):
        sweep_truncated_newton(prob_id)
