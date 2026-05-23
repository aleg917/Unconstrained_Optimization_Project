"""Benchmark: Truncated Newton — ramo classico vs matrix-free.

Due test su Problema 28 (Hessiana densa):
  A) PER-ITER COST: 10 iter da x_bar_28(n) — confronto tempo/memoria
     (la convergenza non e' lo scopo qui; x_bar_28 a n grande e' un punto
     molto duro per P28).
  B) CONVERGENZA: da x0 = ones(n) + perturbazione, fino al criterio
     ||grad||<1e-6. Verifica che il ramo matrix-free produca la stessa
     traiettoria del ramo esatto.

Run:
    python scripts/bench_tn_matrix_free.py
"""
import os
import sys
import time
import tracemalloc

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.functions import f28, x_bar_28
from src.gradients import grad_f28
from src.hessians import hess_f28, hv_fd
from src.methods.truncated_newton import truncated_newton
from src.stopping_criteria import GradNormAbsolute


def bench_one(label, fn):
    tracemalloc.start()
    t0 = time.perf_counter()
    res = fn()
    t1 = time.perf_counter()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    n_iter = res['n_iter']
    cg = res['cg_iters_total']
    return {
        'label': label,
        'time_s': t1 - t0,
        'peak_MB': peak / (1024 ** 2),
        'n_iter': n_iter,
        'cg_iters_total': cg,
        'success': res['success'],
        'grad_norm': res['grad_norm'],
    }


def run_for_n(n, max_outer=10):
    stop = GradNormAbsolute(tol=1e-6)
    x0 = x_bar_28(n)
    rows = []

    # ramo classico — proviamo solo se n e' "ragionevole" (n^2 memoria)
    if n <= 10000:
        rows.append(bench_one(
            f"n={n:>6d}  EXACT     ",
            lambda: truncated_newton(f28, grad_f28, hess_f28, x0,
                                     stopping=stop, max_iter=max_outer,
                                     cg_max_iter=min(n, 100)),
        ))
    else:
        rows.append({'label': f"n={n:>6d}  EXACT     ",
                     'time_s': float('nan'), 'peak_MB': float('nan'),
                     'n_iter': 0, 'cg_iters_total': 0,
                     'success': False, 'grad_norm': float('nan')})

    # ramo matrix-free — sempre
    hv = lambda x, v: hv_fd(grad_f28, x, v, k=8, stencil="centered")
    rows.append(bench_one(
        f"n={n:>6d}  MATRIX-FRE",
        lambda: truncated_newton(f28, grad_f28, x0=x0, stopping=stop,
                                 max_iter=max_outer,
                                 cg_max_iter=min(n, 100),
                                 hv_func=hv),
    ))
    return rows


def run_convergence(n, seed=1234):
    """Da un punto ben condizionato, fino a convergenza. Confronta i due rami."""
    rng = np.random.default_rng(seed)
    x0 = np.ones(n) + 0.1 * rng.standard_normal(n)
    stop = GradNormAbsolute(tol=1e-6)

    rows = []
    if n <= 10000:
        rows.append(bench_one(
            f"n={n:>6d}  EXACT     ",
            lambda: truncated_newton(f28, grad_f28, hess_f28, x0,
                                     stopping=stop, max_iter=200),
        ))

    hv = lambda x, v: hv_fd(grad_f28, x, v, k=8, stencil="centered")
    rows.append(bench_one(
        f"n={n:>6d}  MATRIX-FRE",
        lambda: truncated_newton(f28, grad_f28, x0=x0, stopping=stop,
                                 max_iter=200, hv_func=hv),
    ))
    return rows


def _print_header():
    print(f"{'config':22s}  {'time(s)':>10s}  {'peakMB':>10s}  "
          f"{'iter':>6s}  {'cg_tot':>7s}  {'ok':>5s}  {'||g||':>10s}")
    print('-' * 86)


def _print_row(row):
    print(f"{row['label']:22s}  "
          f"{row['time_s']:10.3f}  "
          f"{row['peak_MB']:10.1f}  "
          f"{row['n_iter']:6d}  "
          f"{row['cg_iters_total']:7d}  "
          f"{str(row['success']):>5s}  "
          f"{row['grad_norm']:10.2e}")


if __name__ == "__main__":
    print("=== A) PER-ITER COST (10 outer iter, x = x_bar_28) ===")
    _print_header()
    for n in (100, 1000, 5000, 10000, 50000):
        for row in run_for_n(n):
            _print_row(row)
        print()

    print("=== B) CONVERGENZA (x = ones + 0.1*N(0,I), tol=1e-6) ===")
    _print_header()
    for n in (100, 1000, 5000, 10000, 50000):
        for row in run_convergence(n):
            _print_row(row)
        print()
