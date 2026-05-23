"""Unit test per Truncated Newton:
- ramo classico (hess_f assemblata)
- ramo matrix-free (hv_func: nessuna Hessiana materializzata)
"""
import os
import sys

import numpy as np
import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.functions import f16, f28, x_bar_16, x_bar_28
from src.gradients import grad_f16, grad_f28
from src.hessians import hess_f16, hess_f28, hv_fd
from src.methods.truncated_newton import _cg_truncated, truncated_newton
from src.stopping_criteria import GradNormAbsolute


# ---------------------------------------------------------------------------
# Ramo classico: Hessiana esatta assemblata
# ---------------------------------------------------------------------------
def test_tn_converges_on_p28_with_exact_hessian():
    n = 5
    res = truncated_newton(f28, grad_f28, hess_f28, x_bar_28(n),
                           stopping=GradNormAbsolute(tol=1e-6))
    assert res['success'], f"Mancata convergenza: {res['stop_reason']}"
    assert np.allclose(res['x_star'], 1.0, atol=1e-4)


def test_tn_converges_on_p16_with_exact_hessian():
    res = truncated_newton(f16, grad_f16, hess_f16, x_bar_16(2),
                           stopping=GradNormAbsolute(tol=1e-6))
    assert res['success']
    assert res['grad_norm'] < 1e-5


# ---------------------------------------------------------------------------
# Ramo matrix-free
# ---------------------------------------------------------------------------
def test_tn_matrix_free_converges_on_p28():
    """Stesso problema del test sopra, ma usando hv_func basata su FD del
    gradiente esatto. Niente Hessiana materializzata."""
    n = 5
    hv = lambda x, v: hv_fd(grad_f28, x, v, k=8, stencil="centered")
    res = truncated_newton(f28, grad_f28, x0=x_bar_28(n),
                           stopping=GradNormAbsolute(tol=1e-6),
                           hv_func=hv)
    assert res['success'], f"Mancata convergenza: {res['stop_reason']}"
    assert np.allclose(res['x_star'], 1.0, atol=1e-4)


def test_tn_matrix_free_matches_exact_branch_on_p28():
    """Verifica che il ramo matrix-free dia risultato sostanzialmente uguale
    al ramo classico, per lo stesso problema/inizializzazione."""
    n = 8
    x0 = x_bar_28(n)
    stop = GradNormAbsolute(tol=1e-8)

    res_ex = truncated_newton(f28, grad_f28, hess_f28, x0, stopping=stop)
    hv = lambda x, v: hv_fd(grad_f28, x, v, k=8, stencil="centered")
    res_mf = truncated_newton(f28, grad_f28, x0=x0, stopping=stop, hv_func=hv)

    assert res_ex['success'] and res_mf['success']
    # Il punto finale deve coincidere entro le tolleranze del solver / FD
    assert np.linalg.norm(res_ex['x_star'] - res_mf['x_star']) < 1e-5


def test_tn_matrix_free_does_not_call_hess_f():
    """Quando hv_func e' fornita, hess_f deve essere ignorata (puo' essere
    None) — niente Hessiana mai materializzata."""
    n = 5
    sentinel = {'called': False}
    def hess_should_not_be_called(x):
        sentinel['called'] = True
        return np.eye(n)
    hv = lambda x, v: hv_fd(grad_f28, x, v, k=8, stencil="centered")
    truncated_newton(f28, grad_f28, hess_f=hess_should_not_be_called,
                     x0=x_bar_28(n),
                     stopping=GradNormAbsolute(tol=1e-6),
                     hv_func=hv)
    assert not sentinel['called'], "hv_func avrebbe dovuto avere precedenza"


# ---------------------------------------------------------------------------
# Validazione argomenti
# ---------------------------------------------------------------------------
def test_tn_requires_hess_f_or_hv_func():
    with pytest.raises(ValueError, match="hess_f or hv_func"):
        truncated_newton(f28, grad_f28, x0=x_bar_28(3),
                         stopping=GradNormAbsolute(tol=1e-6))


def test_tn_requires_x0_and_stopping():
    with pytest.raises(ValueError):
        truncated_newton(f28, grad_f28, hess_f28)


# ---------------------------------------------------------------------------
# _cg_truncated: il refactor accetta sia ndarray sia callable
# ---------------------------------------------------------------------------
def test_cg_truncated_accepts_callable_H():
    """Caso semplice: H = I, soluzione attesa z = b."""
    n = 4
    b = np.array([1.0, -2.0, 0.5, 3.0])
    H_mat = np.eye(n)
    H_callable = lambda d: d  # azione identita'

    z_mat, j_mat, term_mat = _cg_truncated(H_mat, b, eta_k=1e-10, cg_max_iter=20)
    z_cb, j_cb, term_cb = _cg_truncated(H_callable, b, eta_k=1e-10, cg_max_iter=20)

    assert np.allclose(z_mat, b, atol=1e-10)
    assert np.allclose(z_cb, b, atol=1e-10)
    assert np.allclose(z_mat, z_cb)
    assert term_mat == term_cb == "tol"


def test_cg_truncated_callable_handles_negative_curvature_at_j0():
    """Con curvatura negativa al primo step, esce con z = b (steepest)."""
    n = 3
    b = np.ones(n)
    H_callable = lambda d: -d  # curvatura negativa garantita
    z, j, term = _cg_truncated(H_callable, b, eta_k=0.1, cg_max_iter=10)
    assert term == "negcurv_j0"
    assert np.allclose(z, b)
    assert j == 0
