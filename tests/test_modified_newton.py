"""Unit test per Modified Newton + Hessiane esatte."""
import os
import sys

import numpy as np
import pytest

# Assicura che la root del progetto sia su sys.path quando si lancia pytest
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.functions import f16, f28, x_bar_16, x_bar_28
from src.gradients import grad_f16, grad_f28
from src.hessians import hess_f16, hess_f28
from src.methods.modified_newton import (
    _modify_hessian_cholesky,
    modified_newton,
)
from src.stopping_criteria import GradNormAbsolute


# ---------------------------------------------------------------------------
# Test 1 — Convergenza su Problem 28 (convesso, minimo noto x*=(1,...,1))
# ---------------------------------------------------------------------------
def test_modified_newton_converges_on_p28():
    n = 5
    res = modified_newton(f28, grad_f28, hess_f28, x_bar_28(n),
                          stopping=GradNormAbsolute(tol=1e-6))
    assert res['success'], f"Mancata convergenza: {res['stop_reason']}"
    assert res['n_iter'] < 30, f"Troppe iterazioni: {res['n_iter']}"
    assert np.allclose(res['x_star'], 1.0, atol=1e-4), \
        f"x_star non vicino a 1: {res['x_star']}"


# ---------------------------------------------------------------------------
# Test 2 — Convergenza su Problem 16, n=2
# ---------------------------------------------------------------------------
def test_modified_newton_converges_on_p16():
    res = modified_newton(f16, grad_f16, hess_f16, x_bar_16(2),
                          stopping=GradNormAbsolute(tol=1e-6))
    assert res['success'], f"Mancata convergenza: {res['stop_reason']}"
    assert res['grad_norm'] < 1e-5, f"grad_norm troppo grande: {res['grad_norm']}"


# ---------------------------------------------------------------------------
# Test 3 — Regolarizzazione di Cholesky su Hessiana indefinita
# ---------------------------------------------------------------------------
def test_cholesky_regularization_on_indefinite_hessian():
    H = np.diag([-1.0, 2.0])
    beta = 1e-3
    R, tau, n_adj = _modify_hessian_cholesky(H, beta=beta)

    # tau iniziale stimato come beta - min(diag) = 1 + 1e-3, deve bastare
    assert tau >= 1.0 + beta - 1e-12, f"tau troppo piccolo: {tau}"
    # R^T R deve coincidere con H + tau*I (R upper-triangolare, notazione slides)
    reconstructed = R.T @ R
    expected = H + tau * np.eye(2)
    assert np.allclose(reconstructed, expected, atol=1e-10)


# ---------------------------------------------------------------------------
# Test 4 — La direzione di Newton modificata e' di discesa
# ---------------------------------------------------------------------------
def test_modified_newton_direction_is_descent():
    n = 5
    res = modified_newton(f28, grad_f28, hess_f28, x_bar_28(n),
                          stopping=GradNormAbsolute(tol=1e-6),
                          return_history=True)
    # Da history: ad ogni step (tranne lo 0 che e' lo snapshot iniziale),
    # alpha > 0 e' la prova che la line search ha trovato un passo accettabile,
    # il che implica che p era di discesa (g @ p < 0).
    for step in res['history'][1:]:
        assert step['alpha'] is not None and step['alpha'] > 0
        assert step['p_norm'] is not None and step['p_norm'] > 0


# ---------------------------------------------------------------------------
# Test 5 — Hessiane esatte vs differenze finite del gradiente
# ---------------------------------------------------------------------------
def _hess_fd(grad, x, h=1e-6):
    """Hessiana via differenze finite centrali del gradiente."""
    n = len(x)
    H = np.zeros((n, n))
    for i in range(n):
        e = np.zeros(n)
        e[i] = h
        H[:, i] = (grad(x + e) - grad(x - e)) / (2.0 * h)
    return 0.5 * (H + H.T)  # simmetrizza


def test_hess_f28_matches_finite_differences():
    x = x_bar_28(5)
    H_exact = hess_f28(x)
    H_fd = _hess_fd(grad_f28, x)
    assert np.allclose(H_exact, H_fd, atol=1e-4), \
        f"max err = {np.max(np.abs(H_exact - H_fd))}"


def test_hess_f16_matches_finite_differences():
    x = np.array([0.3, 0.7])
    H_exact = hess_f16(x)
    H_fd = _hess_fd(grad_f16, x)
    assert np.allclose(H_exact, H_fd, atol=1e-4), \
        f"max err = {np.max(np.abs(H_exact - H_fd))}"


def test_hess_f16_is_diagonal():
    x = np.array([0.3, 0.7, -0.5])
    H = hess_f16(x)
    off_diag = H - np.diag(np.diag(H))
    assert np.all(off_diag == 0.0), "Hessiana di P16 non e' diagonale"


# ---------------------------------------------------------------------------
# Test 6 — Su H gia' definita positiva, tau resta 0 (no regolarizzazione)
# ---------------------------------------------------------------------------
def test_cholesky_no_adjustment_when_already_pd():
    H = np.array([[4.0, 1.0], [1.0, 3.0]])
    R, tau, n_adj = _modify_hessian_cholesky(H)
    assert tau == 0.0
    assert n_adj == 0
    assert np.allclose(R.T @ R, H, atol=1e-12)
