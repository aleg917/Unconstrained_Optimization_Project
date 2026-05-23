"""Test per src/finite_diff_factories.py.

Sei test, tutti su n=10, deterministici:
  1. make_fd_grad combacia col primitivo grad_fd_forward (identita').
  2. make_fd_grad approssima il gradiente esatto di P16 a ~1e-7.
  3. make_fd_hess combacia col primitivo hess_fd (identita').
  4. make_fd_hess approssima la Hessiana esatta di P28 a ~1e-6.
  5. make_fd_hv combacia con hess_f28(x) @ v.
  6. Closure isolation: factory create in un loop con k diversi NON
     condividono il valore di k (no late-binding bug).
"""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.functions import f16, f28, x_bar_16, x_bar_28
from src.gradients import grad_f16, grad_f28
from src.gradients.finite_diff import grad_fd_forward
from src.hessians import hess_f16, hess_f28
from src.hessians.finite_diff import hess_fd, hv_fd
from src.finite_diff_factories import make_fd_grad, make_fd_hess, make_fd_hv


N = 10


def test_make_fd_grad_matches_primitive():
    x = x_bar_16(N)
    g_factory = make_fd_grad(f16, k=8, scaled=False)(x)
    g_direct = grad_fd_forward(f16, x, k=8, scaled=False)
    np.testing.assert_allclose(g_factory, g_direct, rtol=0, atol=0)


def test_make_fd_grad_approximates_exact_on_p16():
    x = x_bar_16(N)
    g_fd = make_fd_grad(f16, k=8, scaled=False)(x)
    g_ex = grad_f16(x)
    np.testing.assert_allclose(g_fd, g_ex, atol=1e-6)


def test_make_fd_hess_matches_primitive():
    x = x_bar_28(N)
    H_factory = make_fd_hess(grad_f28, k=8, scaled=False, stencil="centered")(x)
    H_direct = hess_fd(grad_f28, x, k=8, scaled=False, stencil="centered")
    np.testing.assert_allclose(H_factory, H_direct, rtol=0, atol=0)


def test_make_fd_hess_approximates_exact_on_p28():
    x = x_bar_28(N)
    H_fd = make_fd_hess(grad_f28, k=8, scaled=False, stencil="centered")(x)
    H_ex = hess_f28(x)
    np.testing.assert_allclose(H_fd, H_ex, rtol=1e-5)


def test_make_fd_hv_matches_dense_hess_times_v():
    rng = np.random.default_rng(0)
    x = x_bar_28(N)
    v = rng.standard_normal(N)
    hv_fd_factory = make_fd_hv(grad_f28, k=8, scaled=False, stencil="centered")(x, v)
    Hv_exact = hess_f28(x) @ v
    np.testing.assert_allclose(hv_fd_factory, Hv_exact, rtol=1e-5)


def test_closure_isolation_no_late_binding():
    """Costruisce due factory con k diversi dentro un loop e verifica che
    non condividano il valore di k. Bug classico: se i factory facessero
    riferimento alla variabile di loop per nome, tutti userebbero l'ultimo k.
    """
    x = x_bar_16(N)
    factories = {}
    for k in (4, 12):
        factories[k] = make_fd_hess(grad_f16, k=k, scaled=False, stencil="centered")

    # I due risultati devono coincidere con chiamate dirette al primitivo
    # con il rispettivo k bound (non con l'ultimo k del loop):
    H_k4_factory = factories[4](x)
    H_k4_direct = hess_fd(grad_f16, x, k=4, scaled=False, stencil="centered")
    np.testing.assert_allclose(H_k4_factory, H_k4_direct, rtol=0, atol=0)

    H_k12_factory = factories[12](x)
    H_k12_direct = hess_fd(grad_f16, x, k=12, scaled=False, stencil="centered")
    np.testing.assert_allclose(H_k12_factory, H_k12_direct, rtol=0, atol=0)

    # E i due output devono essere DIVERSI (altrimenti il bug e' silenzioso):
    assert not np.allclose(H_k4_factory, H_k12_factory)
