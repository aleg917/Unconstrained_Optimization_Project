"""Unit test per le approssimazioni FD della Hessiana.

Verifica le tre routine di src/hessians/finite_diff.py contro le Hessiane
esatte di Problem 16 (diagonale, CPR a 1 colore) e Problem 28 (densa).

Tolleranze: O(h)+O(eps/h) per forward, O(h^2)+O(eps/h) per centered. A k=8
con eps ~ 1e-16, attendiamo ~1e-7 per forward e ~1e-8/1e-7 per centered.
Le tolleranze nei test sono volutamente larghe (1e-4) per non risultare
flaky tra hardware/BLAS, ma in pratica i residui osservati sono ~1e-8/1e-7.
"""
import os
import sys

import numpy as np
import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.functions import x_bar_16, x_bar_28
from src.gradients import grad_f16, grad_f28
from src.hessians import (
    cpr_groups_p16,
    cpr_groups_p28,
    hess_f16,
    hess_f28,
    hess_fd,
    hess_fd_diag,
    hv_fd,
)


# ---------------------------------------------------------------------------
# hess_fd : assemblaggio colonna-per-colonna su P16 (diagonale)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("stencil", ["forward", "centered"])
@pytest.mark.parametrize("scaled", [False, True])
def test_hess_fd_matches_exact_on_p16(stencil, scaled):
    x = np.array([0.3, -0.7, 1.1, 0.0, 2.5])  # include x_i=0 -> esercita fallback scaled
    H_ex = hess_f16(x)
    H_fd = hess_fd(grad_f16, x, k=8, stencil=stencil, scaled=scaled)
    err = np.linalg.norm(H_fd - H_ex, ord=np.inf)
    assert err < 1e-4, f"stencil={stencil} scaled={scaled} err={err:.3e}"
    # simmetria imposta a valle
    assert np.allclose(H_fd, H_fd.T, atol=1e-12)


# ---------------------------------------------------------------------------
# hess_fd : assemblaggio colonna-per-colonna su P28 (densa)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("stencil", ["forward", "centered"])
@pytest.mark.parametrize("scaled", [False, True])
def test_hess_fd_matches_exact_on_p28(stencil, scaled):
    n = 5
    x = x_bar_28(n)  # = 1 - (1..n)/n, no zeri
    H_ex = hess_f28(x)
    H_fd = hess_fd(grad_f28, x, k=8, stencil=stencil, scaled=scaled)
    err = np.linalg.norm(H_fd - H_ex, ord=np.inf)
    rel = err / np.linalg.norm(H_ex, ord=np.inf)
    assert rel < 1e-4, f"stencil={stencil} scaled={scaled} rel-err={rel:.3e}"
    assert np.allclose(H_fd, H_fd.T, atol=1e-12)


# ---------------------------------------------------------------------------
# hess_fd : centered piu' accurato di forward al medesimo h (per H "regolare")
# ---------------------------------------------------------------------------
def test_centered_beats_forward_on_p28():
    """A h=1e-4 la troncatura O(h) di forward domina su quella O(h^2) di centered.

    A k=8 i due stencil sono comparabili (round-off ~ truncation per centered),
    quindi usiamo k=4 dove la verifica e' netta.
    """
    n = 5
    x = x_bar_28(n)
    H_ex = hess_f28(x)
    err_fwd = np.linalg.norm(hess_fd(grad_f28, x, k=4, stencil="forward") - H_ex, ord=np.inf)
    err_cen = np.linalg.norm(hess_fd(grad_f28, x, k=4, stencil="centered") - H_ex, ord=np.inf)
    assert err_cen < err_fwd, f"centered={err_cen:.3e} >= forward={err_fwd:.3e}"


# ---------------------------------------------------------------------------
# hess_fd_diag : path CPR a 1 colore su P16
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("stencil", ["forward", "centered"])
@pytest.mark.parametrize("scaled", [False, True])
def test_hess_fd_diag_matches_diag_on_p16(stencil, scaled):
    n = 7
    x = np.linspace(-1.0, 1.5, n)
    diag_ex = np.diag(hess_f16(x))
    diag_fd = hess_fd_diag(grad_f16, x, k=8, stencil=stencil, scaled=scaled)
    err = np.linalg.norm(diag_fd - diag_ex, ord=np.inf)
    assert err < 1e-4, f"stencil={stencil} scaled={scaled} err={err:.3e}"


def test_hess_fd_diag_agrees_with_hess_fd_full_on_p16():
    """Su P16 (diagonale) le due strade devono dare lo stesso risultato
    (a meno di errore di tondamento sulla simmetrizzazione)."""
    n = 5
    x = np.array([0.3, -0.7, 1.1, 0.0, 2.5])
    d_compact = hess_fd_diag(grad_f16, x, k=8, stencil="centered")
    H_full = hess_fd(grad_f16, x, k=8, stencil="centered", scaled=False)
    err = np.linalg.norm(d_compact - np.diag(H_full), ord=np.inf)
    assert err < 1e-6


# ---------------------------------------------------------------------------
# hv_fd : Hv matrix-free vs H_exact @ v
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("problem", ["p16", "p28"])
@pytest.mark.parametrize("stencil", ["forward", "centered"])
def test_hv_fd_matches_exact_Hv(problem, stencil):
    rng = np.random.default_rng(0)
    if problem == "p16":
        n = 6
        x = rng.uniform(-1.0, 1.5, size=n)
        grad_f = grad_f16
        H_ex = hess_f16(x)
    else:
        n = 6
        x = x_bar_28(n) + 0.1 * rng.standard_normal(n)
        grad_f = grad_f28
        H_ex = hess_f28(x)

    for _ in range(5):
        v = rng.standard_normal(n)
        Hv_ex = H_ex @ v
        Hv = hv_fd(grad_f, x, v, k=8, stencil=stencil)
        rel = np.linalg.norm(Hv - Hv_ex, ord=np.inf) / max(np.linalg.norm(Hv_ex, ord=np.inf), 1e-16)
        assert rel < 1e-4, f"{problem} stencil={stencil} rel={rel:.3e}"


def test_hv_fd_linearity_in_v():
    """Hv e' lineare in v (approssimativamente: l'errore di FD e' O(h)/O(h^2),
    non O(||v||^2))."""
    rng = np.random.default_rng(1)
    n = 5
    x = x_bar_28(n)
    H_ex = hess_f28(x)
    v1 = rng.standard_normal(n)
    v2 = rng.standard_normal(n)
    Hv1 = hv_fd(grad_f28, x, v1, k=8, stencil="centered")
    Hv2 = hv_fd(grad_f28, x, v2, k=8, stencil="centered")
    Hv_sum = hv_fd(grad_f28, x, v1 + v2, k=8, stencil="centered")
    expected = H_ex @ (v1 + v2)
    scale = np.linalg.norm(expected, ord=np.inf)
    # tolleranza relativa: la FD ha errore O(h^2)*||H''|| che su P28 e' grande
    # in modulo assoluto (||H|| ~ n^2) ma ~1e-7 in relativo a k=8
    assert np.linalg.norm(Hv_sum - expected, ord=np.inf) / scale < 1e-6
    assert np.linalg.norm((Hv1 + Hv2) - expected, ord=np.inf) / scale < 1e-6


# ---------------------------------------------------------------------------
# Simmetrizzazione opt-out
# ---------------------------------------------------------------------------
def test_hess_fd_symmetrize_flag():
    """Con symmetrize=False puo' restare un piccolo gap (H_fd - H_fd.T)
    di ordine O(h) (forward) o O(h^2) (centered). Con symmetrize=True deve
    sparire entro tondamento."""
    n = 5
    x = x_bar_28(n)
    H_sym = hess_fd(grad_f28, x, k=4, stencil="forward", symmetrize=True)
    H_raw = hess_fd(grad_f28, x, k=4, stencil="forward", symmetrize=False)
    assert np.allclose(H_sym, H_sym.T, atol=1e-12)
    asym_raw = np.linalg.norm(H_raw - H_raw.T, ord=np.inf)
    # a k=4 con forward l'asimmetria deve essere visibile (>> 1e-12)
    assert asym_raw > 1e-6


# ---------------------------------------------------------------------------
# Sparsity declarations
# ---------------------------------------------------------------------------
def test_cpr_groups_p16_is_single_color():
    for n in (1, 5, 100):
        groups = cpr_groups_p16(n)
        assert len(groups) == 1
        assert sorted(groups[0]) == list(range(n))


def test_cpr_groups_p28_is_one_per_color():
    for n in (1, 5, 100):
        groups = cpr_groups_p28(n)
        assert len(groups) == n
        assert sorted(g[0] for g in groups) == list(range(n))


# ---------------------------------------------------------------------------
# Argomenti invalidi
# ---------------------------------------------------------------------------
def test_hess_fd_rejects_unknown_stencil():
    with pytest.raises(ValueError):
        hess_fd(grad_f16, np.zeros(3), k=8, stencil="bogus")


def test_hv_fd_rejects_unknown_stencil():
    with pytest.raises(ValueError):
        hv_fd(grad_f16, np.zeros(3), np.ones(3), k=8, stencil="bogus")


def test_hess_fd_diag_rejects_unknown_stencil():
    with pytest.raises(ValueError):
        hess_fd_diag(grad_f16, np.zeros(3), k=8, stencil="bogus")
