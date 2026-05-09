"""Criteri di arresto per metodi iterativi.

Sei criteri totali, in due varianti (assoluta e relativa) per ciascuna
quantita monitorata:

    grad_abs  ||g_k|| <= tol
    grad_rel  ||g_k|| / ||g_0|| <= tol
    f_abs     |F_k - F_{k-1}| <= tol
    f_rel     |F_k - F_{k-1}| / max(|F_k|, 1) <= tol
    x_abs     ||x_k - x_{k-1}|| <= tol
    x_rel     ||x_k - x_{k-1}|| / max(||x_k||, 1) <= tol

Il metodo di ottimizzazione riceve UNO di questi criteri come parametro:
ogni run usa una sola strategia di arresto, in modo da poter attribuire
correttamente la terminazione nelle tabelle sperimentali.

Per i razionali sulle scale e tolleranze vedi `stopping_criteria.md`.
"""
from .base import StoppingCriterion
from .absolute import GradNormAbsolute, FChangeAbsolute, XChangeAbsolute
from .relative import GradNormRelative, FChangeRelative, XChangeRelative


def all_criteria(tol_g: float = 1e-6,
                 tol_f: float = 1e-12,
                 tol_x: float = 1e-8):
    """Fabbrica per gli esperimenti: ritorna i 6 criteri pronti.

    Le tolleranze di default differiscono di ordini di grandezza per
    riflettere le scale (||g|| ~ tol_g, |dF| ~ tol_g^2, ||dx|| ~ alpha*tol_g).
    Vedi `stopping_criteria.md` per la giustificazione.
    """
    return [
        GradNormAbsolute(tol_g),
        GradNormRelative(tol_g),
        FChangeAbsolute(tol_f),
        FChangeRelative(tol_f),
        XChangeAbsolute(tol_x),
        XChangeRelative(tol_x),
    ]


__all__ = [
    "StoppingCriterion",
    "GradNormAbsolute", "FChangeAbsolute", "XChangeAbsolute",
    "GradNormRelative", "FChangeRelative", "XChangeRelative",
    "all_criteria",
]
