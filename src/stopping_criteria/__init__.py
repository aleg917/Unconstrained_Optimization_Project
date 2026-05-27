"""Stopping criteria for iterative methods.

Six criteria total, in two variants (absolute and relative) for each
monitored quantity:

    grad_abs  ||g_k|| <= tol
    grad_rel  ||g_k|| / ||g_0|| <= tol
    f_abs     |F_k - F_{k-1}| <= tol
    f_rel     |F_k - F_{k-1}| / max(|F_k|, 1) <= tol
    x_abs     ||x_k - x_{k-1}|| <= tol
    x_rel     ||x_k - x_{k-1}|| / max(||x_k||, 1) <= tol

The optimization method receives ONE of these criteria as a parameter:
each run uses a single stopping strategy, so that termination can be
attributed correctly in the experimental tables.

For rationale on scales and tolerances see `stopping_criteria.md`.
"""
from .base import StoppingCriterion
from .absolute import GradNormAbsolute, FChangeAbsolute, XChangeAbsolute
from .relative import GradNormRelative, FChangeRelative, XChangeRelative


def all_criteria(tol_g: float = 1e-6,
                 tol_f: float = 1e-12,
                 tol_x: float = 1e-8):
    """Factory for experiments: returns all 6 criteria ready to use.

    The default tolerances differ by orders of magnitude to reflect the
    natural scales (||g|| ~ tol_g, |dF| ~ tol_g^2, ||dx|| ~ alpha*tol_g).
    See `stopping_criteria.md` for justification.
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
