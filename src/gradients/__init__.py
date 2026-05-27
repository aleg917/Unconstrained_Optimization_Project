"""Gradients — exact (problem-specific) and finite-difference (generic).

Optimization methods accept `grad_f` as a callable: you can pass either the
exact gradient or an FD wrapper interchangeably.  Example:

    from src.functions import f16
    from src.gradients import grad_f16, grad_fd
    g_ex = grad_f16(x)
    g_fd = grad_fd(f16, x, k=8, scaled=False)
"""
from .problem16 import grad_f16
from .problem28 import grad_f28
from .finite_diff import grad_fd

__all__ = ["grad_f16", "grad_f28", "grad_fd"]
