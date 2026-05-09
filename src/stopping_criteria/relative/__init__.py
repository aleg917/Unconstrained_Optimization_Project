"""Criteri di arresto in forma relativa."""
from .grad_norm import GradNormRelative
from .f_change import FChangeRelative
from .x_change import XChangeRelative

__all__ = ["GradNormRelative", "FChangeRelative", "XChangeRelative"]
