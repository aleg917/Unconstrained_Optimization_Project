"""||grad F(x_k)|| / ||grad F(x_0)|| <= tol  (relative form).

Scale-independent: if F is multiplied by a constant, both the numerator
and denominator scale identically.
"""
import numpy as np

from ..base import StoppingCriterion


class GradNormRelative(StoppingCriterion):
    name = "grad_rel"

    def initialize(self, x0, F0, g0):
        # Save ||g_0|| for the denominator.  The 1e-300 fallback avoids
        # division by zero in the pathological case where x_0 is already
        # stationary.
        self.g0_norm = max(float(np.linalg.norm(g0)), 1e-300)

    def should_stop(self, k, x, F, g, x_prev, F_prev) -> bool:
        return float(np.linalg.norm(g)) / self.g0_norm <= self.tol
