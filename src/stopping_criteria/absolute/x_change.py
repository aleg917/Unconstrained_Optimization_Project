"""||x_k - x_{k-1}|| <= tol  (absolute form).

Detects step collapse: the iterate is no longer moving, even if the
gradient is not yet small.  Near a minimum this decreases as ~ alpha * ||g||.
"""
import numpy as np

from ..base import StoppingCriterion


class XChangeAbsolute(StoppingCriterion):
    name = "x_abs"

    def should_stop(self, k, x, F, g, x_prev, F_prev) -> bool:
        if k == 0:
            return False
        return float(np.linalg.norm(x - x_prev)) <= self.tol
