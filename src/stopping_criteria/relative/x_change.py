"""||x_k - x_{k-1}|| / max(||x_k||, 1) <= tol  (relative form).

Normalized by the CURRENT iterate (not x_0): we assume no prior
knowledge of x_0 after the fact.  The max(..., 1) avoids division
by zero for small iterates.
"""
import numpy as np

from ..base import StoppingCriterion


class XChangeRelative(StoppingCriterion):
    name = "x_rel"

    def should_stop(self, k, x, F, g, x_prev, F_prev) -> bool:
        if k == 0:
            return False
        return (float(np.linalg.norm(x - x_prev))
                / max(float(np.linalg.norm(x)), 1.0)) <= self.tol
