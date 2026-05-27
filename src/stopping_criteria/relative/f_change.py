"""|F(x_k) - F(x_{k-1})| / max(|F(x_k)|, 1) <= tol  (relative form).

Normalized by the CURRENT value of F (not F_0): we assume no prior
knowledge of F at the start of the run.  The max(..., 1) avoids
division by zero when F approaches 0 (e.g. sum-of-squares problems
like Problem 28).
"""
from ..base import StoppingCriterion


class FChangeRelative(StoppingCriterion):
    name = "f_rel"

    def should_stop(self, k, x, F, g, x_prev, F_prev) -> bool:
        if k == 0:
            return False
        return abs(F - F_prev) / max(abs(F), 1.0) <= self.tol
