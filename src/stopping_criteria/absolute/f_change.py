"""|F(x_k) - F(x_{k-1})| <= tol  (absolute form).

Detects function-value stagnation: F is no longer decreasing.
Near a minimum this quantity decreases as ~ alpha * ||g||^2 / 2 (Armijo),
so it typically requires tighter tolerances than the gradient criterion.
"""
from ..base import StoppingCriterion


class FChangeAbsolute(StoppingCriterion):
    name = "f_abs"

    def should_stop(self, k, x, F, g, x_prev, F_prev) -> bool:
        if k == 0:
            return False
        return abs(F - F_prev) <= self.tol
