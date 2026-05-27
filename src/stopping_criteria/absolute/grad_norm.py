"""||grad F(x_k)|| <= tol  (absolute form)."""
import numpy as np

from ..base import StoppingCriterion


class GradNormAbsolute(StoppingCriterion):
    """First-order criterion: stop when the gradient norm drops below the
    tolerance.  Scale-dependent on F."""

    name = "grad_abs"

    def should_stop(self, k, x, F, g, x_prev, F_prev) -> bool:
        return float(np.linalg.norm(g)) <= self.tol
