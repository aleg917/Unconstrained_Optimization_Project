"""||x_k - x_{k-1}|| / max(||x_k||, 1) <= tol  (forma relativa).

Normalizzata sull'iterato CORRENTE (non su x_0): assumiamo di non
conoscere x_0 a posteriori. Il `max(..., 1)` evita la divisione per
zero per iterati piccoli.
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
