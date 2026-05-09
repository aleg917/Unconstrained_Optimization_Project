"""||x_k - x_{k-1}|| <= tol  (forma assoluta).

Diagnostica il collasso del passo: l'iterato non si muove piu, anche se il
gradiente non e ancora piccolo. Vicino a un minimo decresce come ~ alpha * ||g||.
"""
import numpy as np

from ..base import StoppingCriterion


class XChangeAbsolute(StoppingCriterion):
    name = "x_abs"

    def should_stop(self, k, x, F, g, x_prev, F_prev) -> bool:
        if k == 0:
            return False
        return float(np.linalg.norm(x - x_prev)) <= self.tol
