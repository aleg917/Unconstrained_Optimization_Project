"""||grad F(x_k)|| <= tol  (forma assoluta)."""
import numpy as np

from ..base import StoppingCriterion


class GradNormAbsolute(StoppingCriterion):
    """Criterio del primo ordine: arresto quando la norma del gradiente
    scende sotto la tolleranza. Dipende dalla scala di F."""

    name = "grad_abs"

    def should_stop(self, k, x, F, g, x_prev, F_prev) -> bool:
        return float(np.linalg.norm(g)) <= self.tol
