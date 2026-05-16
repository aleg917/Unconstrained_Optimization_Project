"""|F(x_k) - F(x_{k-1})| / max(|F(x_k)|, 1) <= tol  (forma relativa).

Normalizzata sul valore CORRENTE di F (non su F_0): assumiamo di non
conoscere F all'inizio del run. Il `max(..., 1)` evita la divisione per
zero quando F si avvicina a 0 (es. problemi sum-of-squares come Problem 28).
"""
from ..base import StoppingCriterion


class FChangeRelative(StoppingCriterion):
    name = "f_rel"

    def should_stop(self, k, x, F, g, x_prev, F_prev) -> bool:
        if k == 0:
            return False
        return abs(F - F_prev) / max(abs(F), 1.0) <= self.tol
