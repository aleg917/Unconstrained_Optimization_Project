"""|F(x_k) - F(x_{k-1})| <= tol  (forma assoluta).

Diagnostica la stagnazione del valore di funzione: F non scende piu.
Vicino a un minimo questo decresce come ~ alpha * ||g||^2 / 2 (Armijo),
quindi richiede tolleranze tipicamente piu strette del gradiente.
"""
from ..base import StoppingCriterion


class FChangeAbsolute(StoppingCriterion):
    name = "f_abs"

    def should_stop(self, k, x, F, g, x_prev, F_prev) -> bool:
        if k == 0:
            return False
        return abs(F - F_prev) <= self.tol
