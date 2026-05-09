"""||grad F(x_k)|| / ||grad F(x_0)|| <= tol  (forma relativa).

Indipendente dalla scala di F: se F viene moltiplicata per una costante,
sia il numeratore che il denominatore scalano allo stesso modo.
"""
import numpy as np

from ..base import StoppingCriterion


class GradNormRelative(StoppingCriterion):
    name = "grad_rel"

    def initialize(self, x0, F0, g0):
        # Salva ||g_0|| per il denominatore. Il fallback a 1e-300 evita la
        # divisione per zero nel caso patologico in cui x_0 sia gia' stazionario.
        self.g0_norm = max(float(np.linalg.norm(g0)), 1e-300)

    def should_stop(self, k, x, F, g, x_prev, F_prev) -> bool:
        return float(np.linalg.norm(g)) / self.g0_norm <= self.tol
