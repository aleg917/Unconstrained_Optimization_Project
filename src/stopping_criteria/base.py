"""Classe base per i criteri di arresto plug-in.

Pattern a strategia: lo stesso loop di Steepest Descent puo essere eseguito
con criteri di arresto diversi senza modificare il metodo. Ogni criterio:

1. Espone un attributo `name` (stringa) che finisce nel campo `stop_reason`
   del risultato — utile per le tabelle sperimentali.
2. Implementa `initialize(x0, F0, g0)`, chiamato una sola volta prima del
   loop. Le versioni `relative` salvano qui i valori di riferimento (es.
   ||g_0||).
3. Implementa `should_stop(k, x, F, g, x_prev, F_prev) -> bool`, chiamato
   a inizio di ogni iterazione. I criteri 'change' devono ritornare False
   quando k == 0 (non c'e ancora un x_prev sensato).
"""


class StoppingCriterion:
    """Strategia di arresto plug-in per metodi iterativi."""

    name: str = "base"

    def __init__(self, tol: float):
        self.tol = float(tol)

    def initialize(self, x0, F0, g0):
        """Hook chiamato UNA volta prima del loop. Override nelle sottoclassi
        che hanno bisogno di un riferimento iniziale (es. ||g_0||)."""
        pass

    def should_stop(self, k, x, F, g, x_prev, F_prev) -> bool:
        """Decide se fermare il loop all'iterazione k.

        Parametri
        ---------
        k       : indice di iterazione (0-based, prima del passo k-esimo)
        x       : iterato corrente
        F       : F(x) corrente
        g       : grad F(x) corrente
        x_prev  : iterato precedente (uguale a x quando k == 0)
        F_prev  : F(x_prev)

        I criteri 'change' devono restituire False quando k == 0.
        """
        raise NotImplementedError

    def __repr__(self):
        return f"{type(self).__name__}(tol={self.tol:g})"
