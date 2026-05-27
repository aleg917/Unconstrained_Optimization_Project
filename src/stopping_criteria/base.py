"""Base class for plug-in stopping criteria.

Strategy pattern: the same optimization loop can be run with different
stopping criteria without modifying the method.  Each criterion:

1. Exposes a `name` attribute (string) that ends up in the `stop_reason`
   field of the result dict — useful for experimental tables.
2. Implements `initialize(x0, F0, g0)`, called once before the loop.
   The `relative` variants save their reference values here (e.g. ||g_0||).
3. Implements `should_stop(k, x, F, g, x_prev, F_prev) -> bool`, called
   at every iteration.  The 'change' criteria must return False when
   k == 0 (there is no meaningful x_prev yet).
"""


class StoppingCriterion:
    """Plug-in stopping strategy for iterative optimization methods."""

    name: str = "base"

    def __init__(self, tol: float):
        self.tol = float(tol)

    def initialize(self, x0, F0, g0):
        """Hook called ONCE before the loop.  Override in subclasses that
        need an initial reference (e.g. ||g_0||)."""
        pass

    def should_stop(self, k, x, F, g, x_prev, F_prev) -> bool:
        """Decide whether to stop the loop at iteration k.

        Parameters
        ----------
        k       : iteration index (0-based, before the k-th step)
        x       : current iterate
        F       : F(x) at current iterate
        g       : grad F(x) at current iterate
        x_prev  : previous iterate (equal to x when k == 0)
        F_prev  : F(x_prev)

        The 'change' criteria must return False when k == 0.
        """
        raise NotImplementedError

    def __repr__(self):
        return f"{type(self).__name__}(tol={self.tol:g})"
