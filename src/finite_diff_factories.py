"""Factories: bind (k, scaled) once, return clean FD callables.

These replace inline lambda expressions in notebooks and scripts.  They rely
on the existing FD primitives (grad_fd, hess_fd, hv_fd) and introduce no new
numerical logic — they just freeze the parameters into a closure.

Example (Modified Newton with FD Hessian from the exact gradient):

    from src.functions import f16, x_bar_16
    from src.gradients import grad_f16
    from src.finite_diff_factories import make_fd_hess
    from src.methods.modified_newton import modified_newton

    H_fd = make_fd_hess(grad_f16, k=8, scaled=False)
    res = modified_newton(f16, x_bar_16(1000), stopping=...,
                          grad_f=grad_f16, hess_f=H_fd)

Composite example (FD gradient + FD Hessian — "approximate everything"):

    g_fd = make_fd_grad(f28, k=8, scaled=False)
    H_fd = make_fd_hess(g_fd, k=8, scaled=False)
"""
from .gradients.finite_diff import grad_fd
from .hessians.finite_diff import hess_fd, hv_fd


def make_fd_grad(f, k, scaled):
    """Callable g(x) -> ndarray, approximates grad f(x) with centered FD.

    Step: h = 10^{-k}  if scaled=False
          h_i = 10^{-k} * |x_i|  if scaled=True  (falls back to 10^{-k} when x_i = 0)
    """
    return lambda x: grad_fd(f, x, k=k, scaled=scaled)


def make_fd_hess(grad_f, k, scaled):
    """Callable H(x) -> ndarray, approximates hess f(x) via FD of the gradient.

    Centered stencil, O(h^2), 2n gradient calls.  Always symmetrized.
    """
    return lambda x: hess_fd(grad_f, x, k=k, scaled=scaled)


def make_fd_hv(grad_f, k, scaled):
    """Callable hv(x, v) -> ndarray, approximates H(x) v matrix-free.

    Two gradient calls per application.  Used by the inner CG solver of
    Truncated Newton.
    """
    return lambda x, v: hv_fd(grad_f, x, v, k=k, scaled=scaled)
