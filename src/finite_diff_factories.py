"""Factories: bind (k, scaled) once, return clean FD callables.

These replace inline lambda expressions in notebooks and scripts.  They rely
on the existing FD primitives (grad_fd, hess_fd, hv_fd) and introduce no new
numerical logic - they just freeze the parameters into a closure.

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
    """
    Build a gradient callable g(x) -> ndarray that approximates grad f(x) by
    centered finite differences of f, with (k, scaled) frozen into the closure.

    Step: h = 10^{-k}             if scaled=False
          h_i = 10^{-k} * |x_i|   if scaled=True (falls back to 10^{-k} when x_i = 0)

    When to use
    -----------
    To supply an FD gradient wherever an exact grad_f is expected — the grad_f
    argument of Modified / Truncated Newton — or as the input to make_fd_hess /
    make_fd_hv to differentiate twice ("approximate everything").

    Parameters
    ----------
    f : callable
        Objective F(x) -> float.
    k : int
        Step-size exponent: the base step is h = 10**(-k).
    scaled : bool
        If True use h_i = 10**(-k) * |x_i|; otherwise h_i = 10**(-k).

    Returns
    -------
    g : callable
        g(x) -> ndarray of shape (n,), the approximated gradient at x.
    """
    return lambda x: grad_fd(f, x, k=k, scaled=scaled)


def make_fd_hess(grad_f, k, scaled):
    """
    Build a Hessian callable H(x) -> ndarray that approximates hess f(x) by
    centered finite differences of the gradient grad_f, with (k, scaled) frozen
    into the closure.

    Centered stencil, O(h^2) accuracy, 2n gradient calls, always symmetrized;
    returns the full (n, n) matrix.

    When to use
    -----------
    As the hess_f argument of Modified Newton, which factorizes the full matrix
    (Cholesky). Pass an exact grad_f for an FD-of-exact-gradient Hessian, or a
    make_fd_grad callable to approximate everything.

    Parameters
    ----------
    grad_f : callable
        Gradient map x -> grad f(x) (exact, or built with make_fd_grad).
    k : int
        Step-size exponent: the base step is h = 10**(-k).
    scaled : bool
        Forwarded to hess_fd.

    Returns
    -------
    H : callable
        H(x) -> ndarray of shape (n, n), the approximated (symmetrized) Hessian.
    """
    return lambda x: hess_fd(grad_f, x, k=k, scaled=scaled)


def make_fd_hv(grad_f, k, scaled):
    """
    Build a Hessian-vector callable hv(x, v) -> ndarray that approximates the
    product H(x) v matrix-free (without ever assembling H), with (k, scaled)
    frozen into the closure.

    Two gradient calls per application.

    When to use
    -----------
    As the hv_func argument of Truncated Newton, whose inner CG solver needs
    only Hessian-vector products. Pass an exact grad_f, or a make_fd_grad
    callable to approximate everything.

    Parameters
    ----------
    grad_f : callable
        Gradient map x -> grad f(x) (exact, or built with make_fd_grad).
    k : int
        Step-size exponent: the base step is h = 10**(-k).
    scaled : bool
        Forwarded to hv_fd.

    Returns
    -------
    hv : callable
        hv(x, v) -> ndarray of shape (n,), the approximated product H(x) v.
    """
    return lambda x, v: hv_fd(grad_f, x, v, k=k, scaled=scaled)
