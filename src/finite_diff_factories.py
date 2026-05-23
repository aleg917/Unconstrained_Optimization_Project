"""Factories: bind (k, scaled, stencil) once, return clean FD callables.

Sostituiscono le lambda inline sparse in scripts/ e tests/. Si appoggiano
ai primitivi gia esistenti (grad_fd_forward, hess_fd, hv_fd) e non
introducono nuova logica numerica: solo unificano il modo in cui gli
argomenti vengono passati.

Esempio d'uso (steepest descent con gradiente FD):

    from src.functions import f16, x_bar_16
    from src.finite_diff_factories import make_fd_grad
    from src.methods.steepest_descent import steepest_descent

    grad_cb = make_fd_grad(f16, k=8, scaled=False)
    res = steepest_descent(f16, grad_cb, x_bar_16(1000), stopping=...)

Esempio composito (FD del gradiente + FD della Hessiana, per il caso
"approssima tutte le derivate"):

    g_fd = make_fd_grad(f28, k=8, scaled=False)
    H_fd = make_fd_hess(g_fd, k=8, scaled=False, stencil="centered")
"""
from .gradients.finite_diff import grad_fd_forward
from .hessians.finite_diff import hess_fd, hv_fd


def make_fd_grad(f, k, scaled):
    """Callable g(x) -> ndarray, approssima grad f(x) con forward FD.

    Passo: h = 10^{-k}  se scaled=False
           h_i = 10^{-k} * |x_i|  se scaled=True  (fallback a 10^{-k} se x_i=0)
    """
    return lambda x: grad_fd_forward(f, x, k=k, scaled=scaled)


def make_fd_hess(grad_f, k, scaled, stencil="centered"):
    """Callable H(x) -> ndarray, approssima hess f(x) con FD del gradiente.

    stencil 'centered' (O(h^2), 2n chiamate) o 'forward' (O(h), n+1 chiamate).
    Passo come in make_fd_grad. Restituisce sempre una Hessiana simmetrizzata.
    """
    return lambda x: hess_fd(grad_f, x, k=k, scaled=scaled, stencil=stencil)


def make_fd_hv(grad_f, k, scaled, stencil="centered"):
    """Callable hv(x, v) -> ndarray, approssima H(x) v matrix-free.

    Una/due chiamate a grad_f per applicazione. Usato dalla CG interna del
    Truncated Newton. Stesso (k, scaled, stencil) di make_fd_hess.
    """
    return lambda x, v: hv_fd(grad_f, x, v, k=k, scaled=scaled, stencil=stencil)
