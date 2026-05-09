"""Gradienti — esatti (problem-specific) e differenze finite (generico).

I metodi di ottimizzazione accettano `grad_f` come callable: si puo passare
indifferentemente il gradiente esatto o un wrapper FD. Esempio:

    from src.functions import f16
    from src.gradients import grad_f16, grad_fd_forward
    g_ex = grad_f16(x)
    g_fd = grad_fd_forward(f16, x, k=8, scaled=False)
"""
from .problem16 import grad_f16
from .problem32 import grad_f32
from .finite_diff import grad_fd_forward

__all__ = ["grad_f16", "grad_f32", "grad_fd_forward"]
