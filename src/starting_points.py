"""Generazione dei punti iniziali per gli esperimenti.

Specifiche dall'assignment:
    - 1 punto suggerito (x_bar) + 5 punti random uniformi in
      [x_bar_i - 1, x_bar_i + 1] per ogni componente i.
    - seed = minimo student-ID del team (impostato UNA VOLTA SOLA
      prima di generare i punti per tutti i problemi/dimensioni,
      per garantire la riproducibilita).
"""
import numpy as np


def generate_starting_points(x_bar, num_random=5, rng=None):
    """Restituisce una lista di punti iniziali: [x_bar, x_bar + delta_1, ...].

    Parametri
    ---------
    x_bar      : ndarray (n,) — punto suggerito
    num_random : numero di punti random aggiuntivi (default 5)
    rng        : numpy.random.Generator (opzionale).
                 Se None usa np.random globale.
    """
    x_bar = np.asarray(x_bar, dtype=float)
    n = len(x_bar)
    points = [x_bar.copy()]
    sample = rng.uniform if rng is not None else np.random.uniform
    for _ in range(num_random):
        delta = sample(-1.0, 1.0, size=n)
        points.append(x_bar + delta)
    return points
