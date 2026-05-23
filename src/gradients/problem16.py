"""Gradiente esatto del Problem 16 (Banded Trigonometric)."""
import numpy as np


def grad_f16(x):
    """Gradiente esatto del Banded Trigonometric.

    Partendo da
        F(x) = sum_{i=1..n} i * [ (1 - cos x_i) + sin x_{i-1} - sin x_{i+1} ],
        x_0 = x_{n+1} = 0,
    riscriviamo le tre sommatorie come somme su x_j (j = 1, ..., n)
    cambiando l'indice nei termini con shift:

        sum_i i * sin x_{i-1}  -->  sum_j (j+1) * sin x_j,   j = 1..n-1
        sum_i (-i) * sin x_{i+1} --> sum_j -(j-1) * sin x_j, j = 2..n

    (i due bordi che cadono fuori griglia, j=n+1 e j=0, sono nulli per
    la condizione x_0 = x_{n+1} = 0). Differenziando rispetto a x_j si
    ottengono allora tre contributi: j * sin x_j dal termine diagonale
    j * (1 - cos x_j), piu' un coefficiente su cos x_j che dipende dal
    bordo.

    Per j = 1:        coefficiente cos = (1+1) = 2  (solo lo shift a sinistra)
    Per 2 <= j < n:   coefficiente cos = (j+1) - (j-1) = 2
    Per j = n:        coefficiente cos = 0 - (n-1) = -(n-1)
                       (manca lo shift a sinistra perche' sin x_{n+1} = 0)

    Forma chiusa:
        dF/dx_j =   j * sin x_j + 2 * cos x_j,         1 <= j <= n-1
        dF/dx_n =   n * sin x_n - (n-1) * cos x_n.
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    j = np.arange(1, n + 1)
    g = j * np.sin(x) + 2.0 * np.cos(x)
    g[-1] = n * np.sin(x[-1]) - (n - 1) * np.cos(x[-1])
    return g
