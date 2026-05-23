"""Hessiana esatta del Problem 28 — Variably Dimensioned.

Punto di partenza (gia' ricavato nel gradiente):
    dF/dx_j = (x_j - 1) + j * S * (1 + 2 S^2),
    S(x)   = sum_{i=1..n} i (x_i - 1).

Per ottenere l'elemento H_{ij} deriviamo la j-esima componente del
gradiente rispetto a x_i, un addendo per volta:
    d/dx_i (x_j - 1)            = delta_{ij}
    d/dx_i [ j S (1 + 2 S^2) ]  = j * [ (dS/dx_i)(1 + 2 S^2)
                                         + S * 4 S * (dS/dx_i) ]
                                = j * i * (1 + 2 S^2 + 4 S^2)
                                = i * j * (1 + 6 S^2).
Sommando:
    H_{ij} = delta_{ij} + i * j * (1 + 6 S^2).

WARNING: O(n^2) memoria. Per n grandi (tipicamente n >= 1e4) la
materializzazione della matrice n x n non e' sostenibile; in quei
regimi si usa il path matrix-free Hv del Truncated Newton, che
costruisce solo prodotti Hessiana-vettore via differenze finite del
gradiente esatto.
"""
import numpy as np


def hess_f28(x):
    """Hessiana esatta del Variably Dimensioned function (matrice n x n densa).

    Costruzione coerente con la formula scalare
        H_{ij} = delta_{ij} + i * j * (1 + 6 S^2):
    la matrice (i * j) per i, j = 1..n e' l'outer product del vettore
    (1, 2, ..., n) con se stesso; la moltiplichiamo per il fattore
    comune (1 + 6 S^2) e sommiamo l'identita' per il termine
    delta_{ij}.
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    j = np.arange(1, n + 1, dtype=float)
    S = float(np.dot(j, x - 1.0))
    c = 1.0 + 6.0 * S * S
    return np.eye(n) + c * np.outer(j, j)
