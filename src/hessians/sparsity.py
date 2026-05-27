"""Sparsity patterns (CPR / coloring) for the Hessians of the test problems.

Reference: NO4LSP course slides on "GENERALIZATION OF THE SCHEME: COLORING
GRAPHS".  Given the graph
    V = {x_1, ..., x_n},
    {x_i, x_j} in E  <=>  there exists k such that g_k depends on both x_i and x_j,
variables of the same color have disjoint column supports in H and can be
perturbed together in a single gradient call.

For the two problems in our study the patterns follow by inspection, so
no general coloring solver is needed.
"""


def cpr_groups_p16(n):
    """Problem 16 (Banded Trigonometric): DIAGONAL Hessian.

    Each g_j depends only on x_j, so the graph G has no edges: a single
    color suffices.  All variables go into one group => ONE pair of gradient
    calls extracts the entire diagonal (cf. hess_fd_diag), regardless of n.
    """
    return [list(range(n))]


def cpr_groups_p28(n):
    """Problem 28 (Variably Dimensioned): DENSE Hessian.

    Each g_j depends on ALL x_i: the sparsity graph is complete, n distinct
    colors are needed, and coloring produces no savings.  Exposed here for
    symmetry with P16 and to document the absence of structural gains.

    For this problem the only FD-Hessian optimizations are "general purpose":
        - symmetrization: H_fd <- (H_fd + H_fd^T) / 2
        - use of the analytic gradient, O(n) per call, instead of FD on f
          which would cost O(n^2)
        - matrix-free Hv path for Truncated Newton (one gradient per Hv
          product, without materializing the n x n matrix).
    """
    return [[j] for j in range(n)]
