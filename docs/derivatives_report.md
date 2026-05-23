# Analytical derivatives of the two test problems

This file contains a clean, step-by-step derivation of the gradient and
the Hessian of the two test problems used in our study (Problem 16 and
Problem 28 from Lukšan & Vlček, *Test Problems for Unconstrained
Optimization*, Technical Report No. 897, 2003). The text is written in
the form we plan to copy into the *introductory analysis of the test
problems* section of the final report.

Formulas use LaTeX inline (`$...$`) and display (`$$...$$`) math, plus
`align*` blocks when several steps need to be shown together.

---

## 1. Problem 16 — Banded trigonometric function

### 1.1 Statement

For $n \ge 1$ define
$$
F(x) \;=\; \sum_{i=1}^{n} i \,\bigl[(1 - \cos x_i) + \sin x_{i-1} - \sin x_{i+1}\bigr],
\qquad x_0 = x_{n+1} = 0,
$$
with suggested starting point $\bar{x}_i = 1$, $i = 1, \dots, n$.

### 1.2 Gradient

Split the sum into three independent contributions:
$$
F(x) = \underbrace{\sum_{i=1}^{n} i\,(1 - \cos x_i)}_{(A)}
      + \underbrace{\sum_{i=1}^{n} i \sin x_{i-1}}_{(B)}
      - \underbrace{\sum_{i=1}^{n} i \sin x_{i+1}}_{(C)}.
$$

Each $\partial F / \partial x_j$ collects the contributions in which
$x_j$ appears. To make these contributions explicit we re-index the
shifted sums so that the summation variable becomes $j$.

*Term $(B)$.* Setting $i = j+1$ in $\sum_{i=1}^{n} i \sin x_{i-1}$
gives
$$
(B) \;=\; \sum_{j=0}^{n-1} (j+1)\,\sin x_j
       \;=\; \sum_{j=1}^{n-1} (j+1)\,\sin x_j,
$$
where the term $j=0$ has been dropped because $\sin x_0 = 0$.

*Term $(C)$.* Setting $i = j-1$ in $\sum_{i=1}^{n} i \sin x_{i+1}$
gives
$$
(C) \;=\; \sum_{j=2}^{n+1} (j-1)\,\sin x_j
       \;=\; \sum_{j=2}^{n} (j-1)\,\sin x_j,
$$
where the term $j=n+1$ has been dropped because $\sin x_{n+1} = 0$.

The variable $x_j$ thus appears in:
- $(A)$ through $i = j$, contributing the term $j\,(1 - \cos x_j)$;
- $(B)$ with coefficient $+(j+1)$ on $\sin x_j$, only for $1 \le j \le n-1$;
- $(C)$ with coefficient $-(j-1)$ on $\sin x_j$, only for $2 \le j \le n$.

Differentiating each piece with respect to $x_j$ yields a coefficient
$j\sin x_j$ from $(A)$ and a coefficient on $\cos x_j$ given by
$\bigl[(j+1)\,\mathbf{1}_{j \le n-1}\bigr] - \bigl[(j-1)\,\mathbf{1}_{j \ge 2}\bigr]$.
The three cases are:

| index $j$        | coefficient on $\cos x_j$ |
|------------------|---------------------------|
| $j = 1$          | $(1+1) - 0 = 2$           |
| $2 \le j \le n-1$ | $(j+1) - (j-1) = 2$       |
| $j = n$          | $0 - (n-1) = -(n-1)$       |

So the closed-form gradient is
$$
\boxed{\;
\frac{\partial F}{\partial x_j}(x) =
\begin{cases}
j\,\sin x_j + 2\,\cos x_j, & 1 \le j \le n-1,\\[4pt]
n\,\sin x_n - (n-1)\,\cos x_n, & j = n.
\end{cases}
\;}
$$

### 1.3 Hessian

A glance at the formula above shows that
$\partial F / \partial x_j$ depends **only on $x_j$**. Therefore every
mixed second derivative vanishes:
$$
\frac{\partial^{2} F}{\partial x_i\,\partial x_j} = 0
\qquad (i \neq j),
$$
and the Hessian $H(x) = \nabla^{2} F(x)$ is **diagonal**. Differentiating
the gradient component again with respect to the same variable gives:
$$
\boxed{\;
H_{jj}(x) =
\begin{cases}
j\,\cos x_j - 2\,\sin x_j, & 1 \le j \le n-1,\\[4pt]
n\,\cos x_n + (n-1)\,\sin x_n, & j = n,
\end{cases}
\qquad H_{ij}(x) = 0 \text{ for } i \neq j.\;}
$$

The diagonal structure makes Problem 16 the *easy* benchmark in our
comparison: the Newton system $H d = -\nabla F$ reduces to $n$
independent scalar divisions, and finite-difference approximation of
$H$ via column-by-column perturbation of $\nabla F$ requires a single
combined perturbation step (CPR with one color).

### 1.4 Sanity check at $\bar{x} = (1, \dots, 1)$ for $n = 2$

Using $\sin 1 \approx 0.8415$, $\cos 1 \approx 0.5403$:
$$
\nabla F(\bar x) =
\begin{pmatrix}
\sin 1 + 2\cos 1 \\
2\sin 1 - \cos 1
\end{pmatrix}
\approx
\begin{pmatrix}
1.9221 \\
1.1426
\end{pmatrix},
$$
$$
H(\bar x) =
\operatorname{diag}\!\bigl(\cos 1 - 2\sin 1,\; 2\cos 1 + \sin 1\bigr)
\approx
\operatorname{diag}(-1.1426,\; 1.9221).
$$

---

## 2. Problem 28 — Variably dimensioned function

### 2.1 Statement

For $n \ge 1$ and $m = n+2$ define
$$
F(x) = \tfrac12 \sum_{k=1}^{m} f_k(x)^{2},
$$
with
$$
f_k(x) =
\begin{cases}
x_k - 1, & 1 \le k \le n,\\[4pt]
\displaystyle\sum_{i=1}^{n} i\,(x_i - 1), & k = n+1,\\[6pt]
\displaystyle\biggl(\sum_{i=1}^{n} i\,(x_i - 1)\biggr)^{\!2}, & k = n+2,
\end{cases}
$$
suggested starting point $\bar{x}_l = 1 - l/n$, $l = 1, \dots, n$, and
global minimum $x^* = (1, \dots, 1)$ with $F(x^*) = 0$.

To avoid rewriting the same weighted sum three times we introduce the
purely **notational abbreviation**
$$
S(x) \;:=\; \sum_{i=1}^{n} i\,(x_i - 1),
\qquad\text{so that}\qquad
f_{n+1}(x) = S(x),\quad f_{n+2}(x) = S(x)^{2}.
$$
With this shorthand,
$$
F(x) = \tfrac12 \Bigl[\,\sum_{k=1}^{n} (x_k - 1)^{2} \;+\; S(x)^{2} \;+\; S(x)^{4}\,\Bigr].
$$

Note the basic chain-rule identity that we will use throughout:
$$
\frac{\partial S}{\partial x_j} \;=\; j, \qquad j = 1, \dots, n.
$$

### 2.2 Gradient

Differentiate $F$ term by term with respect to $x_j$:
$$
\begin{aligned}
\frac{\partial}{\partial x_j}\;\Bigl[\tfrac12 \sum_{k=1}^{n}(x_k - 1)^{2}\Bigr]
   &= x_j - 1, \\[4pt]
\frac{\partial}{\partial x_j}\;\bigl[\tfrac12\,S^{2}\bigr]
   &= S \cdot \frac{\partial S}{\partial x_j} \;=\; j\,S, \\[4pt]
\frac{\partial}{\partial x_j}\;\bigl[\tfrac12\,S^{4}\bigr]
   &= 2\,S^{3} \cdot \frac{\partial S}{\partial x_j} \;=\; 2\,j\,S^{3}.
\end{aligned}
$$
Summing the three contributions:
$$
\boxed{\;
\frac{\partial F}{\partial x_j}(x) \;=\; (x_j - 1) \;+\; j\,S\,\bigl(1 + 2\,S^{2}\bigr),
\qquad j = 1, \dots, n.\;}
$$

The gradient can be evaluated in $O(n)$: one pass computes $S$ from
$x$, then a second pass forms the closed-form expression above.

### 2.3 Hessian

Differentiate the $j$-th gradient component once more, this time with
respect to $x_i$ ($i \in \{1, \dots, n\}$):
$$
\begin{aligned}
\frac{\partial}{\partial x_i}\bigl[\,x_j - 1\,\bigr]
   &= \delta_{ij}, \\[6pt]
\frac{\partial}{\partial x_i}\Bigl[\,j\,S\,(1 + 2\,S^{2})\Bigr]
   &= j\,\biggl[
       \frac{\partial S}{\partial x_i}\,(1 + 2\,S^{2})
       \;+\; S \cdot 4\,S\,\frac{\partial S}{\partial x_i}
       \biggr] \\[2pt]
   &= j \cdot i \cdot \bigl(1 + 2\,S^{2} + 4\,S^{2}\bigr) \\[2pt]
   &= i\,j\,(1 + 6\,S^{2}).
\end{aligned}
$$
Summing the two contributions yields
$$
\boxed{\;
H_{ij}(x) \;=\; \delta_{ij} \;+\; i\,j\,(1 + 6\,S(x)^{2}),
\qquad i, j = 1, \dots, n.\;}
$$

In contrast with Problem 16, the Hessian of Problem 28 is **dense**:
the off-diagonal entry $H_{ij}$ never vanishes, because $S$ couples
every variable to every other. Each row of $H$ is a scaled copy of the
vector $(1, 2, \dots, n)$ with the diagonal shifted by one. This is
the structural reason why CPR coloring of the gradient does not help
for Problem 28 (the sparsity graph is complete) and why we resort to
matrix-free Hessian-vector products in the Truncated Newton path when
$n$ is large.

### 2.4 Sanity check at $x = (0,\;0.5)$ for $n = 2$

Here $S = 1\cdot(0-1) + 2\cdot(0.5-1) = -2$, so $1 + 2S^{2} = 9$ and
$1 + 6S^{2} = 25$. The closed-form formulas give
$$
\nabla F(x) =
\begin{pmatrix}
(0-1) + 1 \cdot (-2) \cdot 9 \\
(0.5-1) + 2 \cdot (-2) \cdot 9
\end{pmatrix}
=
\begin{pmatrix}
-19 \\
-36.5
\end{pmatrix},
$$
$$
H(x) =
\begin{pmatrix}
1 + 1\cdot 1\cdot 25 & 1\cdot 2 \cdot 25 \\
1\cdot 2 \cdot 25     & 1 + 2\cdot 2 \cdot 25
\end{pmatrix}
=
\begin{pmatrix}
26 & 50 \\
50 & 101
\end{pmatrix}.
$$

These reference values can be used to cross-check both the analytical
implementation in `src/gradients/problem28.py` and
`src/hessians/problem28.py` and the finite-difference approximations in
`src/hessians/finite_diff.py`.
