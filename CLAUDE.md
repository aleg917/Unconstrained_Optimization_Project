# Unconstrained Optimization Project — NO4LSP (PoliTo, A.Y. 2025/2026)

Assignment 2.1: derivative-based unconstrained optimization. Team seed: 323334.

## Project Structure

```
src/
  functions/       problem16.py, problem28.py  — f(x), x_bar(n)
  gradients/       problem16.py, problem28.py  — exact grad_f(x)
                   finite_diff.py              — forward FD gradient (grad_fd)
  hessians/        problem16.py, problem28.py  — exact hess_f(x)
                   finite_diff.py              — FD Hessian (hess_fd) + Hessian-vector (hv_fd)
  methods/         modified_newton.py          — Modified Newton (Cholesky + tau modification)
                   truncated_newton.py         — Truncated Newton (Linesearch Newton-CG)
                   armijo_backtracking.py      — shared Armijo line search
  stopping_criteria/
    base.py                                    — StoppingCriterion ABC
    absolute/  grad_norm, f_change, x_change
    relative/  grad_norm, f_change, x_change
  starting_points.py                           — generate_starting_points(x_bar, num_random, rng)
  finite_diff_factories.py                     — factory for FD gradient/Hessian callables

main.ipynb          — full experiment notebook (tables, plots, convergence analysis)
fine_tuning.ipynb   — grid search for algorithm parameters + stopping criteria
debug_testing.ipynb — scratch/debugging
docs/               — design docs and analysis
results/            — CSV outputs from fine_tuning
```

## Test Problems

### Problem 16 — Banded Trigonometric
- F(x) = sum_i i * [(1 - cos x_i) + sin x_{i-1} - sin x_{i+1}], x_0=x_{n+1}=0
- **Hessian: diagonal** (sparse CSC), O(n) per evaluation
- Non-convex, multiple local minima
- Starting point: x_bar_i = 1 for all i

### Problem 28 — Variably Dimensioned
- F(x) = (1/2) * [sum_k(x_k - 1)^2 + S^2 + S^4], S = sum_i i*(x_i - 1)
- **Hessian: dense** H = I + (1+6S^2)*j*j^T, O(n^2) storage
- Convex, unique minimum at x* = (1,...,1), F(x*) = 0
- Starting point: x_bar_l = 1 - l/n
- ||grad(x_bar)|| = O(n^7) — enormous initial gradient

## Algorithms

### Modified Newton (`modified_newton`)
Key params: `alpha0, c1, rho, max_iter_backtrack` (Armijo) + `beta, max_tau_iter` (Cholesky mod)
- Modifies Hessian via B = H + tau*I until Cholesky succeeds
- Solves B*p = -g exactly

### Truncated Newton (`truncated_newton`)
Key params: `alpha0, c1, rho, max_iter_backtrack` (Armijo) + `forcing, cg_max_iter` (CG inner)
- Solves H*p = -g inexactly via truncated CG
- Forcing sequence: 'linear' (eta=0.5), 'superlinear' (eta=min(0.5,sqrt(||g||))), 'quadratic' (eta=min(0.5,||g||))
- Falls back to steepest descent on negative curvature

### Armijo Backtracking (shared)
`armijo_backtracking(f, x, fx, g, d, alpha0=1.0, c1=1e-4, rho=0.5, max_iter=50)`
- Condition: f(x + alpha*d) <= f(x) + c1*alpha*<g, d>

## Full-mode Results (DIMS=[2,1000,10000,100000], 6 starts, 20s time_limit, max_iter=5000, stop=GradNormAbsolute(1e-8))

Selection hierarchy: avg_success → avg_iter → avg_time (avg_grad sanity check only).

Best Modified Newton: beta=1e-2, rho=0.5 → 61.9% avg success, 14.9 avg iter, 1.13s
  - Overall = Best P16 (success_P16=58.3%, iter_P16=8.8)
  - Best P28: beta=1e-2, rho=0.9 (success_P28=66.7%, iter_P28=21.5) — but β/ρ are inert on P28
  - beta=1e-2 wins because it lets P16 n=1000 converge (success 1.0 vs 0.17 for beta=1e-6/1e-3)
Best Truncated Newton: forcing=quadratic, rho=0.9 → 62.5% avg success, 22.9 avg iter, 0.28s
  - Overall = Best P16 (success_P16=75.0%, iter_P16=23.9)
  - Best P28: forcing=quadratic, rho=0.5 (success_P28=50.0%, iter_P28=21.5)
  - rho=0.9 now wins overall (vs old rho=0.5): more robust on P16 at larger n

## Tuning Decisions (see docs/tuning_analysis.md §0)

Fixed parameters follow the course lab specs (NO4LSP_DellaSanta_2526.pdf §7.4
+ Lab 13 MATLAB script): tolgrad=1e-8, c1=1e-4, max_iter_backtrack=50,
max_iter=5000 (PDF §7.4 value; lab script uses 1000), alpha0=1.0. Forcing
terms (linear/superlinear/quadratic) identical to lab §7.1.

**Fixed parameters (both algorithms):** alpha0=1.0, c1=1e-4, max_iter_backtrack=50
**Fixed MN:** max_tau_iter=100
**Fixed TN:** cg_max_iter=None (lab uses n/2; difference negligible — CG
converges well before n iter)

**Grid (15 combos total):**
- MN (9): beta in {1e-6, 1e-3, 1e-2} x rho in {0.5, 0.75, 0.9}
- TN (6): forcing in {'superlinear', 'quadratic'} x rho in {0.5, 0.75, 0.9}

Note: Modified Newton with τ-adjustment is our extension (lab §6.1 mentions
"H + Correction" but defers to external reference). All β tuning is ours.

## Conventions

- Python 3.12+, NumPy/SciPy
- Exact derivatives always available; FD variants for comparison
- Stopping criterion is pluggable via StoppingCriterion interface
- TIME_LIMIT = 20s per run; MAX_ITER = 5000
- Seed for random starting points: 323334
