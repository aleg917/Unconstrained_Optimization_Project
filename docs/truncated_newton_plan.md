# Plan — Implement `src/methods/truncated_newton.py`

## Context

The file `src/methods/truncated_newton.py` currently contains only pseudocode comments. The user needs the **Linesearch Newton-CG method** (a.k.a. Truncated Newton, Inexact Newton via CG) as the third solver in the project, alongside `steepest_descent` and `modified_newton`. The implementation must:

- Mirror the public API of `modified_newton` (callable signature, return-dict shape, integration with the `StoppingCriterion` plug-in family).
- Follow the slides exactly: CG inner solve for `B_k z = c_k` with negative-curvature exit at any `j`, inexact Newton stop `||B_k z − c_k|| ≤ η_k ||c_k||`, line search with Armijo + backtracking on the outer step.
- Expose the forcing sequence `η_k` as a parameter so the three regimes from Thm 6.2 in the slides (linear / superlinear / quadratic) can be compared empirically.
- Track diagnostics (CG inner iters, negative-curvature events, η_k, termination reason) in `history` so plots/analysis can verify the theoretical behaviour described on the last slide.

## Files

- **Modify**: `src/methods/truncated_newton.py` (currently a stub with pseudocode).
- **Update**: `src/methods/__init__.py` — add `truncated_newton` to exports (currently exports `steepest_descent`, `armijo_backtracking`, `modified_newton`).
- **Reuse (no changes)**:
  - `src/methods/armijo_backtracking.py` → outer line search
  - `src/stopping_criteria/base.py` → `StoppingCriterion` plug-in (init + `should_stop(k+1, ...)`)

## Design

### Signature

```python
def truncated_newton(
    f, grad_f, hess_f, x0,
    stopping: StoppingCriterion,
    *,
    alpha0=1.0, c1=1e-4, rho=0.5,           # outer Armijo
    forcing="superlinear",                  # "linear" | "superlinear" | "quadratic" | float
    eta_fixed=0.5,                          # used when forcing == "linear" or a float in (0,1)
    cg_max_iter=None,                       # default = n (dimension of x0)
    max_iter=1000,
    return_history=False,
) -> dict
```

Hessian interface: **full `hess_f(x)` is called once per outer iter** (matvecs done as `H @ d` on the cached matrix). This matches `modified_newton`'s signature and keeps the test-problem callables unchanged. A truly matrix-free `hessp` variant can be added later if needed; the slides leave the choice open (they only specify the CG recurrence `B_k d`).

### Forcing sequence (Thm 6.2)

```python
g_norm = ||grad_f(x_k)||
if forcing == "linear":      eta_k = float(eta_fixed)              # case i): 0 < η ≤ η_bar < 1
if forcing == "superlinear": eta_k = min(0.5, sqrt(g_norm))        # case ii): η_k → 0
if forcing == "quadratic":   eta_k = min(0.5, g_norm)              # case iii): η_k = O(||g_k||)
if isinstance(forcing, float): eta_k = float(forcing)              # explicit constant
```

### Inner CG (private helper `_cg_truncated`)

Solve `H z = b` with `b = -grad_f(x_k)`, starting from `z = 0`, `r = b`, `d = b`.
Per outer iter, the inner loop runs up to `cg_max_iter` and terminates with one of:

| Termination tag | Condition | Returned step |
|---|---|---|
| `"negcurv_j0"`  | `dᵀHd ≤ 0` at `j == 0` (case ii)  | `z = b = -grad` (steepest descent) |
| `"negcurv_jpos"`| `dᵀHd ≤ 0` at `j > 0`  (case iii) | last valid `z_j` |
| `"tol"`         | `||r_new|| ≤ η_k ||b||` (case i)  | `z` after that update |
| `"maxiter"`     | inner cap hit                     | last `z` |

CG body (standard recurrences, with the sign-corrected residual update; the pseudocode in the stub appears to have `r ← r + αHd` which would diverge — correct form is `r ← r − αHd`):

```python
for j in range(cg_max_iter):
    Hd  = H @ d
    dHd = float(d @ Hd)
    if dHd <= 0:
        return (b.copy(), 0, "negcurv_j0") if j == 0 else (z, j, "negcurv_jpos")
    alpha   = rTr / dHd
    z       = z + alpha * d
    r_new   = r - alpha * Hd
    if np.linalg.norm(r_new) <= eta_k * norm_b:
        return z, j + 1, "tol"
    rTr_new = float(r_new @ r_new)
    beta    = rTr_new / rTr
    d       = r_new + beta * d
    r, rTr  = r_new, rTr_new
return z, cg_max_iter, "maxiter"
```

### Outer loop (mirrors `modified_newton`)

```python
x  = np.asarray(x0, dtype=float).copy()
fx = f(x); g = grad_f(x)
stopping.initialize(x, fx, g)
cg_iters_total = 0
neg_curv_count = 0
for k in range(max_iter):
    H      = hess_f(x)
    b      = -g
    eta_k  = _compute_eta(forcing, eta_fixed, g)
    p, cg_j, cg_term = _cg_truncated(H, b, eta_k, cg_max_iter)
    cg_iters_total  += cg_j
    if cg_term.startswith("negcurv"): neg_curv_count += 1

    alpha, _ = armijo_backtracking(f, x, fx, g, p, alpha0, c1, rho)
    x_prev, fx_prev = x.copy(), fx
    x  = x + alpha * p
    fx = f(x); g = grad_f(x)

    if return_history:
        history.append({"x": x.copy(), "f": float(fx),
                        "grad_norm": float(np.linalg.norm(g)),
                        "alpha": float(alpha), "p_norm": float(np.linalg.norm(p)),
                        "cg_iters": cg_j, "nu_k": float(eta_k),
                        "cg_termination": cg_term})

    if stopping.should_stop(k + 1, x, fx, g, x_prev, fx_prev):
        return _build_result(success=True, stop_reason=stopping.name, ...)

return _build_result(success=False, stop_reason="max_iter", ...)
```

### Returned dict

```
x_star, f_star, grad_norm, n_iter, success, stop_reason,
cg_iters_total, neg_curvature_count,
[history]   # if return_history=True
```

(Per-outer-iter `nu_k`, `cg_iters`, `cg_termination` live inside `history` entries.)

## Verification

1. **Sanity unit run** — at the bottom of the file under `if __name__ == "__main__"` (or in a small ad-hoc script), call on `problem16` / `problem28` from `src/functions/`, with `x_bar_16(n)` / `x_bar_28(n)` and `GradNormAbsolute(tol=1e-6)`. Expect `success=True` and `n_iter` of the same order as `modified_newton` on the same input.
2. **Three-regime sweep** (matches the last slide's prediction): run with `forcing="linear"`, `"superlinear"`, `"quadratic"` and confirm:
   - linear → many outer iters, small `cg_iters_total / n_iter` ratio
   - quadratic → few outer iters, large `cg_iters_total / n_iter` ratio
   - superlinear → in between
3. **Negative-curvature path** — construct a small non-convex quadratic (e.g. `H = diag([1, -1])`) and confirm `neg_curvature_count > 0` and `success=True` (Armijo can still make progress because `p` is a descent direction in all three slide cases).
4. **API parity** — `truncated_newton(...)` should be swappable with `modified_newton(...)` in any existing test invocation (same positional args + same `stopping=` kwarg).
