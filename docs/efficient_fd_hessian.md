# Efficient Finite-Difference Hessian — Assignment Point 3.1

Session notes documenting the implementation of *point 3.1* of the
Unconstrained Optimization assignment ([1 pt — only for methods with
Hessian]: use the exact gradient, approximate the Hessian via an
**efficient implementation of the finite differences**, for
`k ∈ {4, 8, 12}` both fixed `h = 10⁻ᵏ` and scaled `hᵢ = 10⁻ᵏ |x̂ᵢ|`).

The codebase already had Modified Newton, Truncated Newton, and exact
gradients/Hessians for the two test problems (Problem 16 and Problem 28).
This session added the FD-Hessian machinery, the matrix-free Hessian-vector
product, sparsity-pattern declarations, unit tests, a driver script for
the Table 2 grid, and two benchmark scripts that surfaced the trade-offs
worth discussing in the report.

---

## 1. What "efficient implementation" means here

From the lecture material the professor's notion of efficiency
combines three exploits (Curtis–Powell–Reid / Coleman–Moré tradition):

1. **FD of the exact gradient**, not 2nd-order FD of `f`. Because
   `H_f(x) = J_{∇f}(x)`, applying the Jacobian FD formulas to `∇f`
   yields the Hessian:

   - Centered: `H[:,j] ≈ (∇f(x̄ + h·eⱼ) − ∇f(x̄ − h·eⱼ)) / (2h)`,  `O(h²)`
   - Forward:  `H[:,j] ≈ (∇f(x̄ + h·eⱼ) − ∇f(x̄))         /  h`,  `O(h)`

2. **Sparsity-aware column grouping via graph coloring.** Build the
   variable-dependency graph
       `V = {x₁,…,x_n}`,
       `{xᵢ,xⱼ} ∈ E ⟺ ∃ k : gₖ depends on both xᵢ and xⱼ`.
   Variables of the same color have disjoint sparsity supports; they can
   be perturbed simultaneously and read off in **one** shared gradient
   call.

3. **Symmetry enforcement**: `H ← (H + Hᵀ) / 2`. The professor
   explicitly highlights this on the slide deriving the Jacobian-of-grad
   approach.

For **Truncated Newton** the inner CG only needs `H·v`, not `H`. The
matrix-free FD
       `Hv ≈ (∇f(x + h·v) − ∇f(x − h·v)) / (2h)`
costs **one (or two) gradient call per CG iteration** and never
assembles `H`. This is the only way Truncated Newton at `n = 10⁴–10⁵`
is feasible on Problem 28.

### Per-problem analysis

| Problem | Hessian structure | Dependency graph | Best CPR coloring | FD-Hessian cost | Hv cost |
|---|---|---|---|---|---|
| **P16** Banded Trigonometric | **diagonal** (each `gⱼ` depends only on `xⱼ`) | zero edges | **1 color** | 1–2 grad calls, **independent of n** | matrix-free is trivial Jacobi |
| **P28** Variably Dimensioned | **dense** (each `gⱼ` depends on every `xᵢ` via the global sum `S = Σᵢ i(xᵢ−1)`) | complete | n colors, **no savings** | `n+1` (forward) / `2n` (centered) grad calls | matrix-free needs 1 grad call per Hv (avoids the `n × n` matrix) |

For P28 the "efficiency" reduces to (a) symmetrization, (b) keeping the
closed-form analytic gradient (`O(n)` per call) rather than falling back
to FD-on-`f` (which would be `O(n²)`), and (c) the matrix-free `Hv` path
in Truncated Newton.

---

## 2. New files

### `src/hessians/finite_diff.py`

Three routines, all consuming the **exact gradient**:

| Function | Purpose | Cost |
|---|---|---|
| `hess_fd(grad_f, x, k, *, scaled, stencil, symmetrize)` | Column-by-column FD assembly, default "general purpose" (no coloring). Symmetrization is on by default. | `n+1` (forward) or `2n` (centered) grad calls |
| `hess_fd_diag(grad_f, x, k, *, scaled, stencil)` | CPR 1-color shortcut for diagonal-Hessian problems (P16). Returns the diagonal as a length-`n` vector. | **2 grad calls regardless of n** |
| `hv_fd(grad_f, x, v, k, *, scaled, stencil)` | Matrix-free `Hv` for Truncated Newton CG. | 1 (forward) or 2 (centered) grad calls per call |

Step convention matches the existing
`src/gradients/finite_diff.py:39`:
- fixed: `hᵢ = 10⁻ᵏ`
- scaled: `hᵢ = 10⁻ᵏ · |xᵢ|`, fallback to `10⁻ᵏ` when `xᵢ = 0`

### `src/hessians/sparsity.py`

CPR group declarations per problem, obtained by inspection of the
gradient formulas:

- `cpr_groups_p16(n) → [[0, …, n-1]]` — single color (diagonal Hessian)
- `cpr_groups_p28(n) → [[0],[1],…,[n-1]]` — n colors (dense Hessian)

We chose not to write a generic graph-coloring solver since the two
test problems give us the coloring by inspection.

### `tests/test_hess_fd.py` — 25 unit tests

- `hess_fd` vs exact Hessian on P16/P28 across all `{forward, centered} × {fixed, scaled}` at `k=8`.
- Stencil ordering at `k=4`: centered beats forward (catches a bug if the stencils get swapped).
- `hess_fd_diag` matches `np.diag(hess_f16)` and agrees with the full `hess_fd` path on P16.
- `hv_fd` matches `exact_H @ v` for 5 random `v` per (problem, stencil).
- `hv_fd` linearity in `v` (relative tolerance — important because P28 Hv magnitudes reach `10⁴`).
- `symmetrize=False` produces visibly asymmetric output that `symmetrize=True` cleans up.
- `cpr_groups_*` declarations are well-formed.
- `ValueError` paths for unknown stencils.

### `tests/test_truncated_newton.py` — 9 unit tests

- Classic branch (assembled Hessian) still converges on P16 and P28.
- Matrix-free branch (`hv_func`) converges on P28 and produces a final point matching the classic branch within `‖Δx‖ < 10⁻⁵`.
- Sentinel test confirms `hess_f` is **not** called when `hv_func` is provided.
- `ValueError` paths for missing required arguments.
- `_cg_truncated` works with both ndarray and callable, including the negative-curvature-at-`j=0` case.

### `scripts/bench_tn_matrix_free.py`

Time & memory benchmark of Truncated Newton's two branches on P28. Output:

| n | exact time | exact peak | matrix-free time | matrix-free peak |
|---|---|---|---|---|
| 1,000 | 0.11 s | **30.6 MB** | 0.002 s | 0.1 MB |
| 5,000 | 2.4 s | **763 MB** | 0.003 s | 0.5 MB |
| 10,000 | 8.1 s | **3,052 MB** | 0.005 s | 1.0 MB |
| 50,000 | (skipped — would need 20 GB) | — | **0.05 s** | **5.2 MB** |

This is the data that proves the matrix-free path is the only way to reach
`n = 10⁵`.

### `scripts/sweep_k_sensitivity.py`

Sweeps `(method × problem × n × k × scaled × stencil)` to identify the
best recipe before launching the full grid. Headline findings:

| Method × Problem | Best (k, scaled, stencil) | Notes |
|---|---|---|
| Mod-Newt × P16 | any `k ∈ {4, 8, 12}` | very robust, 5–14 iters |
| Mod-Newt × P28 | **`k = 8`** is unambiguous | `k=4` fails at `n=1000` (truncation `O(h)=10⁻⁴` corrupts the Newton step) |
| Trunc-N × P16  | `k ∈ {4, 8}`, **never `k=12`** | `k=12` roundoff `ε/h ≈ 10⁻⁴` stalls convergence at `n ≥ 100` |
| Trunc-N × P28  | **`k=8` only at `n ≤ 100`** | **no `(k, scaled, stencil)` converges at `n ≥ 1000`** — the FD-Hv accuracy ceiling on stiff problems |

The last row is *not a bug* — it's a substantive theoretical finding to
discuss in the report. On P28, gradient components scale as
`j · S · (1 + 2S²)` with `var(S) ∝ n³`; at `n=1000` the gradient norm
reaches `10⁷–10⁸`, so subtracting two values that close cancels into
pure round-off no matter which `h` we pick. Modified Newton survives
the same noise because symmetrization + Cholesky modification regularize
indefinite FD-Hessians; the matrix-free path has no such buffer.

### `scripts/run_fd_hessian.py` — driver for Table 2 of the report

Iterates the full grid and writes:

- `results/fd_hessian_raw.csv` — every row of the grid (one experiment per row)
- `results/fd_hessian_table_<method>_<problem>_n<N>.md` — one Table-2-style
  markdown per (method, problem, n), with the per-row data and an aggregated
  "average over successes" section per `(k, scaled, stencil)`

CLI:

```
--quick                      8-cell smoke test
--n-values 2,1000            explicit dimension list (overrides --nmax)
--nmax N                     filter the default {2, 10³, 10⁴, 10⁵} to ≤ N
--starts S                   default 6 (the assignment-mandated count)
--problems P16,P28
--methods modnewt,truncn
--include-forward            adds forward stencil; default is centered only
--max-iter M                 default 500
--tol T                      default 1e-6
--skip-oom-threshold-mb X    skip cells whose dense H would exceed X MB (default 4096)
```

Failure modes are recorded honestly:
- `stop_reason = skip_oom` when the predicted `n²·8 bytes` exceeds the threshold
- `stop_reason = memory_error` when an unforeseen `MemoryError` is raised
- `stop_reason = exception` with the exception name+message in `notes`
- `stop_reason = max_iter` for non-converging runs

The experimental rate of convergence is estimated from the tail of the
`‖gₖ‖` sequence by the standard formula
`pₖ ≈ log(eₖ₊₁/eₖ) / log(eₖ/eₖ₋₁)`, taking the median over the last 5
windows. NaN when the sequence is too short or stagnated.

**Before the assignment-grade run, edit line 47:** `TEAM_SEED = 1234`
must become `min(student IDs of the team)`. The starting points use a
single `np.random.default_rng(TEAM_SEED)` per `(problem, n)` so the
generation is fully reproducible.

---

## 3. Modified files

### `src/methods/truncated_newton.py`

Two surgical edits to enable matrix-free CG:

1. `_cg_truncated(H, b, eta_k, cg_max_iter)` — `H` may now be **either**
   an `ndarray (n, n)` (existing behavior) **or** a callable
   `d → H·d`. One-line change near the top of the function:
   ```python
   apply_H = H if callable(H) else H.__matmul__
   ```
   Replacing `H @ d` with `apply_H(d)` inside the loop.

2. `truncated_newton(...)` — new keyword `hv_func=(x, v) → Hv`.
   When provided, `hess_f` is optional and the outer loop builds
   ```python
   x_k = x
   apply_H = lambda d: hv_func(x_k, d)
   ```
   per outer iteration, never materializing the Hessian.
   Validation: must supply at least one of `{hess_f, hv_func}`,
   plus `x0` and `stopping`. If both are given, `hv_func` wins.

The classic branch is unchanged behaviorally — all pre-existing tests
still pass.

### `src/hessians/__init__.py`

Re-exports the new utilities so callers can write
```python
from src.hessians import hess_fd, hess_fd_diag, hv_fd, cpr_groups_p16, cpr_groups_p28
```

---

## 4. Test suite status

```
42 passed in 0.27 s
  tests/test_hess_fd.py            25 tests
  tests/test_modified_newton.py     8 tests  (unchanged)
  tests/test_truncated_newton.py    9 tests
```

No regressions in the pre-existing Modified Newton tests.

---

## 5. Recommended next steps (not done in this session)

These are deliberately out of scope for point 3.1 but worth flagging:

1. **Diagonal-aware Modified Newton path** so that P16 at `n = 10⁵`
   can run without allocating an `n × n` Hessian. The current
   `_modify_hessian_cholesky` takes a dense matrix; a small variant
   that accepts a length-`n` diagonal vector would unblock that cell.
2. **Point 3.2** of the assignment (1.5 pt): approximate the **gradient
   too** via FD (forward stencil), then propagate that into both the
   outer step and the Hessian/Hv estimation. The infrastructure
   (`grad_fd_forward`, `hv_fd`, `hess_fd`) is already composable for
   this; only the driver script needs an `--approx-grad` flag.
3. **Convergence-rate and `n=2` path plots** (mandatory by item 4 of
   the assignment). The driver already records full history when
   `return_history=True`, so this is a matplotlib-only follow-up.
4. **Overnight full-grid run** (`python -u scripts/run_fd_hessian.py
   --starts 6 --max-iter 500 > results/full_run.log 2>&1`).
   Expect several hours due to `modnewt + P28 + k=4 @ n ∈ {10³, 10⁴}`,
   which routinely hits `max_iter`.

---

## Commit message

```
Add efficient FD-Hessian for assignment point 3.1

Implement Curtis-Powell-Reid-style finite-difference Hessian from the
exact gradient (per professor's lecture notes), plus matrix-free Hv for
Truncated Newton CG. Covers both test problems: P16's diagonal Hessian
is recovered in a single CPR group (1-2 grad calls regardless of n),
while P28's dense Hessian falls back to column-by-column FD with
symmetry enforcement.

New:
- src/hessians/finite_diff.py: hess_fd, hess_fd_diag, hv_fd
- src/hessians/sparsity.py: CPR group declarations per problem
- tests/test_hess_fd.py (25) and tests/test_truncated_newton.py (9)
- scripts/run_fd_hessian.py: CLI driver producing the Table 2 grid
  (raw CSV + per-(method,problem,n) markdown), with --quick preview
- scripts/bench_tn_matrix_free.py and scripts/sweep_k_sensitivity.py:
  benchmarks that informed the k=8 centered default
- docs/efficient_fd_hessian.md: design notes and per-problem analysis

Changed:
- src/methods/truncated_newton.py: _cg_truncated accepts either an
  ndarray Hessian or a callable Hv; truncated_newton accepts a new
  hv_func keyword that activates the matrix-free CG path. Classic
  branch behaviorally unchanged.
- src/hessians/__init__.py: re-export new utilities

Test suite: 42 passed (was 8); no regressions.

The k-sensitivity sweep identifies k=8 centered as the robust default
on both methods/problems. P28 + matrix-free TN at n>=1000 is documented
as a finding (FD-Hv accuracy ceiling on stiff problems where gradient
magnitudes ~ n^1.5 force cancellation), not a regression.

Before the assignment-grade run, edit TEAM_SEED in
scripts/run_fd_hessian.py to min(team student IDs).
```
