# Stopping Criteria: Design Rationale and Tolerance Selection

## 1. Why Three Criteria

We implement three stopping criteria, each monitoring a different quantity:

1. **Gradient norm** $\|\nabla f(x^{(k)})\|$ — tests the first-order necessary condition for optimality. When the gradient is small, the iterate is near a stationary point. This is the most direct theoretical measure.

2. **Function change** $|F(x^{(k)}) - F(x^{(k-1)})|$ — detects stagnation. When the objective function stops decreasing meaningfully, the method has either converged or is stuck in a flat region. This catches cases where the gradient is still large but the iterates zig-zag without progress (common in ill-conditioned problems).

3. **Iterate change** $\|x^{(k)} - x^{(k-1)}\|$ — detects step collapse. When the Armijo backtracking reduces the step size $\alpha$ to tiny values, the iterates barely move even though the gradient may not be small. This is an important safety net for problems where the Hessian is nearly singular.

Each criterion catches failure modes the others miss. Using them in OR-combination (stop when **any** fires) provides robust termination.

## 2. Absolute vs Relative Variants

Each criterion has two variants:

### Absolute
Uses a fixed tolerance threshold:
- $\|\nabla f(x^{(k)})\| \leq \tau_g$
- $|F^{(k)} - F^{(k-1)}| \leq \tau_f$
- $\|x^{(k)} - x^{(k-1)}\| \leq \tau_x$

**Pros**: Simple, interpretable. The tolerance directly specifies the desired accuracy.
**Cons**: Not scale-invariant. The same $\tau_g = 10^{-6}$ means very different things depending on the problem's natural scale.

### Relative
Normalizes by a reference value:
- $\|\nabla f(x^{(k)})\| / \|\nabla f(x^{(0)})\| \leq \tau_g$ — relative to the initial gradient
- $|F^{(k)} - F^{(k-1)}| / \max(|F^{(k)}|, 1) \leq \tau_f$ — relative to the current function value
- $\|x^{(k)} - x^{(k-1)}\| / \max(\|x^{(k)}\|, 1) \leq \tau_x$ — relative to the current iterate norm

**Pros**: Scale-invariant. Works robustly across problems with different magnitudes.
**Cons**: Can be misleading when $\|\nabla f(x^{(0)})\|$ or $\|x^{(k)}\|$ is very small (the denominator approaches zero, making the criterion trivially easy or impossible to satisfy).

**Important** (from course slides): Setting TOL equal to machine precision $\varepsilon_m \approx 10^{-16}$ is **not acceptable** for relative criteria, because rounding errors dominate at that scale and the criterion may never be satisfied.

## 3. Tolerance Magnitude Relationships

### The key insight: different criteria operate on different scales

Near a minimum $x^*$, with Newton-type step $\alpha \approx 1$:

$$\|x^{(k+1)} - x^{(k)}\| = \alpha \|p^{(k)}\| \approx \|\nabla^2 f(x^{(k)})^{-1} \nabla f(x^{(k)})\| \approx \frac{\|\nabla f(x^{(k)})\|}{\lambda_{\min}(\nabla^2 f)}$$

For well-conditioned problems where $\lambda_{\min} \approx O(1)$:
$$\|x^{(k+1)} - x^{(k)}\| \approx \|\nabla f(x^{(k)})\|$$

so $\tau_x$ and $\tau_g$ can be on the same order.

For the function change, by Taylor expansion:
$$|F(x^{(k+1)}) - F(x^{(k)})| \approx |\nabla f(x^{(k)})^T (x^{(k+1)} - x^{(k)})| \approx \|\nabla f(x^{(k)})\| \cdot \|x^{(k+1)} - x^{(k)}\| \approx \|\nabla f(x^{(k)})\|^2$$

Therefore:
$$\tau_f \approx \tau_g^2$$

### Example

If we want gradient accuracy $\tau_g = 10^{-6}$:
- $\tau_x \approx 10^{-6}$ to $10^{-8}$ (same order, possibly tighter to account for condition number)
- $\tau_f \approx (10^{-6})^2 = 10^{-12}$

## 4. Three Tolerance Levels

Based on the course slides (defined for relative x-change) and the magnitude relationships above:

### Rough ($10^{-4}$)
| Criterion | Tolerance |
|-----------|-----------|
| grad norm | $10^{-4}$ |
| x change | $10^{-4}$ |
| f change | $10^{-8}$ |

Fast convergence, low precision. Useful for getting a rough estimate quickly, or when the solution only needs a few significant digits.

### Good ($10^{-8}$)
| Criterion | Tolerance |
|-----------|-----------|
| grad norm | $10^{-8}$ |
| x change | $10^{-8}$ |
| f change | $10^{-16}$ |

Recommended default. Provides a good balance of precision and computational cost.

**Warning**: $\tau_f = 10^{-16}$ is at double-precision machine epsilon. The f-change criterion at this band is borderline — it may be affected by floating-point rounding. In practice, the gradient or x-change criterion will fire first.

### Very Good ($10^{-12}$)
| Criterion | Tolerance |
|-----------|-----------|
| grad norm | $10^{-12}$ |
| x change | $10^{-12}$ |
| f change | $10^{-24}$ (not feasible) |

Very demanding. Often unnecessary for practical applications. The f-change criterion is **not usable** at this band because $10^{-24}$ is far below machine precision.

## 5. Combined (OR) Criterion

When using multiple criteria in OR-combination:
- The method stops as soon as **any** criterion fires
- The `stop_reason` field records which criterion triggered first
- The "rough" band will typically be dominated by gradient or x-change (f-change fires later due to the quadratic relationship)
- The "good" and "very good" bands will almost certainly be triggered by gradient norm (f-change hits machine precision before it can fire)

### Practical recommendation

For production use:
- **Gradient norm (absolute)** as the primary criterion — it directly measures optimality
- **x-change (absolute)** as a safety net — catches step collapse
- **f-change** only at the "rough" band where it is numerically meaningful

The combined criterion `OR(grad_abs, x_abs)` at the "good" band ($\tau_g = \tau_x = 10^{-8}$) is a robust default that provides reliable convergence detection without machine-precision issues.
