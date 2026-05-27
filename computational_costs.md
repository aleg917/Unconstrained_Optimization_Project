# Computational Costs — Analisi per Metodo e Setup

---

## 1. Costo delle primitive

Tutte le funzioni obiettivo (f16, f28) sono vettorizzate NumPy, O(n) per valutazione.

| Primitiva | Costo | Note |
|-----------|-------|------|
| `f(x)` | O(n) | Singola valutazione funzione |
| `grad_f16(x)`, `grad_f28(x)` | O(n) | Gradiente esatto, vettorizzato |
| `hess_f16(x)` | O(n) | Diagonale sparsa (CSC) |
| `hess_f28(x)` | O(n^2) | Densa n x n (rank-1 update di I) |
| `grad_fd(f, x)` | 2n f-evals = **O(n^2)** | Loop Python, non vettorizzabile |
| `hess_fd(grad_f, x)` | 2n grad-evals | Se grad esatto: O(n^2). Se grad FD: O(n^3) |
| `hess_fd_diag(grad_f, x)` | 2 grad-evals | Se grad esatto: O(n). Se grad FD: O(n^2) |
| `hv_fd(grad_f, x, v)` | 2 grad-evals | Se grad esatto: O(n). Se grad FD: O(n^2) |

---

## 2. Costo per iterazione esterna

### Modified Newton

Ogni iterazione esterna: 1 Hessiana + 1 fattorizzazione/solve + 1 gradiente + Armijo (poche f-evals).

| Setup (grad, hess) | Costo dominante | Per iterazione |
|---------------------|-----------------|----------------|
| exact, exact (P16 diag) | O(n) Hessian + O(n) solve | **O(n)** |
| exact, exact (P28 densa) | O(n^2) Hessian + O(n^3) Cholesky | **O(n^3)** |
| exact, fd_diag (P16) | O(n) CPR diag | **O(n)** |
| exact, fd_col (P28) | 2n grad esatti = O(n^2) | **O(n^2)** |
| fd, fd_diag (P16) | 2 grad_fd = O(n^2) | **O(n^2)** |
| fd, fd_col (P28) | 2n grad_fd = O(n^3) | **O(n^3)** |

### Truncated Newton

Ogni iterazione esterna: 1 gradiente + J passi CG (ciascuno con 1 prodotto Hv) + Armijo.

| Setup (grad, hess_mode) | Costo per CG step | Per iter esterna (J steps) |
|--------------------------|-------------------|---------------------------|
| exact, exact (P16 diag) | O(n) diag mul | **O(Jn)** |
| exact, exact (P28 densa) | O(n^2) matmul | **O(Jn^2)** |
| exact, fd (hv_fd) | 2 grad esatti = O(n) | **O(Jn)** |
| fd, fd (hv_fd) | 2 grad_fd = O(n^2) | **O(Jn^2)** |

Dove J = numero iterazioni CG interne (fino a `cg_max_iter`, tipicamente O(n) nel caso peggiore).

---

## 3. Fattibilita per dimensione

### n = 100,000

| Metodo | GRAD | HESS | Costo/iter | Tempo stimato/iter | Fattibile? |
|--------|------|------|------------|-------------------|------------|
| ModNewton | exact | exact P16 (diag) | O(n) | < 0.01s | **Si** |
| ModNewton | exact | exact P28 (densa) | O(n^3) | ore (Cholesky 100k) | **No** (anche per RAM: 80 GB) |
| ModNewton | exact | fd_diag P16 | O(n) | < 0.01s | **Si** |
| ModNewton | exact | fd_col P28 | O(n^2) | ~10s | **Borderline** |
| ModNewton | fd | fd_diag P16 | O(n^2) | ~10s | **Borderline** |
| ModNewton | fd | fd_col P28 | O(n^3) | ore | **No** |
| TruncNewton | exact | exact P16 | O(Jn) | < 1s (J piccolo) | **Si** |
| TruncNewton | exact | exact P28 | O(Jn^2) | ~60s per J=100 | **Borderline** |
| TruncNewton | exact | fd (hv_fd) | O(Jn) | < 1s | **Si** |
| TruncNewton | fd | fd (hv_fd) | O(Jn^2) | >> 30s per 1 CG step | **No** |

### n = 1,000

Tutte le combinazioni sono fattibili a n = 1,000 (il costo massimo O(n^3) = 10^9 e gestibile in pochi secondi).

### n = 10,000

Le combinazioni O(n^3) diventano lente (~15 min per iterazione). Le O(n^2) funzionano (~1s per iterazione).

---

## 4. Perche grad FD + hess FD e proibitivo a n grande

Il collo di bottiglia e `grad_fd` (`src/gradients/finite_diff.py:35-42`):

```python
for i in range(n):            # loop Python, n = 100,000
    x_fwd = x.copy()          # O(n)
    x_bwd = x.copy()          # O(n)
    x_fwd[i] += h             # ...
    x_bwd[i] -= h
    g[i] = (f(x_fwd) - f(x_bwd)) / (2*h)  # 2 f-evals, O(n) ciascuna
```

Costo totale: n iterazioni * O(n) per iterazione = **O(n^2)**. A n = 100,000: ~10^10 operazioni in un loop Python (non vettorizzabile perche ogni perturbazione e su una singola componente). Tempo: **10-30 secondi** per un singolo gradiente.

`hv_fd` chiama `grad_f` due volte. Con `grad_f = grad_fd`: 2 * O(n^2) = O(n^2). **Un singolo prodotto hessiana-vettore puo superare il time limit.**

Il check del tempo nel CG (`_cg_truncated`, riga 114) e **tra** le iterazioni CG, ma una singola chiamata `apply_H(d)` a riga 119 puo durare piu di 30 secondi — il check arriva troppo tardi.

---

## 5. Combinazioni raccomandate per il progetto

### P16 (hessiana diagonale)

| n | Combinazione raccomandata | Note |
|---|---------------------------|------|
| 2-1,000 | qualsiasi | Tutto e veloce |
| 10,000 | exact/fd_diag o exact/hv_fd | Evitare grad FD se possibile |
| 100,000 | exact/exact o exact/fd_diag o exact/hv_fd | **grad deve essere exact** |

### P28 (hessiana densa, rank-1 update)

| n | Combinazione raccomandata | Note |
|---|---------------------------|------|
| 2-1,000 | qualsiasi | Tutto e veloce |
| 10,000 | exact/hv_fd (TN) o exact/exact (MN) | Evitare hess_fd col-by-col |
| 100,000 | **Solo TN con exact grad + hv_fd** | MN non fattibile (H densa 80GB) |

**Regola pratica**: a n >= 10,000, usare sempre il gradiente esatto. Il gradiente FD e O(n^2) e diventa il collo di bottiglia.
