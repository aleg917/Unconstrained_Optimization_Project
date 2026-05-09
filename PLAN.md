# Piano: Assignment 2.1 – Derivative-based Optimization (Steepest Descent)

## Context
Implementazione del Assignment 2.1 del corso NO4LSP (Politecnico di Torino, A.Y. 2025/2026).
Si implementa il **Steepest Descent + Backtracking** (punto 1.1, 0.5 pt) su **due funzioni test** da V897-03.pdf.
Il codice viene scritto dallo studente; questo piano serve da guida strutturata.

---

## Scelta delle funzioni test

### Funzione 1 — Problem 16: Banded Trigonometric
**Perché**: gradiente pulitissimo, struttura sparsa, ottima per visualizzare il path in 2D, nessuna esplosione numerica.

```
F(x) = Σᵢ₌₁ⁿ  i · [(1 - cos xᵢ) + sin xᵢ₋₁ - sin xᵢ₊₁]
```
con `x₀ = x_{n+1} = 0` (condizioni al bordo fisse) e **punto iniziale** `x̄ᵢ = 1` per ogni i.

**Gradiente esatto** (da calcolare nella relazione):
- Caso generale (1 < j < n): `∂F/∂xⱼ = j·sin(xⱼ) + (j+1)·cos(xⱼ) - (j-1)·cos(xⱼ)`
  → semplifica in: `∂F/∂xⱼ = j·sin(xⱼ) + 2·cos(xⱼ)`
- Bordo j=1: `∂F/∂x₁ = sin(x₁) + 2·cos(x₁)`
- Bordo j=n: `∂F/∂xₙ = n·sin(xₙ) - (n-1)·cos(xₙ)`

### Funzione 2 — Problem 32: Generalized Broyden Tridiagonal
**Perché**: struttura sum-of-squares (F_min = 0), gradiente sparso tridiagonale, ben contrastante con Prob.16.

```
F(x) = ½ Σₖ₌₁ⁿ  fₖ(x)²
fₖ(x) = (3 - 2xₖ)·xₖ + 1 - xₖ₋₁ - xₖ₊₁
```
con `x₀ = x_{n+1} = 0` e **punto iniziale** `x̄ₗ = -1` per ogni l.

**Gradiente esatto** (tramite regola della catena su somma di quadrati):
```
∂F/∂xⱼ = fⱼ·(3 - 4xⱼ) - fⱼ₋₁ - fⱼ₊₁
```
dove `fⱼ₋₁ = 0` se j=1, `fⱼ₊₁ = 0` se j=n.

---

## Insight dal caso n=2 (per discussione sperimentale)

L'analisi analitica del caso n=2 (vedi `report_notes.md` §2.4 e §3.4) rivela due topografie qualitativamente diverse:

- **Problem 16**: per n=2 la funzione è **separabile**, i minimi locali sono **tutti globali** (stesso valore F* = 3 - 2√5) e formano un reticolo periodico di passo 2π. I bacini di attrazione sono simmetrici per traslazione: la *qualità* del minimo raggiunto non dipende dal punto iniziale.

- **Problem 32**: la condizione F = 0 (somma di quadrati nulla) ammette **4 zeri distinti** in ℝ² (2 sulla diagonale + 2 off-diagonale, dalle radici di `(2u² - 2u - 1)(2u² - 4u + 1) = 0`). I 4 bacini di attrazione **non sono** legati da simmetria → punti iniziali diversi conducono a minimi globali diversi.

**Implicazioni per il setup sperimentale**:
- Per Problem 32, n=2: oltre alle metriche standard, **classificare ciascun run** in base al bacino raggiunto (A, B, C, D). È materiale narrativo forte per il report.
- Per n > 2: la separabilità (P16) si rompe e il numero di zeri di P32 cresce → ci si aspetta più variabilità tra punti iniziali random in entrambi i problemi.
- Per il top-view a n=2: usare **`LogNorm`** sul colormap di P32 (la funzione cresce ~x⁴, le valli a F→0 spariscono in scala lineare) e clipping di Z per il 3D.

---

## Steepest Descent + Armijo Backtracking

**Algoritmo**:
```
x⁽⁰⁾ = x̄  (o punto random)
Per k = 0, 1, 2, ...:
  gₖ = ∇f(x⁽ᵏ⁾)
  Se ‖gₖ‖ ≤ tolleranza: STOP
  dₖ = -gₖ          ← direzione di discesa
  αₖ = backtracking_armijo(x⁽ᵏ⁾, f, gₖ, dₖ)
  x⁽ᵏ⁺¹⁾ = x⁽ᵏ⁾ + αₖ·dₖ
```

**Backtracking Armijo**:
```
Input: x, f, g, d, α₀=1, ρ∈(0,1), c₁∈(0,0.5)
α = α₀
while f(x + α·d) > f(x) + c₁·α·gᵀd:
    α = ρ·α
return α
```
Parametri di default suggeriti: `c₁ = 1e-4`, `ρ = 0.5`, `α₀ = 1`, `max_iter = 1000`, `tol = 1e-6`.

**Criteri di arresto**:
1. `‖∇f(x⁽ᵏ⁾)‖ ≤ tol`  (gradiente assoluto)
2. `‖∇f(x⁽ᵏ⁾)‖ / ‖∇f(x⁽⁰⁾)‖ ≤ tol_rel`  (gradiente relativo)
3. `k ≥ max_iter`  (fallimento)

---

## Finite Differences (punto 3 del assignment)

Per Steepest Descent (nessuna Hessiana), si applica il punto "1.5 pts": approssimare il **gradiente** con FD.

**Forward difference** componente per componente:
```
∂F/∂xᵢ ≈ [F(x + hᵢ·eᵢ) - F(x)] / hᵢ
```

Due versioni da testare:
- **Fixed h**: `hᵢ = 10⁻ᵏ` con k ∈ {4, 8, 12}
- **Scaled h**: `hᵢ = 10⁻ᵏ · |x̂ᵢ|` con k ∈ {4, 8, 12}

**Implementazione efficiente** (importante per n=10⁵):
Per entrambi i problemi, ogni componente xⱼ appare solo in 3 termini consecutivi della somma.
Quindi `F(x + h·eⱼ) - F(x)` richiede di ricalcolare solo quei ~3 termini, non l'intera F.
Questo porta a O(n) operazioni totali per il gradiente FD anziché O(n²).

---

## Setup esperimenti

**Dimensioni**: n ∈ {2, 10³, 10⁴, 10⁵}

**Punti iniziali** (da spec assignment):
```python
np.random.seed(min_student_id)  # seed = minimo student ID del team

def generate_starting_points(x_bar, n, num_random=5):
    """x_bar è il vettore n-dim del punto iniziale suggerito."""
    points = [x_bar.copy()]  # primo punto: x̄ suggerito
    for _ in range(num_random):
        # uniforme in [x̄ᵢ - 1, x̄ᵢ + 1] per ogni i
        delta = np.random.uniform(-1, 1, n)
        points.append(x_bar + delta)
    return points
```
Attenzione: il seed va impostato **una volta sola** prima di generare tutti i punti di tutte le dimensioni e problemi, per riproducibilità.

**Parametri di backtracking da sintonizzare e discutere**:
- `c₁` (Armijo constant): tipicamente 1e-4, valori più alti = passi più piccoli
- `ρ` (reduction factor): tipicamente 0.5, valori più bassi = riduzione più aggressiva
- `α₀` (initial step): tipicamente 1 (passo unitario di Newton)
- `max_iter`: 1000, discutere se sufficiente per convergenza a n grande
- `tol`: 1e-6 sulla norma del gradiente

Il report deve descrivere test preliminari con parametri diversi e motivare la scelta finale.

---

## Struttura del progetto (consigliata)

```
Unconstrained_Optimization_Project/
├── src/
│   ├── methods/
│   │   └── steepest_descent.py   ← algoritmo + backtracking
│   ├── problems/
│   │   ├── problem16.py          ← F(x), grad_F(x), x_bar(n)
│   │   └── problem32.py          ← F(x), grad_F(x), x_bar(n)
│   └── finite_differences.py     ← grad_fd(f, x, h_type, k)
├── experiments/
│   └── run_experiments.py        ← loop su n, starting points, tabelle
├── plots/
│   └── plot_results.py           ← top-view n=2, convergence rates
└── main.py
```

---

## Pacchetti Python da usare

| Pacchetto | Uso |
|-----------|-----|
| `numpy` | vettori, operazioni array, random seed |
| `matplotlib` | grafici (top view, convergence rates) |
| `time` | misurare tempo di esecuzione |
| `scipy.sparse` | (opzionale) per strutture sparse a n grande |
| `pandas` | (opzionale) per esportare tabelle risultati |

Installazione (nel venv già esistente):
```bash
pip install numpy matplotlib scipy pandas
```

**Non servono** librerie di ottimizzazione come `scipy.optimize` — l'implementazione va fatta a mano.

---

## Output obbligatori (da assignment)

### Tabelle (una per ogni metodo × problema × dimensione n)
Colonne: `start_pt_ID | grad_norm | iters/max_iters | success | rate_of_conv | time`

### Figure obbligatorie
1. **Top view** della funzione + path di ogni starting point, per n=2 (una figura per metodo, esatta vs FD)
2. **Experimental convergence rates** per ogni sequenza convergente, per ogni n (separata per FD)

---

## Verifica implementazione (checklist)

- [ ] Per n=2: visualizzare F(x) con `plt.contourf` e sovrapporre il path iterativo
- [ ] `‖∇f(x*)‖` molto piccolo al punto di convergenza (< 1e-5)
- [ ] Confrontare gradiente esatto vs FD con h=1e-4 su punto noto: errore ~1e-4
- [ ] Per n=10³ il codice gira in tempo ragionevole (< 60s per run)
- [ ] Seed random corretto → risultati riproducibili
