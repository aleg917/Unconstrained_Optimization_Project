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

### Funzione 2 — Problem 28: Variably Dimensioned
**Perché**: classico Moré-Garbow-Hillström; sum-of-squares (F_min = 0 a x*=(1,...,1)); **Hessiana piena** (rank-1 update di I), in **contrasto netto** con la banded di Prob.16.

```
F(x) = ½ Σₖ₌₁ⁿ⁺² fₖ(x)²
fₖ(x) = xₖ - 1,                       1 ≤ k ≤ n
fₙ₊₁(x) = Σᵢ₌₁ⁿ i (xᵢ - 1)             ≡ S(x)
fₙ₊₂(x) = S(x)²
```
**Punto iniziale**: `x̄ₗ = 1 - l/n` per `l = 1, ..., n`.

Posto `S := Σᵢ i (xᵢ - 1)`, si semplifica:
```
F(x) = ½ [ Σₖ (xₖ - 1)² + S² + S⁴ ]
```

**Gradiente esatto** (O(n) dopo una sola sweep per S):
```
∂F/∂xⱼ = (xⱼ - 1) + j · S · (1 + 2 S²)
```

**Hessiana** (per la relazione, non serve nel metodo):
```
∂²F/∂xᵢ∂xⱼ = δᵢⱼ + i j (1 + 6 S²)
H = I + (1 + 6 S²) · jvec · jvecᵀ,  jvec = (1, ..., n)ᵀ
```

---

## Insight dal caso n=2 (per discussione sperimentale)

L'analisi analitica del caso n=2 (vedi `report_notes.md` §2.4 e §3.4) rivela due topografie qualitativamente diverse:

- **Problem 16**: per n=2 la funzione è **separabile**, i minimi locali sono **tutti globali** (stesso valore F* = 3 - 2√5) e formano un reticolo periodico di passo 2π. I bacini di attrazione sono simmetrici per traslazione: la *qualità* del minimo raggiunto non dipende dal punto iniziale.

- **Problem 28**: F è **convessa** (somma di norme quadratiche e di S², S⁴ con S lineare in x), quindi il minimo globale è **unico** a x* = (1, 1). La narrativa si sposta dalla *molteplicità di bacini* (P16) alla **struttura della curvatura**:
  - Per n=2: `H = [[1+(1+6S²), 2(1+6S²)], [2(1+6S²), 1+4(1+6S²)]]`. A x* (S=0) → H = I + jjᵀ con jvec=(1,2), autovalori ~ {0.83, 6.17} → condition number ~7.4. Per n=10⁵ a x* il condition number cresce come ~n³/3 (autovalore massimo di jvecᵀjvec/||jvec||² nella direzione jvec).
  - Lontano da x* (es. x̄): `S(x̄) = -Σl²/n² ≈ -n/3`, quindi `S² ≈ n²/9`, `S⁴ ≈ n⁴/81` → l'Hessiana ha autovalore *enorme* lungo jvec e ~1 in tutte le altre direzioni → terreno classico in cui lo **steepest descent zig-zaga molto**.

**Implicazioni per il setup sperimentale**:
- Per Problem 28, n=2: top-view con `LogNorm` (F cresce come S⁴, valli a F→0 sparirebbero in scala lineare); evidenziare nel report l'ellitticità delle curve di livello (in contrasto con i bacini multipli di una P32-like).
- A n grandi (10⁴, 10⁵): `F(x̄)` cresce come ~n⁸, `‖∇F(x̄)‖` come ~n⁵-n⁶. Backtracking di Armijo deve scegliere α iniziale piccolo (oppure rho aggressivo) per evitare overflow nei primi step. Materiale narrativo: discutere il fallimento di `α₀ = 1` standard e la sintonizzazione di α₀ e ρ.
- Convergenza di steepest descent attesa **lineare** (problema convesso ben definito) ma con costante vicina a 1 per n grande → molte iterazioni. Confronto con P16 (Hessiana banded, ben condizionata localmente) sarà istruttivo.

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

## Struttura del progetto (attuale)

`functions/` e `gradients/` sono **separati** per poter sostituire il gradiente esatto con uno approssimato (FD) senza toccare il codice del metodo: il metodo `steepest_descent` riceve `grad_f` come callable e non sa né si interessa di come è calcolato.

```
Unconstrained_Optimization_Project/
├── src/
│   ├── functions/                  ← solo F(x) e punti iniziali suggeriti
│   │   ├── problem16.py            → f16, x_bar_16
│   │   └── problem28.py            → f28, x_bar_28
│   ├── gradients/                  ← problem-specific (esatti) + generico (FD)
│   │   ├── problem16.py            → grad_f16  (esatto)
│   │   ├── problem28.py            → grad_f28  (esatto)
│   │   └── finite_diff.py          → grad_fd_forward(f, x, k, scaled)
│   ├── stopping_criteria/          ← strategie plug-in di arresto
│   │   ├── base.py                 → StoppingCriterion (classe base)
│   │   ├── absolute/
│   │   │   ├── grad_norm.py        → GradNormAbsolute
│   │   │   ├── f_change.py         → FChangeAbsolute
│   │   │   └── x_change.py         → XChangeAbsolute
│   │   ├── relative/
│   │   │   ├── grad_norm.py        → GradNormRelative
│   │   │   ├── f_change.py         → FChangeRelative
│   │   │   └── x_change.py         → XChangeRelative
│   │   └── __init__.py             → re-exports + all_criteria(tol_g, tol_f, tol_x)
│   ├── methods/
│   │   └── steepest_descent.py     → steepest_descent, armijo_backtracking
│   └── starting_points.py          → generate_starting_points
├── main.ipynb                      ← notebook di visualizzazione e demo
├── PLAN.md
├── report_notes.md
└── stopping_criteria.md            ← analisi teorica delle 6 condizioni
```

**Pattern di chiamata** (futuro `experiments/run_experiments.py`):

```python
from src.functions import f16, x_bar_16
from src.gradients import grad_f16, grad_fd_forward
from src.methods import steepest_descent
from src.stopping_criteria import all_criteria, GradNormAbsolute

# Run singolo: gradiente esatto, criterio assoluto sul gradiente
res = steepest_descent(f16, grad_f16, x_bar_16(1000),
                       stopping=GradNormAbsolute(tol=1e-6))

# Loop sperimentale: 6 criteri sullo stesso punto iniziale
for crit in all_criteria(tol_g=1e-6, tol_f=1e-12, tol_x=1e-8):
    r = steepest_descent(f16, grad_f16, x_bar_16(1000), stopping=crit)
    print(f"{crit.name:10s} | iters={r['n_iter']:4d} | stop={r['stop_reason']}")

# Gradiente FD (basta un wrapper lambda)
grad_fd = lambda x: grad_fd_forward(f16, x, k=8, scaled=False)
res_fd = steepest_descent(f16, grad_fd, x_bar_16(1000),
                          stopping=GradNormAbsolute(tol=1e-6))
```

> Da aggiungere quando servono: cartelle `experiments/` (loop sperimentali e generazione tabelle) e `plots/` (top-view per n=2, convergence rates).

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
