# Report Notes — Assignment 2.1 (NO4LSP)

Materiale per il report. Stato attuale: **scelta funzioni test**, **derivazione dei gradienti esatti**, **implementazione vettorizzata** e **visualizzazione del paesaggio per n=2**.

---

## 1. Scelta delle funzioni test

Dal catalogo Lukšan–Vlček (V897-03.pdf) sono stati selezionati due problemi di natura strutturale diversa, in modo da contrastare i comportamenti dei metodi:

| # | Nome | Tipo | Perché è interessante |
|---|------|------|-----------------------|
| 16 | Banded Trigonometric | Funzione trigonometrica banded | Gradiente sparso (tridiagonale), struttura periodica → molti minimi locali, ottimo per visualizzazione |
| 32 | Generalized Broyden Tridiagonal | Sum-of-squares | $F_{\min}=0$, gradiente tridiagonale, contrasta con Prob.16 |

I problemi non rientrano nelle famiglie escluse dall'assignment (Rosenbrock).

---

## 2. Problem 16 — Banded Trigonometric

### 2.1 Formulazione

$$
F(x) \;=\; \sum_{i=1}^{n} i\,\bigl[(1 - \cos x_i) + \sin x_{i-1} - \sin x_{i+1}\bigr],
\qquad x_0 = x_{n+1} = 0.
$$

Punto iniziale suggerito: $\bar{x}_i = 1$ per ogni $i$.

### 2.2 Derivazione del gradiente

La variabile $x_j$ compare in **tre** termini consecutivi della somma:

- nel termine $i=j$, attraverso $j(1 - \cos x_j)$;
- nel termine $i=j-1$, attraverso $-(j-1)\sin x_j$;
- nel termine $i=j+1$, attraverso $+(j+1)\sin x_{j-1}$ → contributo $+(j+1)\cos x_j$ se $i-1 = j$.

Sommando le derivate parziali si ottiene:

$$
\frac{\partial F}{\partial x_j}
\;=\; j\sin x_j \;+\; (j+1)\cos x_j \;-\; (j-1)\cos x_j
\;=\; j\sin x_j + 2\cos x_j.
$$

**Bordi** (l'indice fuori range non contribuisce):

- $j=1$: il termine $i = j-1 = 0$ non esiste → $\partial F/\partial x_1 = \sin x_1 + 2\cos x_1$ (coincide con la formula generale per $j=1$ poiché il fattore $-(j-1)=0$).
- $j=n$: il termine $i = j+1 = n+1$ non esiste → $\partial F/\partial x_n = n\sin x_n - (n-1)\cos x_n$.

### 2.3 Caso $n=2$ (esplicito)

$$
F(x_1, x_2) \;=\; (1 - \cos x_1) - \sin x_2 + 2(1 - \cos x_2) + 2\sin x_1.
$$

Verifica numerica al punto iniziale $\bar{x} = (1, 1)$:

```
F(1,1)        = 2.2206
grad F(1,1)   = [1.9221, 1.1426]
```

### 2.4 Analisi di $n=2$: separabilità e minimi locali

Per $n=2$ la funzione è **separabile**: $F(x_1, x_2) = g_1(x_1) + g_2(x_2)$, con

- $g_1(x_1) = (1 - \cos x_1) + 2\sin x_1$,
- $g_2(x_2) = 2(1 - \cos x_2) - \sin x_2$.

I punti critici sono quindi prodotti cartesiani delle radici 1D:

$$
\tan x_1 = -2 \;\Rightarrow\; x_1^* = \arctan(-2) + k\pi,
\qquad
\tan x_2 = \tfrac{1}{2} \;\Rightarrow\; x_2^* = \arctan(\tfrac{1}{2}) + k\pi.
$$

Il test della derivata seconda (separato per ogni asse) mostra che **i minimi si alternano con i massimi** ad ogni $\pi$: i minimi cadono sui rami con $\cos x_1 > 0$ e $2\cos x_2 + \sin x_2 > 0$, quindi sono **periodici di periodo $2\pi$**:

$$
\bigl(x_1^{\min},\, x_2^{\min}\bigr) \;=\; \bigl(\arctan(-2) + 2k\pi,\; \arctan(\tfrac{1}{2}) + 2m\pi\bigr).
$$

Nella finestra $[-2\pi, 2\pi]^2$ se ne contano **quattro**, tutti con stesso valore di funzione

$$
F^* \;=\; 3 - 2\sqrt{5} \;\approx\; -1.4721.
$$

Per $n > 2$ la separabilità si rompe: i termini accoppiano $x_{i-1}, x_i, x_{i+1}$ e la struttura dei minimi va analizzata numericamente.

---

## 3. Problem 32 — Generalized Broyden Tridiagonal

### 3.1 Formulazione

$$
F(x) \;=\; \tfrac{1}{2}\sum_{k=1}^{n} f_k(x)^2,
\qquad
f_k(x) \;=\; (3 - 2x_k)\,x_k + 1 - x_{k-1} - x_{k+1},
$$

con $x_0 = x_{n+1} = 0$. Punto iniziale suggerito: $\bar{x}_l = -1$.

### 3.2 Derivazione del gradiente

Dalla regola della catena su una somma di quadrati, $\partial F/\partial x_j = \sum_k f_k\,\partial f_k/\partial x_j$. Le derivate $\partial f_k/\partial x_j$ sono non nulle solo per $k \in \{j-1, j, j+1\}$:

- $\partial f_j / \partial x_j = 3 - 4 x_j$,
- $\partial f_{j-1} / \partial x_j = -1$ (compare come $-x_{(j-1)+1}$),
- $\partial f_{j+1} / \partial x_j = -1$ (compare come $-x_{(j+1)-1}$).

Quindi

$$
\frac{\partial F}{\partial x_j} \;=\; f_j\,(3 - 4x_j) \;-\; f_{j-1} \;-\; f_{j+1},
\qquad f_0 = f_{n+1} = 0.
$$

### 3.3 Caso $n=2$ — verifica al punto iniziale

In $\bar{x} = (-1, -1)$:

$$
f_1 = f_2 = (3 + 2)(-1) + 1 - 0 - (-1) = -3 \;\Rightarrow\; F = \tfrac{1}{2}(9 + 9) = 9.
$$

$$
\nabla F(\bar{x}) = \bigl[-3 \cdot 7 - 0 - (-3),\; -3 \cdot 7 - (-3) - 0\bigr] = [-18,\; -18].
$$

### 3.4 Minimi globali per $n=2$

Trattandosi di una somma di quadrati, $F(x) = 0$ se e solo se $f_1 = f_2 = 0$:

$$
\begin{cases}
3x_1 - 2x_1^2 + 1 - x_2 = 0 \\
3x_2 - 2x_2^2 + 1 - x_1 = 0
\end{cases}
$$

Sostituendo $x_2 = 3x_1 - 2x_1^2 + 1$ nella seconda si arriva al polinomio quartico

$$
4u^4 - 12u^3 + 8u^2 + 2u - 1 \;=\; (2u^2 - 2u - 1)\,(2u^2 - 4u + 1) \;=\; 0,
$$

che dà **quattro** radici in $\mathbb{R}^2$:

| Punto | Coordinate | Tipo |
|-------|------------|------|
| $A$ | $\bigl(\tfrac{1+\sqrt{3}}{2},\; \tfrac{1+\sqrt{3}}{2}\bigr) \approx (1.366,\, 1.366)$ | diagonale |
| $B$ | $\bigl(\tfrac{1-\sqrt{3}}{2},\; \tfrac{1-\sqrt{3}}{2}\bigr) \approx (-0.366,\, -0.366)$ | diagonale |
| $C$ | $\bigl(\tfrac{2+\sqrt{2}}{2},\; \tfrac{2-\sqrt{2}}{2}\bigr) \approx (1.707,\, 0.293)$ | off-diagonale |
| $D$ | $\bigl(\tfrac{2-\sqrt{2}}{2},\; \tfrac{2+\sqrt{2}}{2}\bigr) \approx (0.293,\, 1.707)$ | off-diagonale |

Tutti e quattro sono **minimi globali** con $F = 0$. La presenza di più bacini disgiunti rende interessante l'analisi del Steepest Descent al variare del punto iniziale: a seconda della regione di partenza converge a un minimo diverso.

> **Nota sulla visualizzazione**: poiché $F$ cresce come $\sim x^4$, in scala lineare le valli sono visivamente schiacciate. Nel notebook si usa `LogNorm` per il contour 2D e si effettua un *clipping* di $Z$ per la superficie 3D, in modo da rendere visibili gli avvallamenti vicino ai 4 zeri.

---

## 4. Implementazione vettorizzata (NumPy)

### 4.1 Trucco delle condizioni al bordo

Per evitare casistiche ai bordi, si estende il vettore di stato:

```python
x_ext = np.concatenate(([0.0], x, [0.0]))    # lunghezza n+2
# x_ext[i]   == x_i        per i = 1..n
# x_ext[i-1] == x_{i-1}    (con x_0 = 0)
# x_ext[i+1] == x_{i+1}    (con x_{n+1} = 0)
```

Tutta la somma diventa **un'unica espressione vettoriale** con slicing.

### 4.2 Problem 16

```python
def f16(x):
    n = len(x)
    x_ext = np.concatenate(([0.0], x, [0.0]))
    i = np.arange(1, n + 1)
    terms = i * ((1 - np.cos(x_ext[1:n+1]))
                 + np.sin(x_ext[0:n])
                 - np.sin(x_ext[2:n+2]))
    return float(terms.sum())

def grad_f16(x):
    n = len(x)
    j = np.arange(1, n + 1)
    g = j * np.sin(x) + 2 * np.cos(x)                 # formula generale
    g[-1] = n * np.sin(x[-1]) - (n - 1) * np.cos(x[-1])  # bordo j = n
    return g
```

Costo: $O(n)$ in tempo e memoria per ciascuna chiamata.

### 4.3 Problem 32

```python
def f32(x):
    n = len(x)
    x_ext = np.concatenate(([0.0], x, [0.0]))
    fk = (3 - 2 * x) * x + 1 - x_ext[0:n] - x_ext[2:n+2]
    return 0.5 * float(np.dot(fk, fk))

def grad_f32(x):
    n = len(x)
    x_ext = np.concatenate(([0.0], x, [0.0]))
    fk = (3 - 2 * x) * x + 1 - x_ext[0:n] - x_ext[2:n+2]
    f_ext = np.concatenate(([0.0], fk, [0.0]))
    return fk * (3 - 4 * x) - f_ext[0:n] - f_ext[2:n+2]
```

Costo: $O(n)$. Il gradiente riusa il vettore $f_k$ già calcolato.

---

## 5. Organizzazione del codice

I file sono separati in tre livelli — **funzioni**, **gradienti**, **metodi** — in modo che lo stesso metodo di ottimizzazione possa ricevere indifferentemente un gradiente esatto o un'approssimazione (differenze finite). Questo è essenziale perché l'assignment richiede di confrontare proprio queste due varianti.

```
src/
├── functions/                 ← solo F(x) e punti iniziali suggeriti
│   ├── problem16.py           → f16, x_bar_16
│   └── problem32.py           → f32, x_bar_32
├── gradients/                 ← problem-specific (esatti) + generico (FD)
│   ├── problem16.py           → grad_f16  (esatto)
│   ├── problem32.py           → grad_f32  (esatto)
│   └── finite_diff.py         → grad_fd_forward(f, x, k, scaled)
├── methods/
│   └── steepest_descent.py    → steepest_descent, armijo_backtracking
└── starting_points.py         → generate_starting_points
```

**Pattern di chiamata**: il metodo riceve `f` e `grad_f` come callable e non sa né si interessa di quale gradiente stia usando.

```python
from src.functions import f16, x_bar_16
from src.gradients import grad_f16, grad_fd_forward
from src.methods import steepest_descent

# (a) gradiente esatto
res_ex = steepest_descent(f16, grad_f16, x_bar_16(1000))

# (b) gradiente FD — basta un wrapper lambda
grad_fd = lambda x: grad_fd_forward(f16, x, k=8, scaled=False)
res_fd = steepest_descent(f16, grad_fd, x_bar_16(1000))
```

Nel report si descrive questa decisione architetturale come una conseguenza diretta del requisito sperimentale (confronto esatto vs FD), e non un dettaglio implementativo.

---

## 6. Visualizzazione $n=2$

Per ogni problema si genera una griglia $400 \times 400$ su $[-2\pi, 2\pi]^2$ (Problem 16) e $[-3, 3]^2$ (Problem 32). $F$ viene valutata in tutti i punti tramite `np.vectorize`.

**Output prodotti** (vedi `main.ipynb`):

1. **Problem 16 — contour 2D**: minimi locali analitici evidenziati con marker stellati, punto iniziale $\bar{x}=(1,1)$ in rosso. Si nota la struttura periodica.
2. **Problem 16 — surface 3D** (`plot_surface`) con isolinee proiettate sul piano di base.
3. **Problem 32 — contour 2D**: minimo globale ($F=0$) in basso a destra rispetto a $\bar{x}=(-1,-1)$.

Queste figure costituiscono materiale per la sezione "top view per $n=2$" obbligatoria dal testo dell'assignment.

---

## 7. Implicazioni per il Steepest Descent (analisi $n=2$)

Il confronto delle due topografie suggerisce comportamenti **diversi** del Steepest Descent al variare del punto iniziale, ed è un punto da sviluppare nella discussione dei risultati sperimentali.

### Problem 16 — minimi periodici equivalenti

Tutti i minimi locali sono **globali** (stesso valore $F^* = 3 - 2\sqrt{5}$) e disposti su un reticolo periodico di passo $2\pi$ in entrambe le direzioni. Ne deriva che:

- ogni punto iniziale "ragionevolmente vicino" a un minimo del reticolo viene attratto dal minimo più prossimo (i bacini di attrazione sono simmetrici per traslazione);
- statistiche come `grad_norm`, `iters` e `rate_of_conv` non dipendono dalla *qualità* del minimo raggiunto, ma solo dalla geometria locale del bacino;
- la variabilità tra punti iniziali random è quindi **piccola** in termini di valore finale di $F$.

### Problem 32 — bacini disgiunti con stesso valore di funzione

I 4 minimi globali (tutti con $F = 0$) **non** sono legati da una simmetria di traslazione: due sono sulla diagonale, due fuori diagonale. Ne deriva che:

- punti iniziali in regioni diverse possono essere attratti da minimi diversi → traiettorie qualitativamente diverse pur convergendo allo stesso valore $F^*=0$;
- è interessante (e produce buon materiale per la relazione) classificare ciascuno dei 6 punti iniziali in base al bacino raggiunto;
- in dimensione $n>2$ ci si attende una proliferazione del numero di bacini, e questo è uno dei motivi per cui Steepest Descent può "rallentare" in regioni piatte fra minimi vicini.

> **Take-away per il report**: i due problemi sono complementari. Prob.16 mette alla prova la rapidità di convergenza in una geometria pulita; Prob.32 mette alla prova la **sensibilità ai punti iniziali**. Per la sezione sperimentale conviene riportare, oltre alle metriche standard, anche **a quale minimo** ciascun run converge per Prob.32 ($n=2$).

---

## 8. Da fare

- ✅ Funzioni (`f16`, `f32`) e gradienti esatti (`grad_f16`, `grad_f32`) — fatti.
- ✅ Scheletro di `steepest_descent` con backtracking di Armijo — fatto (`src/methods/steepest_descent.py`).
- ✅ Skeleton di `grad_fd_forward` (forward-difference, fisso e scalato) — fatto (`src/gradients/finite_diff.py`).
- ✅ Helper per i 6 punti iniziali — fatto (`src/starting_points.py`).
- ⬜ Decidere il `seed` (= minimo student-ID del team).
- ⬜ Eseguire gli esperimenti per $n \in \{10^3, 10^4, 10^5\}$ con entrambi i gradienti (esatto e FD per $k\in\{4,8,12\}$, fisso e scalato).
- ⬜ Compilare le tabelle obbligatorie: `start_pt_ID | grad_norm | iters/max_iters | success | rate_of_conv | time`.
- ⬜ Per Prob.32 ($n=2$): classificare ciascun run per **bacino di attrazione** raggiunto.
- ⬜ Generare le figure di convergence rates sperimentali per ogni $n$.
