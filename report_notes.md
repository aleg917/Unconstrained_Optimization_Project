# Report Notes — Assignment 2.1 (NO4LSP)

Materiale per il report. Stato attuale: **scelta funzioni test**, **derivazione dei gradienti esatti**, **implementazione vettorizzata** e **visualizzazione del paesaggio per n=2**.

---

## 1. Scelta delle funzioni test

Dal catalogo Lukšan–Vlček (V897-03.pdf) sono stati selezionati due problemi di natura strutturale diversa, in modo da contrastare i comportamenti dei metodi:

| # | Nome | Tipo | Perché è interessante |
|---|------|------|-----------------------|
| 16 | Banded Trigonometric | Funzione trigonometrica banded | Gradiente sparso (tridiagonale), struttura periodica → molti minimi locali, ottimo per visualizzazione |
| 28 | Variably Dimensioned (Moré–Garbow–Hillström) | Sum-of-squares | $F_{\min}=0$ a $x^*=(1,\dots,1)$; **Hessiana piena** (rank-1 update di $I$), contrasto netto con la banded di Prob.16 |

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

## 3. Problem 28 — Variably Dimensioned

### 3.1 Formulazione

$$
F(x) \;=\; \tfrac{1}{2}\sum_{k=1}^{n+2} f_k(x)^2,
\qquad
\begin{cases}
f_k(x) = x_k - 1, & 1 \le k \le n,\\
f_{n+1}(x) = \displaystyle\sum_{i=1}^{n} i\,(x_i - 1) \;\equiv\; S(x),\\
f_{n+2}(x) = S(x)^2.
\end{cases}
$$

Punto iniziale suggerito: $\bar{x}_l = 1 - l/n$, $l = 1, \dots, n$.

Posto $S := \sum_i i\,(x_i - 1)$, l'espressione si compatta in

$$
F(x) \;=\; \tfrac{1}{2}\Bigl[\,\sum_{k=1}^{n}(x_k - 1)^2 \;+\; S^2 \;+\; S^4\,\Bigr].
$$

### 3.2 Derivazione del gradiente

Dalla forma compatta, $\partial S/\partial x_j = j$ e

$$
\frac{\partial F}{\partial x_j}
\;=\; (x_j - 1) \;+\; j\,S \;+\; 2j\,S^3
\;=\; (x_j - 1) \;+\; j\,S\,(1 + 2 S^2).
$$

Calcolabile in $O(n)$: una sweep per ottenere $S$, poi una sweep vettorizzata sul gradiente.

### 3.3 Hessiana — struttura piena

$$
\frac{\partial^2 F}{\partial x_i \partial x_j}
\;=\; \delta_{ij} \;+\; i\,j\,(1 + 6 S^2),
\qquad
H \;=\; I \;+\; (1 + 6 S^2)\,\mathbf{j}\,\mathbf{j}^\top,\quad \mathbf{j}=(1,2,\dots,n)^\top.
$$

$H$ è **piena** (rank-1 update di $I$) — contrasto qualitativo con la struttura banded di Prob.16. Autovalori: $1$ con molteplicità $n-1$ (sottospazio ortogonale a $\mathbf{j}$) e $1 + (1+6S^2)\|\mathbf{j}\|^2$ lungo $\mathbf{j}$. Poiché $\|\mathbf{j}\|^2 = n(n+1)(2n+1)/6 \sim n^3/3$, il **numero di condizionamento** cresce come

$$
\kappa(H) \;\sim\; 1 + (1 + 6 S^2)\,\tfrac{n^3}{3}.
$$

### 3.4 Caso $n=2$ — verifica al punto iniziale

In $\bar{x} = (1 - 1/2,\, 1 - 1) = (1/2,\, 0)$:

$$
d = (-1/2,\, -1), \quad S = 1\cdot(-1/2) + 2\cdot(-1) = -5/2,
$$
$$
F(\bar{x}) = \tfrac{1}{2}\bigl[\tfrac{1}{4} + 1 + \tfrac{25}{4} + \tfrac{625}{16}\bigr] = \tfrac{1}{2}\cdot\tfrac{745}{16} = \tfrac{745}{32} \approx 23.28.
$$
$$
\nabla F(\bar{x}) = d + \mathbf{j}\,S(1+2S^2)
= \begin{pmatrix}-1/2\\ -1\end{pmatrix} + \begin{pmatrix}1\\ 2\end{pmatrix}\cdot(-5/2)\cdot\bigl(1 + 25/2\bigr)
= \begin{pmatrix}-1/2 - 135/4\\ -1 - 135/2\end{pmatrix}.
$$

Numericamente: $\nabla F(\bar{x}) \approx (-34.25,\; -68.5)$, $\|\nabla F\| \approx 76.6$.

### 3.5 Convessità e minimo unico

$F$ è somma di funzioni convesse (norme quadratiche, $S^2$ e $S^4$ con $S$ lineare in $x$), quindi è **convessa**. L'unico punto critico è quindi il **minimo globale**:

$$
x^* = (1, 1, \dots, 1), \qquad F^* = 0.
$$

A differenza di una funzione con bacini multipli (es. Broyden-tridiagonal per $n=2$), qui la narrativa sperimentale si sposta dalla *molteplicità di minimi* alla **struttura della curvatura** e alla **sensibilità al condizionamento** a $n$ grande.

> **Nota sulla visualizzazione**: $F$ cresce come $\sim S^4 \sim n^4 \|x-1\|^4$ lungo la direzione $\mathbf{j}$; in scala lineare le curve di livello vicino al minimo sono invisibili. Nel notebook si usa `LogNorm` per il contour 2D e si effettua un *clipping* di $Z$ per la superficie 3D.

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

### 4.3 Problem 28

```python
def f28(x):
    n = len(x)
    j = np.arange(1, n + 1)
    d = x - 1.0
    S = float(np.dot(j, d))
    return 0.5 * (float(np.dot(d, d)) + S * S + S ** 4)

def grad_f28(x):
    n = len(x)
    j = np.arange(1, n + 1)
    d = x - 1.0
    S = float(np.dot(j, d))
    return d + j * (S * (1.0 + 2.0 * S * S))
```

Costo: $O(n)$. Niente trucco di slicing perché qui l'accoppiamento è **globale** (tramite $S$), non locale; il calcolo si chiude in due dot-product e una sweep elementwise.

---

## 5. Organizzazione del codice

I file sono separati in tre livelli — **funzioni**, **gradienti**, **metodi** — in modo che lo stesso metodo di ottimizzazione possa ricevere indifferentemente un gradiente esatto o un'approssimazione (differenze finite). Questo è essenziale perché l'assignment richiede di confrontare proprio queste due varianti.

```
src/
├── functions/                  ← solo F(x) e punti iniziali suggeriti
│   ├── problem16.py            → f16, x_bar_16
│   └── problem28.py            → f28, x_bar_28
├── gradients/                  ← problem-specific (esatti) + generico (FD)
│   ├── problem16.py            → grad_f16  (esatto)
│   ├── problem28.py            → grad_f28  (esatto)
│   └── finite_diff.py          → grad_fd_forward(f, x, k, scaled)
├── stopping_criteria/          ← strategie plug-in di arresto (6 totali)
│   ├── base.py                 → StoppingCriterion (classe base)
│   ├── absolute/               → 3 criteri "absolute"
│   │   ├── grad_norm.py
│   │   ├── f_change.py
│   │   └── x_change.py
│   └── relative/               → 3 criteri "relative" (no F_0, no x_0)
│       ├── grad_norm.py
│       ├── f_change.py
│       └── x_change.py
├── methods/
│   └── steepest_descent.py     → steepest_descent, armijo_backtracking
└── starting_points.py          → generate_starting_points
```

Tre livelli di pluggability nel metodo:

1. **funzione** (`f`): il problema test.
2. **gradiente** (`grad_f`): esatto problem-specific oppure approssimato `grad_fd_forward` (FD).
3. **stopping criterion**: una delle 6 strategie, passata come oggetto (`stopping=GradNormAbsolute(1e-6)`). Ogni run usa **una** sola strategia, in modo da poter attribuire la terminazione nelle tabelle.

**Pattern di chiamata**:

```python
from src.functions import f16, x_bar_16
from src.gradients import grad_f16, grad_fd_forward
from src.methods import steepest_descent
from src.stopping_criteria import all_criteria, GradNormAbsolute

# Run singolo
res = steepest_descent(f16, grad_f16, x_bar_16(1000),
                       stopping=GradNormAbsolute(tol=1e-6))

# Confronto sperimentale: 6 criteri sullo stesso x_0
for crit in all_criteria():
    r = steepest_descent(f16, grad_f16, x_bar_16(1000), stopping=crit)
    print(crit.name, r['n_iter'], r['stop_reason'])

# Gradiente FD via wrapper lambda
grad_fd = lambda x: grad_fd_forward(f16, x, k=8, scaled=False)
res_fd = steepest_descent(f16, grad_fd, x_bar_16(1000),
                          stopping=GradNormAbsolute(tol=1e-6))
```

Le tre dimensioni di sostituzione sono **ortogonali**: nel ciclo sperimentale si itera su (gradiente × stopping × punto iniziale) senza modificare il metodo.

Per la teoria dietro le 6 stopping conditions vedi `stopping_criteria.md` (ordini di scala $\tau_g$, $\alpha\tau_g$, $\alpha\tau_g^2$ e diagnosi distinte per ciascun criterio).

---

## 6. Visualizzazione $n=2$

Per ogni problema si genera una griglia $400 \times 400$ su $[-2\pi, 2\pi]^2$ (Problem 16) e su $[-1, 3]^2$ centrato attorno a $x^*=(1,1)$ (Problem 28). $F$ viene valutata in tutti i punti tramite `np.vectorize`.

**Output prodotti** (vedi `main.ipynb`):

1. **Problem 16 — contour 2D**: minimi locali analitici evidenziati con marker stellati, punto iniziale $\bar{x}=(1,1)$ in rosso. Si nota la struttura periodica.
2. **Problem 16 — surface 3D** (`plot_surface`) con isolinee proiettate sul piano di base.
3. **Problem 28 — contour 2D** (in `LogNorm`): unico minimo globale a $x^*=(1,1)$; le curve di livello hanno forte ellitticità lungo la direzione $\mathbf{j}=(1,2)$ (autovettore associato all'autovalore grande di $H$). Punto iniziale $\bar{x}_2=(1/2,0)$.

Queste figure costituiscono materiale per la sezione "top view per $n=2$" obbligatoria dal testo dell'assignment.

---

## 7. Implicazioni per il Steepest Descent (analisi $n=2$)

Il confronto delle due topografie suggerisce comportamenti **diversi** del Steepest Descent al variare del punto iniziale, ed è un punto da sviluppare nella discussione dei risultati sperimentali.

### Problem 16 — minimi periodici equivalenti

Tutti i minimi locali sono **globali** (stesso valore $F^* = 3 - 2\sqrt{5}$) e disposti su un reticolo periodico di passo $2\pi$ in entrambe le direzioni. Ne deriva che:

- ogni punto iniziale "ragionevolmente vicino" a un minimo del reticolo viene attratto dal minimo più prossimo (i bacini di attrazione sono simmetrici per traslazione);
- statistiche come `grad_norm`, `iters` e `rate_of_conv` non dipendono dalla *qualità* del minimo raggiunto, ma solo dalla geometria locale del bacino;
- la variabilità tra punti iniziali random è quindi **piccola** in termini di valore finale di $F$.

### Problem 28 — minimo unico, curvatura fortemente anisotropa

L'unico minimo globale è $x^*=(1,\dots,1)$ (problema convesso). Ne deriva che:

- la *qualità* finale è la stessa per ogni punto iniziale (tutti convergono a $x^*$);
- la variabilità si manifesta nel **numero di iterazioni**, nello **stop reason** e nell'andamento del rate sperimentale di convergenza;
- l'autovalore massimo di $H$ lungo $\mathbf{j}=(1,2,\dots,n)$ cresce come $\sim (1+6S^2)\,n^3/3$ e fa esplodere $\kappa(H)$ a $n$ grande → terreno classico in cui lo **Steepest Descent zig-zaga**, con rate lineare ma costante asintotica $(\kappa-1)/(\kappa+1)$ vicina a 1;
- al punto iniziale $\bar{x}_l = 1 - l/n$ si ha $S(\bar{x}) = -\sum_l l^2/n = -(n+1)(2n+1)/6 \sim -n^2/3$, da cui $F(\bar{x}) \sim S^4/2 \sim n^8/162$ e $\|\nabla F(\bar{x})\| \sim \|\mathbf{j}\|\cdot S^3 \sim n^{5/2}\cdot n^6 = n^{17/2}$ — quindi $\bar{x}$ è **lontanissimo** dal minimo a $n$ grande e Armijo deve scegliere $\alpha$ molto piccolo per evitare overflow.

> **Take-away per il report**: i due problemi sono complementari. Prob.16 mette alla prova la rapidità di convergenza in una geometria *banded e periodica*; Prob.28 mette alla prova il metodo su una geometria *convessa ma fortemente mal condizionata* con Hessiana **piena**. Per la sezione sperimentale conviene riportare, oltre alle metriche standard, anche il valore di $\alpha_0$ effettivamente scelto dal backtracking nelle prime iterazioni (diagnostico del condizionamento) e il rate di convergenza sperimentale (atteso lineare, con costante che peggiora con $n$).

---

## 8. Da fare

- ✅ Funzioni (`f16`, `f28`) e gradienti esatti (`grad_f16`, `grad_f28`) — fatti.
- ✅ `steepest_descent` con backtracking di Armijo e stopping plug-in — fatto.
- ✅ Sei stopping criteria (3 absolute + 3 relative) in `src/stopping_criteria/` — fatti.
- ✅ Skeleton di `grad_fd_forward` (forward-difference, fisso e scalato) — fatto (`src/gradients/finite_diff.py`).
- ✅ Helper per i 6 punti iniziali — fatto (`src/starting_points.py`).
- ⬜ Decidere il `seed` (= minimo student-ID del team).
- ⬜ Eseguire gli esperimenti per $n \in \{10^3, 10^4, 10^5\}$ con (gradiente esatto + FD per $k\in\{4,8,12\}$, fisso e scalato) × (6 stopping criteria) × (6 punti iniziali).
- ⬜ Compilare le tabelle obbligatorie: `start_pt_ID | stop_reason | grad_norm | iters | success | |F-F*| | ||x-x*|| | time`.
- ⬜ Per Prob.28 ($n$ grande): registrare l'evoluzione di $\alpha$ scelto da Armijo nelle prime iterazioni come diagnostico del condizionamento.
- ⬜ Generare le figure di convergence rates sperimentali per ogni $n$.
