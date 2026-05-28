# Analisi Teorica per il Tuning dei Parametri

Obiettivo: giustificare teoricamente una griglia di tuning ridotta (da 120 a 8
configurazioni) per i metodi Modified Newton (MN) e Truncated Newton (TN),
sui problemi P16 (Banded Trigonometric) e P28 (Variably Dimensioned).

---

## 0. Conformità con le specifiche del lab

Per scegliere quali parametri *fissare* e quali *tunare*, ci siamo riferiti
a due fonti del corso:

1. **Lecture notes**: `NO4LSP_DellaSanta_2526.pdf`, §6 (Newton with
   Backtracking) e §7 (Inexact Newton, esercizio 7.4 su Rosenbrock 100D).
2. **Script MATLAB del Lab 13**: codice effettivamente eseguito dalla
   docente — fonte autorevole quando diverge dal PDF teorico.

### 0.1 Allineamenti

| Parametro      | Lab CODE        | Nostro                       | Status |
|----------------|-----------------|------------------------------|--------|
| `tolgrad`      | 10⁻⁸            | 10⁻⁸                         | ✓      |
| `c1` (Armijo)  | 10⁻⁴            | 10⁻⁴                         | ✓      |
| `rho` default  | 0.8             | {0.5, 0.8}                   | ✓      |
| `kmax`         | 1000            | 1000                         | ✓      |
| `btmax`        | 50              | 50                           | ✓      |
| Starting pts   | 1 det + 5 rand  | 1 det + 5 rand               | ✓      |
| Forcing terms  | linear/superlinear/quadratic | identici          | ✓      |
| Stop criterion | `‖∇f‖_2 ≤ Ftol` | `GradNormAbsolute(tol=1e-8)` | ✓      |
| Box random pts | `[x_bar-1, x_bar+1]^n` (assignment) | `[x_bar-1, x_bar+1]^n` | ✓      |

> **Nota su `kmax`**: il PDF teorico (§7.4) dichiara `kmax=5000`, ma lo
> script effettivo del Lab 13 usa `kmax=1000`. Seguiamo il codice
> esistente.

### 0.2 Deviazioni minori (motivate)

| Parametro       | Lab CODE | Nostro     | Motivo |
|-----------------|----------|------------|--------|
| `cg_max_iter` (TN) | `n/2` | `None` (=n) | Fattore 2 di differenza; il CG converge molto prima di n iter in pratica, quindi la differenza è trascurabile su problemi grandi. |
| Griglia `rho`   | esplora {0.9, 0.75, 0.5, 0.25, 0.1} (visualizzazione 2D nei labs); usa 0.8 come default 100D | {0.5, 0.8} | Il range esteso è per visualizzazione; per tuning numerico il lab fissa 0.8. Noi testiamo 0.8 (default) + 0.5 (alternativo). |
| Seed RNG        | `rng(42)` (lab) | 323334 (team-specific) | Da assignment: `seed = min(student_IDs)`. |

### 0.3 Estensioni nostre (oltre lo scope del lab)

- **Modified Newton con `τ`-adjustment** (Cholesky correction): il PDF
  (§6.1) menziona `B_k := H_k + Correction` ma rimanda a riferimento
  esterno [2]. La nostra implementazione (Cholesky con `τ` raddoppiato
  finché la fattorizzazione riesce) è additiva. Il tuning di **`beta`**
  (perturbazione iniziale di `τ`) è interamente nostro.
- **Dimensioni**: lab fa Rosenbrock 100D; noi P16/P28 con n ∈
  {2, 10³, 10⁴, 10⁵}.
- **6 starting points** (1 deterministic + 5 random) per stabilità
  statistica vs il singolo `x⁰=0` del lab.

---

## 1. Armijo Backtracking — Analisi Approfondita

La line search di Armijo con backtracking è **condivisa** tra MN e TN. I suoi
parametri sono `alpha0`, `c1`, `rho`, `max_iter_backtrack`.

### 1.1 alpha0 = 1.0 — FISSO, mai tunare

I metodi di Newton (sia MN che TN) calcolano una direzione di ricerca
$p_k = -B_k^{-1} \nabla f(x_k)$ che minimizza il **modello quadratico locale**:

$$m_k(p) = f(x_k) + \nabla f(x_k)^T p + \tfrac{1}{2} p^T B_k \, p$$

dove $B_k$ è l'Hessiana (possibilmente modificata per MN, approssimata per TN).
Il minimo di $m_k$ si raggiunge esattamente per $p = p_k$ con passo $\alpha = 1$.

Se si usasse $\alpha_0 \neq 1$:
- $\alpha_0 < 1$: si limita artificialmente il passo, impedendo convergenza
  quadratica nelle iterazioni finali (dove il modello quadratico è accurato).
- $\alpha_0 > 1$: si overshoot il modello quadratico, sprecando backtracking
  step per rientrare.

Il passo unitario $\alpha_0 = 1$ è l'unica scelta che permette **convergenza
quadratica** quando l'Hessiana è ben approssimata (Nocedal-Wright, Theorem 3.5).

### 1.2 c1 = 1e-4 — FISSO

La condizione di Armijo (sufficient decrease) è:

$$f(x_k + \alpha \, p_k) \leq f(x_k) + c_1 \, \alpha \, \nabla f(x_k)^T p_k$$

Poiché $p_k$ è una direzione di discesa, $\nabla f(x_k)^T p_k < 0$, e il
lato destro è $f(x_k) - c_1 \, \alpha \, |\nabla f(x_k)^T p_k|$.

**Quanto è facile soddisfare questa condizione per il passo Newton pieno?**

Vicino alla soluzione, il modello quadratico è accurato, quindi:

$$f(x_k + p_k) \approx f(x_k) + \nabla f^T p_k + \tfrac{1}{2} p_k^T H_k \, p_k$$

Usando $p_k = -H_k^{-1} \nabla f_k$:

$$= f(x_k) + \nabla f^T (-H^{-1} \nabla f) + \tfrac{1}{2} (-H^{-1} \nabla f)^T H (-H^{-1} \nabla f)$$
$$= f(x_k) - \nabla f^T H^{-1} \nabla f + \tfrac{1}{2} \nabla f^T H^{-1} \nabla f$$
$$= f(x_k) - \tfrac{1}{2} \nabla f^T H^{-1} \nabla f$$

Il **decremento reale** è $\approx \frac{1}{2} |\nabla f^T p_k|$.

La condizione di Armijo con $\alpha=1$ richiede solo
$c_1 \, |\nabla f^T p_k| = 10^{-4} \, |\nabla f^T p_k|$.

Il margine è di **5000:1** — il passo Newton pieno viene accettato con margine
enorme. La differenza tra $c_1 = 10^{-4}$ e $c_1 = 10^{-3}$ cambia il margine
da 5000:1 a 500:1, comunque irrilevante.

**Conferma empirica**: nel quick-mode, MN ottiene il best con $c_1 = 10^{-3}$ e
TN con $c_1 = 10^{-4}$, entrambi al 100% success rate. Il parametro non
discrimina.

### 1.3 rho — unico parametro Armijo da testare

$\rho$ è il fattore di riduzione del passo ad ogni backtracking step:
$\alpha \leftarrow \rho \cdot \alpha$.

| rho | Sequenza step sizes | Step per raggiungere α ≈ 0.01 |
|-----|--------------------|----|
| 0.5 | 1, 0.5, 0.25, 0.125, 0.0625, ... | 7 |
| 0.8 | 1, 0.8, 0.64, 0.512, 0.410, 0.328, 0.262, 0.210, ... | 21 |

**Trade-off**:
- **rho = 0.5** (halving): raggiunge step piccoli rapidamente. Se il passo
  ottimale è piccolo, lo trova in pochi step. Se il passo ottimale è 0.7,
  accetta 0.5 (undershoot del 29%).
- **rho = 0.8**: granularità più fine. Se il passo ottimale è 0.7, accetta
  0.64 (undershoot dell'8.6%). Ma servono 3x più function evaluations.

Per i metodi Newton, il **costo dominante** per iterazione è:
- MN: Cholesky factorization $O(n^3)$ per Hessiana densa (P28),
  $O(n)$ per diagonale (P16)
- TN: CG inner iterations, ciascuna con Hessian-vector product $O(n)$

Le function evaluations nella line search costano $O(n)$, quindi sono
**trascurabili** rispetto al linear solve. L'extra costo di rho=0.8 è
irrilevante.

**Quando rho potrebbe fare la differenza?**

Nelle prime iterazioni su P28, dove $\|\nabla f(x_0)\| = O(n^7)$ e il modello
quadratico è una pessima approssimazione della funzione reale (il termine $S^4$
domina). Il passo Newton pieno $\alpha=1$ potrebbe overshootare enormemente.
Con rho=0.8 si potrebbe trovare un $\alpha$ migliore (più grande di quello
che rho=0.5 troverebbe), riducendo il numero totale di iterazioni esterne.

**Risultato quick-mode**: MN preferisce rho=0.5, TN preferisce rho=0.8.
Questo suggerisce che la direzione TN (inesatta) beneficia di una ricerca
più fine dello step size. Vale la pena testare entrambi.

**Aggiornamento full-mode** (DIMS=[2,1000,10000,100000], 6 starting points):
- MN: rho non discrimina (mean_iter=130.4 con rho=0.5, 130.4 con rho=0.8 a beta=1e-3).
  L'effetto del passo iniziale e del backtracking aggressivo è trascurabile.
- TN: rho=0.5 è leggermente migliore (mean_iter=146.1 vs 149.7 a forcing=quadratic).
  In particolare su P16: 18.2 iter (rho=0.5) vs 25.7 iter (rho=0.8) — la preferenza
  per rho=0.5 si manifesta nettamente al crescere della dimensione. Inversione
  rispetto al quick-mode.

La spiegazione è che il passo unitario alpha=1 viene accettato dall'Armijo nella
grande maggioranza delle iterazioni (specialmente vicino alla soluzione, dove il
modello quadratico è accurato). Le rare riduzioni di passo non beneficiano della
granularità fine di rho=0.8; conviene il halving aggressivo di rho=0.5.

### 1.4 max_iter_backtrack = 50 — FISSO

Questo è un parametro di **safety**. Con $\rho = 0.5$, dopo 50 riduzioni:
$\alpha = 0.5^{50} \approx 10^{-15}$. Se servono step così piccoli, la
direzione di ricerca è essenzialmente inutile (numericamente zero). Qualsiasi
valore $\geq 30$ è equivalente. Mai tunare.

### 1.5 Armijo: stesso per entrambi gli algoritmi? Stesso per entrambi i problemi?

**Sì a entrambe le domande.**

La funzione `armijo_backtracking(f, x, fx, g, d, ...)` riceve:
- Un punto $x$
- Una direzione di ricerca $d$
- La funzione $f$ e il gradiente $g$

È **completamente agnostica** su come $d$ è stata calcolata. Non sa (e non
deve sapere) se $d$ viene da:
- Un sistema lineare esatto $B_k d = -g$ (Modified Newton)
- Un sistema risolto approssimatamente via CG (Truncated Newton)
- Steepest descent $d = -g$ (fallback su curvatura negativa)

La **qualità della direzione** influenza:
1. Se il passo pieno $\alpha = 1$ viene accettato (vicino alla soluzione, con
   direzione Newton accurata: sì; lontano dalla soluzione: probabilmente no)
2. Quanti backtracking step servono (direzione scadente → più step)

Ma questi effetti sono **gestiti automaticamente** dalla line search: se la
direzione è peggiore, la line search backtrack di più. I parametri $c_1$ e
$\rho$ non devono cambiare.

Analogamente, le proprietà del problema (condizionamento, struttura
dell'Hessiana, non-convessità) influenzano la qualità della direzione Newton
e quindi il comportamento della line search, ma non i valori ottimali di
$c_1$ e $\rho$. La condizione di Armijo è **scale-invariant**: funziona in
termini di decremento relativo rispetto alla derivata direzionale.

**In sintesi**: usare gli stessi parametri Armijo per MN e TN, su P16 e P28.
L'unico parametro da testare è $\rho \in \{0.5, 0.8\}$.

---

## 2. Modified Newton — Parametri Specifici

### 2.1 beta — DA TUNARE {1e-6, 1e-3}

Quando l'Hessiana $H_k$ non è definita positiva, il Modified Newton calcola
$B_k = H_k + \tau_k I$ dove $\tau_k$ parte da $\beta$ (o da
$\beta - \min(\text{diag}(H))$ se il minimo diagonale è negativo) e
**raddoppia** fino a che la fattorizzazione di Cholesky riesce.

$\beta$ controlla l'**aggressività della perturbazione iniziale**:

| beta | Effetto |
|------|---------|
| 1e-6 | Perturbazione minima → $B_k \approx H_k$ → direzione vicina a Newton puro. Ma servono più raddoppi per superare l'indefinitezza. |
| 1e-3 | Perturbazione più forte → $B_k$ più regolarizzata → direzione più verso steepest descent. Meno raddoppi necessari. |

**Analisi per problema**:

**P16 (Hessiana diagonale)**: Al punto iniziale $\bar{x} = (1, \ldots, 1)$:
$$H_{jj} = j \cos(1) - 2 \sin(1) \approx 0.5403 \, j - 1.6829$$

- $j = 1$: $H_{11} \approx -1.14$ (negativo!)
- $j = 2$: $H_{22} \approx -0.60$ (negativo)
- $j = 3$: $H_{33} \approx -0.06$ (negativo)
- $j = 4$: $H_{44} \approx +0.48$ (positivo)

La modifica $\tau$ deve essere almeno $|\min_j H_{jj}| \approx 1.14$.

Con $\beta = 10^{-6}$: $\tau_0 = 10^{-6} - (-1.14) = 1.14$, poi si parte
già quasi abbastanza alto. In realtà $\tau_0 = \beta - \min(\text{diag}(H))$,
e siccome $\min(\text{diag}) < 0$, si ha $\tau_0 = \beta + 1.14 \approx 1.14$.
Per la Cholesky su matrice diagonale, $B_{jj} = H_{jj} + \tau > 0$ per tutti
i $j$ basta che $\tau > 1.14$. Quindi la Cholesky riesce subito o dopo 1
raddoppio. **Il valore di $\beta$ quasi non conta su P16** perché il
meccanismo $\beta - \min(\text{diag})$ già lo compensa.

**P28 (Hessiana densa)**: Al punto iniziale $\bar{x}_l = 1 - l/n$:
$$S = \sum_i i \, (x_i - 1) = \sum_i i \cdot (-i/n) = -\frac{1}{n} \sum_{i=1}^n i^2 \approx -\frac{n^2}{3}$$

$$H = I + (1 + 6S^2) \, \mathbf{j} \, \mathbf{j}^T$$

dove $\mathbf{j} = (1, 2, \ldots, n)^T$. Il fattore $(1 + 6S^2) \approx
\frac{2n^4}{3}$ è enorme e positivo. $H$ è la somma di $I$ (pos. def.)
più un termine rank-1 positivo semidefinito → **$H$ è sempre definita positiva**
al punto iniziale di P28. La modifica Cholesky non serve mai (a meno che il
punto corrente si allontani molto da $\bar{x}$). **$\beta$ è irrilevante per P28.**

**Verdetto preliminare (quick-mode)**: $\beta$ ha impatto solo su P16 e solo
marginale. Testiamo $\{10^{-6}, 10^{-3}\}$ per completezza, ma ci aspettiamo
poca differenza.

**Aggiornamento full-mode — smentita parziale**: il quick-mode aveva mascherato
l'effetto. Sui CSV full-mode (`results/fine_tuning_modified_newton.csv`):

| beta | mean_iter (aggregato) | mean_iter P16 | mean_iter P28 | success aggreg. |
|------|----------------------|---------------|---------------|-----------------|
| 1e-6 | 168.9                | 284.4         | 14.3          | 0.726           |
| 1e-3 | 130.4 (**-23%**)     | 217.7         | 14.3          | 0.738           |

L'effetto è significativo su P16 (~30% iter risparmiate), inesistente su P28
(coerente con la teoria: $H$ è già SPD al punto iniziale, nessuna modifica
richiesta). La spiegazione: nelle iterazioni intermedie di P16, $x_k$ si
allontana dal punto iniziale e $\min(\text{diag}(H))$ può cambiare segno o
ampiezza. Un $\beta$ iniziale più grande riduce il numero di **raddoppi di
$\tau$** necessari ad ogni iterazione per ottenere una matrice SPD —
fattorizzazione di Cholesky più veloce e meno tentativi falliti per step.

**Verdetto finale**: $\beta = 10^{-3}$ è la scelta consigliata (default in
`main.ipynb`). $\beta = 10^{-6}$ resta accettabile ma costa il 23% in più di
iterazioni su P16.

### 2.2 max_tau_iter = 100 — FISSO

Il meccanismo raddoppia $\tau$ ad ogni tentativo: $\tau_{j+1} = 2 \, \tau_j$.
Dopo $k$ raddoppi, $\tau = \tau_0 \cdot 2^k$.

Per P16, $\tau$ deve raggiungere ~1.14. Con $\tau_0 = \beta - \min(\text{diag})
\approx 1.14$, basta 0-1 raddoppio.

Per P28, nessuna modifica necessaria → 0 raddoppi.

Anche nel **worst case** teorico (Hessiana con autovalore minimo $-10^6$),
servirebbero ~20 raddoppi ($2^{20} \approx 10^6$). Il limite di 100 è
vastamente sovradimensionato. Qualsiasi valore $\geq 25$ è equivalente.

---

## 3. Truncated Newton — Parametri Specifici

### 3.1 forcing — IL parametro critico, DA TUNARE

La sequenza di forcing $\eta_k$ controlla la **precisione del CG inner loop**:
la condizione di stop è $\|r_j\| \leq \eta_k \|b\|$ dove $b = -\nabla f_k$.

| Tipo | Formula | Convergenza outer |
|------|---------|-------------------|
| `'linear'` | $\eta_k = 0.5$ (costante) | Lineare |
| `'superlinear'` | $\eta_k = \min(0.5, \sqrt{\|\nabla f_k\|})$ | Superlineare |
| `'quadratic'` | $\eta_k = \min(0.5, \|\nabla f_k\|)$ | Quadratica |

**Perché `'linear'` è escluso**: con $\eta = 0.5$, il CG risolve il sistema
al 50% di precisione relativa ad ogni iterazione. Questo è equivalente a usare
sempre una direzione Newton molto approssimata. Il rate di convergenza dell'outer
loop è lineare — non competitivo con il Newton esatto (che ha convergenza
quadratica).

**`'superlinear'` vs `'quadratic'`**: la differenza è nel rate con cui la
precisione CG aumenta man mano che $\|\nabla f_k\| \to 0$:
- Superlineare: $\eta_k \propto \|\nabla f_k\|^{1/2}$. Quando $\|\nabla f\|$
  è piccolo, $\eta$ scende lentamente → CG fa poche iterazioni inner.
- Quadratico: $\eta_k \propto \|\nabla f_k\|$. $\eta$ scende più rapidamente →
  CG fa più iterazioni inner ma l'outer converge più velocemente.

Nelle ultime iterazioni (vicino alla soluzione), quadratic esige una
risoluzione CG quasi esatta, ottenendo una direzione quasi-Newton-esatta
e convergenza quadratica. Superlinear è meno esigente e costa meno per
iterazione inner, ma l'outer converge più lentamente.

**Analisi per problema**:

**P16**: Hessiana diagonale → CG converge in **1 sola iterazione** (sistemi
diagonali sono banali). Indipendentemente da $\eta_k$, la soluzione CG è
esatta. $\Rightarrow$ **forcing irrilevante su P16**.

**P28**: $H = I + c \, \mathbf{j} \mathbf{j}^T$ (identità + rank-1). CG
converge in al massimo **2 iterazioni** (il Krylov subspace per una matrice
con 2 cluster di autovalori ha dimensione 2). Anche qui, forcing ha impatto
limitato sul costo inner. Ma il rate di convergenza outer dipende dalla
qualità della direzione, che è determinata dalla forcing sequence.

Il risultato quick-mode conferma: **quadratic è il migliore** (19.9 avg iter
vs attese più alte per superlinear).

**Aggiornamento full-mode — risultato problema-dipendente**:

| forcing     | rho | success P16 | mean_iter P16 | success P28 | mean_iter P28 |
|-------------|-----|-------------|---------------|-------------|---------------|
| superlinear | 0.5 | 0.917       | 19.9          | 0.542       | 274.9         |
| quadratic   | 0.5 | 0.875       | 18.2          | 0.625       | 273.9         |

- Su **P28**, quadratic vince nettamente (success +8 punti percentuali) — la
  teoria regge: l'Hessiana $I + c\, \mathbf{j}\mathbf{j}^T$ ha 2 cluster di
  autovalori, CG converge in 2 iterazioni, e una direzione Newton accurata
  è essenziale per la convergenza outer.

- Su **P16** invece superlinear vince in success rate (+4 punti). Il motivo:
  l'Hessiana diagonale di P16 ha *autovalori molto sparsi* (non 2 cluster), e
  il CG NON converge in poche iterazioni — anzi, fa ~108 iter inner per
  ogni iter outer (vedi §3.2). Con quadratic, si richiede una precisione CG
  altissima vicino alla soluzione → CG fa moltissime iterazioni inner →
  costo computazionale alto e maggior rischio di mancata convergenza per
  time_limit. Superlinear richiede meno precisione e completa più velocemente.

**Verdetto finale**: quadratic resta la scelta di default (vince P28 e
aggregato), ma su problemi con Hessiana "ben dispersa" (autovalori non
clusterizzati) superlinear può essere preferibile per ragioni di costo
inner.

### 3.2 cg_max_iter = None — FISSO (con caveat)

$\text{None} \Rightarrow$ CG può fare fino a $n$ iterazioni. L'analisi
preliminare aveva previsto:
- P16: CG converge in 1 iterazione (Hessiana diagonale ⇒ sistema banale).
- P28: CG converge in $\leq 2$ iterazioni (Hessiana = $I + c\,\mathbf{j}\mathbf{j}^T$,
  Krylov subspace di dimensione 2).

**Verifica empirica full-mode (cg_total dai CSV)**:

| Problema | mean_outer_iter | mean_cg_total | CG iter/outer |
|----------|-----------------|---------------|---------------|
| P16      | 18.2            | 2074          | **~114**      |
| P28      | 273.9           | 274.7         | ~1.003        |

- **P28**: predizione confermata (~1 iter inner per outer).
- **P16**: predizione **falsa** — il CG fa in media 114 iter inner per outer.
  L'analisi "sistema diagonale banale" era ingenua: il CG non sfrutta la
  diagonalità (è un metodo iterativo generale che vede solo prodotti
  $H \cdot v$). Su una matrice diagonale con autovalori molto sparsi, il
  rate di convergenza del CG è governato dal numero di condizionamento
  $\kappa(H) = \lambda_{\max}/\lambda_{\min}$, che può essere grande.

**Implicazione**: il default `cg_max_iter=n` permette al CG di lavorare il
necessario senza cap. Su P16 con $n=100000$, il CG può in linea di principio
arrivare a $n$ iter, e il cap teorico non è mai sotto-dimensionato. Tuttavia
un cap esplicito (es. `cg_max_iter=50`) potrebbe ridurre il costo inner
accettando direzioni più approssimate — esperimento fuori dallo scope di
questa griglia.

---

## 4. Stopping Criteria — rinvio

Il tuning degli stopping criteria segue una filosofia **diversa** da quello
dei parametri del metodo: gli stopping criteria non alterano la traiettoria
di convergenza, quindi si può registrare **una sola** traiettoria (con
tolleranza molto stretta o fino a `max_iter`) e poi valutare *post-hoc*
qualunque criterio sulla history salvata. Ogni criterio diventa una semplice
funzione di check sulla sequenza $\{x_k, f_k, \|\nabla f_k\|\}$.

L'analisi completa — filosofia del tuning post-hoc, scelta delle tolleranze
per criterio (incluso il riscalamento $\text{tol}_f \approx \text{tol}_g^2$
suggerito dallo sviluppo di Taylor $|\Delta F| \approx \frac{1}{2}\|\nabla f\|^2$),
pericoli dei criteri relativi su P28, struttura della Phase 2 del notebook,
e template di motivazione per il report — è contenuta in un documento separato:

> **→ Vedi [`docs/Stopping_criteria_explanation.md`](Stopping_criteria_explanation.md)**.

La griglia del notebook è stata aggiornata per testare ~22 configurazioni
(6 criteri base × 3 band + 2 combined × 2 band) anziché le 7 originali —
costo computazionale aggiuntivo nullo grazie all'approccio post-hoc.

---

## 5. Griglia Ridotta Finale

### Parametri fissi (entrambi gli algoritmi)

| Parametro | Valore | Motivazione |
|-----------|--------|-------------|
| `alpha0` | 1.0 | Passo Newton naturale, prerequisito per convergenza quadratica |
| `c1` | 1e-4 | Nocedal-Wright standard; margine 5000:1 sul passo Newton pieno |
| `max_iter_backtrack` | 50 | Safety parameter; 50 step → α ≈ 10⁻¹⁵ |
| `max_tau_iter` (MN) | 100 | Safety; mai più di 25 raddoppi necessari |
| `cg_max_iter` (TN) | None (=n) | CG converge in 1-2 iter per P16/P28 |

### Griglia da testare

**Modified Newton (4 configurazioni):**

| Config | beta | rho |
|--------|------|-----|
| MN-1 | 1e-6 | 0.5 |
| MN-2 | 1e-6 | 0.8 |
| MN-3 | 1e-3 | 0.5 |
| MN-4 | 1e-3 | 0.8 |

**Truncated Newton (4 configurazioni):**

| Config | forcing | rho |
|--------|---------|-----|
| TN-1 | superlinear | 0.5 |
| TN-2 | superlinear | 0.8 |
| TN-3 | quadratic | 0.5 |
| TN-4 | quadratic | 0.8 |

**Totale: 8 configurazioni** (vs 120 nella griglia originale = **93% in meno**).

### Stima tempi

Con DIMS = [2, 1000, 10000, 100000], 6 starting points per cella:

- Celle per combo: ~8 (2 problemi × 4 dim, con OOM skip per P28 n=100000 su MN)
- Run totali: 8 combo × ~8 celle × 6 starts ≈ **384 run**
- vs originale: 120 × ~8 × 6 ≈ 5760 run
- **Speedup: ~15x**

---

## 6. Validazione Empirica (Full-mode)

I risultati sui CSV `results/fine_tuning_modified_newton.csv` e
`results/fine_tuning_truncated_newton.csv` (DIMS=[2, 1000, 10000, 100000],
6 starting points, 60s time limit, criterio fisso `GradNormAbsolute(1e-4)`).

### 6.0 Criterio di selezione del "best"

Per ogni metodo definiamo il best parameter set tramite una gerarchia
esplicita di metriche:

1. **`avg_success`** — primary. Frazione di run convergenti (sul totale
   `problemi × dimensioni × starting points`). Una configurazione che non
   converge non è utilizzabile.
2. **`avg_iter`** — secondary. Media delle iterazioni sulle run di successo.
   Misura efficienza algoritmica indipendente dalla macchina.
3. **`avg_time`** — tertiary tiebreaker. Wall-clock medio sulle run di
   successo. Confronto solo a parità delle prime due.

**Perché `avg_grad` non entra nel ranking**: la colonna `avg_grad` riporta
`||∇f||` all'ultimo iterato (= momento di stop), mediato sulle run di
successo. Per definizione del criterio fissato `GradNormAbsolute(tol=1e-4)`,
ogni run con `success=True` soddisfa già `||∇f|| ≤ 1e-4`. Empiricamente si
osservano mediani 1e-6 a 1e-5 in tutte le configurazioni — la metrica è
sostanzialmente uniforme tra config diverse e quindi **non discrimina**.
Resta in tabella come *sanity check*: se vedessimo `avg_grad > 1e-4`,
ci sarebbe un bug.

**Importante**: le metriche `avg_iter`, `avg_time`, `avg_grad` qui riportate
sono calcolate **solo sulle run che hanno avuto successo**. Le run fallite
per `max_iter` o `time_limit` venivano originariamente aggregate con un
`grad_norm` finale enorme che inquinava le medie (vedi
[Stopping_criteria_explanation.md](Stopping_criteria_explanation.md) §4 per
lo stesso pattern di bug nella Phase 2).

### 6.1 Modified Newton

**Ranking globale (sort: avg_success → avg_iter → avg_time)**:

| rank | beta  | rho | avg_success | avg_iter | avg_time (s) |
|------|-------|-----|-------------|----------|--------------|
| 1    | 1e-3  | 0.8 | 0.738       | **15.2** | **0.99**     |
| 2    | 1e-3  | 0.5 | 0.738       | 15.5     | 1.06         |
| 3    | 1e-6  | 0.5 | 0.738       | 72.3     | 1.82         |
| 4    | 1e-6  | 0.8 | 0.714       | 35.6     | 1.59         |

**Best per problema** (ranking ricomputato sul subset del problema):

| Scope     | Config         | success | iter | time (s) |
|-----------|----------------|---------|------|----------|
| Overall   | β=1e-3, ρ=0.8  | 0.738   | 15.2 | 0.99     |
| Best P16  | β=1e-3, ρ=0.8  | 0.792   | 11.6 | 0.013    |
| Best P28  | β=1e-3, ρ=0.8  | 0.667   | 21.0 | 2.54     |

**Success rate per (problem, n, beta, rho)**:

| problem, n  | β=1e-6 ρ=0.5 | β=1e-6 ρ=0.8 | β=1e-3 ρ=0.5 | β=1e-3 ρ=0.8 |
|-------------|--------------|--------------|--------------|--------------|
| P16, 2      | 1.00         | 1.00         | 1.00         | 1.00         |
| P16, 1000   | 1.00         | 1.00         | 1.00         | 1.00         |
| P16, 10000  | 1.00         | 0.83         | 1.00         | 1.00         |
| P16, 100000 | 0.17         | 0.17         | 0.17         | 0.17         |
| P28, 2      | 1.00         | 1.00         | 1.00         | 1.00         |
| P28, 1000   | 1.00         | 1.00         | 1.00         | 1.00         |
| P28, 10000  | 0            | 0            | 0            | 0            |
| P28, 100000 | 0            | 0            | 0            | 0            |

**Conclusioni MN**:
1. **β=1e-3 è il chiaro vincitore su P16**: avg_iter scende da ~36–72 a ~12
   (fattore 5x), e il tempo da ~0.3–0.6s a ~0.013s (fattore 30x). La predizione
   teorica "β marginale" era basata sulle prime iterazioni; in realtà il
   beneficio si accumula su tutte le iter intermedie.
2. **Su P28 β e ρ sono inerti** (21.0 iter sempre): l'Hessiana è già SPD al
   punto iniziale, la modifica $\tau$ non scatta mai. Tutte le 4 config sono
   equivalenti su P28.
3. **ρ è un tiebreaker fine**: tra ρ=0.5 e ρ=0.8 a β=1e-3, la differenza è
   ~0.3 iter (15.5 vs 15.2). Il sort gerarchico privilegia ρ=0.8.

**Best MN scelto**: **β = 1e-3, ρ = 0.8** (vince in tutti i 3 scope:
Overall, P16-only, P28-only). Sostituisce il default precedente (ρ=0.5)
basato su quick-mode.

### 6.2 Truncated Newton

**Ranking globale (sort: avg_success → avg_iter → avg_time)**:

| rank | forcing     | rho | avg_success | avg_iter | avg_time (s) |
|------|-------------|-----|-------------|----------|--------------|
| 1    | quadratic   | 0.5 | 0.750       | 21.1     | 4.63         |
| 2    | quadratic   | 0.8 | 0.750       | 23.1     | 4.65         |
| 3    | superlinear | 0.5 | 0.729       | 20.9     | 3.88         |
| 4    | superlinear | 0.8 | 0.729       | 24.8     | 2.93         |

**Best per problema**:

| Scope     | Config                 | success | iter | time (s) |
|-----------|------------------------|---------|------|----------|
| Overall   | quadratic, ρ=0.5       | 0.750   | 21.1 | 4.63     |
| Best P16  | **superlinear, ρ=0.5** | 0.917   | 19.6 | 4.12     |
| Best P28  | quadratic, ρ=0.5       | 0.625   | 26.5 | 9.22     |

Notare: sulla scelta P16-only, superlinear (success_P16=91.7%) vince su
quadratic (87.5%). La gerarchia premia il success rate prima delle iter.

**Success rate per (problem, n, forcing, rho)**:

| problem, n  | quad ρ=0.5 | quad ρ=0.8 | super ρ=0.5 | super ρ=0.8 |
|-------------|------------|------------|-------------|-------------|
| P16, 2      | 1.00       | 1.00       | 1.00        | 1.00        |
| P16, 1000   | 1.00       | 1.00       | 1.00        | 1.00        |
| P16, 10000  | 1.00       | 1.00       | 1.00        | 1.00        |
| P16, 100000 | 0.50       | 0.50       | 0.67        | 0.67        |
| P28, 2      | 1.00       | 1.00       | 1.00        | 1.00        |
| P28, 1000   | 1.00       | 1.00       | 1.00        | 1.00        |
| P28, 10000  | 0.50       | 0.50       | 0.17        | 0.17        |
| P28, 100000 | 0          | 0          | 0           | 0           |

**Conclusioni TN**:
1. **Su P16**, superlinear è più robusto: success_P16=91.7% vs 87.5%.
   Quadratic richiede meno iter (17.3 vs 19.6) ma il vantaggio in efficienza
   viene perso se non converge.
2. **Su P28**, quadratic è strettamente meglio: success_P28=62.5% vs 54.2%.
   Confermata la teoria — l'Hessiana $I + c\mathbf{j}\mathbf{j}^T$ con 2
   cluster di autovalori richiede una direzione Newton accurata.
3. **Best globale**: quadratic ρ=0.5 perché vince sia in success aggregato
   (75% vs 72.9%) sia in iter (a parità di success, ρ=0.5 < ρ=0.8 in iter).

**Best TN scelto**:
- Default (`main.ipynb`): **forcing = quadratic, ρ = 0.5** (vince Overall e
  Best P28).
- Su P16-only: superlinear, ρ=0.5 sarebbe preferibile. Trade-off documentato
  nel report.

### 6.3 Verdetto pratico per produzione

| Metodo | Default consigliato      | Best P16-only       | Best P28-only     |
|--------|--------------------------|---------------------|-------------------|
| MN     | β=1e-3, ρ=0.8            | β=1e-3, ρ=0.8       | β=1e-3, ρ=0.8     |
| TN     | forcing=quadratic, ρ=0.5 | forcing=superlinear, ρ=0.5 | forcing=quadratic, ρ=0.5 |

### 6.4 Limite di applicabilità della griglia

Sulla griglia DIMS = [2, 1000, 10000, 100000], alcuni fallimenti sono inerenti
all'esperimento e non al metodo:

- **P16 a n=100000 (success ~17–67%)**: l'algoritmo raggiunge `max_iter=1000`
  ma il gradiente è ancora sopra `1e-4`. Aumentando `max_iter` ci si
  aspetterebbe convergenza, ma il costo per iterazione su P16-large
  rende il completamento più lento del time_limit.

- **P28 a n=10000 (success ~17–50%) e n=100000 (success 0%)**: tutte le run
  che falliscono lo fanno per **`stop_reason='time_limit'`** (60s). Su
  P28 con dimensione grande, una sola iterazione Newton richiede la
  fattorizzazione di una matrice $n \times n$ densa — costa $O(n^3)$.
  Per n=10000 questa è ~10⁹ operazioni → secondi per iterazione → il
  time limit viene raggiunto prima della convergenza. **Non significa
  che il metodo non converge**: significa che 60s non bastano. Aumentare
  `TIME_LIMIT` (a costo di tempi di esperimento più lunghi) recupererebbe
  questi casi.

Le celle `(P28, n=10000)` e `(P28, n=100000)` appaiono come **NaN nelle
pivot di `mean_iter`** perché il filter `success=True` esclude tutte le run.
La pivot di `success_rate` mostra invece il valore corretto (0.00) e
documenta l'esistenza di queste configurazioni.

Per gli stopping criteria, vedi documento dedicato
[Stopping_criteria_explanation.md](Stopping_criteria_explanation.md).
