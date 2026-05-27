# Analisi Teorica per il Tuning dei Parametri

Obiettivo: giustificare teoricamente una griglia di tuning ridotta (da 120 a 8
configurazioni) per i metodi Modified Newton (MN) e Truncated Newton (TN),
sui problemi P16 (Banded Trigonometric) e P28 (Variably Dimensioned).

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

**Verdetto**: $\beta$ ha impatto solo su P16 e solo marginale. Testiamo
$\{10^{-6}, 10^{-3}\}$ per completezza, ma ci aspettiamo poca differenza.

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

### 3.2 cg_max_iter = None — FISSO

$\text{None} \Rightarrow$ CG può fare fino a $n$ iterazioni. Come analizzato sopra:
- P16: CG converge in 1 iterazione
- P28: CG converge in $\leq 2$ iterazioni

Il cap non viene mai raggiunto. Per problemi con Hessiane più complesse (non
i nostri), potrebbe avere senso limitare il CG per bilanciare costo inner/outer.
Ma per P16 e P28, il valore di default è ottimale.

---

## 4. Stopping Criteria — Analisi e Riduzione

### 4.1 Insight chiave: valutazione post-hoc

Lo stopping criterion **non altera la traiettoria di convergenza** — decide
solo quando dichiarare convergenza. Il notebook attuale ri-esegue l'algoritmo
22 volte per ogni (metodo, problema, dim, start) cambiando solo il criterio.

**Approccio efficiente**: eseguire l'algoritmo **una sola volta** con
`return_history=True` e una tolleranza molto stretta (o fino a `max_iter`),
poi valutare post-hoc ogni criterio sulla history registrata.

Questo elimina il fattore 22x dal tempo di esecuzione.

### 4.2 Criteri assoluti vs relativi

I criteri **relativi** normalizzano per il valore iniziale:
$$\frac{\|\nabla f(x_k)\|}{\|\nabla f(x_0)\|} \leq \tau$$

Per P28, $\|\nabla f(x_0)\| = O(n^7)$. Con $\tau = 10^{-8}$:
$$\|\nabla f(x_k)\| \leq 10^{-8} \cdot O(n^7)$$

| n | $\|\nabla f(x_0)\|$ (ordine) | Soglia effettiva grad_rel @ good |
|---|---|---|
| 100 | $10^{14}$ | $10^{6}$ (pessimo!) |
| 1000 | $10^{21}$ | $10^{13}$ (catastrofico) |
| 10000 | $10^{28}$ | $10^{20}$ (privo di senso) |

I criteri relativi **dichiarano convergenza quando la norma del gradiente
è ancora enorme**. I risultati quick-mode confermano: `grad_rel @ good`
produce $\|\nabla f\| \approx 10^{12}$.

I criteri **assoluti** usano soglie fisse indipendenti dal punto iniziale:
sempre affidabili.

### 4.3 Criteri basati su f-change

Vicino al minimo, $|\Delta F| \approx \|\nabla f\|^2$ (sviluppo di Taylor).
Quindi la tolleranza su $|F_k - F_{k-1}|$ corrisponde a:

| Soglia $|\Delta F|$ | Soglia $\|\nabla f\|$ equivalente |
|---|---|
| $10^{-8}$ (rough) | $10^{-4}$ |
| $10^{-16}$ (good) | $10^{-8}$ |
| (very_good) | **non fattibile** — sotto la precisione macchina |

I criteri f-change sono limitati alla banda "rough". Per soglie più strette,
$|\Delta F|$ è dominato dal rumore di arrotondamento.

### 4.4 Riduzione proposta

Da **22 configurazioni** a **7**:

| # | Criterio | Banda | Motivazione |
|---|----------|-------|-------------|
| 1 | `grad_abs` | rough ($10^{-4}$) | Gold standard, banda larga |
| 2 | `grad_abs` | good ($10^{-8}$) | Gold standard, banda media |
| 3 | `grad_abs` | very_good ($10^{-12}$) | Gold standard, banda stretta |
| 4 | `x_abs` | rough ($10^{-4}$) | Conferma secondaria |
| 5 | `x_abs` | good ($10^{-8}$) | Conferma secondaria |
| 6 | `grad_rel` | rough ($10^{-4}$) | Mostrare che funziona solo a soglia larga |
| 7 | `combined_abs` | good | Mostrare il combined approach |

Con l'approccio post-hoc, queste 7 configurazioni non richiedono nessuna run
aggiuntiva rispetto alla Phase 1.

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

## 6. Validazione Empirica (Quick-mode)

I risultati della griglia originale in quick-mode (DIMS=[2,1000], 2 starting
points) confermano le scelte:

### Best configurations trovate

| Metodo | c1 | rho | Parametro specifico | Success | Avg iter |
|--------|-----|-----|---------------------|---------|----------|
| MN (best) | 1e-3 | 0.5 | beta=1e-3, max_tau=50, max_bt=30 | 100% | 15.6 |
| TN (best) | 1e-4 | 0.8 | forcing=quadratic, cg_max=50, max_bt=30 | 100% | 19.9 |

### Conferme

1. **c1**: MN best usa 1e-3, TN best usa 1e-4 → il parametro non discrimina
   (entrambi al 100%). Fissare c1=1e-4 è sicuro.

2. **rho**: MN preferisce 0.5, TN preferisce 0.8 → la differenza è reale,
   vale la pena testarla. La griglia ridotta include entrambi i valori.

3. **beta (MN)**: best = 1e-3. La griglia ridotta include sia 1e-6 che 1e-3.

4. **forcing (TN)**: best = quadratic. La griglia ridotta include sia
   superlinear che quadratic.

5. **Parametri eliminati**: max_tau_iter (50 vs 100), max_iter_bt (30 vs 50 vs
   100), cg_max_iter (None vs 50) — tutti al 100% success rate su ogni valore
   testato. La loro eliminazione dalla griglia non causa perdita di informazione.

6. **Stopping criteria relativi**: grad_rel/combined_rel mostrano
   $\|\nabla f\| \approx 10^{12}$ alla "convergenza" (band=good) → confermano
   che i criteri relativi sono pericolosi per P28.
