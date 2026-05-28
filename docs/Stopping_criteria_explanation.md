# Stopping Criteria — Filosofia del tuning post-hoc e analisi

Documento di accompagnamento a `tuning_analysis.md`. Tratta esclusivamente
il tuning degli stopping criteria — un'attività concettualmente distinta dal
tuning dei parametri del metodo, che richiede una metodologia specifica.

---

## 1. Filosofia: cosa stiamo facendo nella Phase 2 del notebook

### 1.1 Il punto di partenza concettuale

Il tuning "classico" che si insegna è: definisci una griglia di parametri,
esegui l'algoritmo per ogni combinazione, scegli quella che ottimizza una
metrica (success rate, numero di iterazioni, tempo). È esattamente quello
che fa la **Phase 1** del notebook per `beta`, `forcing`, `rho`.

Per gli stopping criteria, **questo approccio è inefficiente e teoricamente
non necessario**, per un motivo fondamentale:

> **Lo stopping criterion non altera la traiettoria di ottimizzazione.**

Modified Newton e Truncated Newton, dati un punto iniziale $x_0$ e un set
di parametri (β/forcing/ρ/c₁/α₀), generano una sequenza deterministica
$\{x_0, x_1, x_2, \ldots\}$. Lo stopping criterion decide **soltanto in
quale punto** della sequenza ci fermiamo e dichiariamo "convergenza". Non
cambia la sequenza stessa.

### 1.2 Conseguenza pratica: la valutazione post-hoc

Se la sequenza è la stessa, possiamo:

1. **Eseguire l'algoritmo una sola volta**, con il criterio di stop più
   *permissivo* possibile (in pratica: `GradNormAbsolute(tol=1e-12)`, che
   fa girare l'algoritmo quasi sempre fino a `max_iter`).
2. **Salvare la history completa** (`x_k`, `‖∇f_k‖`, `f_k` per ogni $k$).
3. **Valutare offline** N stopping criteria diversi sulla stessa history,
   cercando per ciascuno la prima iterazione in cui scatterebbe.

Costo: 1 run vs N run. Su una griglia di 22 criteri e ~50 celle problema×dim,
risparmiamo ~21 × 50 × 6 = 6300 run di algoritmo. Le valutazioni post-hoc sono
puramente aritmetiche e impercettibili in tempo.

Questa è esattamente la struttura della **Phase 2** del notebook.

### 1.3 Cosa stiamo *scoprendo* — è davvero "tuning"?

Sì, è tuning a pieno titolo. Per ciascun criterio (c, tol) impariamo:

- **`fire_rate`**: il criterio si attiva mai sulla traiettoria, o "manca"
  l'iterazione di convergenza? Esempio fallimento: `grad_rel` con
  $\tau=10^{-12}$ su un problema dove $\|\nabla f_0\|$ è enorme — il
  rapporto $\|\nabla f_k\|/\|\nabla f_0\|$ resta sempre sopra $10^{-12}$.
- **`stop_iter`**: quanto presto si attiva. Un criterio che si attiva alla
  prima iterazione è inutile (siamo lontani dal minimo); uno che si attiva
  a `max_iter` non discrimina.
- **`grad_norm_at_stop`** (e `f_at_stop`): la qualità della soluzione al
  momento del trigger. Un criterio che si attiva quando $\|\nabla f\|$ è
  ancora $10^{12}$ è un **falso positivo**.

Il tuning consiste nel selezionare il criterio sulla **frontiera di Pareto**
tra "early stop" (`stop_iter` basso) e "soluzione accurata"
(`grad_norm_at_stop` basso).

---

## 2. Risposte alle domande aperte

### 2.1 "Posso sostituire il criterio della Phase 1 con quello migliore trovato in Phase 2?"

**Sì, senza esitazione.** È esattamente il motivo per cui stiamo facendo la
Phase 2. Il criterio fisso usato in Phase 1 (`GradNormAbsolute(1e-4)`) serve
solo a non far girare l'algoritmo all'infinito durante la grid search dei
parametri. Una volta scelti i best parametri MN e TN, la Phase 2 ci dice
quale criterio usare in `main.ipynb` e nei test finali.

Il fatto che il criterio cambi tra Phase 1 e produzione è benigno: la
**traiettoria** generata dai best parametri è la stessa, cambia solo il punto
in cui ci fermiamo.

### 2.2 "È possibile che la Phase 2 trovi qualcosa di meglio della Phase 1?"

**Sì, ed è il motivo per cui la facciamo.** Esempi concreti:

- Phase 1 ha usato `grad_abs @ rough = 10^{-4}`. La Phase 2 può rivelare che
  `combined_abs @ good` (che richiede $\|\nabla f\| \leq 10^{-8}$ *OR*
  $\|\Delta x\| \leq 10^{-8}$ *OR* $|\Delta F| \leq 10^{-16}$) si attiva
  poche iterazioni dopo, dando una soluzione molto più accurata
  ($\|\nabla f\| \approx 10^{-8}$ invece di $10^{-4}$).
- La Phase 2 può rivelare che `grad_rel @ good` si attiva *molto presto* su
  P28 ma con $\|\nabla f\| \approx 10^{12}$ — un falso positivo da
  evitare (vedi §3.2).

### 2.3 Cosa scrivere nel report (specifica del progetto)

> *"Concerning any possible parameter of the methods implemented [...] tune
> it and discuss/motivate your choice. If you perform preliminary tests with
> parameters that are not your final choice, you do not need to include the
> numerical results in the report, but you must briefly describe what
> happened in the preliminary tests."*

Pattern consigliato per la sezione "Stopping Criteria" del report:

1. **Filosofia**: una frase per spiegare che il criterio non altera la
   traiettoria → giustifica la valutazione post-hoc su 1 sola run lunga.
2. **Set testato**: elencare i criteri considerati (6 base × 3 band + 2
   combined × 2 band ≈ 22 configurazioni).
3. **Test preliminari** (1-2 frasi): "Durante la fase iniziale abbiamo usato
   `GradNormAbsolute(tol=10^{-4})` come criterio di lavoro per la grid
   search dei parametri del metodo. Questa scelta minimale serviva solo a
   garantire la terminazione."
4. **Risultato della Phase 2**: tabella con `fire_rate`, `mean_stop_iter`,
   `mean_grad_at_stop` per le ~5 configurazioni Pareto-ottimali. Discussione.
5. **Scelta finale**: criterio + tolleranza usati in `main.ipynb`, con
   motivazione (es. "Combined absolute @ good: fire rate 100%, attivazione
   ~20 iter più tardi di grad_abs@rough, ma `‖∇f‖` finale ≈ 10⁻⁸ vs 10⁻⁴").

---

## 3. Analisi teorica delle tolleranze per criterio

Tutte le tolleranze derivano dal comportamento asintotico vicino al minimo,
ottenuto via sviluppo di Taylor. Sia $x^*$ il minimo e $H^* = \nabla^2 F(x^*)$
SPD. Vicino a $x^*$, in un'iterata Newton/quasi-Newton:

$$\|\nabla f(x_k)\| \to 0, \qquad x_k - x^* = -H_{x_k}^{-1} \nabla f(x_k) + O(\|x_k-x^*\|^2).$$

### 3.1 Riepilogo

| Criterio       | Quantità                              | Scala asintotica          | Tolleranza sensata |
|----------------|---------------------------------------|---------------------------|---------------------|
| `grad_abs`     | $\|\nabla f(x_k)\|$                   | $\to 0$                   | $10^{-4} / 10^{-8} / 10^{-12}$ |
| `grad_rel`     | $\|\nabla f_k\|/\|\nabla f_0\|$       | dipende da $\|\nabla f_0\|$ | **pericoloso su P28** |
| `f_abs`        | $\|F_k - F_{k-1}\|$                   | $\approx \frac{1}{2}\|\nabla f\|^2$ | $\text{tol}_f \approx \text{tol}_g^2$ |
| `f_rel`        | $\|\Delta F\|/\max(\|F\|, 1)$         | dipende da $|F^*|$        | $\geq \varepsilon_{\text{mach}}$ |
| `x_abs`        | $\|x_k - x_{k-1}\|$                   | $\approx \|H^{-1}\nabla f\| \sim \kappa \cdot \|\nabla f\|$ | $\text{tol}_x \approx \kappa \cdot \text{tol}_g$ |
| `x_rel`        | $\|\Delta x\|/\max(\|x\|, 1)$         | come x_abs normalizzato   | come x_abs |

### 3.2 Pericolo dei criteri relativi su P28

La definizione `grad_rel`:

$$\frac{\|\nabla f(x_k)\|}{\|\nabla f(x_0)\|} \leq \tau$$

Su P28, $\|\nabla f(x_0)\| = O(n^7)$ (vedi `functions/problem28.py`). Con
$\tau = 10^{-8}$ ("good"):

| $n$    | $\|\nabla f(x_0)\|$ ordine | Soglia effettiva $\|\nabla f_k\|$  |
|--------|-------------------|------------------------------------|
| 100    | $10^{14}$         | $10^{6}$  (pessimo)                |
| 1000   | $10^{21}$         | $10^{13}$ (catastrofico)           |
| 10000  | $10^{28}$         | $10^{20}$ (privo di senso)         |
| 100000 | $10^{35}$         | $10^{27}$ (irreale)                |

Il criterio "fires" quando il gradiente è ancora enorme — falso positivo.
Su P16 il problema non c'è perché $\|\nabla f_0\|$ è $O(n)$.

### 3.3 Derivazione $\text{tol}_f \approx \text{tol}_g^2$

Tra due iterati successivi $x_k, x_{k+1}$ con $x_{k+1} = x_k + \alpha_k p_k$
e $p_k$ direzione di Newton ($H_k p_k = -\nabla f_k$, con $\alpha_k \approx 1$
vicino al minimo):

$$F(x_{k+1}) - F(x_k) \approx \nabla f_k^T (\alpha_k p_k) + \tfrac{1}{2} (\alpha_k p_k)^T H_k (\alpha_k p_k)$$

Per $\alpha_k = 1$ e $p_k = -H_k^{-1}\nabla f_k$:

$$\Delta F = -\nabla f_k^T H_k^{-1} \nabla f_k + \tfrac{1}{2} \nabla f_k^T H_k^{-1} \nabla f_k = -\tfrac{1}{2}\nabla f_k^T H_k^{-1} \nabla f_k.$$

Se $H_k$ è ben condizionata ($\kappa = O(1)$), $|\Delta F| \approx \frac{1}{2}\|\nabla f_k\|^2$.

Conseguenza: per essere coerente con `grad_abs @ tol_g`, la tolleranza
$\text{tol}_f$ deve essere dello stesso ordine di $\text{tol}_g^2$.

| Band       | $\text{tol}_g$ | $\text{tol}_f$ coerente | $\text{tol}_x$ coerente |
|------------|----------------|--------------------------|--------------------------|
| rough      | $10^{-4}$      | $10^{-8}$                | $10^{-4}$                |
| good       | $10^{-8}$      | $10^{-16}$               | $10^{-8}$                |
| very_good  | $10^{-12}$     | **non fattibile** (sotto $\varepsilon_{\text{mach}}$) | $10^{-12}$ |

Il `TOLERANCE_BANDS` attuale del notebook (`fine_tuning.ipynb` linea 2092)
rispetta già questa convenzione. È importante mantenerla anche quando si
testa `f_abs` come criterio autonomo.

### 3.4 Limite delle tolleranze very_good su `f_abs`

A precisione macchina IEEE 754 double, $\varepsilon_{\text{mach}} \approx 2.2 \cdot 10^{-16}$.
Quando $|F_k - F_{k-1}| < \varepsilon_{\text{mach}} \cdot |F_k|$, la sottrazione
è dominata dal rumore di arrotondamento e il criterio non è più affidabile.
Per P28 con $|F^*| = 0$ e $|F_k|$ piccolo vicino al minimo, $\text{tol}_f = 10^{-16}$
è già al limite. La banda very_good ($\text{tol}_f$ richiederebbe $10^{-24}$)
**non è fattibile** per `f_abs` ed è impostata a `None` nel notebook.

---

## 4. Il bug della Phase 2 — diagnosi e fix

### 4.1 Sintomo osservato

Le tabelle di Phase 2 stampate dal notebook mostrano:

```
=== Stopping Criteria — ModNewton ===
   crit_type      band  success_rate  mean_stop_iter  mean_grad_at_stop
combined_abs      good      0.857143       15.738095       6.131136e+27
   grad_abs      good      0.500000      364.476190        6.131136e+27
      x_abs      good      0.666667       15.976190        6.131136e+27
   grad_abs     rough      0.738095      130.285714        6.131136e+27
   grad_rel     rough      0.857143        8.761905        6.131136e+27
      x_abs     rough      0.857143       14.619048        6.131136e+27
   grad_abs very_good      0.452381      411.880952        6.131136e+27
```

`mean_grad_at_stop` è **identico** ($\approx 6.13 \cdot 10^{27}$) per tutte
le righe — impossibile in un'analisi corretta. Ogni criterio fa un trigger
diverso, dovrebbe leggere il gradiente in un'iterazione diversa.

### 4.2 Causa

In `fine_tuning.ipynb`, intorno alla cella 2399-2408:

```python
for entry in log[1:]:
    k = entry['k']; gn = entry['grad_norm']; ...
    if fired:
        return k, gn, f_val, reason   # <-- OK: legge gn[k] alla iter del trigger
return None

# Caller:
result = eval_criterion_on_log(log, crit_type, band)
if result is not None:
    stop_iter, grad_at_stop, f_at_stop, reason = result
    sc_rows.append(dict(
        ...
        grad_norm_at_stop=grad_at_stop,   # <-- corretto per fired runs
        ...))
else:
    final_g = log[-1]['grad_norm']        # <-- BUG: gradiente finale della baseline
    sc_rows.append(dict(
        ...
        grad_norm_at_stop=final_g,        # <-- stesso valore per tutti i crit_type
        ...))
```

Quando un criterio post-hoc non si attiva sulla traiettoria baseline (es.
`grad_abs @ very_good = 10^{-12}` su P28 dove `‖∇f‖` non raggiunge mai
$10^{-12}$ in 1000 iter), il caller scrive `grad_norm_at_stop = log[-1]['grad_norm']`,
ovvero il **gradiente finale della run baseline**.

La baseline è `MetricsLogger(GradNormAbsolute(tol=1e-12))` (cella 2357):
una tolleranza così stretta non viene mai raggiunta su P28 dove
$\|\nabla f_0\| = O(n^7)$. La run finisce a `max_iter` con un gradiente
ancora enorme — $\sim 10^{27}$ a $n=10000$, $\sim 10^{35}$ a $n=100000$.

Successivamente l'aggregazione (cella ~2464):

```python
mean_grad_at_stop=('grad_norm_at_stop', lambda s: np.nanmean(s)),
```

calcola la media su sia i casi "fired" (valori sensati) che "not-fired"
(valori enormi). I valori enormi dominano numericamente, e si propagano in
modo identico a tutte le righe.

### 4.3 Fix

Cambiare il ramo `else` per inserire `NaN`, non `final_g`:

```python
else:
    sc_rows.append(dict(
        method=method_name, problem=prob_id, n=n,
        start_idx=si, crit_type=crit_type, band=band,
        stop_iter=len(log) - 1, success=False,
        grad_norm_at_stop=np.nan,    # <-- era final_g
        f_at_stop=np.nan,            # <-- era final_f
        stop_reason='max_iter'))
```

L'aggregazione con `np.nanmean` poi calcola correttamente la media solo
sulle run dove il criterio si è attivato (`success=True`). Le run failed
si vedono nella colonna `success_rate` (frazione di run dove il criterio
si è attivato) — che è la quantità giusta per giudicare l'utilità del
criterio.

### 4.4 Rinominare per chiarezza: `success_rate` → `fire_rate`

Nell'aggregazione, è opportuno aggiungere o rinominare la colonna in
`fire_rate` (frazione di run dove il criterio post-hoc si attiva), per
distinguerla concettualmente dal `success_rate` della Phase 1 (frazione di
run dove l'algoritmo ha convergito).

---

## 5. Espansione del set di stopping criteria testati

### 5.1 Set originale (7 configurazioni)

Definito in `fine_tuning.ipynb` cella 2120:

```python
SC_CONFIGS = [
    ('grad_abs',     'rough'),
    ('grad_abs',     'good'),
    ('grad_abs',     'very_good'),
    ('x_abs',        'rough'),
    ('x_abs',        'good'),
    ('grad_rel',     'rough'),
    ('combined_abs', 'good'),
]
```

Mancano (criteri implementati ma mai testati):
- `f_abs` (tutte le band)
- `f_rel` (tutte le band)
- `x_rel` (tutte le band)
- `grad_rel @ good`, `grad_rel @ very_good` (importante per *mostrare* il
  falso positivo su P28)
- `combined_rel` (non implementato; va aggiunto)
- `combined_abs @ rough`, `combined_abs @ very_good`

### 5.2 Set espanso (≈22 configurazioni)

```python
BASE_CRITS = ['grad_abs', 'grad_rel', 'f_abs', 'f_rel', 'x_abs', 'x_rel']
BANDS      = ['rough', 'good', 'very_good']
COMBINED   = ['combined_abs', 'combined_rel']

SC_CONFIGS = (
    [(c, b) for c in BASE_CRITS for b in BANDS]       # 6 × 3 = 18
    + [(c, b) for c in COMBINED for b in ['rough', 'good']]  # 2 × 2 = 4
)
# Totale: 22 configurazioni
```

Alcune combinazioni *non* sono fattibili e vanno saltate gracefully:
- `f_abs @ very_good` e `f_rel @ very_good`: $\text{tol}_f$ sotto
  $\varepsilon_{\text{mach}}$ (vedi §3.4). Si possono includere ma ci si
  aspetta `fire_rate = 0` consistentemente.

### 5.3 Nuove classi/mapping necessari

Tutti i criteri base sono già implementati in `src/stopping_criteria/`:
- `GradNormAbsolute` ✓, `FChangeAbsolute` ✓, `XChangeAbsolute` ✓
- `GradNormRelative` ✓, `FChangeRelative` ✓, `XChangeRelative` ✓

Da aggiungere nel notebook:

1. **`CombinedRelStoppingCriterion`**: specchio dell'attuale
   `CombinedStoppingCriterion` (cella ~145), ma componendo
   `GradNormRelative + FChangeRelative + XChangeRelative` invece delle
   varianti absolute.

2. **Estensione di `eval_criterion_on_log`** (cella ~2283): aggiungere il
   mapping per i nuovi `crit_type`:

   ```python
   CRIT_FACTORIES = {
       'grad_abs': lambda tol_g, tol_f, tol_x: GradNormAbsolute(tol_g),
       'grad_rel': lambda tol_g, tol_f, tol_x: GradNormRelative(tol_g),
       'f_abs':    lambda tol_g, tol_f, tol_x: FChangeAbsolute(tol_f),
       'f_rel':    lambda tol_g, tol_f, tol_x: FChangeRelative(tol_f),
       'x_abs':    lambda tol_g, tol_f, tol_x: XChangeAbsolute(tol_x),
       'x_rel':    lambda tol_g, tol_f, tol_x: XChangeRelative(tol_x),
       'combined_abs': lambda tol_g, tol_f, tol_x: CombinedStoppingCriterion(
           GradNormAbsolute(tol_g), FChangeAbsolute(tol_f) if tol_f else None,
           XChangeAbsolute(tol_x)),
       'combined_rel': lambda tol_g, tol_f, tol_x: CombinedRelStoppingCriterion(
           GradNormRelative(tol_g), FChangeRelative(tol_f) if tol_f else None,
           XChangeRelative(tol_x)),
   }
   ```

3. **Skip dei `tol_f = None` per `f_abs`/`f_rel`**: quando la band è
   `very_good` e $\text{tol}_f$ è `None`, registrare `fire_rate=0` senza
   far girare il criterio.

---

## 6. Lettura dei risultati Phase 2 dopo il fix

Tre metriche da leggere **insieme** (mai una sola):

| Metrica              | Cosa significa                                       | Valore "buono" |
|----------------------|------------------------------------------------------|----------------|
| `fire_rate`          | Frazione di run dove il criterio si attiva           | $\geq 0.95$    |
| `mean_stop_iter`     | Iterazione tipica del trigger                        | piccolo        |
| `mean_grad_at_stop`  | $\|\nabla f\|$ tipico al trigger (solo run *fired*)   | piccolo        |

### 6.1 Pattern di errore tipici da cercare

- `fire_rate = 0`: il criterio non si attiva mai. Esempio: `grad_abs @ very_good`
  su P28 (tol=$10^{-12}$, ma $\|\nabla f\|$ non scende sotto $10^{-6}$ in 1000 iter).
- `fire_rate alto` ma `mean_grad_at_stop` grande: falso positivo. Esempio
  classico: `grad_rel @ good` su P28 (vedi §3.2).
- `fire_rate alto`, `mean_grad_at_stop` ragionevole, ma `mean_stop_iter` enorme:
  il criterio è corretto ma non risparmia iterazioni. Esempio: `grad_abs @ very_good`
  su problemi facili (su P16 a n piccolo si attiva, ma molto tardi).

### 6.2 Visualizzazione: frontiera di Pareto

Tracciare scatter:
- asse x: `mean_stop_iter`
- asse y: `mean_grad_at_stop` (scala log)
- dimensione marker: `fire_rate`

Frontiera Pareto-ottimale: punti in basso-a-sinistra con marker grande
(early stop, soluzione accurata, fire rate alto). Una facet per metodo
(MN / TN) e una per problema (P16 / P28).

---

## 7. Scelta finale e template per il report

### 7.1 Criterio di scelta finale (da applicare dopo il rerun)

Per ciascun metodo, scegliere il criterio Pareto-ottimale che soddisfi:

1. `fire_rate ≥ 0.95` su tutte le coppie (problem, n) del set di test (esclusi
   i casi dove l'algoritmo stesso fallisce per `time_limit`/`max_iter`).
2. `mean_grad_at_stop ≤ 10^{-6}` (soluzione "decente": gradiente piccolo in
   valore assoluto).
3. `mean_stop_iter` minimo tra i candidati che soddisfano (1) e (2).

### 7.2 Configurazione consigliata di default (best guess pre-rerun)

Sulla base della teoria (§3) e dei dati pre-fix:

| Metodo | Criterio default | tol_g | Motivazione |
|--------|-----------------|-------|-------------|
| MN     | `combined_abs @ good` | $10^{-8}$ | Robusto: scatta se grad O X change diventano piccoli. tol_f=$10^{-16}$, tol_x=$10^{-8}$. |
| TN     | `combined_abs @ good` | $10^{-8}$ | Stesso ragionamento. |

`grad_abs @ good` come secondario, semplice ma rischia di richiedere troppe
iter su problemi mal-condizionati.

### 7.3 Template di paragrafo per il report

```
We tested 22 stopping criterion configurations (6 base criteria —
absolute and relative variants of gradient norm, function change, and step
length — applied across 3 tolerance bands, plus 2 combined criteria across
2 bands). Because the stopping criterion does not influence the iteration
trajectory, we evaluated all 22 configurations offline on a single
high-resolution run per (method, problem, dimension, starting point), saving
factor-22 of compute.

Preliminary tests used a loose absolute gradient criterion
GradNormAbsolute(tol=1e-4) to enable the grid search over method parameters;
this is not the final choice.

Our analysis revealed that relative criteria are unreliable on Problem 28,
where ||grad f(x_0)|| = O(n^7) makes the relative threshold ineffective even
at "good" tolerance levels. Absolute criteria on the gradient norm are the
gold standard; combined criteria (logical OR of absolute gradient, function
change with tol_f = tol_g^2, and step length) provide robustness at near-zero
extra cost.

Final choice: CombinedAbsoluteStoppingCriterion with tol_g=1e-8,
tol_f=1e-16, tol_x=1e-8 ("good" band), used in main.ipynb for the production
runs of both Modified Newton and Truncated Newton.
```

Da rivedere e popolare con i numeri reali dopo il rerun della Phase 2 con il bug fixato.
