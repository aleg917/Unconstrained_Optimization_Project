# Stopping Criteria — Analisi, Scelta e Trappole

---

## 1. I tre criteri

Un metodo iterativo di ottimizzazione puo fermarsi monitorando tre quantita:

1. **Norma del gradiente** — condizione necessaria del primo ordine:
   $$\|\nabla F(x^{(k)})\| \le \tau_g$$
   "Sono in un punto stazionario."

2. **Differenza dei valori di F** — rileva stagnazione:
   $$|F(x^{(k)}) - F(x^{(k-1)})| \le \tau_f$$
   "La funzione non scende piu."

3. **Differenza degli iterati** — rileva collasso del passo:
   $$\|x^{(k)} - x^{(k-1)}\| \le \tau_x$$
   "Non mi sto piu muovendo."

A questi si aggiunge sempre `k >= max_iter` come fail-safe.

---

## 2. Perche le tolleranze NON sono intercambiabili

Vicino a un minimo, con passo di tipo Newton e $\alpha \approx 1$:

| Criterio | Scala rispetto a $\|\nabla F\|$ |
|----------|----------------------------------|
| $\|\nabla F\|$ | $\|\nabla F\|$ |
| $\|x^{(k+1)} - x^{(k)}\|$ | $\approx \alpha \cdot \|\nabla F\| / \lambda_{\min}(H)$ |
| $|F^{(k+1)} - F^{(k)}|$ | $\approx \|\nabla F\|^2$ (Taylor) |

**Conseguenza**: $\tau_f \approx \tau_g^2$. Se $\tau_g = 10^{-6}$, serve $\tau_f = 10^{-12}$.

### Esempio numerico

Con $\alpha \approx 0.5$ e $\tau = 10^{-6}$ uguale per tutti:
- $\|\nabla F\| \le 10^{-6}$ : ti fermi a $\|\nabla F\| \approx 10^{-6}$ (OK)
- $\|\Delta x\| \le 10^{-6}$ : ti fermi a $\|\nabla F\| \approx 2 \cdot 10^{-6}$ (OK, simile)
- $|\Delta F| \le 10^{-6}$ : ti fermi a $\|\nabla F\| \approx 2 \cdot 10^{-3}$ (**troppo presto!**)

---

## 3. Perche servono tutti e tre (in OR)

I criteri diagnosticano cose diverse:

- **Gradiente**: condizione necessaria vera. E' il criterio "scientificamente corretto" da riportare.
- **|Delta F|**: rileva stagnazione in valli strette o regioni piatte, dove il metodo zigzaga senza migliorare F ma il gradiente non e ancora piccolo.
- **||Delta x||**: rileva collasso del passo. Armijo puo ridurre $\alpha$ a $10^{-10}$ se la line search e in difficolta; senza questo check il loop gira a vuoto.

---

## 4. Varianti assolute vs relative

| Forma assoluta | Forma relativa |
|----------------|-----------------|
| $\|\nabla F_k\| \le \tau_g$ | $\|\nabla F_k\| / \|\nabla F_0\| \le \tau_g$ |
| $|F_k - F_{k-1}| \le \tau_f$ | $|F_k - F_{k-1}| / \max(|F_k|, 1) \le \tau_f$ |
| $\|x_k - x_{k-1}\| \le \tau_x$ | $\|x_k - x_{k-1}\| / \max(\|x_k\|, 1) \le \tau_x$ |

Il `max(..., 1)` evita divisione per zero.

**Assoluta**: semplice, interpretabile. Ma dipende dalla scala di F.

**Relativa**: scale-invariant rispetto a F. Ma attenzione: se il denominatore e enorme, la soglia effettiva diventa troppo permissiva (vedi sezione 6).

**Nota** (dalle slide): impostare TOL pari a $\varepsilon_m \approx 10^{-16}$ e inaccettabile per criteri relativi perche gli errori di arrotondamento dominano.

---

## 5. Livelli di tolleranza

Basati sulle slide del corso e sulle relazioni di scala:

### Rough ($10^{-4}$)

| Criterio | Tolleranza |
|----------|-----------|
| grad norm | $10^{-4}$ |
| x change | $10^{-4}$ |
| f change | $10^{-8}$ |

Poca precisione, convergenza veloce.

### Good ($10^{-8}$)

| Criterio | Tolleranza |
|----------|-----------|
| grad norm | $10^{-8}$ |
| x change | $10^{-8}$ |
| f change | $10^{-16}$ |

Default raccomandato. $\tau_f = 10^{-16}$ e al limite della macchina: in pratica scattera prima il gradiente o $\Delta x$.

### Very Good ($10^{-12}$)

| Criterio | Tolleranza |
|----------|-----------|
| grad norm | $10^{-12}$ |
| x change | $10^{-12}$ |
| f change | non fattibile ($10^{-24}$ sotto la macchina) |

Molto esigente. Il criterio su f non e utilizzabile a questo livello.

---

## 6. Dipendenza dalla dimensione: la trappola della 2-norma

Tutti i criteri sopra usano la 2-norma $\|g\|_2 = \sqrt{\sum_{j=1}^n g_j^2}$. Questa **scala con** $\sqrt{n}$: a parita di qualita per-componente, la norma cresce con la dimensione.

### Il problema concreto

Se ogni $|g_j| \approx 10^{-6}$ (ottimo per-componente):

| n | $\|g\|_2$ | Esito con $\tau = 10^{-4}$ |
|---|-----------|----------------------------|
| 2 | $1.4 \times 10^{-6}$ | OK |
| 1,000 | $3.2 \times 10^{-5}$ | OK |
| 100,000 | $3.2 \times 10^{-4}$ | **FAIL** |

La **stessa qualita di soluzione** viene dichiarata OK a n=2 e FAIL a n=100,000.

### Perche il criterio relativo non risolve

Si potrebbe pensare che normalizzare per $\|\nabla F_0\|$ compensi. In realta crea il problema opposto.

$\|\nabla F_0\|$ scala come $O(n^{3/2})$ o peggio (P28: $O(n^7)$). Con $\tau = 10^{-4}$:

$$\text{soglia effettiva} = \tau \cdot \|\nabla F_0\| = 10^{-4} \cdot 10^7 = 10^3$$

Il metodo si ferma con $\|g\| = 600$ e dichiara "OK". Risultati osservati:

| Run | Criterio | iter | f* | $\|g\|$ | Esito |
|-----|----------|------|----|---------|-------|
| P16 n=100k, $\bar{x}$ | grad_rel $10^{-4}$ | 4 | -4.144350e+04 | 1.18 | "OK" |
| P16 n=100k, rand | grad_rel $10^{-4}$ | 21 | **-3.901e+04** | **609** | "OK" |
| P28 n=1000, $\bar{x}$ | grad_rel $10^{-4}$ | 8 | ... | **8.08e+16** | "OK" |

Con grad_rel, il metodo si ferma **molto prematuramente**: f* non e il minimo, il gradiente e enorme, ma il rapporto $\|g\|/\|g_0\|$ e gia sotto tolleranza.

### Soluzione: norma infinito

La norma infinito $\|g\|_\infty = \max_j |g_j|$ e **indipendente dalla dimensione**. Se ogni componente ha $|g_j| < \tau$, allora $\|g\|_\infty < \tau$, qualunque sia n.

| Norma | Significato di $\tau = 10^{-4}$ a n=100,000 |
|-------|----------------------------------------------|
| $\|g\|_2 \le 10^{-4}$ | Media quadratica $< 3.2 \times 10^{-7}$ per componente |
| $\|g\|_\infty \le 10^{-4}$ | **Ogni** componente $< 10^{-4}$ |

La seconda ha un significato fisico diretto e non dipende da n.

---

## 7. Stagnazione con alpha scalare su problemi separabili

Anche con il criterio "giusto", i metodi possono stagnare per un motivo strutturale.

### Il meccanismo

Quando l'hessiana e diagonale (P16), il passo di Newton e per-componente: $p_j = -g_j / H_{jj}$. Ma la line search di Armijo sceglie un **unico** $\alpha$ per tutte le n componenti.

Se le scale sono diverse (per P16: $H_{jj} \approx j$, da 1 a 100,000):
1. Le componenti con j piccolo hanno $p_j$ grande e possono overshooting
2. Armijo riduce $\alpha$ per accomodare le componenti "difficili"
3. Le componenti con j grande (che hanno gia $p_j$ minuscolo) subiscono un $\alpha$ moltiplicato per un passo microscopico
4. Il passo effettivo $\alpha \cdot p_j$ si perde nel floating point

**Risultato**: $\|g_{k+1}\|/\|g_k\| \approx 1.0$ — il metodo e completamente fermo. Non sta convergendo lentamente: non si muove.

### Perche da $\bar{x}$ funziona

Da $\bar{x} = (1, \ldots, 1)$, tutte le componenti partono dallo stesso valore. I passi Newton sono uniformi, non c'e conflitto di scala, $\alpha \approx 1$ viene accettato, convergenza quadratica in 6 iterazioni.

### Per il report

Questo non e un difetto del metodo: e una conseguenza nota dell'uso di un passo scalare su un problema con scale diverse. Il minimo viene comunque trovato correttamente (f* identico da tutti gli starting point). La "non convergenza" e solo nella certificazione tramite il gradiente, non nella qualita della soluzione.

---

## 8. Combinazione (OR) e criterio combinato

Quando si usano piu criteri in OR:
- Il metodo si ferma appena **uno qualsiasi** scatta
- Il campo `stop_reason` registra quale criterio ha scattato
- Alla banda "rough" domina il gradiente o $\Delta x$ (f-change scatta dopo per la relazione quadratica)
- Alla banda "good"/"very good" domina il gradiente (f-change raggiunge i limiti della macchina)

### Raccomandazione pratica

Per il nostro assignment:
- **Gradiente (assoluto)** come criterio primario — misura direttamente l'ottimalita
- **$\Delta x$ (assoluto)** come safety net — cattura il collasso del passo
- **$\Delta F$** solo alla banda "rough" dove e numericamente significativo
- Riportare sempre il **motivo** di terminazione, non solo OK/FAIL

---

## 9. Pseudocodice

```
g_norm0 = ||grad_f(x0)||
F_prev  = F(x0)
x_prev  = x0

for k in 1..max_iter:
    g = grad_f(x)
    g_norm = ||g||     # o ||g||_inf per indipendenza dalla dimensione

    # Stop sul gradiente
    if g_norm <= tol_g_abs:           stop("grad_abs")
    if g_norm <= tol_g_rel * g_norm0: stop("grad_rel")

    # Esegui un passo (Armijo + update)
    alpha = armijo(...)
    x_new = x + alpha * p
    F_new = F(x_new)

    # Stop sui progressi
    if |F_new - F_prev| <= tol_f * max(|F_prev|, 1):
        stop("f_change")
    if ||x_new - x_prev|| <= tol_x * max(||x_prev||, 1):
        stop("x_change")

    x_prev, F_prev = x_new, F_new
    x = x_new

else:
    stop("max_iter")
```
