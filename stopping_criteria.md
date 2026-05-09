# Stopping criteria per Steepest Descent — analisi e scelta

Apri questo file nel **preview Markdown** di VSCode (Ctrl+Shift+V) per vederlo formattato bene. Se lo apri come testo, è comunque leggibile.

---

## 1. I tre criteri possibili

Quando il loop di Steepest Descent decide se fermarsi, può guardare tre quantità:

1. **Norma del gradiente**
   $$\|\nabla F(x_k)\| \le \tau_g$$
   "Sono arrivato in un punto stazionario."

2. **Differenza dei valori di F (function progress)**
   $$|F(x_{k+1}) - F(x_k)| \le \tau_f$$
   "La funzione non scende più, sono fermo."

3. **Differenza dei punti (iterate change)**
   $$\|x_{k+1} - x_k\| \le \tau_x$$
   "I miei spostamenti sono ormai trascurabili."

A questi si aggiunge sempre `k >= max_iter` come fail-safe.

---

## 2. Perché le tre tolleranze NON sono intercambiabili

Vicino a un minimo, lo Steepest Descent fa il passo

$$x_{k+1} - x_k = -\alpha_k \, g_k$$

dove `α_k ∈ (0, 1]` è il passo di Armijo (tipicamente 0.5, 0.25, 0.125...).

Da qui si ricavano le tre quantità di stop:

| Criterio | Vale circa | In funzione di `||g||` |
|----------|------------|-------------------------|
| `||grad||` | `||g_k||` | **`||g||`** |
| `||x_{k+1} - x_k||` | `α · ||g_k||` | **`α · ||g||`** |
| `\|F_{k+1} - F_k\|` | `α · ||g_k||² / 2` (da Armijo) | **`α · ||g||² / 2`** |

### Esempio numerico

Supponi α ≈ 0.5 e usi la stessa tolleranza `τ = 10⁻⁶` per tutti e tre:

- `||grad|| ≤ 10⁻⁶` → ti fermi a `||g|| ≈ 10⁻⁶` ✅
- `||Δx|| ≤ 10⁻⁶` → ti fermi a `||g|| ≈ 2·10⁻⁶` ✅ (ordine simile)
- `|ΔF| ≤ 10⁻⁶` → ti fermi a `||g|| ≈ √(2·10⁻⁶/α) ≈ **2·10⁻³**` ❌ **molto presto!**

**Conclusione**: per ottenere precisioni paragonabili sulla stazionarietà,

```
τ_g   ≈ 10⁻⁶      (tolleranza sul gradiente)
τ_x   ≈ 10⁻⁶      (stesso ordine di τ_g)
τ_f   ≈ 10⁻¹²     (deve essere ~ τ_g², perché va come ||g||²)
```

Se tu usassi `τ_f = 10⁻⁶` ti fermeresti molto presto e diresti "convergente" quando non lo è.

---

## 3. Perché tenerli comunque tutti e tre (in OR)

Anche se ridimensionati correttamente, i tre criteri **diagnosticano cose diverse** e non si sostituiscono uno all'altro.

### 3.1 `||grad||`
- È la vera condizione necessaria del primo ordine: in un minimo `∇F = 0`.
- È quella "scientificamente corretta" da riportare nel paper.
- **Difetto**: dipende dalla scala di F. Se moltiplichi F per 1000, anche il gradiente scala per 1000. Per questo si usa la versione **relativa** `||g_k|| / ||g_0||` ≤ τ_rel.

### 3.2 `|ΔF|` — **rileva stagnazione**
- Se F non scende più ma `||grad||` resta ancora "grande", siamo in una zona piatta o mal condizionata.
- Tipico per SD: in valli strette, scendi a zig-zag e fai migliaia di iterazioni senza migliorare F.
- Se non lo metti, il loop continua fino a `max_iter` per niente.

### 3.3 `||Δx||` — **rileva collasso del passo**
- Armijo può ridurre α a valori minuscoli (10⁻¹⁰ e oltre) se la line-search è in difficoltà.
- Quando succede, ti muovi pochissimo anche se il gradiente non è piccolo.
- Senza questo controllo si "gira a vuoto" sul posto.

### 3.4 Cosa fare in pratica

Combinarli in **OR**, ciascuno con la **propria tolleranza** della scala giusta. Quando il loop si ferma, salvare il *motivo*:

```
stop_reason ∈ { "grad_abs", "grad_rel", "f_change", "x_change", "max_iter" }
```

Questo finisce in tabella nel report e dice esattamente *come* è terminato ogni run.

---

## 4. Versioni assolute vs relative

Per rendere le tolleranze indipendenti dalla scala di F (e quindi dal problema):

| Forma assoluta | Forma relativa |
|----------------|-----------------|
| `||g_k||` ≤ τ_g | `||g_k|| / ||g_0||` ≤ τ_g_rel |
| `\|F_{k+1} - F_k\|` ≤ τ_f | `\|F_{k+1} - F_k\| / max(\|F_k\|, 1)` ≤ τ_f_rel |
| `||x_{k+1} - x_k||` ≤ τ_x | `||x_{k+1} - x_k|| / max(\|\|x_k\|\|, 1)` ≤ τ_x_rel |

Il `max(..., 1)` evita la divisione per zero quando il denominatore è piccolo.

**In pratica si usano entrambe le versioni** in OR — quella assoluta vince vicino a F=0 (es. Prob.32), quella relativa vince per F grandi (Prob.16 a `n=10⁵` ha F dell'ordine di 10⁵).

---

## 5. Tolleranze consigliate per il nostro assignment

```python
tol_g_abs   = 1e-6        # ||grad|| <= tol_g_abs
tol_g_rel   = 1e-6        # ||grad_k|| / ||grad_0|| <= tol_g_rel
tol_f_rel   = 1e-12       # |F_{k+1} - F_k| / max(|F_k|, 1) <= tol_f_rel
tol_x_rel   = 1e-8        # ||x_{k+1} - x_k|| / max(||x_k||, 1) <= tol_x_rel
max_iter    = 1000        # safety
```

Ratio:
- `tol_f_rel = 10⁻¹²` perché `|ΔF| ~ ||g||²`, quindi per coerenza con `tol_g = 10⁻⁶` serve il quadrato.
- `tol_x_rel = 10⁻⁸` un po' più stretto del gradiente (perché `||Δx|| ~ α||g||` con α ≤ 1: per la stessa precisione su `||g||`, `||Δx||` è leggermente più piccolo).

---

## 6. Cosa scrivere nel report

**Sezione "Stopping criteria"** (mezza pagina, con la tabella della §2 e il paragrafo §3.4 riadattati). Punti da toccare:

1. Tre criteri usati in OR, ciascuno con la sua tolleranza.
2. Le tolleranze sono **diverse di ordini di grandezza** perché le quantità misurate hanno scale diverse (gradiente lineare, ΔF quadratico in `||g||`).
3. I criteri non sono ridondanti: il gradiente è la condizione "vera", `|ΔF|` rileva stagnazione, `||Δx||` rileva collasso del passo.
4. La forma relativa è preferita per essere indipendente dalla scala di F (importante a `n=10⁵` dove F può valere `10⁵`).
5. Per ogni run riportare il **motivo** di terminazione, non solo "success/fail".

---

## 7. Pseudocodice della logica di stop

```
g_norm0 = ||grad_f(x0)||
F_prev  = F(x0)
x_prev  = x0

for k in 1..max_iter:
    g = grad_f(x)
    g_norm = ||g||

    # Stop sul gradiente
    if g_norm <= tol_g_abs:           stop("grad_abs")
    if g_norm <= tol_g_rel * g_norm0: stop("grad_rel")

    # Esegui un passo (Armijo + update)
    alpha = armijo(...)
    x_new = x + alpha * (-g)
    F_new = F(x_new)

    # Stop sui progressi
    if |F_new - F_prev| <= tol_f_rel * max(|F_prev|, 1):
        stop("f_change")
    if ||x_new - x_prev|| <= tol_x_rel * max(||x_prev||, 1):
        stop("x_change")

    x_prev, F_prev = x_new, F_new
    x = x_new

else:
    stop("max_iter")
```

L'ordine dei controlli è: prima il gradiente (fa scattare la convergenza "vera"), poi i progressi (fanno scattare le diagnosi di stagnazione/collasso).
