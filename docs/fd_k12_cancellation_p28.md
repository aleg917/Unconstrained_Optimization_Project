# k=12 su P28: ‖∇f‖ = 0 in tabella ma path assurdi nel grafico

Analisi di **TN, P28, n=2, both_fd, h=fixed** al variare di k (Table 143/144/145
e relativi grafici di traiettoria). A k=4 e k=8 i path sono puliti e convergono a
(1,1); a k=12 il grafico mostra zig-zag e rette che attraversano tutto il dominio,
mentre la tabella riporta ‖∇f‖ = 0.00e+00 (esatto) e "success" per tutti i punti.
Sono **due artefatti numerici distinti** che convivono. Verificato empiricamente.

## Setup: in `both_fd` ci sono DUE differenze finite annidate

Per TN P28 `both_fd` (grad_f=None, hess_f=None) sia il gradiente sia la curvatura
sono FD del valore di funzione:

- **gradiente**: `g_FD = [f(x+h eᵢ) − f(x−h eᵢ)] / (2h)`,  `h = 10⁻ᵏ`
- **curvatura** (CG usa `hv_fd`): `Hv = [g_FD(x+h·v) − g_FD(x−h·v)] / (2h)`
  → una **FD di una FD** = differenza finita applicata DUE volte a `f`.

Floor di roundoff di una FD centrata: il numeratore sottrae due valori quasi
uguali, ognuno con errore relativo `ε ≈ 2.2e-16`, quindi errore assoluto del
numeratore `~ε·|f|`, e diviso per `2h`:

| | floor gradiente `ε|f|/(2h)` | floor `hv_fd` `ε|f|/(4h²)` |
|---|---|---|
| **k=4** (h=1e-4) | ~2.5e-11 (a x̄, \|f\|=23) | ~1.3e-7 |
| **k=8** (h=1e-8) | ~2.5e-7 | ~12 |
| **k=12** (h=1e-12) | ~2.5e-3 | **~1.3e9** |

La seconda differenza finita di `hv_fd` **riapplica un fattore 1/h**: a k=12
l'amplificazione è `1/h² = 10²⁴`, quindi la curvatura FD è **rumore ~10⁹**.

## Perché i path sono assurdi (k=12)

Diagnostico (TN both_fd, P28 n=2, da x̄=(0.5,0)):

| k | iter | ‖g_final‖ | neg_curv | p_norm max | distanza max dal min |
|---|---|---|---|---|---|
| 4 | 7 | 4.3e-9 | 0 | 0.40 | 1.0 |
| 8 | 7 | 3.7e-9 | 0 | 0.40 | 1.0 |
| **12** | **137** | **0.00e+00** | **116** | **3.1e10** | **1.1e8** |

A k=12 i prodotti `Hv` sono spazzatura (~10⁹) → CG produce direzioni senza senso,
con curvatura negativa spuria 116 volte → **passi giganteschi (p_norm fino a
3·10¹⁰)** → gli iterati schizzano fino a **~10⁸** dal minimo. Là `F` è enorme (il
termine `S⁴` di P28: a x~10⁸, `F~10³³`), quindi il floor di rumore esplode ancora
di più → vagabondaggio caotico. Quegli zig-zag e quelle rette che attraversano il
dominio **sono il rumore della doppia FD, non vera ottimizzazione**. A k=4/k=8
invece `Hv` è accurato → vera convergenza di Newton (7 iter, conv_rate ~2,
neg_curv=0, path pulito verso (1,1)).

## Perché la tabella mostra ‖∇f‖ = 0.00e+00 esatto

Il criterio di arresto misura `‖g_FD(x)‖`. La differenza centrata restituisce
**esattamente 0** quando il vero salto `2h·gᵢ` scende sotto la granularità di
arrotondamento (ULP) di f:

```
2h·|gᵢ|  <  ε·|f|     ⟹   f(x+h eᵢ)  e  f(x−h eᵢ)  danno lo STESSO float64
                      ⟹   numeratore = 0  ⟹  g_FD = 0  esatto
```

A k=12 (`2h = 2e-12`) questo accade facilmente lungo il percorso caotico: appena
l'iterato capita in un punto dove `f`, perturbata di ±10⁻¹², non cambia
rappresentazione in virgola mobile, **la FD non riesce più a misurare il gradiente
e ritorna il vettore nullo**. Il criterio `‖g‖ ≤ 1e-8` vede `0` e dichiara
`grad_abs / success`.

È un **falso positivo**: non è che il gradiente vero sia zero (l'iterato può non
essere affatto stazionario), è che la FD ha perso tutta la cifra significativa per
cancellazione catastrofica. Lo confermano:
- `conv_rate ≈ 0.9` a k=12 (vs ~2 a k=4/8): nessuna vera convergenza quadratica.
- iter erratici (7 → 137): dipende solo dalla fortuna di quando `g_FD` va in
  underflow a 0.

## Nota: il gradiente FD NON è rumoroso ovunque

Controintuitivo ma importante: vicino al minimo `|f|` è piccolo (a x=(1+1e-6,…),
`f≈5.5e-12`), quindi il floor `ε|f|/(2h)` è minuscolo e `grad_fd` resta accurato
anche a k=12 (dà 8.06e-6 vs esatto 8.06e-6). Il rumore del gradiente scala con
`|f|`: è grande solo nelle regioni ad alto `F` (il giallo del contour). Il vero
detonatore del caos è la **doppia** FD di `hv_fd` (fattore 1/h²), non il gradiente.

## In una riga

A k=12 il passo `h=10⁻¹²` è così piccolo che la cancellazione catastrofica governa
entrambe le FD: la doppia FD di `hv_fd` esplode in rumore ~10⁹ (→ direzioni e passi
assurdi nel grafico), mentre la FD del gradiente collassa a esattamente 0 quando
`2h·g < ε|f|` (→ ‖∇f‖=0 e "success" spurio in tabella). Convivono perché misurano
`f` con lo stesso `h` rovinoso, ma una *amplifica* il roundoff (curvatura) e
l'altra lo *annulla* (gradiente).

## Riferimenti nel codice
- `src/gradients/finite_diff.py` — `grad_fd` (FD centrata, floor `ε|f|/(2h)`).
- `src/hessians/finite_diff.py` — `hv_fd` (FD del gradiente → doppia FD, floor
  `ε|f|/(4h²)`); usato da TN in `both_fd`.
- `src/methods/truncated_newton.py` — `_cg_truncated` (curvatura negativa spuria);
  criterio `GradNormAbsolute` ingannato dal gradiente FD nullo.
