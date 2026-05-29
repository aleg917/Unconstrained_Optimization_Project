# L'enigma del "successo in 1 iterazione" di TN su P28 (both_fd)

Analisi di **TN, P28, n=1000** confrontando `exact` (Table 166), `only_hess_fd`
(Table 167/168, k=4/8) e `both_fd` (Table 173/174/175, k=4/8/12). Verificato
empiricamente.

Fenomeno apparente:
- `only_hess_fd` (grad esatto, hv_fd): FALLISCE, ‖∇f‖ bloccato ~1e21, `max_iter`, 0/6.
- `both_fd` (grad FD, hv_fd): "risolve" in **1 sola iterazione**, ‖∇f‖=0.00e+00,
  6/6, per **ogni k** — apparentemente meglio del caso `exact` (37 iter).

**Colpo di scena: `both_fd` NON raggiunge la soluzione. Fugge a ~10¹⁷ da x*, e il
gradiente FD viene ingannato nel leggere zero. `only_hess_fd` invece fallisce in
modo onesto.**

## Evidenza misurata (da x̄, S₀≈−3.34e5, ‖g(x₀)‖≈1.36e21)

| iterazione 1 | `only_hess_fd` (grad **esatto**) | `both_fd` (grad **FD**) |
|---|---|---|
| `alpha` | 1.0 | 5.15e-3 (max backtracking) |
| `p_norm` | **3.3e-31** (≈0) | **1.36e21** (enorme) |
| terminazione CG | `tol` (curvatura +∞) | `negcurv_j0` (curvatura ≤ 0) |
| S(x₁) | −3.34e5 (= S₀, fermo) | **1.28e23** (catapultato) |
| ‖x₁ − x*‖∞ | 1.0 (non si muove) | **3.8e17** (lontanissimo) |
| ‖grad **esatto**(x₁)‖ | 1.36e21 | **7.67e73** |
| ‖grad **FD**(x₁)‖ | 1.36e21 | **0.00e+00** ← bugia |
| esito | `max_iter`, 0/6 (fallimento onesto) | "success" iter=1 (falso positivo) |

## Radice comune: gradiente iniziale enorme + passo hv_fd non normalizzato

A x̄, `‖∇f‖ ≈ 1.4·10²¹` (cresce come n⁷). TN è matrix-free:
`Hv ≈ [∇f(x+h·v) − ∇f(x−h·v)]/(2h)`, con `h` **non** normalizzato per ‖v‖. La
prima direzione CG è `v=−g`, ‖v‖~10²¹ → la perturbazione `h·‖v‖ ~ 10¹⁷` anche con
h=1e-4: fuori da qualsiasi regime lineare → la "curvatura" misurata è nonsenso.
Fin qui i due casi sono identici; divergono per QUALE gradiente entra in hv_fd.

## Caso A — `only_hess_fd` (grad esatto): paralisi onesta

Gradiente esatto in hv_fd → il nonsenso esce come curvatura **enormemente
positiva** → CG fa un passo `α·p ≈ 10⁻³¹` → il punto **non si muove** → ‖g‖ resta
10²¹ → `max_iter`. Il criterio di arresto usa il gradiente esatto (mai ingannato):
vede 10²¹ e dichiara correttamente fallimento.

## Caso B — `both_fd` (grad FD): catapulta + bugia autoconsistente

Gradiente FD in hv_fd → curvatura **negativa** (`negcurv_j0`) → CG ripiega su
steepest descent `p=−g` (10²¹). Anche col backtracking massimo (α=5e-3) il passo è
~10¹⁸ → il punto è **catapultato a S≈10²³, 3.8·10¹⁷ lontano da x***. Là
`F(x₁)≈½S⁴≈10⁹²`. Il criterio misura ‖grad_FD(x₁)‖, ma:

```
ΔF = F(x₁+h eᵢ) − F(x₁−h eᵢ) ≈ |gᵢ|·2h ≈ 10⁷³·h
ULP(F) = ε·F ≈ 2.2e-16·10⁹² ≈ 10⁷⁶
⟹  ΔF ≪ ULP(F)  ⟹  F(x₁+h)=F(x₁−h) in float64  ⟹  grad_FD(x₁)=0 esatto
```

Il gradiente vero a x₁ è 7.67·10⁷³ (misurato), ma quello FD va in underflow a 0 →
il test di convergenza (stesso gradiente FD) è ingannato → "success" a iter 1.

## La "coerenza" tra g e H: nel senso peggiore

La stessa patologia (cancellazione catastrofica del gradiente FD) agisce due volte
in modo autoconsistente: (1) dentro hv_fd → curvatura sbagliata → catapulta;
(2) nel criterio di arresto → gradiente nullo fasullo → falso successo. Il
gradiente FD mente coerentemente sia come bussola sia come traguardo.
`only_hess_fd` non viene ingannato perché il suo traguardo (grad esatto) è immune
alla cancellazione — per questo "fallisce", ma è l'unico onesto.

## Perché identico per k=4, 8, 12

Al punto catapultato `|g|/F ~ 10⁻¹⁸`, quindi `ΔF/ULP ~ 10⁻⁴·h < 1` per qualunque
h ≤ 1 → cancellazione totale del gradiente FD a ogni k → stesso identico falso
"1 iterazione, ‖∇f‖=0".

## Quadro completo dei tre comportamenti su P28 n=1000

| caso | Hessiano usato | esito | natura |
|---|---|---|---|
| **exact** | `I+(1+6S²)jjᵀ` esatta (matmul) | 37 iter, x→(1,…,1) esatto | **vera** convergenza Newton |
| **only_hess_fd** | hv_fd + grad esatto | max_iter, ‖g‖~10²¹ | **paralisi onesta** (curvatura sovrastimata → passo ≈0) |
| **both_fd** | hv_fd + grad FD | "1 iter, ‖g‖=0" | **fuga + bugia** (catapulta a 10¹⁷, FD-grad fasullo) |

In una frase: l'unica differenza tra "fallimento totale" e "successo perfetto in
una mossa" è se il gradiente che alimenta hv_fd è esatto (→ paralisi visibile) o
approssimato (→ il punto schizza all'infinito e la stessa approssimazione che l'ha
lanciato dichiara vittoria leggendo uno zero di pura cancellazione). Il "meglio
dell'exact" è in realtà il risultato più sbagliato della batteria.

## Riferimenti nel codice
- `src/methods/truncated_newton.py` — `_cg_truncated` (curvatura `dHd≤0` →
  ritorno `b=−g`; ramo `negcurv_j0`); `apply_H` = `hv_fd` quando `hess_f=None`.
- `src/hessians/finite_diff.py` — `hv_fd` (passo `h=10⁻ᵏ`, NON normalizzato per
  ‖v‖ → perturbazione `h·‖v‖` esplode con il gradiente O(n⁷) di P28).
- `src/gradients/finite_diff.py` — `grad_fd` (cancellazione → 0 esatto dove F è
  astronomico).
- Collegato a [[tn_hvfd_step_mismatch]] (stessa causa su P16) e
  [[fd_k12_cancellation_p28]] (cancellazione del gradiente FD).
