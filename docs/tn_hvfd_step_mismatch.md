# Truncated Newton e il passo di `hv_fd`: perché k=4 si blocca e k=8 converge (P16)

Analisi del confronto **TN, P16, n=1000, only_hess_fd, h=specific** tra k=4
(Table 50) e k=8 (Table 51), con verifica empirica.

## Premessa: per TN, `only_hess_fd` su P16 NON usa la diagonale FD

Nel notebook (`build_method_kwargs`):

```python
use_p16_sparse_hess = (method_name == 'MN' and pinfo['id'] == 'P16')   # FALSE per TN
kw['hess_f'] = make_sparse_diag_fd_hess(...) if use_p16_sparse_hess else None
```

Per **MN** P16 usa la Hessiana diagonale FD (`hess_fd_diag`, quasi esatta a ogni k).
Per **TN** invece `hess_f = None` → TN usa `hv_fd` **matrix-free**: prodotti
Hessiano-vettore via FD del gradiente esatto, con **passo scalare**:

```
H(x) v ≈ [∇f(x + h·v) − ∇f(x − h·v)] / (2h),   h = 10⁻ᵏ · max(‖x‖∞, 1)   (scaled)
```

Punto chiave: `h` **non** è normalizzato per `‖v‖`. La perturbazione effettiva
applicata a x è **`h·‖v‖`**. Quando la direzione `v` (= residuo CG, inizialmente
`−g`) ha norma enorme, la perturbazione esce dal regime lineare e `hv_fd` misura
una differenza *attraverso* le oscillazioni trigonometriche di P16, restituendo
una curvatura sbagliata.

## Risultati verificati (diagnostico, time_limit=4s)

| | SP0=x̄, k=4 | SP0, k=8 | SP1 (random), k=4 | SP1, k=8 |
|---|---|---|---|---|
| **h·‖g₀‖** (pert. effettiva) | 1.54 | 1.5e-4 | **2.86** | 2.9e-4 |
| iter | **24 ✓** | 98 (TL) | 839 (TL) | 84 (TL) |
| ‖g_final‖ | 6.8e-9 | 2.2e-7 | **1.26e4 (bloccato)** | 2.4e-6 |
| neg_curv_count | 0 | 0 | **839 / 839** | 12 |
| CG termination | tol | maxiter ×80 | **negcurv_j0 ×814** | maxiter ×56 |
| alpha (mediana) | 1.0 | 0.21 | **5.15e-3 (max backtrack)** | 0.39 |

(TL = time_limit. Con il budget pieno di 20s gli iter scalano, ma gli ordini di
grandezza combaciano con Table 50/51.)

## Meccanica del confronto

### k=4 sui random start (SP1–5): la trappola della curvatura negativa
Con `‖g₀‖ ≈ 1.4e4` e `h = 2e-4`, la perturbazione `h·‖v‖ ≈ 2.9` è **enorme**:
`hv_fd` campiona il gradiente attraverso le oscillazioni di P16 → il prodotto Hv è
**spazzatura**. A ogni iterazione il test di curvatura esce `dᵀHd ≤ 0`
(`negcurv_j0` in **814/839** iter) → **CG ritorna `b = −g`, cioè steepest descent
col gradiente esatto**. Su P16 il condizionamento è ~n=1000 (autovalori
`j·cos xⱼ − 2 sin xⱼ`, da O(1) a ~1000): steepest descent ha rate ≈ 1 − 2/κ ≈
0.998, lentissimo. In più il passo è gigante (`p_norm` mediana 1.26e4, max 8e22)
e la line-search lo schiaccia ad `alpha ≈ 5e-3` (50 backtrack). →
**nessun progresso, gradiente fermo a ~1.3e4, migliaia di iter a vuoto** (Table 50).

### k=8: stessa meccanica, ma perturbazione piccola
`h·‖v‖ ≈ 2.9e-4` → `hv_fd` resta nel regime lineare → curvatura corretta
(neg_curv raro) → CG costruisce buone direzioni di Newton → il gradiente scende di
11 ordini (1.5e4 → ~1e-6). Vicino alla soluzione `eta_k = min(0.5, ‖g‖) = ‖g‖`
diventa minuscolo: CG deve risolvere quasi-esattamente e **satura il cap interno**
(`cg_max_iter = n`, → `maxiter`) → ogni iterazione esterna costa moltissimo →
poche iter entrano in 20s → time_limit a ‖g‖~1e-7 (Table 51).

### Perché SP0 = x̄ fa l'opposto (k=4 vince, 24 iter, neg_curv=0)
In x̄ tutte le componenti valgono 1: la simmetria fa cadere bene la prima
direzione e la curvatura resta positiva. Lì la perturbazione grande di k=4 non
innesca la trappola e k=4 converge superlinearmente (conv_rate 1.32), mentre k=8 è
solo lento. È la riga "1/6 success" della Table 50.

## Due firme generali per TN

- **Molte iter + gradiente ALTO** → il metodo NON progredisce: direzioni cattive,
  non vere direzioni di Newton. Tipicamente CG cade su steepest descent
  (curvatura negativa/sballata) su problema mal condizionato, con `alpha`
  schiacciato dal backtracking. Fingerprint: `neg_curvature_count` alto,
  `cg_termination` = **negcurv_j0**, `alpha` minimo, `‖g‖` piatto. Fallimento di
  **qualità della direzione**.
- **Molte iter + gradiente BASSO** → il metodo STA convergendo ma è lento
  per-iterazione o non aggancia la tolleranza stretta. Vicino alla soluzione
  `eta_k → 0` forza CG a risolvere quasi-esattamente → satura `cg_max_iter`
  (fingerprint: `cg_termination` = **maxiter**/tol, neg_curv raro) → poche iter
  esterne nel time budget. Oppure floor di rumore FD che ferma le ultime cifre.
  Non è un vero fallimento: convergenza limitata da **costo per iterazione** o
  **precisione FD**.

## Verdetto sull'ipotesi (k=4 → vᵀHv≤0 → CG ritorna −g → steepest descent scadente)

- **Corretta nel meccanismo osservabile**: a k=4 `vᵀHv ≤ 0` quasi sempre → CG
  restituisce il gradiente esatto come direzione → steepest descent su P16 mal
  condizionato → nessun progresso, tante iterazioni. Provato: `negcurv_j0` in
  814/839 iter, `alpha` 5e-3.
- **Causa radice (precisazione)**: non è l'Hessiano FD a essere intrinsecamente
  impreciso (la diagonale FD sarebbe accuratissima a ogni k). È che TN usa
  `hv_fd` **matrix-free con passo non normalizzato per `‖v‖`**: con `‖g‖ ~ 1e4`,
  a k=4 la perturbazione `h·‖v‖ ~ 3` esce dal regime lineare. Il "k troppo grande"
  agisce *tramite* `hv_fd`, non tramite una diagonale sbagliata.
- **Correzione canonica**: perturbare lungo `v/‖v‖` (cioè `h_eff = h/‖v‖`)
  renderebbe TN robusto al variare di k. La formula delle slide usa invece un `h`
  globale, da cui la fragilità.

## Riferimenti nel codice
- `src/methods/truncated_newton.py` — `_cg_truncated` (test `dHd ≤ 0`, ritorno
  `b = −g` su `negcurv_j0`); selezione operatore `apply_H`.
- `src/hessians/finite_diff.py` — `hv_fd` (passo `h = 10⁻ᵏ·max(‖x‖∞,1)`, **non**
  normalizzato per `‖v‖`); `hess_fd_diag` (usato solo da MN su P16).
