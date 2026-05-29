# Il "falso fallimento" delle differenze finite: floor di roundoff del gradiente (k=8)

Analisi del fenomeno osservato in **Table 24: TN, P16, n=2, both_fd, h=fixed, k=8**,
dove 5 starting point su 6 risultano `no / max_iter` con `||∇f||` bloccato appena
sopra la tolleranza.

## Premessa: `grad_fd` è a differenze centrate

`src/gradients/finite_diff.py` calcola il gradiente con **differenze centrate**
(O(h²)), con passo `h = 10^{-k}`:

```
df/dx_i ≈ [f(x + h e_i) − f(x − h e_i)] / (2 h)
```

Con `k = 8` → `h = 10⁻⁸`. (Nota: il `CLAUDE.md` dice "forward FD" ma il codice
è centrato; la formula d'errore qui sotto è quella centrata.)

## Cosa dice la tabella (e perché è un "falso fallimento")

| SP | ‖∇f‖ | iter/max | success | flag | conv_rate | time (s) |
|----|--------|----------|---------|------------|-----------|----------|
| 0  | 6.66e-08 | 5000/5000 | no  | max_iter | —    | 1.00 |
| 1  | 5.55e-08 | 5000/5000 | no  | max_iter | —    | 0.98 |
| 2  | 5.55e-08 | 5000/5000 | no  | max_iter | —    | 0.98 |
| 3  | 0.00e+00 | 26/5000   | yes | grad_abs | 0.85 | 0.01 |
| 4  | 7.02e-08 | 5000/5000 | no  | max_iter | —    | 0.98 |
| 5  | 8.88e-08 | 5000/5000 | no  | max_iter | —    | 0.93 |
| Avg| 0.00e+00 | 26.0/5000 | 1/6 | —        | 0.85 | 0.01 |

5 run su 6 si fermano con `||∇f||` tra **5.5e-8 e 8.9e-8**, cioè appena **sopra**
la tolleranza `STOP_TOL = 1e-8`. **Non** è un fallimento dell'ottimizzazione: gli
iterati **sono già in un minimo** (gradiente ~10⁻⁷), ma sprecano 5000 iterazioni
girando a vuoto. Il criterio `GradNormAbsolute(1e-8)` è semplicemente
**irraggiungibile** con il gradiente FD a k=8.

## Perché il gradiente si blocca a ~5e-8: il roundoff delle FD

La differenza centrata ha due fonti d'errore:

| Fonte | Ordine | Con h=10⁻⁸ |
|---|---|---|
| Troncamento | ~ (h²/6)·\|f‴\| | ~10⁻¹⁷ (trascurabile) |
| **Roundoff** (cancellazione) | ~ ε·\|f\| / (2h) | **~1.7e-8** |

con ε ≈ 2.2e-16 (epsilon macchina). Vicino al minimo di P16 n=2 si ha |F| ≈ 1.5,
quindi:

```
errore ≈ (2.2e-16 · 1.5) / (2 · 10⁻⁸) ≈ 1.7e-8  per componente
       ⇒ ‖·‖₂ ≈ 2–5e-8
```

**Combacia con il plateau osservato.** Con h=10⁻⁸ il passo è **troppo piccolo**:
la sottrazione `f(x+h) − f(x−h)` tra due numeri quasi identici perde cifre per
cancellazione, e il gradiente non può scendere sotto ~10⁻⁸. Per le differenze
centrate il passo ottimale è h ≈ ε^{1/3} ≈ 6e-6 (cioè **k≈5**); k=8 è già nel
regime dominato dal roundoff.

### Sensibilità ai tre valori di k dell'assignment

| k | h | floor del gradiente | esito vs tol=1e-8 |
|---|------|---------------------|-------------------|
| 4  | 10⁻⁴  | ~10⁻⁹ (troncamento) | **sotto** → converge |
| 8  | 10⁻⁸  | ~2e-8 (roundoff)    | **appena sopra** → fallisce di poco (questo caso) |
| 12 | 10⁻¹² | ~10⁻⁴ (roundoff)    | enorme → fallisce clamorosamente |

k=8 è esattamente il **caso di confine**: spiega perché la tabella mostra valori
appena sopra la soglia.

## Perché SP 3 "riesce" con ‖∇f‖ = 0.00e+00 esatto

Non è 5e-8, è **esattamente zero**. In un minimo la funzione è localmente
**simmetrica** (f'≈0, domina il termine pari ½f″h²), quindi `f(x+h)` e `f(x−h)`
restituiscono lo **stesso identico float** e la differenza è bit-a-bit 0. SP 3 è
atterrato abbastanza vicino al minimo da innescare questa cancellazione esatta →
gradiente FD esattamente 0 → criterio soddisfatto a iter 26. Gli altri 5 si
fermano leggermente "di fianco", dove il rumore antisimmetrico (~5e-8) non si
annulla mai. È una lotteria di pochi ULP.

## Il grafico

P16 con n=2 è **separabile**:

```
F(x₁,x₂) = (1 − cos x₁ + 2 sin x₁) + (2 − 2 cos x₂ − sin x₂)
```

Essendo periodica (sin/cos), il paesaggio è una **tassellatura** di bacini:
giallo = massimi, viola scuro = minimi locali → **non convessa, multimodale**.

- Il **pallino rosso grande** in (1,1) è `x_bar`; gli altri 5 start sono random
  in [0,2]².
- Le **traiettorie corte** vicino al centro: i run fanno pochi passi e si
  depositano nel minimo locale più vicino, poi restano lì 5000 iter a oscillare
  nel rumore FD.
- La **lunga retta diagonale fino a (6,−6)**: un singolo passo enorme. Tipico di
  TN quando il CG interno incontra **curvatura negativa** e ripiega su steepest
  descent (o quando la line-search accetta α=1 con direzione grande): in un
  paesaggio periodico il salto attraversa più periodi e finisce in un bacino
  lontano (equivalente per simmetria).

## In sintesi (per il report)

Questi `max_iter` **non sono fallimenti dell'algoritmo** ma un **artefatto della
precisione FD**: a k=8 il floor di roundoff del gradiente centrato (~2e-8) sta
sopra la soglia 1e-8, quindi il criterio del primo ordine è irraggiungibile anche
se gli iterati hanno di fatto raggiunto un minimo. È la dimostrazione attesa della
sensibilità al passo h e del motivo per cui servono derivate esatte (o un k più
moderato) per soddisfare tolleranze strette.
