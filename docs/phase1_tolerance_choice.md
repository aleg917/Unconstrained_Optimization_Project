# Scelta del criterio di stop in Phase 1: passaggio a tol=1e-8

Documento che motiva il passaggio del criterio di stop usato nella Phase 1
(grid search dei parametri MN/TN) da `GradNormAbsolute(tol=1e-4)` a
`GradNormAbsolute(tol=1e-8)`.

## 1. Convention del corso (PDF + lab MATLAB + slide)

Il criterio scelto è la combinazione canonica della professoressa,
documentata in tre fonti convergenti:

- **PDF lecture notes** (`NO4LSP_DellaSanta_2526.pdf`, esercizio 7.4 §7):
  raccomanda esplicitamente `tolgrad = 10⁻⁸` per il tuning di Rosenbrock
  100D.

- **Lab MATLAB del Lab 13** (script su Rosenbrock 100D):
  ```matlab
  while k < kmax && normFk >= Ftol
  ```
  dove `normFk = norm(F(xk))`, `Ftol = 1e-8`. Per ottimizzazione
  `F = ∇f`, quindi il criterio è $\|\nabla f(x_k)\|_2 \leq \texttt{Ftol}$
  — **norma euclidea assoluta del gradiente**.

- **Slide teoriche** (lezioni frontali): valori tipici per la tolleranza
  in doppia precisione:
  - $10^{-4}$ — rough precision
  - $10^{-8}$ — **good solution**
  - $10^{-12}$ — very good solution ("very demanding and quite often
    unnecessary")

Per il tuning Phase 1 usiamo `Ftol = 10⁻⁸` (banda "good"): è la convention
standard del corso, allineata con il valore esatto usato dalla docente
nello script del lab.

## 2. Perché non tol=1e-4

Il valore inizialmente usato (`tol=1e-4`) era pragmatico ma — come emerge
dal sensitivity check qui sotto — **troppo permissivo**: dichiarava successo
troppo presto, nascondendo la differenza fra configurazioni robuste e
configurazioni fragili. Il valore "good solution" (1e-8) della slide è
quello giusto per il tuning.

### 2.1 Caveat sulla scala con n (discusso e accettato)

La norma euclidea $\|g\|_2 \approx c \cdot \sqrt{n}$ per componenti di
ordine $c$. Quindi richiedere $\|g\|_2 \leq 10^{-8}$ su $n=100\,000$
corrisponde a $\max_i |g_i| \lesssim 3 \cdot 10^{-11}$ — più stretto che
chiedere lo stesso valore su $n=2$.

Abbiamo valutato alternative scale-invariant (norma infinito; step
relativo `||Δx||/||x||` come da slide 2 teoria), ma la **convention del
corso usa la norma 2 assoluta**, quindi manteniamo questa scelta. Le
conseguenze osservate (stagnazione su n grande per config fragili) sono
**informazione utile**, non artefatto: la fragilità è una proprietà reale
del metodo a fronte di un criterio richiesto in modo uniforme.

## 3. Sensitivity check (prova empirica)

Per misurare l'impatto della scelta della tolleranza sul ranking dei
parametri, abbiamo eseguito una run mirata: 1 run per ciascuna delle 4
config MN e 4 config TN, sul punto `x_bar` di tre celle rappresentative
`(P16, n=1000)`, `(P16, n=10000)`, `(P28, n=1000)`. Per ciascuna run, abbiamo
estratto dalla `history` il numero di iterazioni necessario a scendere sotto
le soglie `10⁻⁴`, `10⁻⁸`, `10⁻¹²`.

### 3.1 Modified Newton

| config         | iter ≤ 10⁻⁴ | iter ≤ 10⁻⁸ | iter ≤ 10⁻¹² | final ‖g‖ (max_iter=1000)  |
|----------------|-------------|-------------|--------------|----------------------------|
| β=1e-6, ρ=0.5  | 6           | 6           | **mai**      | 9.6·10⁻¹¹ (stagna)         |
| β=1e-6, ρ=0.8  | 6           | 6           | **mai**      | 9.6·10⁻¹¹ (stagna)         |
| β=1e-3, ρ=0.5  | 6           | 7           | 7            | 2.8·10⁻¹³ ✓                |
| β=1e-3, ρ=0.8  | 6           | 7           | 7            | 2.8·10⁻¹³ ✓                |

**Lettura**: a `tol=10⁻⁴` tutte le 4 config arrivano in 6 iter — appaiono
equivalenti. Ma `β=10⁻⁶` **stagna** intorno a `||g||≈10⁻¹⁰` e non scende mai
a `10⁻¹²`, mentre `β=10⁻³` raggiunge praticamente la precisione macchina.
Il ranking è qualitativamente lo stesso, ma a `tol=10⁻⁴` il margine reale è
**completamente invisibile**.

### 3.2 Truncated Newton — qui il ranking si ribalta

| config              | iter@10⁻⁴ | iter@10⁻⁸ (P16,1000)  | iter@10⁻⁸ (P16,10000) | iter@10⁻⁸ (P28,1000) |
|---------------------|-----------|------------------------|------------------------|----------------------|
| superlinear, ρ=0.5  | 20        | **mai**                | **mai** (final 5·10⁻⁸) | **mai**              |
| superlinear, ρ=0.8  | 20        | 139                    | 47                     | **mai**              |
| quadratic, ρ=0.5    | 18        | **mai** (final 1.9·10⁻⁶)| 22                    | 38                   |
| quadratic, ρ=0.8    | 18        | 19                     | 22                     | 38                   |

**Lettura**:
- A **`tol=10⁻⁴`**, `quadratic ρ=0.5` sembrava il best (18 iter, qualche
  iter in meno di tutti gli altri).
- A **`tol=10⁻⁸`**:
  - Su `(P16, n=1000)`: `quadratic ρ=0.5` **stagna** a ~`10⁻⁶` e *non scende
    mai* a `10⁻⁸`. Solo `quadratic ρ=0.8` ce la fa.
  - Su `(P28, n=1000)`: `superlinear` stagna ovunque; solo `quadratic`
    converge (in entrambe le ρ).

**Conseguenza**: il best TN scelto dalla Phase 1 a `tol=10⁻⁴` era
`quadratic, ρ=0.5`. Una misurazione a `tol=10⁻⁸` rivela che `ρ=0.5` non è
realmente robusto sulla zona finale della convergenza, e che `ρ=0.8` è
qualitativamente preferibile. **Il ranking *cambia* cambiando tolleranza**.

## 4. Conseguenza: rerun di Phase 1 con `tol=10⁻⁸`

In base all'evidenza del sensitivity check:

- Per **MN** la conclusione (`β=10⁻³` migliore di `β=10⁻⁶`) resta valida,
  ma a `tol=10⁻⁴` il margine quantitativo era sottostimato.
- Per **TN** la conclusione cambia: `ρ=0.8` è preferibile a `ρ=0.5`.

Per ottenere un ranking robusto è opportuno **ripetere la Phase 1** con un
criterio più stretto: `GradNormAbsolute(tol=10⁻⁸)` (banda "good").

### Dettagli operativi del rerun

- **Notebook**: `fine_tuning.ipynb`, celle MN grid e TN grid — il criterio è
  `GradNormAbsolute(tol=1e-8)`.
- **Griglia ampliata nel rerun**: MN `β∈{1e-6,1e-3,1e-2}×ρ∈{0.5,0.75,0.9}`
  (9 combo), TN `forcing∈{superlinear,quadratic}×ρ∈{0.5,0.75,0.9}` (6 combo);
  `max_iter=5000`.
- **Time limit**: ridotto a **20 s** per run nel rerun (era 60 s). Le run che
  già fallivano per `time_limit` (P28 a n ≥ 10000, P16 a n=100000) falliscono
  comunque — non è un nuovo problema, è una caratteristica reale della
  griglia. Rispetto a `tol=10⁻⁴`, alcune run che prima finivano in pochi
  secondi ora arrivano un po' più in là (Newton ha convergenza quadratica:
  ~2 iter in più per scendere a `10⁻⁸`).
- **Output**: i nuovi CSV sovrascriveranno `results/fine_tuning_modified_newton.csv`
  e `..._truncated_newton.csv`. Tutte le tabelle e analisi a valle leggono
  questi CSV → si aggiorneranno automaticamente.
- **Tempo stimato**: 1-4 ore (come la run precedente; la maggior parte del
  tempo è già "consumata" dalle run che falliscono per time_limit, che non
  cambia tra `10⁻⁴` e `10⁻⁸`).

## 5. Cosa aggiornare dopo il rerun

1. **`docs/tuning_analysis.md` §6**: ri-estrarre il ranking dai CSV
   aggiornati. Probabili aggiornamenti:
   - MN: best **cambiato** → `β=10⁻², ρ=0.5` (la griglia è stata ampliata
     aggiungendo `β=10⁻²` e `ρ∈{0.5,0.75,0.9}`; `β=10⁻²` vince perché fa
     convergere P16 a n=1000, dove `β≤10⁻³` falliva).
   - TN: best Overall/P16 spostato a `ρ=0.9` (forcing `quadratic`). Confermata
     la direzione del sensitivity check (ρ più alto è più robusto), ma il punto
     ottimo è 0.9, non 0.8.

2. **`CLAUDE.md`**: aggiornare il blocco "Full-mode Results" con i nuovi
   best.

3. **Questa pagina**: integrare una sezione "§5 Risultati post-rerun" con
   il confronto numerico tra Phase 1 originale (tol=10⁻⁴) e rerun
   (tol=10⁻⁸): success_rate per config, ranking finale, e una nota sui
   casi in cui `ρ=0.8` ha effettivamente ribaltato il best (validazione del
   sensitivity check).

## 6. Risultati post-rerun

Rerun eseguito con `GradNormAbsolute(tol=1e-8)`, `time_limit=20s`,
`max_iter=5000`, griglia ampliata (MN 3×3 = 9 combo, TN 2×3 = 6 combo).
CSV in `results/fine_tuning_{modified,truncated}_newton.csv`.
MN: 378 run, 54.0% success. TN: 288 run, 55.9% success.

| Metodo | Best Overall (tol=10⁻⁸)  | Best P16          | Best P28          | Δ vs tol=10⁻⁴ (griglia vecchia)        |
|--------|--------------------------|-------------------|-------------------|----------------------------------------|
| MN     | β=10⁻², ρ=0.5 (61.9%)    | β=10⁻², ρ=0.5     | β=10⁻², ρ=0.9     | era β=10⁻³, ρ=0.8 → ora β=10⁻², ρ=0.5  |
| TN     | quadratic, ρ=0.9 (62.5%) | quadratic, ρ=0.9  | quadratic, ρ=0.5  | era quadratic, ρ=0.5 → ora ρ=0.9       |

### Confronto sintetico

Il ranking cambia in modo sostanziale rispetto a `tol=10⁻⁴`:
- **MN**: l'aggiunta di `β=10⁻²` alla griglia ribalta la scelta. Con `tol=10⁻⁸`
  il discriminante è la cella **(P16, n=1000)**: `β=10⁻²` la fa convergere
  (success 1.00) mentre `β≤10⁻³` resta a 0.17 — da qui il salto di success
  aggregato (61.9% vs 50.0%). `ρ` resta quasi inerte (le tre ρ danno lo stesso
  success aggregato; differenza di ~0.2 iter).
- **TN**: confermata la direzione del sensitivity check (ρ più alto = più
  robusto sulla coda di convergenza), ma il punto migliore è **ρ=0.9**, non 0.8:
  con forcing `quadratic` porta P16 a n=1000 e n=10000 a success 1.00. Su P28
  resta meglio `ρ=0.5` (P28 è ρ-inerte: fallisce comunque per `time_limit` da
  n=10000 in su).
