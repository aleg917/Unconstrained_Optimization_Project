"""Driver per il punto 3.1 dell'assignment (Hessiana via FD del gradiente esatto).

Genera la griglia di esperimenti richiesta dalla Tabella 2 dell'assignment:

    metodi    : Modified Newton (FD-Hessian assemblata)
                Truncated Newton (Hv matrix-free)
    problemi  : P16 (Banded Trigonometric, Hessiana diagonale)
                P28 (Variably Dimensioned, Hessiana densa)
    n         : 2, 10^3, 10^4, 10^5
    starting  : x_bar + 5 punti casuali in [x_bar-1, x_bar+1]^n
                (seed = min student ID, definito sotto)
    k         : 4, 8, 12
    scaled    : False, True
    stencil   : 'centered' (default); --include-forward aggiunge 'forward'

Per ogni cella: tempo, iterazioni, ||grad|| finale, success, tasso di
convergenza sperimentale, motivo di terminazione, eventuali OOM/eccezioni.

Output:
    results/fd_hessian_raw.csv         (una riga per esperimento)
    results/fd_hessian_table_*.md      (un Table-2 markdown per
                                        (method, problem, n))

Flag CLI:
    --quick                  preset rapido per validare il driver
                             (n=[2], 1 starting point, k=[8], scaled=[False])
    --nmax N                 limita n a valori <= N
    --starts S               numero di starting points (default 6)
    --methods modnewt,truncn (sottoinsieme di metodi)
    --problems P16,P28       (sottoinsieme di problemi)
    --include-forward        aggiunge stencil forward (raddoppia i run)
    --max-iter M             default 500
    --tol T                  default 1e-6
    --skip-oom-threshold-mb X  salta cella se H assemblata > X MB
                               (default 4096 = 4 GB)
"""
import argparse
import csv
import os
import sys
import time
import traceback
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.functions import f16, f28, x_bar_16, x_bar_28
from src.gradients import grad_f16, grad_f28
from src.hessians import hess_fd, hv_fd
from src.methods.modified_newton import modified_newton
from src.methods.truncated_newton import truncated_newton
from src.starting_points import generate_starting_points
from src.stopping_criteria import GradNormAbsolute


# ----------------------------------------------------------------------------
# Configurazione di base
# ----------------------------------------------------------------------------
TEAM_SEED = 1234  # placeholder: sostituire con min(student IDs)

PROBLEMS = {
    'P16': dict(f=f16, grad=grad_f16, x_bar=x_bar_16,
                desc='Banded Trigonometric (H diagonale, 1 colore CPR)'),
    'P28': dict(f=f28, grad=grad_f28, x_bar=x_bar_28,
                desc='Variably Dimensioned (H densa, no CPR)'),
}

METHODS = ('modnewt', 'truncn')
N_VALUES_FULL = (2, 1000, 10000, 100000)
K_VALUES = (4, 8, 12)
SCALED_VALUES = (False, True)


# ----------------------------------------------------------------------------
# Tasso di convergenza sperimentale
# ----------------------------------------------------------------------------
def experimental_rate(g_norms):
    """Stima il tasso di convergenza p da una sequenza di ||g_k||.

    Modello e_k = ||g_k|| ~ c * (e_{k-1})^p  =>
        p ~ log(e_{k+1}/e_k) / log(e_k/e_{k-1})

    Calcoliamo p su finestre di 3 iterazioni consecutive nella coda della
    sequenza (dove dominano i termini di convergenza, non transitori).
    Restituiamo la mediana, robusta a outlier. NaN se non calcolabile.
    """
    e = np.asarray(g_norms, dtype=float)
    if e.size < 4:
        return float('nan')
    # taglia tutto cio' che e' 0 o vicino a precisione di macchina
    e = e[e > 1e-14]
    if e.size < 4:
        return float('nan')
    # ratios consecutivi
    log_e = np.log(e)
    # p_k = (log e_{k+1} - log e_k) / (log e_k - log e_{k-1})
    num = np.diff(log_e)[1:]    # log(e_{k+1}/e_k)
    den = np.diff(log_e)[:-1]   # log(e_k/e_{k-1})
    mask = (np.abs(den) > 1e-12) & np.isfinite(num) & np.isfinite(den)
    if not mask.any():
        return float('nan')
    p_vals = num[mask] / den[mask]
    # prendi le ultime stime (coda della sequenza)
    p_tail = p_vals[-min(5, p_vals.size):]
    return float(np.median(p_tail))


# ----------------------------------------------------------------------------
# Predicato di OOM: skippa se l'assemblaggio della Hessiana eccede il budget
# ----------------------------------------------------------------------------
def expected_hessian_mb(n):
    """Memoria stimata per una Hessiana densa n x n float64 (MB)."""
    return n * n * 8 / (1024 ** 2)


# ----------------------------------------------------------------------------
# Singolo esperimento: una cella della griglia
# ----------------------------------------------------------------------------
def run_one(method, prob_id, n, start_idx, x0, k, scaled, stencil,
            tol, max_iter, oom_threshold_mb):
    pinfo = PROBLEMS[prob_id]
    f, grad_f = pinfo['f'], pinfo['grad']
    stop = GradNormAbsolute(tol=tol)
    base = dict(method=method, problem=prob_id, n=n, start_idx=start_idx,
                k=k, scaled=scaled, stencil=stencil, max_iter=max_iter)

    # --- skip preventivi -----------------------------------------------------
    if method == 'modnewt':
        h_mb = expected_hessian_mb(n)
        if h_mb > oom_threshold_mb:
            return {**base, 'iters': 0, 'cg_iters': 0,
                    'success': False, 'grad_norm': float('nan'),
                    'rate': float('nan'), 'time_s': 0.0,
                    'stop_reason': 'skip_oom',
                    'notes': f'expected H size {h_mb:.0f} MB > {oom_threshold_mb} MB'}

    # --- esegui --------------------------------------------------------------
    t0 = time.perf_counter()
    history = None
    try:
        if method == 'modnewt':
            hess_cb = lambda x: hess_fd(grad_f, x, k=k, scaled=scaled, stencil=stencil)
            res = modified_newton(f, grad_f, hess_cb, x0,
                                  stopping=stop, max_iter=max_iter,
                                  return_history=True)
            cg_iters = 0
        else:  # truncn matrix-free
            hv_cb = lambda x, v: hv_fd(grad_f, x, v, k=k, scaled=scaled, stencil=stencil)
            res = truncated_newton(f, grad_f, x0=x0, stopping=stop,
                                   max_iter=max_iter, hv_func=hv_cb,
                                   return_history=True)
            cg_iters = res.get('cg_iters_total', 0)
        t = time.perf_counter() - t0

        history = res.get('history', None)
        if history is not None:
            g_norms = [h['grad_norm'] for h in history]
            rate = experimental_rate(g_norms)
        else:
            rate = float('nan')

        return {**base,
                'iters': res['n_iter'],
                'cg_iters': cg_iters,
                'success': res['success'],
                'grad_norm': res['grad_norm'],
                'rate': rate,
                'time_s': t,
                'stop_reason': res.get('stop_reason', '?'),
                'notes': ''}

    except MemoryError:
        return {**base, 'iters': 0, 'cg_iters': 0,
                'success': False, 'grad_norm': float('nan'),
                'rate': float('nan'),
                'time_s': time.perf_counter() - t0,
                'stop_reason': 'memory_error',
                'notes': 'MemoryError during run'}
    except Exception as e:
        return {**base, 'iters': 0, 'cg_iters': 0,
                'success': False, 'grad_norm': float('nan'),
                'rate': float('nan'),
                'time_s': time.perf_counter() - t0,
                'stop_reason': 'exception',
                'notes': f'{type(e).__name__}: {e!s:.80s}'}


# ----------------------------------------------------------------------------
# Main: itera la griglia, scrive CSV e markdown
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--quick', action='store_true',
                    help='Preset rapido per validare il driver')
    ap.add_argument('--nmax', type=int, default=None,
                    help='Limita n a valori <= NMAX')
    ap.add_argument('--n-values', type=str, default=None,
                    help='Lista esplicita di n separati da virgola '
                         '(es. "2,1000"). Sovrascrive --nmax.')
    ap.add_argument('--starts', type=int, default=6,
                    help='Numero di starting points (1 x_bar + (S-1) random)')
    ap.add_argument('--methods', type=str, default='modnewt,truncn',
                    help='Sottoinsieme di metodi (csv)')
    ap.add_argument('--problems', type=str, default='P16,P28',
                    help='Sottoinsieme di problemi (csv)')
    ap.add_argument('--include-forward', action='store_true',
                    help='Aggiunge stencil forward (default solo centered)')
    ap.add_argument('--max-iter', type=int, default=500)
    ap.add_argument('--tol', type=float, default=1e-6)
    ap.add_argument('--skip-oom-threshold-mb', type=float, default=4096.0)
    ap.add_argument('--outdir', type=str, default=str(ROOT / 'results'))
    args = ap.parse_args()

    if args.quick:
        n_values = (2,)
        starts = 2  # x_bar + 1 random
        k_values = (8,)
        scaled_vals = (False,)
        stencils = ('centered',)
        max_iter = 100
    else:
        if args.n_values:
            n_values = tuple(int(x) for x in args.n_values.split(',') if x.strip())
        else:
            n_values = tuple(n for n in N_VALUES_FULL
                             if args.nmax is None or n <= args.nmax)
        starts = args.starts
        k_values = K_VALUES
        scaled_vals = SCALED_VALUES
        stencils = ('centered',) + (('forward',) if args.include_forward else ())
        max_iter = args.max_iter

    methods_sel = tuple(m.strip() for m in args.methods.split(',') if m.strip())
    problems_sel = tuple(p.strip() for p in args.problems.split(',') if p.strip())

    Path(args.outdir).mkdir(parents=True, exist_ok=True)
    csv_path = Path(args.outdir) / 'fd_hessian_raw.csv'

    # Prepara starting points una volta per (problem, n), seedando in modo
    # riproducibile e indipendente dall'ordine di iterazione
    starts_cache = {}
    for prob_id in problems_sel:
        x_bar_fn = PROBLEMS[prob_id]['x_bar']
        for n in n_values:
            rng = np.random.default_rng(TEAM_SEED)
            pts = generate_starting_points(x_bar_fn(n), num_random=starts - 1, rng=rng)
            starts_cache[(prob_id, n)] = pts

    rows = []
    n_cells = (len(methods_sel) * len(problems_sel) * len(n_values) *
               starts * len(k_values) * len(scaled_vals) * len(stencils))
    cell_idx = 0
    t_total0 = time.perf_counter()

    for method in methods_sel:
        for prob_id in problems_sel:
            for n in n_values:
                for start_idx, x0 in enumerate(starts_cache[(prob_id, n)]):
                    for stencil in stencils:
                        for scaled in scaled_vals:
                            for k in k_values:
                                cell_idx += 1
                                row = run_one(method, prob_id, n, start_idx, x0,
                                              k, scaled, stencil,
                                              tol=args.tol, max_iter=max_iter,
                                              oom_threshold_mb=args.skip_oom_threshold_mb)
                                rows.append(row)
                                elapsed = time.perf_counter() - t_total0
                                tag = 'OK ' if row['success'] else (
                                    'SKIP' if row['stop_reason'] == 'skip_oom' else 'X  ')
                                print(f"[{cell_idx:>4d}/{n_cells}] {tag} "
                                      f"{method:7s} {prob_id} n={n:<6d} "
                                      f"s={start_idx} k={k:>2d} sc={int(scaled)} "
                                      f"st={stencil[:3]} | "
                                      f"it={row['iters']:>3d} "
                                      f"||g||={row['grad_norm']:.2e} "
                                      f"t={row['time_s']:.2f}s "
                                      f"[{elapsed:.0f}s tot]")

    # ----- scrivi CSV ------------------------------------------------------
    fields = ['method', 'problem', 'n', 'start_idx', 'k', 'scaled', 'stencil',
              'iters', 'max_iter', 'cg_iters', 'success', 'grad_norm', 'rate',
              'time_s', 'stop_reason', 'notes']
    with open(csv_path, 'w', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, '') for k in fields})
    print(f"\nCSV scritto: {csv_path}  ({len(rows)} righe)")

    # ----- markdown table per (method, problem, n) -------------------------
    by_group = defaultdict(list)
    for r in rows:
        by_group[(r['method'], r['problem'], r['n'])].append(r)

    for (method, prob, n), grp in sorted(by_group.items()):
        md_path = Path(args.outdir) / f'fd_hessian_table_{method}_{prob}_n{n}.md'
        write_markdown_table(md_path, method, prob, n, grp)
    print(f"Markdown scritti in: {args.outdir}/")


def write_markdown_table(path, method, prob, n, rows):
    """Tabella stile Table 2 dell'assignment.
    Colonne: start.pt ID | k | scaled | stencil | grad.norm | iters/max | success | rate | time
    + righe aggregate Avg (successes) per ciascun k.
    """
    rows = sorted(rows, key=lambda r: (r['start_idx'], r['stencil'], r['scaled'], r['k']))
    method_label = {'modnewt': 'Modified Newton', 'truncn': 'Truncated Newton (matrix-free)'}[method]
    lines = [f"# {method_label} — {prob}, n = {n}", "",
             f"_{PROBLEMS[prob]['desc']}_", ""]

    header = ('| start | k | scaled | stencil | grad.norm | iters/max | '
              'success | rate | time (s) | note |')
    sep = '|' + '|'.join(['---'] * 10) + '|'
    lines += [header, sep]
    for r in rows:
        line = (f"| {r['start_idx']} | {r['k']} | {int(r['scaled'])} | "
                f"{r['stencil']} | {r['grad_norm']:.2e} | "
                f"{r['iters']}/{r['max_iter']} | "
                f"{'yes' if r['success'] else 'no'} | "
                f"{r['rate']:.2f} | {r['time_s']:.2f} | "
                f"{r['stop_reason']} {r['notes']}|")
        lines.append(line)

    # Aggregati per (k, scaled, stencil), media solo sui successi
    lines += ["", "## Aggregato: media tra successi", "",
              "| k | scaled | stencil | n. successi | avg grad.norm | "
              "avg iters | avg rate | avg time (s) |",
              '|' + '|'.join(['---'] * 8) + '|']
    grouped = defaultdict(list)
    for r in rows:
        grouped[(r['k'], r['scaled'], r['stencil'])].append(r)
    for key in sorted(grouped):
        k, sc, st = key
        grp = grouped[key]
        succ = [r for r in grp if r['success']]
        if not succ:
            lines.append(f"| {k} | {int(sc)} | {st} | 0 / {len(grp)} | - | - | - | - |")
            continue
        avg_g = np.mean([r['grad_norm'] for r in succ])
        avg_it = np.mean([r['iters'] for r in succ])
        rates = [r['rate'] for r in succ if np.isfinite(r['rate'])]
        avg_r = np.mean(rates) if rates else float('nan')
        avg_t = np.mean([r['time_s'] for r in succ])
        lines.append(f"| {k} | {int(sc)} | {st} | {len(succ)} / {len(grp)} | "
                     f"{avg_g:.2e} | {avg_it:.1f} | "
                     f"{(f'{avg_r:.2f}' if np.isfinite(avg_r) else '-')} | "
                     f"{avg_t:.2f} |")

    path.write_text('\n'.join(lines), encoding='utf-8')


if __name__ == '__main__':
    main()
