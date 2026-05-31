"""Generate a LaTeX `figures.tex` including every image in results/plots.

Usage
-----
    python scripts/gen_latex_figures.py
    python scripts/gen_latex_figures.py --src results/plots --out results/plots/figures.tex
    python scripts/gen_latex_figures.py --per-figure 2     # 2 subfigures per figure

On Overleaf: upload the images into a folder called `plots/`, upload the
generated `figures.tex`, then in your main.tex put

    \\usepackage{graphicx}
    \\usepackage{subcaption}   % only needed when --per-figure > 1
    \\graphicspath{{plots/}}
    ...
    \\input{figures.tex}

The figures reference images by base name, so `\\graphicspath{{plots/}}`
makes LaTeX find them inside the Overleaf `plots/` folder.
"""
import argparse
import re
from pathlib import Path

# Map filename tokens to human-readable caption pieces.
TOKENS = {
    "contour": "solution path (contour)",
    "conv": "convergence",
    "MN": "Modified Newton",
    "TN": "Truncated Newton",
    "PTN": "Preconditioned Truncated Newton",
    "P16": "Problem 16",
    "P28": "Problem 28",
    "exact": "exact derivatives",
    "only": "FD Hessian",            # 'only_hess_fd' -> 'only' + 'hess' + 'fd'
    "both": "FD gradient + Hessian",  # 'both_fd' -> 'both' + 'fd'
    "special": "special step",
    "fixed": "fixed step",
    "specific": "specific step",
    "fd": None, "hess": None,         # drop these standalone fragments
}


def natural_key(s):
    """Sort key so k4 < k8 < k12 and n2 < n1000 < n100000."""
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", s)]


def humanize(stem):
    """Turn 'contour_MN_P16_only_hess_fd_fixed_k8' into a readable caption."""
    pieces = []
    for tok in stem.split("_"):
        m = re.fullmatch(r"k(\d+)", tok)              # step exponent k8 -> $k=8$
        n = re.fullmatch(r"n(\d+)", tok)              # dimension n1000 -> $n=1000$
        kg = re.fullmatch(r"kg(\d+)h(\d+)", tok)      # mixed kg8h4 -> $k_g=8,\,k_H=4$
        if m:
            pieces.append(rf"$k={m.group(1)}$")
        elif n:
            pieces.append(rf"$n={n.group(1)}$")
        elif kg:
            pieces.append(rf"$k_g={kg.group(1)},\,k_H={kg.group(2)}$")
        elif tok in TOKENS:
            if TOKENS[tok] is not None:
                pieces.append(TOKENS[tok])
        else:
            pieces.append(tok.replace("_", " "))
    # Deduplicate consecutive repeats while preserving order.
    out = []
    for p in pieces:
        if not out or out[-1] != p:
            out.append(p)
    return ", ".join(out)


def figure_block(name, caption):
    """A standalone figure for one image."""
    return (
        "\\begin{figure}[htbp]\n"
        "    \\centering\n"
        f"    \\includegraphics[width=0.85\\textwidth]{{{name}}}\n"
        f"    \\caption{{{caption}}}\n"
        f"    \\label{{fig:{name}}}\n"
        "\\end{figure}\n"
    )


def grouped_block(items):
    """A figure with several side-by-side subfigures (items = [(name, caption)])."""
    width = 0.99 / len(items) - 0.02
    lines = ["\\begin{figure}[htbp]", "    \\centering"]
    for name, caption in items:
        lines += [
            f"    \\begin{{subfigure}}{{{width:.3f}\\textwidth}}",
            "        \\centering",
            f"        \\includegraphics[width=\\linewidth]{{{name}}}",
            f"        \\caption{{{caption}}}",
            f"        \\label{{fig:{name}}}",
            "    \\end{subfigure}\\hfill",
        ]
    lines += ["\\end{figure}", ""]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", default="results/plots",
                    help="folder to scan for images (default: results/plots)")
    ap.add_argument("--out", default="results/plots/figures.tex",
                    help="output .tex file (default: results/plots/figures.tex)")
    ap.add_argument("--ext", nargs="+", default=[".png", ".pdf"],
                    help="image extensions to include")
    ap.add_argument("--per-figure", type=int, default=1,
                    help="images per figure (>1 uses subfigure)")
    ap.add_argument("--include", default=None,
                    help="only include images whose filename contains this "
                         "substring (e.g. 'conv' or 'contour')")
    ap.add_argument("--clearpage-every", type=int, default=12,
                    help="insert \\clearpage after this many figures to avoid "
                         "LaTeX's 'Too many unprocessed floats' (0 disables)")
    args = ap.parse_args()

    src = Path(args.src)
    exts = {e.lower() if e.startswith(".") else "." + e.lower() for e in args.ext}
    images = sorted((p for p in src.iterdir()
                     if p.is_file() and p.suffix.lower() in exts
                     and (args.include is None or args.include in p.name)),
                    key=lambda p: natural_key(p.name))
    if not images:
        hint = f" matching '{args.include}'" if args.include else ""
        raise SystemExit(f"No images with {sorted(exts)}{hint} found in {src}")

    rows = [(p.name, humanize(p.stem)) for p in images]

    chunks = ["% Auto-generated by scripts/gen_latex_figures.py - do not edit by hand.",
              f"% {len(rows)} figures from {src.as_posix()}", ""]
    if args.per_figure <= 1:
        blocks = [figure_block(name, cap) for name, cap in rows]
    else:
        blocks = [grouped_block(rows[i:i + args.per_figure])
                  for i in range(0, len(rows), args.per_figure)]
    # Periodic \clearpage flushes pending floats so LaTeX never exceeds its
    # ~18 unprocessed-float limit (which aborts the build with no PDF).
    every = args.clearpage_every
    for i, block in enumerate(blocks):
        chunks.append(block)
        if every and (i + 1) % every == 0 and (i + 1) < len(blocks):
            chunks.append("\\clearpage")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(chunks), encoding="utf-8")
    print(f"Wrote {len(rows)} figures to {out}")


if __name__ == "__main__":
    main()
