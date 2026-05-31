"""Insert a \\label{tab:...} after every \\caption in all_tables.tex.

The label slug is derived from the `% ===== <key> =====` comment that
precedes each table, e.g.

    % ===== P16 | n=2 | MN | only_hess_fd | h=fixed | k=4 =====
        -> \\label{tab:mn_p16_n2_only_hess_fd_fixed_k4}

Idempotent: tables that already carry a \\label are left untouched.
A .bak copy is written before overwriting (the file is not tracked by git).
"""
import re
from pathlib import Path

PATH = Path("results/tables/all_tables.tex")


def slug_from_tag(tag_inner):
    """Build the label slug from a comment key like 'P16 | n=2 | MN | ...'."""
    parts = [p.strip() for p in tag_inner.split("|")]
    prob = parts[0].lower()                 # P16 -> p16
    n = parts[1].split("=", 1)[1].strip()   # n=2 -> 2
    method = parts[2].lower()               # MN -> mn
    deriv = parts[3].strip()                # only_hess_fd / both_fd / ...
    out = [method, prob, f"n{n}", deriv]
    if deriv == "both_fd_special":
        htype = parts[4].split("=", 1)[1].strip()
        out += [htype, "kg8h4"]
    elif deriv != "exact":
        htype = parts[4].split("=", 1)[1].strip()
        kv = parts[5].split("=", 1)[1].strip()
        out += [htype, f"k{kv}"]
    return "_".join(out)


def main():
    text = PATH.read_text(encoding="utf-8")

    # Position -> comment key, for the nearest preceding header of each table.
    comment_re = re.compile(r"^%\s*=+\s*(.*?)\s*=+\s*$", re.M)
    comments = [(m.start(), m.group(1)) for m in comment_re.finditer(text)]

    def key_before(pos):
        key = None
        for start, k in comments:
            if start < pos:
                key = k
            else:
                break
        return key

    seen, added = {}, 0

    def uniquify(slug):
        if slug not in seen:
            seen[slug] = 0
            return slug
        seen[slug] += 1
        return f"{slug}_{seen[slug]}"

    table_re = re.compile(r"\\begin\{table\}.*?\\end\{table\}", re.S)
    caption_re = re.compile(r"\\caption\{.*\}")

    def repl(m):
        nonlocal added
        block = m.group(0)
        if "\\label{" in block:                       # already labelled
            return block
        key = key_before(m.start())
        if key is None:
            return block
        slug = uniquify(slug_from_tag(key))
        cap = caption_re.search(block)
        if not cap:
            return block
        new_block = (block[:cap.end()]
                     + "\n\\label{tab:" + slug + "}"
                     + block[cap.end():])
        added += 1
        return new_block

    new_text = table_re.sub(repl, text)

    if added:
        PATH.with_suffix(".tex.bak").write_text(text, encoding="utf-8")
        PATH.write_text(new_text, encoding="utf-8")
    print(f"Added {added} labels (backup: {PATH.with_suffix('.tex.bak')})")


if __name__ == "__main__":
    main()
