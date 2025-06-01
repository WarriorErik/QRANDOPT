# strip_hrules.py — put this in your book/ folder next to _toc.yml
import re, nbformat, os

# matches any Markdown “horizontal rule” line: 3+ of -, *, or _ (with optional spaces)
HRULE = re.compile(r'^(?:\s*[-*_]\s*){3,}$')

folder = "content"
for fname in os.listdir(folder):
    if not fname.endswith(".ipynb"):
        continue
    path = os.path.join(folder, fname)
    nb = nbformat.read(path, as_version=4)
    changed = False

    for cell in nb.cells:
        if cell.cell_type != "markdown":
            continue
        lines = cell.source.splitlines()
        # keep only those lines that are NOT a pure horizontal rule
        new_lines = [L for L in lines if not HRULE.match(L)]
        if len(new_lines) != len(lines):
            changed = True
            cell.source = "\n".join(new_lines)

    if changed:
        nbformat.write(nb, path)
        print(f"Cleaned {fname}")
