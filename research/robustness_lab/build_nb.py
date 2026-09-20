#!/usr/bin/env python
"""Build (and optionally execute) robustness_lab.ipynb from the percent-format
sources in nbsrc/.

Why this exists: a 3,000-line notebook is painful to author and impossible to
diff as raw .ipynb JSON. So the notebook is authored as ordinary Python files
using the `# %%` cell convention (the same format jupytext uses), which means
the source stays greppable, diffable, syntax-checkable with py_compile, and
directly runnable as a plain script for fast iteration. This script is the
only thing that knows how to turn that into a notebook.

Usage
-----
  python build_nb.py              # build .ipynb + flat .py
  python build_nb.py --exec       # build, then execute from a clean kernel
  python build_nb.py --exec --mode QUICK
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC_DIR = HERE / "nbsrc"
NB_PATH = HERE / "robustness_lab.ipynb"
FLAT_PATH = HERE / "robustness_lab_flat.py"

CELL_RE = re.compile(r"^#\s*%%(.*)$")


def read_parts() -> str:
    """Concatenate nbsrc/part*.py in lexical order into one percent-format doc."""
    parts = sorted(SRC_DIR.glob("part*.py"))
    if not parts:
        raise SystemExit(f"no part*.py files found in {SRC_DIR}")
    chunks = []
    for p in parts:
        text = p.read_text()
        if not text.startswith("# %%"):
            raise SystemExit(f"{p.name} must begin with a '# %%' cell marker")
        chunks.append(text.rstrip("\n"))
    return "\n\n".join(chunks) + "\n"


def split_cells(text: str) -> list[tuple[str, str]]:
    """Split percent-format text into [(kind, source), ...] where kind is
    'markdown' or 'code'. Raises on content appearing before the first marker,
    which would otherwise be silently dropped."""
    lines = text.split("\n")
    cells: list[tuple[str, list[str]]] = []
    current_kind: str | None = None
    current: list[str] = []

    for i, line in enumerate(lines):
        m = CELL_RE.match(line)
        if m:
            if current_kind is not None:
                cells.append((current_kind, current))
            tag = m.group(1).strip()
            current_kind = "markdown" if tag.startswith("[markdown]") else "code"
            current = []
        else:
            if current_kind is None:
                if line.strip():
                    raise SystemExit(f"line {i+1} has content before the first '# %%' marker: {line!r}")
                continue
            current.append(line)
    if current_kind is not None:
        cells.append((current_kind, current))

    out: list[tuple[str, str]] = []
    for kind, body in cells:
        if kind == "markdown":
            # Markdown cells are authored as comment blocks; strip one leading
            # '# ' (or a bare '#') from each line.
            md = []
            for ln in body:
                if ln.startswith("# "):
                    md.append(ln[2:])
                elif ln.strip() == "#":
                    md.append("")
                elif not ln.strip():
                    md.append("")
                else:
                    md.append(ln.lstrip("#").lstrip() if ln.startswith("#") else ln)
            src = "\n".join(md).strip("\n")
        else:
            src = "\n".join(body).strip("\n")
        if src.strip():
            out.append((kind, src))
    return out


def build_notebook(cells: list[tuple[str, str]]) -> dict:
    import nbformat

    nb = nbformat.v4.new_notebook()
    nb.cells = [
        nbformat.v4.new_markdown_cell(src) if kind == "markdown" else nbformat.v4.new_code_cell(src)
        for kind, src in cells
    ]
    nb.metadata = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": sys.version.split()[0]},
    }
    return nb


def execute(nb, kernel_name: str, timeout: int):
    """Run every cell from a clean kernel. Returns (ok, [failure dicts])."""
    import nbformat
    from nbclient import NotebookClient
    from nbclient.exceptions import CellExecutionError

    client = NotebookClient(
        nb,
        timeout=timeout,
        kernel_name=kernel_name,
        allow_errors=True,           # keep going so one failure reveals the rest
        resources={"metadata": {"path": str(HERE)}},
    )
    t0 = time.time()
    client.execute()
    elapsed = time.time() - t0

    failures = []
    for idx, cell in enumerate(nb.cells):
        if cell.cell_type != "code":
            continue
        for out in cell.get("outputs", []):
            if out.get("output_type") == "error":
                failures.append({
                    "cell": idx,
                    "ename": out.get("ename"),
                    "evalue": out.get("evalue"),
                    "traceback": "\n".join(out.get("traceback", [])),
                    "source_head": "\n".join(cell.source.split("\n")[:6]),
                })
    return failures, elapsed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exec", dest="do_exec", action="store_true")
    ap.add_argument("--mode", default=None, help="sets ROBUSTLAB_MODE for the executed kernel")
    ap.add_argument("--kernel", default="robustlab-py314")
    ap.add_argument("--timeout", type=int, default=7200)
    args = ap.parse_args()

    text = read_parts()
    cells = split_cells(text)

    # A syntax error in any code cell should fail the build, not the kernel.
    for i, (kind, src) in enumerate(cells):
        if kind == "code":
            try:
                compile(src, f"<cell {i}>", "exec")
            except SyntaxError as e:
                print(f"SYNTAX ERROR in cell {i}: {e}")
                print("\n".join(src.split("\n")[:12]))
                return 1

    FLAT_PATH.write_text(text)
    nb = build_notebook(cells)

    n_code = sum(1 for k, _ in cells if k == "code")
    n_md = len(cells) - n_code
    print(f"built {len(cells)} cells ({n_code} code, {n_md} markdown)")

    if not args.do_exec:
        import nbformat
        nbformat.write(nb, str(NB_PATH))
        print(f"wrote {NB_PATH}")
        return 0

    if args.mode:
        os.environ["ROBUSTLAB_MODE"] = args.mode
    failures, elapsed = execute(nb, args.kernel, args.timeout)

    import nbformat
    nbformat.write(nb, str(NB_PATH))
    print(f"wrote {NB_PATH} (executed in {elapsed:.1f}s)")

    if failures:
        print(f"\n{'='*70}\n{len(failures)} CELL FAILURE(S)\n{'='*70}")
        for f in failures:
            print(f"\n--- cell {f['cell']}: {f['ename']}: {f['evalue']}")
            print(f"    source: {f['source_head'][:300]}")
            tb = re.sub(r"\x1b\[[0-9;]*m", "", f["traceback"])
            print("\n".join(tb.split("\n")[-18:]))
        return 1

    print("\nALL CELLS EXECUTED CLEANLY")
    return 0


if __name__ == "__main__":
    sys.exit(main())
