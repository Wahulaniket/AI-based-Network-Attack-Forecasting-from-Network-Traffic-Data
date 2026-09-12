#!/usr/bin/env python3
"""Generate the CyberCast_Complete.ipynb notebook.

Writes each cell's source lines to a JSON notebook file.
"""

import json, os

def md(lines):
    """Create a markdown cell from a list of source lines."""
    return {
        "cell_type": "markdown", "metadata": {},
        "source": [l + "\n" for l in lines[:-1]] + [lines[-1]] if lines else []
    }

def code(lines):
    """Create a code cell from a list of source lines."""
    return {
        "cell_type": "code", "metadata": {},
        "source": [l + "\n" for l in lines[:-1]] + [lines[-1]] if lines else [],
        "outputs": [], "execution_count": None
    }

cells = []

# Instead of building cells inline (which has escaping issues),
# read cell sources from separate .py / .md files.
# But for simplicity, let's write each cell source to a temp file,
# then read it back.

# We'll build the cells list by reading from a big text file
# that separates cells with special markers.

# Write the cell-source master file
CELL_MARKER = "###CELL_SEPARATOR###"

cell_sources_path = os.path.join(os.path.dirname(__file__), "cell_sources.txt")

# Actually, let's just construct the notebook JSON directly
# by reading from a Python source file that we can execute.

# Simplest approach: write the notebook as a .py percent-script,
# then convert to .ipynb using jupytext-like logic.

# The percent format uses # %% to separate cells:
# # %% [markdown]
# # markdown content
# # %%
# code content

print("Building notebook from cell source blocks...")

# Read the cell source blocks from a companion file
blocks_path = os.path.join(os.path.dirname(__file__), "nb_cells.py")
with open(blocks_path, "r", encoding="utf-8") as f:
    content = f.read()

# Parse percent-format
import re
raw_blocks = re.split(r'^# %%', content, flags=re.MULTILINE)

cells = []
for block in raw_blocks:
    if not block.strip():
        continue
    # Do not strip before checking the markdown tag, or checking '[markdown]'
    if block.strip().startswith("[markdown]"):
        # Markdown cell
        lines = block.strip()[len("[markdown]"):].strip().split("\n")
        # Remove leading '# ' from each line
        md_lines = []
        for l in lines:
            if l.startswith("# "):
                md_lines.append(l[2:])
            elif l == "#":
                md_lines.append("")
            else:
                md_lines.append(l)
        cells.append(md(md_lines))
    else:
        # Code cell - remove leading newline if present
        lines = block.strip().split("\n")
        cells.append(code(lines))

notebook = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.11.9"
        }
    },
    "cells": cells,
}

out_path = r"D:\working_projects\SIH\cyberCast\nootebooks\CyberCast_Complete.ipynb"
with open(out_path, 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=1, ensure_ascii=False)

print(f"Notebook written to: {out_path}")
print(f"   Cells: {len(cells)}")
print(f"   Size: {os.path.getsize(out_path) / 1024:.1f} KB")
