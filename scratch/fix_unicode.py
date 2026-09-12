import re

path = r'D:\working_projects\SIH\cyberCast\scratch\generate_notebook.py'
with open(path, 'r', encoding='utf-8') as f:
    data = f.read()

# Replace problematic Unicode characters with ASCII equivalents
replacements = {
    '\u2014': '--',   # em dash
    '\u2013': '-',    # en dash
    '\u2018': "'",    # left single quote
    '\u2019': "'",    # right single quote
    '\u201c': '"',    # left double quote
    '\u201d': '"',    # right double quote
    '\u2026': '...',  # ellipsis
    '\u00d7': 'x',    # multiplication sign
    '\u2265': '>=',   # >=
    '\u2264': '<=',   # <=
    '\u2260': '!=',   # !=
    '\u00b1': '+/-',  # plus-minus
    '\u221e': 'inf',  # infinity
    '\u2192': '->',   # right arrow
    '\u2190': '<-',   # left arrow
    '\u2191': '^',    # up arrow
    '\u2193': 'v',    # down arrow
    '\u2502': '|',    # box drawing vertical
    '\u2500': '-',    # box drawing horizontal
    '\u250c': '+',    # box drawing top-left
    '\u2510': '+',    # box drawing top-right
    '\u2514': '+',    # box drawing bottom-left
    '\u2518': '+',    # box drawing bottom-right
    '\u2524': '+',    # box drawing vertical-left
    '\u251c': '+',    # box drawing vertical-right
    '\u2534': '+',    # box drawing horizontal-up
    '\u252c': '+',    # box drawing horizontal-down
    '\u253c': '+',    # box drawing cross
}

for old, new in replacements.items():
    data = data.replace(old, new)

with open(path, 'w', encoding='utf-8') as f:
    f.write(data)

print("Fixed Unicode characters")
