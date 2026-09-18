import json

nb = json.load(open('nootebooks/CyberCast_Complete.ipynb', 'r', encoding='utf-8'))
print(f"Total cells: {len(nb['cells'])}")
print()

for i, c in enumerate(nb['cells']):
    ctype = c['cell_type'][:4]
    src = ''.join(c['source'])
    preview = src[:150].replace('\n', '\\n')
    print(f"Cell {i:3d} [{ctype}]: {preview}")
