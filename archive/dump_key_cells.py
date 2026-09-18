import json

nb = json.load(open('nootebooks/CyberCast_Complete.ipynb', 'r', encoding='utf-8'))

# Print full source of key cells
for idx in [14, 18, 20, 28, 31, 33, 35, 39, 41, 43, 45]:
    src = ''.join(nb['cells'][idx]['source'])
    print(f"{'='*80}")
    print(f"CELL {idx}")
    print(f"{'='*80}")
    print(src)
    print()
