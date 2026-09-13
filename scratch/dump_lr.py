import json

nb = json.load(open('nootebooks/CyberCast_Complete.ipynb', 'r', encoding='utf-8'))

# Cell 20 = LR baseline, Cell 37 = permutation importance
for idx in [20, 37, 47, 48]:
    src = ''.join(nb['cells'][idx]['source'])
    print(f"{'='*80}")
    print(f"CELL {idx}")
    print(f"{'='*80}")
    print(src)
    print()
