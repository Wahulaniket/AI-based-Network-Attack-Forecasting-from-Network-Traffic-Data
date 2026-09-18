import json

nb = json.load(open('nootebooks/CyberCast_Complete.ipynb', 'r', encoding='utf-8'))

for idx in [10, 14]:
    src = ''.join(nb['cells'][idx]['source'])
    print(f"{'='*80}")
    print(f"CELL {idx}")
    print(f"{'='*80}")
    print(src)
    print()
