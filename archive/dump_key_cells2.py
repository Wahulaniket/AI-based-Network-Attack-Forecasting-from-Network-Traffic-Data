import json

nb = json.load(open('nootebooks/CyberCast_Complete.ipynb', 'r', encoding='utf-8'))

# Print full source of key cells - setup, split, model, training, evaluation
for idx in [0, 2, 4, 6, 10, 14, 16, 22, 24, 28, 29]:
    src = ''.join(nb['cells'][idx]['source'])
    print(f"{'='*80}")
    print(f"CELL {idx}")
    print(f"{'='*80}")
    print(src)
    print()
