import json
import os

os.makedirs('scratch', exist_ok=True)

with open(r'nootebooks\CyberCast_Complete.ipynb', 'r', encoding='utf-8-sig') as fp:
    nb = json.load(fp)

cells = nb['cells']
print(f'Total cells: {len(cells)}')

with open('scratch/nb_dump.txt', 'w', encoding='utf-8') as f:
    for i, c in enumerate(cells):
        ctype = c['cell_type']
        src = ''.join(c['source'])
        f.write(f'=== CELL {i} ({ctype}) ===\n{src}\n\n')

print('Done - wrote scratch/nb_dump.txt')
