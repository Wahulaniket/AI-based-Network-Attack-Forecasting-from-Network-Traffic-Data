import json

nb1 = json.load(open('notebooks/CyberCast_Complete.ipynb', encoding='utf-8'))
nb2 = json.load(open('nootebooks/CyberCast_Complete.ipynb', encoding='utf-8'))

print(f"notebooks/CyberCast_Complete.ipynb: {len(nb1['cells'])} cells")
for i, c in enumerate(nb1['cells']):
    src = "".join(c['source']).strip().replace('\n', ' ')[:90]
    print(f"  [{c['cell_type'][:4]}] {i:02d}: {src}")

print(f"\nnootebooks/CyberCast_Complete.ipynb: {len(nb2['cells'])} cells")
for i, c in enumerate(nb2['cells']):
    src = "".join(c['source']).strip().replace('\n', ' ')[:90]
    print(f"  [{c['cell_type'][:4]}] {i:02d}: {src}")
