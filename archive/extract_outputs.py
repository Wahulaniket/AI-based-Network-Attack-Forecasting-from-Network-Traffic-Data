import json
j = json.load(open(r'nootebooks\CyberCast_Model_From_Scratch.ipynb', 'r', encoding='utf-8'))
for cell in j['cells']:
    if cell['cell_type'] == 'code' and 'outputs' in cell:
        for out in cell['outputs']:
            if 'text' in out:
                print(''.join(out['text']))
