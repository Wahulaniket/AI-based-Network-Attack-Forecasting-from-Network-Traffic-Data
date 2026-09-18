import os
import sys
import json
import gc
import traceback
import io
import contextlib

print("================================================================================")
print("FIXING CELL 12 & RE-RUNNING ALL NOTEBOOK CELLS")
print("================================================================================")

NB_SRC = r'd:\working_projects\SIH\cyberCast\nootebooks\CyberCast_Complete.ipynb'
NB_DEST1 = r'd:\working_projects\SIH\cyberCast\notebooks\CyberCast_Complete.ipynb'
NB_DEST2 = r'd:\working_projects\SIH\cyberCast\nootebooks\CyberCast_Complete.ipynb'

with open(NB_SRC, 'r', encoding='utf-8') as f:
    nb = json.load(f)

cells = nb['cells']

# Define map_attack_family helper in Cell 12
c12_source = """# Label mapping summary
def map_attack_family(label):
    l = str(label).lower()
    if 'benign' in l: return 'Benign'
    if 'brute' in l or 'ftp' in l or 'ssh' in l: return 'Brute Force'
    if 'dos' in l: return 'DoS'
    if 'ddos' in l: return 'DDoS'
    if 'web' in l or 'xss' in l or 'sql' in l: return 'Web Attack'
    if 'infil' in l: return 'Infiltration'
    if 'bot' in l: return 'Botnet'
    return 'Other Attack'

print("Label -> Attack Family -> Binary:")
print(f"{'Original Label':40s} {'Family':20s} {'Binary':>6s}")
print("-" * 68)
for lab in sorted(global_label_counts.keys()):
    fam = map_attack_family(lab)
    b = 0 if lab == 'Benign' else 1
    print(f"{lab:40s} {fam:20s} {b:>6d}")
print("\\nNote: Attack families are dataset-specific groupings, NOT MITRE ATT&CK stages.")
"""

def make_source_list(text):
    lines = text.split('\n')
    return [l + '\n' if i < len(lines)-1 else l for i, l in enumerate(lines)]

cells[12]['source'] = make_source_list(c12_source)

# Shared environment
exec_env = {
    '__name__': '__main__',
    '__file__': NB_SRC
}

executed_cells = 0
passed_cells = 0
error_cells = 0

for i, cell in enumerate(cells):
    if cell['cell_type'] == 'code':
        source = "".join(cell['source'])
        if not source.strip():
            cell['outputs'] = []
            cell['execution_count'] = None
            continue
        
        executed_cells += 1
        stdout_buf = io.StringIO()
        try:
            with contextlib.redirect_stdout(stdout_buf):
                exec(source, exec_env)
            
            output_text = stdout_buf.getvalue()
            cell['outputs'] = []
            if output_text:
                cell['outputs'].append({
                    "name": "stdout",
                    "output_type": "stream",
                    "text": output_text.splitlines(keepends=True)
                })
            cell['execution_count'] = executed_cells
            passed_cells += 1
            print(f"Cell {i:02d}: PASSED")
        except Exception as e:
            error_cells += 1
            output_text = stdout_buf.getvalue()
            err_msg = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
            print(f"Cell {i:02d}: FAILED: {type(e).__name__}: {e}")
            cell['outputs'] = []
            if output_text:
                cell['outputs'].append({
                    "name": "stdout",
                    "output_type": "stream",
                    "text": output_text.splitlines(keepends=True)
                })
            cell['outputs'].append({
                "ename": type(e).__name__,
                "evalue": str(e),
                "output_type": "error",
                "traceback": err_msg.splitlines(keepends=True)
            })
            cell['execution_count'] = executed_cells

print("\n================================================================================")
print(f"TOTAL EXECUTED: {executed_cells} code cells | PASSED: {passed_cells} | ERRORS: {error_cells}")
print("================================================================================")

# Save updated notebook to both locations
nb['cells'] = cells

with open(NB_DEST1, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

with open(NB_DEST2, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"Successfully saved executed notebook to:")
print(f"  1. {NB_DEST1}")
print(f"  2. {NB_DEST2}")
