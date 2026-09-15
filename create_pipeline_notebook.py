import os
import nbformat as nbf
import re

def convert_py_to_cells(py_file, phase_name):
    cells = []
    cells.append(nbf.v4.new_markdown_cell(f'# {phase_name}\\n\\n---\\n\\n## Script: `{os.path.basename(py_file)}`'))
    
    with open(py_file, 'r', encoding='utf-8') as f:
        content = f.read()

    # Split by comment blocks that look like headers: # 1., # 2. or # ====
    # Actually, let's just split by double newlines combined with comments if possible.
    # To keep it simple but structured, we will split by lines that start with '# ===' or '# ---' or '# 1.', '# 2.'
    
    chunks = re.split(r'\n(?=# =================|# 1\.|# 2\.|# 3\.|# 4\.|# 5\.|# 6\.|# 7\.|# 8\.|# 9\.)', content)
    
    for chunk in chunks:
        if not chunk.strip():
            continue
        cells.append(nbf.v4.new_code_cell(chunk.strip()))
        
    return cells

def main():
    # 1. Base notebook is CyberCast_Complete.ipynb
    base_nb_path = r'D:\working_projects\SIH\cyberCast\notebooks\CyberCast_Complete.ipynb'
    print(f"Loading {base_nb_path}")
    nb = nbf.read(base_nb_path, as_version=4)
    
    # Update title
    nb.cells[0].source = nb.cells[0].source.replace('World Model Architecture', 'World Model Architecture\\n\\n*(Note: This notebook has been expanded to include Phase 2.2 and Phase 2.3 training pipelines)*')

    # 2. Phase 2.2
    phase2_2_script = r'D:\working_projects\SIH\cyberCast\scripts\phase2_2_experiments.py'
    if os.path.exists(phase2_2_script):
        print(f"Processing {phase2_2_script}")
        p2_cells = convert_py_to_cells(phase2_2_script, 'CyberCast Phase 2.2: Strict Causal Baseline')
        nb.cells.extend(p2_cells)

    # 3. Phase 2.3
    phase2_3_script = r'D:\working_projects\SIH\cyberCast\scripts\phase2_3_experiments.py'
    if os.path.exists(phase2_3_script):
        print(f"Processing {phase2_3_script}")
        p3_cells = convert_py_to_cells(phase2_3_script, 'CyberCast Phase 2.3: Rich Feature Recovery')
        nb.cells.extend(p3_cells)
        
    out_path = r'D:\working_projects\SIH\cyberCast\notebooks\CyberCast_All_Pipelines.ipynb'
    with open(out_path, 'w', encoding='utf-8') as f:
        nbf.write(nb, f)
        
    print(f"Successfully generated {out_path} with {len(nb.cells)} cells.")

if __name__ == '__main__':
    main()
