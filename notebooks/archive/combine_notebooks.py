import os
import nbformat as nbf
import glob

def combine_notebooks():
    input_files = [
        'notebooks/Phase_2/01_Phase_2_Model_Training.ipynb',
        'notebooks/Phase_2/02_Phase_2_Model_Evaluation.ipynb',
        'notebooks/Phase_2_1/01_Phase_2_1_Experiments_Training.ipynb',
        'notebooks/Phase_2_1/02_Phase_2_1_Evaluation.ipynb',
        'notebooks/Phase_2_2/01_Phase_2_2_Experiments_Training.ipynb',
        'notebooks/Phase_2_2/02_Phase_2_2_Evaluation.ipynb',
        'notebooks/Phase_2_3/01_Phase_2_3_Training.ipynb',
        'notebooks/Phase_2_3/02_Phase_2_3_Final_Evaluation.ipynb'
    ]
    
    combined_nb = nbf.v4.new_notebook()
    combined_nb.cells.append(nbf.v4.new_markdown_cell('# CyberCast -- All Phases\\n\\nThis notebook combines all experimental phases (Phase 2, 2.1, 2.2, and 2.3) into a single document.'))
    
    for filepath in input_files:
        if not os.path.exists(filepath):
            print(f"Skipping {filepath} (does not exist)")
            continue
            
        print(f"Reading {filepath}...")
        nb = nbf.read(filepath, as_version=4)
        
        # Add a clear separator for each notebook
        filename = os.path.basename(filepath)
        combined_nb.cells.append(nbf.v4.new_markdown_cell(f'---\\n## Source: `{filename}`\\n---'))
        
        for cell in nb.cells:
            combined_nb.cells.append(cell)
            
    out_path = 'notebooks/CyberCast_All_Phases.ipynb'
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        nbf.write(combined_nb, f)
    
    print(f"\\nSuccessfully combined into {out_path}")

if __name__ == '__main__':
    combine_notebooks()
