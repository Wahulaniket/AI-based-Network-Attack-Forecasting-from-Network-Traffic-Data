import nbformat as nbf
import sys
import re

def convert(py_file, ipynb_file):
    with open(py_file, 'r', encoding='utf-8') as f:
        content = f.read()

    cells = []
    # Split by # %%
    parts = re.split(r'^# %%\s*(.*)$', content, flags=re.MULTILINE)
    
    # The first part might be preamble
    if parts[0].strip():
        cells.append(nbf.v4.new_code_cell(parts[0].strip()))
        
    for i in range(1, len(parts), 2):
        header = parts[i].strip()
        cell_content = parts[i+1].strip()
        
        if not cell_content:
            continue
            
        if header.startswith('[markdown]'):
            lines = cell_content.split('\n')
            md_lines = []
            for line in lines:
                if line.startswith('# '):
                    md_lines.append(line[2:])
                elif line == '#':
                    md_lines.append('')
                else:
                    md_lines.append(line)
            cells.append(nbf.v4.new_markdown_cell('\n'.join(md_lines)))
        else:
            cells.append(nbf.v4.new_code_cell(cell_content))

    nb = nbf.v4.new_notebook()
    nb['cells'] = cells
    
    with open(ipynb_file, 'w', encoding='utf-8') as f:
        nbf.write(nb, f)

if __name__ == '__main__':
    convert(sys.argv[1], sys.argv[2])
