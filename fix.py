import os
import sys

lines = open('scripts/phase2_3_experiments.py', 'r', encoding='utf-8').readlines()
new_lines = []
in_main = False
for i, l in enumerate(lines):
    if 'print("\\n=== STAGE 1:' in l:
        new_lines.append('if __name__ == "__main__":\n')
        in_main = True
    
    if in_main:
        new_lines.append('    ' + l)
    else:
        new_lines.append(l)

open('scripts/phase2_3_experiments.py', 'w', encoding='utf-8').writelines(new_lines)
print('Guarded phase2_3_experiments.py')
