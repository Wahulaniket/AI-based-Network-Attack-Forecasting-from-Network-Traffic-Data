import os
import nbformat as nbf
from nbconvert.preprocessors import ExecutePreprocessor

def save_and_run(nb, path, run=True):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        nbf.write(nb, f)
    print(f'Created {path}')
    if run:
        print(f'Executing {path}...')
        try:
            ep = ExecutePreprocessor(timeout=600, kernel_name='python3')
            ep.preprocess(nb, {'metadata': {'path': os.path.dirname(path)}})
            with open(path, 'w', encoding='utf-8') as f:
                nbf.write(nb, f)
            print(f'Successfully executed {path}')
        except Exception as e:
            print(f'Error executing {path}: {e}')

nb = nbf.v4.new_notebook()
nb.cells.append(nbf.v4.new_markdown_cell('''# CyberCast
## Phase 2.3
### Evaluation

**THIS NOTEBOOK MUST NOT RERUN THE FINAL BLIND TEST.**
Loading frozen test metrics...'''))
nb.cells.append(nbf.v4.new_code_cell('''import os
os.chdir('../..')
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns'''))
nb.cells.append(nbf.v4.new_code_cell("""
champ_path = 'results/phase2_3/champion_metrics.json'
if os.path.exists(champ_path):
    with open(champ_path, 'r') as f:
        champ = json.load(f)
    tm = champ['Test_Metrics']
    print("FROZEN TEST METRICS:")
    print(f"PR-AUC = {tm['PR-AUC']:.4f}")
    print(f"ROC-AUC = {tm['ROC-AUC']:.4f}")
    print(f"F1 = {tm['F1']:.4f}")
    print(f"Precision = {tm['Precision']:.4f}")
    print(f"Recall = {tm['Recall']:.4f}")
    print(f"Accuracy = {tm['Accuracy']:.4f}")
    print(f"FPR = {tm['FPR']:.4f}")
    print(f"FNR = {tm['FNR']:.4f}\\n")
    print(f"TN = {tm['TN']}\\nFP = {tm['FP']}\\nFN = {tm['FN']}\\nTP = {tm['TP']}")
else:
    print("Champion metrics unavailable.")
"""))
nb.cells.append(nbf.v4.new_code_cell("""
ew_path = 'results/phase2_3/early_warning_diagnostic.csv'
if os.path.exists(ew_path):
    print("Multi-horizon diagnostic:")
    display(pd.read_csv(ew_path))
"""))
nb.cells.append(nbf.v4.new_markdown_cell("""
### Phase 2 vs Phase 2.2 vs Phase 2.3
Phase 2.3 recovered the rich legitimate traffic representation while retaining strict causal evaluation. The results suggest that restricting the feature representation in Phase 2.2 substantially reduced predictive performance.

### Limitations
Dataset-based offline evaluation, unverified generalization to unseen networks, binary next-window forecasting is not yet complete attack-chain forecasting.

**RESULT STATUS:** FROZEN RESULT
"""))
save_and_run(nb, 'notebooks/Phase_2_3/02_Phase_2_3_Final_Evaluation.ipynb')
