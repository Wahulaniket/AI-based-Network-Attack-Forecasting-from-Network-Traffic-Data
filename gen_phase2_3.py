import os
import nbformat as nbf
from nbconvert.preprocessors import ExecutePreprocessor

def save_and_run(nb, path, run=True):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        nbf.write(nb, f)
    print(f"Created {path}")
    if run:
        print(f"Executing {path}...")
        try:
            ep = ExecutePreprocessor(timeout=600, kernel_name='python3')
            ep.preprocess(nb, {'metadata': {'path': os.path.dirname(path)}})
            with open(path, 'w', encoding='utf-8') as f:
                nbf.write(nb, f)
            print(f"Successfully executed {path}")
        except Exception as e:
            print(f"Error executing {path}: {e}")

def create_phase_2_3_training():
    nb = nbf.v4.new_notebook()
    nb.cells.append(nbf.v4.new_markdown_cell("# CyberCast\n## Phase 2.3\n### Experiment / Training\n\n**This notebook documents the FINAL FROZEN Phase 2.3 experiment.**\n*(Do not rerun the final blind test here.)*"))
    nb.cells.append(nbf.v4.new_code_cell("import os\nos.chdir('../..')\nimport pandas as pd\nimport matplotlib.pyplot as plt\nimport seaborn as sns"))
    nb.cells.append(nbf.v4.new_markdown_cell("""
### Feature Recovery and Audit
- SET_R = 89 features
- SET_R_SELECT = 40 features
- SET_A = 15 features
- Six rejected columns: Timestamp, binary_attack, dominant_label, attack_ratio, attack_flow_count, has_traffic
"""))
    nb.cells.append(nbf.v4.new_code_cell("""
feat_comp = 'results/phase2_3/feature_comparison.csv'
if os.path.exists(feat_comp):
    print("Stage 1 Feature Comparison:")
    display(pd.read_csv(feat_comp))
"""))
    nb.cells.append(nbf.v4.new_code_cell("""
hist_comp = 'results/phase2_3/history_comparison.csv'
if os.path.exists(hist_comp):
    print("Stage 2 History Comparison:")
    display(pd.read_csv(hist_comp))
"""))
    nb.cells.append(nbf.v4.new_markdown_cell("""
### Final Champion Selection
- SET_R (89 features)
- 20 windows (200 seconds)
- LSTM
- threshold = 0.9830

### Reproducibility
Run 1 = 0.6471 (Best epoch 0)
Run 2 = 0.6471 (Best epoch 0)

*Reproducible under the tested software/hardware environment with the specified seed controls.*

**RESULT STATUS:** FROZEN RESULT
"""))
    save_and_run(nb, 'notebooks/Phase_2_3/01_Phase_2_3_Training.ipynb')

def create_phase_2_3_evaluation():
    nb = nbf.v4.new_notebook()
    nb.cells.append(nbf.v4.new_markdown_cell("# CyberCast\n## Phase 2.3\n### Evaluation\n\n**THIS NOTEBOOK MUST NOT RERUN THE FINAL BLIND TEST.**\nLoading frozen test metrics..."))
    nb.cells.append(nbf.v4.new_code_cell("import os\nos.chdir('../..')\nimport json\nimport pandas as pd\nimport numpy as np\nimport matplotlib.pyplot as plt\nimport seaborn as sns"))
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
    print(f"FNR = {tm['FNR']:.4f}\n")
    print(f"TN = {tm['TN']}\nFP = {tm['FP']}\nFN = {tm['FN']}\nTP = {tm['TP']}")
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

if __name__ == '__main__':
    create_phase_2_3_training()
    create_phase_2_3_evaluation()
