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

def create_phase_2_1_training():
    nb = nbf.v4.new_notebook()
    nb.cells.append(nbf.v4.new_markdown_cell("# CyberCast\n## Phase 2.1\n### Experiment / Training"))
    nb.cells.append(nbf.v4.new_code_cell("import os\nos.chdir('../..')\nimport pandas as pd\nimport json\nimport matplotlib.pyplot as plt\nimport seaborn as sns"))
    nb.cells.append(nbf.v4.new_markdown_cell("""
### 1-5. Objective and Configuration
Phase 2.1 explored class weights, loss functions, and architectural modifications to stabilize training.
"""))
    nb.cells.append(nbf.v4.new_markdown_cell("""
### 6-8. Experiment Comparison and Winner Selection
Actual validation configuration winner:
- Base LSTM 64
- pos_weight = 4.0
- Validation PR-AUC = 0.5087
- Validation F1 = 0.3727
"""))
    nb.cells.append(nbf.v4.new_code_cell("""
exp_path = 'results/phase2_1/experiment_comparison.csv'
if os.path.exists(exp_path):
    df = pd.read_csv(exp_path)
    display(df)
else:
    print("Experiment comparison artifact unavailable.")
"""))
    nb.cells.append(nbf.v4.new_markdown_cell("""
### 9-10. Locked Threshold and Artifacts
Locked threshold = 0.19

**RESULT STATUS:** HISTORICAL RESULT
"""))
    save_and_run(nb, 'notebooks/Phase_2_1/01_Phase_2_1_Experiments_Training.ipynb')

def create_phase_2_1_evaluation():
    nb = nbf.v4.new_notebook()
    nb.cells.append(nbf.v4.new_markdown_cell("# CyberCast\n## Phase 2.1\n### Evaluation\n\nHistorical Phase 2.1 Test Result:\n- PR-AUC = 0.6931\n- F1 = 0.6739\n- Recall = 0.7392\n- Precision = 0.6192\n- FPR = 0.5717"))
    nb.cells.append(nbf.v4.new_code_cell("import os\nos.chdir('../..')\nimport json"))
    nb.cells.append(nbf.v4.new_code_cell("""
test_path = 'results/phase2_1/final_test_metrics.json'
if os.path.exists(test_path):
    with open(test_path, 'r') as f:
        print(json.dumps(json.load(f), indent=2))
else:
    print("Final test metrics artifact unavailable.")
"""))
    nb.cells.append(nbf.v4.new_markdown_cell("**RESULT STATUS:** HISTORICAL RESULT"))
    save_and_run(nb, 'notebooks/Phase_2_1/02_Phase_2_1_Evaluation.ipynb')

if __name__ == '__main__':
    create_phase_2_1_training()
    create_phase_2_1_evaluation()
