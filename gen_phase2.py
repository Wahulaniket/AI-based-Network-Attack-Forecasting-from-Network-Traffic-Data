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
            # If path is notebooks/Phase_2/..., depth is 2, so we chdir ../.. in the notebook
            ep.preprocess(nb, {'metadata': {'path': os.path.dirname(path)}})
            with open(path, 'w', encoding='utf-8') as f:
                nbf.write(nb, f)
            print(f"Successfully executed {path}")
        except Exception as e:
            print(f"Error executing {path}: {e}")

def create_phase_2_training():
    nb = nbf.v4.new_notebook()
    nb.cells.append(nbf.v4.new_markdown_cell("# CyberCast\n## Phase 2\n### Experiment / Training"))
    nb.cells.append(nbf.v4.new_code_cell("import os\nos.chdir('../..')\nimport pandas as pd\nimport matplotlib.pyplot as plt\nimport seaborn as sns"))
    
    nb.cells.append(nbf.v4.new_markdown_cell("""
### 1-6. Objective, Dataset, and Sequences
The objective of Phase 2 was to explore various model architectures for next-window attack forecasting using a rich feature representation.
The dataset was transformed into 10-second temporal windows with approximately 70 features.
"""))
    
    nb.cells.append(nbf.v4.new_markdown_cell("""
### 7-18. Model Architectures and Comparison
We evaluated LSTM, Attention LSTM, TCN, GRU, Transformer, Logistic Regression, XGBoost, and Random Forest.
"""))
    nb.cells.append(nbf.v4.new_code_cell("""
mc_path = 'results/model_comparison.csv'
if os.path.exists(mc_path):
    df = pd.read_csv(mc_path)
    display(df)
    plt.figure(figsize=(10, 5))
    sns.barplot(data=df.sort_values('Val_PR_AUC', ascending=False), x='Model', y='Val_PR_AUC')
    plt.xticks(rotation=45)
    plt.title("Phase 2 Model Comparison (Val PR-AUC)")
    plt.show()
else:
    print("Model comparison artifact unavailable.")
"""))

    nb.cells.append(nbf.v4.new_markdown_cell("""
### 19-20. Champion Selection and Artifacts
The LSTM was selected as the champion due to superior Validation PR-AUC.

**RESULT STATUS:** HISTORICAL RESULT
"""))
    save_and_run(nb, 'notebooks/Phase_2/01_Phase_2_Model_Training.ipynb')

def create_phase_2_evaluation():
    nb = nbf.v4.new_notebook()
    nb.cells.append(nbf.v4.new_markdown_cell("# CyberCast\n## Phase 2\n### Evaluation\n\n**PHASE 2 HISTORICAL BASELINE**"))
    nb.cells.append(nbf.v4.new_code_cell("import os\nos.chdir('../..')\nimport pandas as pd\nimport json"))
    nb.cells.append(nbf.v4.new_markdown_cell("""
### Final Evaluation
Historical Phase 2 LSTM result:
- PR-AUC = 0.7371
- F1 = 0.6887
"""))
    nb.cells.append(nbf.v4.new_code_cell("""
met_path = 'results/metrics.csv'
if os.path.exists(met_path):
    df = pd.read_csv(met_path)
    display(df)
else:
    print("Metrics artifact unavailable. Showing recorded historical values: PR-AUC=0.7371, F1=0.6887.")
"""))
    nb.cells.append(nbf.v4.new_markdown_cell("**RESULT STATUS:** HISTORICAL RESULT"))
    save_and_run(nb, 'notebooks/Phase_2/02_Phase_2_Model_Evaluation.ipynb')

if __name__ == '__main__':
    create_phase_2_training()
    create_phase_2_evaluation()
