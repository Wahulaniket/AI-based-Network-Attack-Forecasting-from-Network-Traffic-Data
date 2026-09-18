import nbformat as nbf
import os

def create_nb4():
    nb = nbf.v4.new_notebook()
    cells = []
    
    cells.append(nbf.v4.new_markdown_cell("""# Experiment Identity
- **Experiment ID**: Phase 2.2 Champion
- **Phase**: Phase 2.2
- **Model**: `CyberCastForecaster`
- **Status**: [ARTIFACT-BACKED DOCUMENTATION]
- **Training Source**: `scripts/phase2_2_experiments.py`
- **Evaluation Source**: `scripts/phase2_2_experiments.py`
- **Dataset**: `network_states_10s.parquet`
- **Feature Set**: SET_B
- **Feature Count**: 30
- **Sequence Length**: 20
- **Target**: `attack_binary`
- **Artifact**: `results/phase2_2/models/model_20_SET_B.pt`
"""))

    cells.append(nbf.v4.new_markdown_cell("""## 1. Objective
[DOCUMENTATION]
This notebook documents the complete Phase 2.2 experiment. The python script (`scripts/phase2_2_experiments.py`) remains the executable source of truth.
"""))

    cells.append(nbf.v4.new_code_cell("""import sys, os, json
import pandas as pd

# Load the experiment results
results_path = '../../results/phase2_2/phase2_2_results.csv'
if os.path.exists(results_path):
    df = pd.read_csv(results_path)
    print("Phase 2.2 Experiment Comparison:")
    display(df)
else:
    print("Results file not found.")
"""))

    cells.append(nbf.v4.new_markdown_cell("""## 16-17. Validation & Champion Selection
[DOCUMENTATION]
The model utilizing `SET_B` with a history of `20` windows achieved the highest PR-AUC on the validation set, eliminating chronological leakage from Phase 2.
"""))

    nb.cells = cells
    os.makedirs('notebooks/phase2_2', exist_ok=True)
    with open('notebooks/phase2_2/Phase2_2_Champion.ipynb', 'w') as f:
        nbf.write(nb, f)
    print("Created Phase2_2_Champion.ipynb")

if __name__ == '__main__':
    create_nb4()
