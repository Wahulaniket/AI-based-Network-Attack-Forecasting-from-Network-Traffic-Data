import nbformat as nbf
import os

def create_nb2():
    nb = nbf.v4.new_notebook()
    cells = []
    
    cells.append(nbf.v4.new_markdown_cell("""# Experiment Identity
- **Experiment ID**: Phase 2 SET_A Historical
- **Phase**: Phase 2
- **Model**: LSTM Baseline (`model_SET_A_h20.pt`)
- **Status**: [HISTORICAL]
- **Training Source**: MISSING
- **Evaluation Source**: Legacy evaluation notebooks
- **Dataset**: `network_states_10s.parquet` (approximate)
- **Feature Set**: SET_A (approximate)
- **Feature Count**: 15 (approximate)
- **Sequence Length**: 20 (based on filename `h20`)
- **Target**: `attack_binary`
- **Artifact**: `models/archive/model_SET_A_h20.pt`
"""))

    cells.append(nbf.v4.new_markdown_cell("""## 1. Objective
[DOCUMENTATION]
This notebook documents the known facts about the Phase 2 `model_SET_A_h20.pt` artifact, whose training source code was lost during the repository migration. It acts as an artifact-backed historical reconstruction.
"""))

    cells.append(nbf.v4.new_markdown_cell("""## 2. Source Code Lineage
[NOT-REPRODUCIBLE]
The original standalone Python training script is missing. The notebook `01_Phase_2_Model_Training.ipynb` only visualized results, it did not contain the training code.
"""))
    
    cells.append(nbf.v4.new_markdown_cell("""## 3. Environment
[DOCUMENTATION]
Requires PyTorch and Scikit-learn.
"""))

    cells.append(nbf.v4.new_code_cell("""import sys, os
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")
"""))

    cells.append(nbf.v4.new_markdown_cell("""## 4-13. Architecture and Data Pipeline
[HISTORICAL]
Based on the artifact filename (`model_SET_A_h20.pt`), it is inferred that the model utilized the base `SET_A` features and a sequence length of `20`. The exact architecture definition is missing, but it is presumed to be a standard PyTorch LSTM.
"""))

    cells.append(nbf.v4.new_markdown_cell("""## 25. Artifact Provenance
[FROZEN-ARTIFACT]
Artifact Path: `models/archive/model_SET_A_h20.pt`
"""))
    cells.append(nbf.v4.new_code_cell("""model_path = '../../models/archive/model_SET_A_h20.pt'
if os.path.exists(model_path):
    print(f"Found frozen artifact: {model_path}")
else:
    print(f"Artifact not found at {model_path}")
"""))

    cells.append(nbf.v4.new_markdown_cell("""## 26. Final Results
[HISTORICAL]
Historically, the Phase 2 baseline models achieved ~0.7371 PR-AUC on the test set, but this metric could be subject to chronological leakage since strict chronological splitting was introduced later in Phase 2.2.
"""))

    cells.append(nbf.v4.new_markdown_cell("""## 28. Limitations
[NOT-REPRODUCIBLE]
This is a purely historical reconstruction. The model cannot be retrained from this notebook, and the exact weights cannot be reproduced.
"""))

    nb.cells = cells
    os.makedirs('notebooks/phase2', exist_ok=True)
    with open('notebooks/phase2/Phase2_SET_A_Historical.ipynb', 'w') as f:
        nbf.write(nb, f)
    print("Created Phase2_SET_A_Historical.ipynb")

if __name__ == '__main__':
    create_nb2()
