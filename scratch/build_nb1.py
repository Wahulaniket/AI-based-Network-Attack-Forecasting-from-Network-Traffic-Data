import nbformat as nbf
import os

def create_nb1():
    nb = nbf.v4.new_notebook()
    cells = []
    
    cells.append(nbf.v4.new_markdown_cell("""# Experiment Identity
- **Experiment ID**: Phase 2 Deep Learning Baselines
- **Phase**: Phase 2
- **Model**: LSTM Baseline (`cybercast_best_model.pt`)
- **Status**: [HISTORICAL]
- **Training Source**: MISSING (Synthetically reconstructed in `generate_notebook.py`)
- **Evaluation Source**: `generate_notebook.py`
- **Dataset**: `network_states_10s.parquet` (approximate)
- **Feature Set**: Base 15 features
- **Feature Count**: 15
- **Sequence Length**: 10
- **Target**: `attack_binary`
- **Artifact**: `models/archive/cybercast_best_model.pt`
"""))

    cells.append(nbf.v4.new_markdown_cell("""## 1. Objective
[DOCUMENTATION]
This notebook documents the recovered historical training logic for the early Phase 2 deep learning baselines, which were found embedded inside a script that generates notebooks (`generate_notebook.py`).
"""))

    cells.append(nbf.v4.new_markdown_cell("""## 2. Source Code Lineage
[NOT-REPRODUCIBLE]
The original standalone Python training script is missing. The logic here was recovered from string literals in `generate_notebook.py`.
"""))

    cells.append(nbf.v4.new_markdown_cell("""## 3. Environment
[DOCUMENTATION]
Requires standard data science stack and PyTorch.
"""))

    cells.append(nbf.v4.new_code_cell("""import sys, os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, DataLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")
"""))

    cells.append(nbf.v4.new_markdown_cell("""## 4-12. Data Pipeline
[HISTORICAL]
Historical preprocessing mapped temporal blocks of length 10 into tensors of `(Batch, 10, 15)`.
"""))
    cells.append(nbf.v4.new_code_cell("""# Data pipeline was historically executed on the raw CSVs, mapping them to 10-second windows.
# X_train shape: (N, 10, 15), y_train shape: (N,)
"""))

    cells.append(nbf.v4.new_markdown_cell("""## 13. Architecture
[SOURCE-REPRODUCED]
Recovered architecture from `generate_notebook.py`.
"""))
    cells.append(nbf.v4.new_code_cell("""class LSTMModel(nn.Module):
    def __init__(self, input_size=15, hidden_size=64, num_layers=2, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.head = nn.Sequential(nn.Linear(hidden_size, 32), nn.ReLU(), nn.Dropout(dropout), nn.Linear(32, 1))
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :])

model = LSTMModel()
print(model)
"""))

    cells.append(nbf.v4.new_markdown_cell("""## 14-17. Training and Validation
[SOURCE-REPRODUCED]
The training loop optimized BCEWithLogitsLoss with Adam.
"""))
    cells.append(nbf.v4.new_code_cell("""# Recovered snippet from `generate_notebook.py`:
'''
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
# ... inside loop ...
loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
optimizer.step()
'''
"""))
    
    cells.append(nbf.v4.new_markdown_cell("""## 25. Artifact Provenance
[FROZEN-ARTIFACT]
The recovered logic was historically used to generate `models/archive/cybercast_best_model.pt`.

## 28. Limitations
[DOCUMENTATION]
This is a historical reconstruction and cannot be natively executed end-to-end to reproduce the exact weights due to the missing data parsing code.
"""))

    nb.cells = cells
    os.makedirs('notebooks/phase2', exist_ok=True)
    with open('notebooks/phase2/Phase2_CyberCast_BestModel.ipynb', 'w') as f:
        nbf.write(nb, f)
    print("Created Phase2_CyberCast_BestModel.ipynb")

if __name__ == '__main__':
    create_nb1()
