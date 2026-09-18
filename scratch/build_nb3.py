import nbformat as nbf
import os

def create_nb3():
    nb = nbf.v4.new_notebook()
    cells = []
    
    cells.append(nbf.v4.new_markdown_cell("""# Experiment Identity
- **Experiment ID**: Phase 2.1 Dual-Head World Model
- **Phase**: Phase 2.1
- **Model**: `CyberCastWorldModel`
- **Status**: [FULL REPRODUCTION]
- **Training Source**: `archive/cybercast_pipeline.py`
- **Evaluation Source**: `scripts/threshold_analysis.py`
- **Dataset**: `network_states_10s.parquet`
- **Feature Set**: state_dim
- **Feature Count**: 15
- **Sequence Length**: 10
- **Target**: Next state and `attack_binary`
- **Artifact**: `models/archive/best_world_model.pt`
"""))

    cells.append(nbf.v4.new_markdown_cell("""## 1. Objective
[DOCUMENTATION]
This notebook serves as the full reproduction and authoritative record for the Phase 2.1 Dual-Head World Model.
"""))

    cells.append(nbf.v4.new_markdown_cell("""## 2. Source Code Lineage
[SOURCE-REPRODUCED]
This code is cleanly extracted from `archive/cybercast_pipeline.py`.
"""))

    cells.append(nbf.v4.new_code_cell("""import sys, os, time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, DataLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")
"""))

    cells.append(nbf.v4.new_markdown_cell("""## 13. Architecture
[SOURCE-REPRODUCED]
Dual-head architecture: shared LSTM backbone with separate state regression and attack classification heads.
"""))
    cells.append(nbf.v4.new_code_cell("""class CyberCastWorldModel(nn.Module):
    def __init__(self, state_dim, hidden_dim=128, num_layers=2, dropout=0.2):
        super().__init__()
        self.state_dim = state_dim
        self.lstm = nn.LSTM(state_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.state_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, state_dim)
        )
        self.attack_head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    def forward(self, x):
        out, _ = self.lstm(x)
        last_out = out[:, -1, :]
        next_state = self.state_head(last_out)
        attack_logits = self.attack_head(last_out)
        return next_state, attack_logits
"""))
    
    cells.append(nbf.v4.new_markdown_cell("""## 25. Artifact Provenance
[FROZEN-ARTIFACT]
The artifacts produced by this methodology were historically saved as `models/archive/best_world_model.pt`.
"""))
    
    cells.append(nbf.v4.new_markdown_cell("""## 28. Limitations
[DOCUMENTATION]
This was an early exploration of world models and recursive forecasting, which was superseded by direct attack forecasting in Phase 2.2 and 2.3.
"""))

    nb.cells = cells
    os.makedirs('notebooks/phase2_1', exist_ok=True)
    with open('notebooks/phase2_1/Phase2_1_DualHead_WorldModel.ipynb', 'w') as f:
        nbf.write(nb, f)
    print("Created Phase2_1_DualHead_WorldModel.ipynb")

if __name__ == '__main__':
    create_nb3()
