import nbformat as nbf
import os

def create_nb5():
    nb = nbf.v4.new_notebook()
    cells = []
    
    cells.append(nbf.v4.new_markdown_cell("""# Experiment Identity
- **Experiment ID**: Phase 2.3 SET_R H20 Production
- **Phase**: Phase 2.3
- **Model**: `CyberCastForecaster`
- **Status**: [ARTIFACT-BACKED DOCUMENTATION]
- **Training Source**: `scripts/phase2_3_experiments.py`
- **Evaluation Source**: `scripts/phase2_3_experiments.py`
- **Dataset**: `network_states_10s.parquet`
- **Feature Set**: SET_R
- **Feature Count**: 89
- **Sequence Length**: 20
- **Target**: `binary_attack`
- **Artifact**: `models/production/model_SET_R_h20.pt`
"""))

    cells.append(nbf.v4.new_markdown_cell("""## 1. Objective
[DOCUMENTATION]
This notebook documents the final production pipeline. The Python script MUST remain the authoritative source.
"""))

    cells.append(nbf.v4.new_markdown_cell("""## 4-5. Dataset and Data Cleaning
[HISTORICAL]
- Raw flows: 16,233,002
- Invalid timestamps: 73
- Clean flows: 16,232,929
"""))

    cells.append(nbf.v4.new_markdown_cell("""## 25. Artifact Provenance
[FROZEN-ARTIFACT]
- **Model Checkpoint**: `models/production/model_SET_R_h20.pt`
- **Scaler**: `models/production/scaler_SET_R_h20.joblib`
- **Features**: `results/phase2_3/feature_sets.json`
- **Metrics**: `results/phase2_3/champion_metrics.json`
"""))

    cells.append(nbf.v4.new_code_cell("""import json
import os

metrics_path = '../../results/phase2_3/champion_metrics.json'
if os.path.exists(metrics_path):
    with open(metrics_path, 'r') as f:
        metrics = json.load(f)
    print("FROZEN FINAL TEST RESULT")
    print("------------------------")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"{k}: {v:.4f}")
        else:
            print(f"{k}: {v}")
else:
    print("Metrics artifact not found. Please ensure it exists.")
"""))

    cells.append(nbf.v4.new_markdown_cell("""## 26. Final Results
[FROZEN-ARTIFACT]
Final verified test metrics (loaded from JSON):
PR-AUC: 0.7309
ROC-AUC: 0.8878
F1: 0.7018
Precision: 0.8046
Recall: 0.6222
Accuracy: 0.9176
FPR: 0.0279
FNR: 0.3778
TP: 280
TN: 2370
FP: 68
FN: 170
"""))

    nb.cells = cells
    os.makedirs('notebooks/phase2_3', exist_ok=True)
    with open('notebooks/phase2_3/Phase2_3_SET_R_H20_Production.ipynb', 'w') as f:
        nbf.write(nb, f)
    print("Created Phase2_3_SET_R_H20_Production.ipynb")

if __name__ == '__main__':
    create_nb5()
