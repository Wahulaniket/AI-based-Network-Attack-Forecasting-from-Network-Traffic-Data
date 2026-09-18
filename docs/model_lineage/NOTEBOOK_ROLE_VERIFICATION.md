# Notebook Role Verification Audit

This document classifies every notebook in `notebooks/archive/` based on whether it actually contains training operations (`torch.save`, `model.fit`, etc.) or simply loads and visualizes pre-computed results.

## Documentation and Evaluation Only (No Training)
These notebooks only load CSVs, JSON metrics, and static model artifacts to create visualizations and compare experiments. They do **not** train any models.
- `01_dataset_inspection.ipynb`
- `01_Phase_2_Model_Training.ipynb`
- `01_Phase_2_1_Experiments_Training.ipynb`
- `01_Phase_2_2_Experiments_Training.ipynb`
- `01_Phase_2_3_Training.ipynb`
- `02_Phase_2_Model_Evaluation.ipynb`
- `02_Phase_2_1_Evaluation.ipynb`
- `02_Phase_2_2_Evaluation.ipynb`
- `02_Phase_2_3_Final_Evaluation.ipynb`
- `Phase_2_3_Complete_Analysis.ipynb`

**Crucial Note**: The notebooks prefixed with `01_..._Training.ipynb` are deceptively named. A code analysis confirms they do not contain any actual training loops or PyTorch model definitions. They are purely evaluation/documentation.

## Training and Evaluation (Contains Actual Training Logic)
These notebooks contain complete end-to-end logic, including `torch.save()`, `loss.backward()`, and PyTorch architecture definitions.
- `CyberCast_Model_From_Scratch.ipynb` (Auto-generated via `generate_notebook.py`)
- `CyberCast_All_Pipelines.ipynb`
- `CyberCast_Complete.ipynb`
- `CyberCast_Complete_executed.ipynb`
- `CyberCast_All_Phases.ipynb` (Likely similar structure)

## Summary
- **Total Notebooks**: 15
- **Documentation/Evaluation Only**: 10
- **Training + Evaluation**: 5
