# Final Notebook Plan: One-Model, One-Notebook

This document outlines the final notebook structure to achieve the "one-model, one-notebook" standard. The exact training source for each model dictates whether the corresponding notebook will serve as a **FULL REPRODUCTION**, an **ARTIFACT-BACKED DOCUMENTATION** viewer, or a **HISTORICAL RECONSTRUCTION**.

## 1. Phase 2: Exploratory Deep Learning Baselines
- **Notebook**: `notebooks/01_Phase_2_Exploration.ipynb`
- **Phase**: Phase 2
- **Model**: LSTM Baseline (`model_SET_A_h20.pt`, `cybercast_best_model.pt`)
- **Actual Training Source**: MISSING (Synthetically reconstructed in `generate_notebook.py`)
- **Evaluation Source**: Legacy evaluation notebooks / `generate_notebook.py`
- **Artifact**: `models/archive/model_SET_A_h20.pt`, `models/archive/cybercast_best_model.pt`
- **Reproducibility**: Unreproducible from standalone `.py`
- **Purpose**: **HISTORICAL RECONSTRUCTION**. The notebook will document the early experimental process using the string literals recovered from `generate_notebook.py`.

## 2. Phase 2.1: Dual-Head World Model
- **Notebook**: `notebooks/02_Phase_2_1_World_Model.ipynb`
- **Phase**: Phase 2.1
- **Model**: `CyberCastWorldModel`
- **Actual Training Source**: `archive/cybercast_pipeline.py`
- **Evaluation Source**: `scripts/threshold_analysis.py`
- **Artifact**: `models/archive/best_world_model.pt`
- **Reproducibility**: Fully Reproducible
- **Purpose**: **FULL REPRODUCTION**. This notebook will extract the clean training loop and model definition directly from `archive/cybercast_pipeline.py` and replace the large, messy duplicate notebooks (`CyberCast_Complete.ipynb`, etc.).

## 3. Phase 2.2: Feature Sets and Leakage Elimination
- **Notebook**: `notebooks/03_Phase_2_2_Champion.ipynb`
- **Phase**: Phase 2.2
- **Model**: `CyberCastForecaster` (History=20, Features=SET_B)
- **Actual Training Source**: `scripts/phase2_2_experiments.py` (Authoritative)
- **Evaluation Source**: `scripts/phase2_2_experiments.py`
- **Artifact**: `results/phase2_2/models/model_20_SET_B.pt`
- **Reproducibility**: Fully Reproducible
- **Purpose**: **ARTIFACT-BACKED DOCUMENTATION**. The notebook will NOT train the model to avoid duplicating logic. It will invoke or document the results produced by the authoritative python script.

## 4. Phase 2.3: Current Production Pipeline
- **Notebook**: `notebooks/04_Phase_2_3_Production.ipynb`
- **Phase**: Phase 2.3
- **Model**: `CyberCastForecaster` (History=20, Features=SET_R)
- **Actual Training Source**: `scripts/phase2_3_experiments.py` (Authoritative)
- **Evaluation Source**: `scripts/phase2_3_experiments.py`
- **Artifact**: `models/production/model_SET_R_h20.pt`
- **Reproducibility**: Fully Reproducible
- **Purpose**: **ARTIFACT-BACKED DOCUMENTATION**. The python script MUST remain the authoritative source. The notebook will serve strictly to load the artifacts and visualize the champion metrics, training curves, and threshold analysis.

## Summary of Deprecations
The following mega-notebooks and legacy files are marked as **DUPLICATE** or **LEGACY** and should be archived or deleted in the next step:
- `CyberCast_Model_From_Scratch.ipynb`
- `CyberCast_All_Pipelines.ipynb`
- `CyberCast_Complete.ipynb`
- `CyberCast_Complete_executed.ipynb`
- `CyberCast_All_Phases.ipynb`
- `archive/nb_cells.py`
- `archive/scratch.py`
