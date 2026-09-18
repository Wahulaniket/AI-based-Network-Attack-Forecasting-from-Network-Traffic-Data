# Lineage Gaps

This document identifies missing or unconfirmed training sources across the repository.

## Unrecoverable Artifacts
1. **Phase 2 Models (`models/archive/model_*.pt`)**: 
   - **Status**: MISSING
   - **Description**: Dozens of model checkpoints exist in `models/archive/` (e.g., `model_10_SET_A.pt`, `cybercast_best_model.pt`). However, the training script that generated these models no longer exists in the repository. The notebooks (e.g., `01_Phase_2_Model_Training.ipynb`) only load aggregate CSV results.
   
2. **Phase 2.1 Dual-Head World Model (`models/archive/best_world_model.pt`)**:
   - **Status**: MISSING
   - **Description**: This model is explicitly loaded and evaluated in `scripts/threshold_analysis.py`, but the script used to train this specific checkpoint with `state_dim` features is missing.

3. **Phase 2.1 Scaler and Logistic Regression Baseline**:
   - **Status**: MISSING
   - **Description**: `models/archive/scaler.pkl` and `models/archive/logistic_regression.pkl` are used during evaluation in `threshold_analysis.py`, but the code that fit them is absent.

## Confirmed Provenance
- **Phase 2.2**: Training source is strictly confirmed via `scripts/phase2_2_experiments.py`.
- **Phase 2.3 (Production)**: Training source is strictly confirmed via `scripts/phase2_3_experiments.py`.

## Summary
The migration/cleanup has left Phase 2 and Phase 2.1 without their original training source code. However, the current production pipeline (Phase 2.3) and its immediate predecessor (Phase 2.2) are fully reproducible from source.
