# Phase 2 Lineage Audit

## Overview
Phase 2 focused on exploring various model architectures (LSTM, Attention LSTM, TCN, GRU, Transformer, Logistic Regression, XGBoost, Random Forest) for next-window attack forecasting using a rich feature representation.

## Data Pipeline
- **Dataset**: Transformed into 10-second temporal windows.
- **Features**: ~70 features.
- **Sequence Length**: Generally 20 historical windows.

## Artifacts
- **Models**: Located in `models/archive/` (e.g., `model_SET_A_h20.pt`, `model_10_SET_A.pt`).
- **Results**: Evaluated in `notebooks/archive/01_Phase_2_Model_Training.ipynb` and `notebooks/archive/02_Phase_2_Model_Evaluation.ipynb`.
- **Performance**: The LSTM baseline achieved a Test PR-AUC of 0.7371 (as referenced in historical comparisons).

## Lineage Gaps
- **Training Script**: MISSING. The notebooks in `notebooks/archive/` claiming to be training scripts only load pre-computed results from CSV files (e.g., `results/model_comparison.csv`) and plot them. The actual code that produced these models is no longer present in the repository.
- **Status**: LEGACY.
