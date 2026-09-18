# Phase 2.1 Lineage Audit

## Overview
Phase 2.1 focused on Threshold Analysis and the evaluation of a Dual-head LSTM World Model for network state forecasting. 

## Data Pipeline
- **Dataset**: `data/processed/network_states_10s.parquet`
- **Features**: Features are specified in `models/feature_names.json`.
- **Sequence Length**: 10 historical windows.
- **Forecast Horizon**: 5 steps ahead.

## Artifacts
- **Model**: `models/archive/best_world_model.pt` (Dual-head LSTM World Model).
- **Evaluation Script**: `scripts/threshold_analysis.py`.
- **Performance**: 
  - Validates threshold operating points strictly on the Validation set. 
  - Produces `results/threshold_analysis_val.csv` and `results/plots/threshold_analysis.png`.

## Lineage Gaps
- **Training Script**: MISSING. Similar to Phase 2, the notebook `notebooks/archive/01_Phase_2_1_Experiments_Training.ipynb` only loads `results/phase2_1/experiment_comparison.csv` and displays results. The actual training script that produced `best_world_model.pt` is missing from the repository.
- **Status**: LEGACY.
