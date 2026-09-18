# Phase 2.2 Lineage Audit

## Overview
Phase 2.2 aimed to eliminate chronological leakage by creating explicitly defined feature sets (SET_A, SET_B, SET_C) and testing different history lengths. 

## Data Pipeline
- **Dataset**: `data/processed/network_states_10s.parquet`
- **Features**: 
  - `SET_A`: 15 base features.
  - `SET_B`: 15 base features + 15 selected features via VarianceThreshold and Mutual Information (fitted strictly on train).
  - `SET_C`: SET_B + 5 explicit causal difference features.
- **Data Splitting**: Strict chronological split (Train: 0-80%, Val: 80-90%, Test: 90-100%).

## Artifacts
- **Training Script**: `scripts/phase2_2_experiments.py`
- **Models**: Saved in `results/phase2_2/models/` (e.g., `model_20_SET_B.pt`).
- **Results**: 
  - `results/phase2_2/phase2_2_results.csv` (Experiment results)
  - `results/phase2_2/champion_metrics.json`
- **Champion Model**: The model with `History=20` and `Features=SET_B` achieved a Test PR-AUC of 0.5571. 
- **Status**: CONFIRMED. Fully reproducible via `scripts/phase2_2_experiments.py`.
