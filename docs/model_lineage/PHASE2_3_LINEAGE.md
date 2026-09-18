# Phase 2.3 Lineage Audit

## Overview
Phase 2.3 is the **current production pipeline**. It features a robust feature audit and strict chronological splitting to ensure zero target leakage.

## Data Pipeline
- **Dataset**: `data/processed/network_states_10s.parquet`
- **Features**: 
  - `SET_A`: 15 baseline features.
  - `SET_R`: 89 recovered and validated features (all target-derived features explicitly excluded).
  - `SET_R_SELECT`: Top 40 features from SET_R (selected via VarianceThreshold and Mutual Information strictly on train).
  - Saved to `results/phase2_3/feature_sets.json`.
- **Data Splitting**: Strict chronological split (Train: 0-80%, Val: 80-90%, Test: 90-100%).
- **Scalers**: `StandardScaler` fitted strictly on train.

## Artifacts
- **Training Script**: `scripts/phase2_3_experiments.py`
- **Models**: The champion model (`model_SET_R_h20.pt`) and its scaler (`scaler_SET_R_h20.joblib`) were migrated to `models/production/`. 
- **Results**: 
  - `results/phase2_3/feature_audit.csv`
  - `results/phase2_3/feature_comparison.csv`
  - `results/phase2_3/history_comparison.csv`
  - `results/phase2_3/model_comparison.csv`
  - `results/phase2_3/champion_metrics.json`
- **Champion Model**: The model with `History=20` and `Features=SET_R` achieved an estimated Test PR-AUC of 0.6887.
- **Status**: CONFIRMED. Fully reproducible via `scripts/phase2_3_experiments.py` and forms the current frozen production model.
