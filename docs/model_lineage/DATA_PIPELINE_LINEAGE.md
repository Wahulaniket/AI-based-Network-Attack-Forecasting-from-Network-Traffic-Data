# Data Pipeline Lineage

This document traces the data lifecycle for the Phase 2.2 and Phase 2.3 pipelines.

## Raw Data to Processed
- **Source**: `data/processed/network_states_10s.parquet` is the definitive starting point for all Phase 2.x experiments.
- **Aggregation**: Data is aggregated into 10-second temporal windows.

## Train/Val/Test Splits
All confirmed pipelines (Phase 2.2 and Phase 2.3) use strict chronological splits to prevent data leakage:
- **Train**: First 80% (or 70% in Phase 2.1 evaluation) of the chronological data.
- **Validation**: Next 10% (or 15%).
- **Test**: Final 10% (or 15%).

## Feature Engineering and Selection
- **Phase 2.2 (`scripts/phase2_2_experiments.py`)**: 
  - `SET_A`: 15 base features.
  - `SET_B`: 30 features total (15 base + 15 selected via VarianceThreshold/Mutual Info strictly on Train).
  - `SET_C`: 35 features total (`SET_B` + 5 explicit causal diff features like `delta_Tot Fwd Pkts`).
- **Phase 2.3 (`scripts/phase2_3_experiments.py`)**:
  - `SET_R`: 89 recovered features, systematically excluding target-derived features (`binary_attack`, `dominant_label`, `attack_ratio`, `has_traffic`).
  - `SET_R_SELECT`: Top 40 features derived from `SET_R` via VarianceThreshold and Mutual Info (strictly on Train).
  - **Feature Tracking**: Saved formally to `results/phase2_3/feature_sets.json` to ensure reproducible inference.

## Scaling
- **StandardScaler**: Exclusively fitted on the Training split.
- **Phase 2.3 Production**: The resulting scaler is saved as `models/production/scaler_SET_R_h20.joblib`.

## Sequence Generation
- A `SequenceDataset` maps temporal blocks of length $H$ into tensors `(Batch, History, Features)`.
- The target $y$ is strictly selected at time $t+H$ to predict the next window without future overlap.
