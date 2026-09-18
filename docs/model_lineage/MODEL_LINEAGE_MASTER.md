# CyberCast Model Lineage Master Index

This directory contains the complete lineage audit of all models and data pipelines across CyberCast's experimental phases (Phase 2 through Phase 2.3). 

## Included Documents
1. **[MODEL_LINEAGE_TABLE.csv](file:///d:/working_projects/SIH/cyberCast/docs/model_lineage/MODEL_LINEAGE_TABLE.csv)** - A tabular summary of all significant models across all phases.
2. **[PHASE2_LINEAGE.md](file:///d:/working_projects/SIH/cyberCast/docs/model_lineage/PHASE2_LINEAGE.md)** - Audit of the initial Phase 2 model training (Exploration of architectures).
3. **[PHASE2_1_LINEAGE.md](file:///d:/working_projects/SIH/cyberCast/docs/model_lineage/PHASE2_1_LINEAGE.md)** - Audit of Phase 2.1 (Threshold Analysis and Dual-head LSTM World Model).
4. **[PHASE2_2_LINEAGE.md](file:///d:/working_projects/SIH/cyberCast/docs/model_lineage/PHASE2_2_LINEAGE.md)** - Audit of Phase 2.2 (Feature set and chronological leakage elimination).
5. **[PHASE2_3_LINEAGE.md](file:///d:/working_projects/SIH/cyberCast/docs/model_lineage/PHASE2_3_LINEAGE.md)** - Audit of Phase 2.3 (Current Production Pipeline).
6. **[DATA_PIPELINE_LINEAGE.md](file:///d:/working_projects/SIH/cyberCast/docs/model_lineage/DATA_PIPELINE_LINEAGE.md)** - Map of how data flows into features, splits, and models.
7. **[NOTEBOOK_LINEAGE.md](file:///d:/working_projects/SIH/cyberCast/docs/model_lineage/NOTEBOOK_LINEAGE.md)** - Analysis of legacy notebooks and their reproducibility.
8. **[DUPLICATE_PIPELINES.md](file:///d:/working_projects/SIH/cyberCast/docs/model_lineage/DUPLICATE_PIPELINES.md)** - Documentation of redundant scripts.
9. **[LINEAGE_GAPS.md](file:///d:/working_projects/SIH/cyberCast/docs/model_lineage/LINEAGE_GAPS.md)** - Explicit list of missing or unconfirmed training sources.

## Current Production System
The current frozen model used for inference is the **Phase 2.3 Champion**.
- **Model Checkpoint**: `models/production/model_SET_R_h20.pt`
- **Scaler**: `models/production/scaler_SET_R_h20.joblib`
- **Feature Set**: `results/phase2_3/feature_sets.json` (SET_R: 89 features)
- **Training Script**: `scripts/phase2_3_experiments.py`
