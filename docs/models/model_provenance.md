# Model Provenance

## Overview
This document maps the evolution of models in the CyberCast repository.

## Production Frozen
- `models/phase2_3/model_SET_R_h20.pt`: The final Phase 2.3 LSTM champion.

## Baselines
Models retained for scientific comparison (e.g. SET_A, SET_R_SELECT).
- `models/phase2_3/model_SET_A_h20.pt`
- `models/phase2_3/model_SET_R_SELECT_h20.pt`
- `models/phase2_3/scaler_SET_A_h20.joblib`
- `models/phase2_3/scaler_SET_R_SELECT_h20.joblib`

## Experimental
Models trained during architecture/hyperparameter search (Phase 2.2 and Phase 2.3 history/reproducibility experiments).
These should be archived, not deleted, to preserve scientific reproducibility.

## Legacy & Orphans
Old models from Phase 1/2.1 or models with no references in code. Can be safely archived to a legacy folder.
