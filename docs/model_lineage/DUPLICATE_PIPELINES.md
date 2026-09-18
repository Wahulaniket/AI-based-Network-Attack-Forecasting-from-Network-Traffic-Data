# Duplicate Pipelines

## Overview
This document records instances of redundant or duplicate scripts across the CyberCast repository. 

## Python Scripts vs Notebooks
During the migration and cleanup phase, some notebooks were found to have identically named `.py` equivalents or functionally identical logic scattered across the repository.

1. **`models/train.py` vs `scripts/phase2_3_experiments.py`**:
   - `models/train.py` exists but is currently EMPTY (0 bytes). 
   - All actual training occurs in the explicit `scripts/phase2_x_experiments.py` files.
   
2. **`models/baseline.py`**:
   - Exists but is currently EMPTY (0 bytes).

3. **`notebooks/archive/` Notebooks**:
   - `01_Phase_2_Model_Training.ipynb`
   - `01_Phase_2_1_Experiments_Training.ipynb`
   - `01_Phase_2_2_Experiments_Training.ipynb`
   - `01_Phase_2_3_Training.ipynb`
   - **Resolution**: None of these contain actual training loops; they are merely result viewers. The source of truth for Phase 2.2 and 2.3 lies exclusively in the `scripts/` directory.

## Threshold Analysis
- `scripts/threshold_analysis.py` contains the complete script to evaluate the Dual-head LSTM World Model (Phase 2.1). This script is comprehensive and acts as the singular source of truth for threshold evaluation on the validation set.
