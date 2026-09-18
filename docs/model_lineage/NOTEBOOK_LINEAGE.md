# Notebook Lineage Analysis

## Overview
This document catalogs the state of Jupyter notebooks found in `notebooks/archive/` and addresses reproducibility concerns.

## Legacy Notebooks
The following notebooks were reviewed:
- `01_Phase_2_Model_Training.ipynb`
- `01_Phase_2_1_Experiments_Training.ipynb`
- `01_Phase_2_2_Experiments_Training.ipynb`
- `01_Phase_2_3_Training.ipynb`

## Analysis Results
**ALL of the above notebooks are non-reproducible for training purposes.**
Despite their naming conventions containing "Training", none of these notebooks actually contain the PyTorch model definitions, dataset parsing, or training loops used to generate the models. 

Instead, these notebooks simply:
1. Load CSV artifacts (e.g., `results/model_comparison.csv`, `results/phase2_1/experiment_comparison.csv`).
2. Generate tabular displays (`display(df)`).
3. Plot static charts (`sns.barplot(...)`).

## Conclusion
The notebooks are strictly for **results visualization** and do not serve as canonical training sources. Formal training code exists only in the python scripts (`scripts/phase2_2_experiments.py`, `scripts/phase2_3_experiments.py`) for the latter phases. Training code for Phase 2 and 2.1 is definitively absent from the repository.
