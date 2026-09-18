# Python Training Script Map

This document maps all discovered training logic back to `.py` files to provide a source-of-truth reference for where actual operations occur.

## Active Production Scripts
- **`scripts/phase2_3_experiments.py`**: The canonical source for training the Phase 2.3 production pipeline. Generates `model_SET_R_h20.pt`.
- **`scripts/phase2_2_experiments.py`**: The canonical source for training the Phase 2.2 pipeline. Generates `model_20_SET_B.pt`.

## Archive / Legacy Scripts
- **`archive/cybercast_pipeline.py`** (and `archive/nb_cells.py`, `archive/scratch.py`): Legacy scripts that represent the training logic for Phase 2.1 (the Dual-head World Model). These generate `best_world_model.pt` and `scaler.pkl`.
- **`generate_notebook.py`**: A unique script that embeds Phase 2 training code inside string literals to procedurally generate the notebook `CyberCast_Model_From_Scratch.ipynb`.

## Missing Scripts
- **Phase 2 Baseline (`model_SET_A_h20.pt`)**: The original standalone Python script used to train the Phase 2 base models is missing from the repository.

## Evaluation Scripts (Not Training)
- **`scripts/threshold_analysis.py`**: Used exclusively to load `best_world_model.pt` and evaluate the threshold on the validation set.
