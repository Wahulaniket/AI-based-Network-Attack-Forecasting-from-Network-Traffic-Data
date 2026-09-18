# Training Source Verification Audit

## 1. Goal
This document records the definitive, actual `.py` source that trained each CyberCast model artifact. This audit distinguishes between evaluation sources (scripts that load a model to score it) and training sources (scripts that run `loss.backward()` and `torch.save()`).

## 2. Phase 2 Analysis
- **Artifacts**: `models/archive/model_*.pt` (e.g., `model_10_SET_A.pt`), `models/archive/cybercast_best_model.pt`
- **Training Source**: MISSING (for `model_SET_A_h20.pt` and variants). The original Python scripts that produced these artifacts no longer exist in the repository. The file `generate_notebook.py` contains string literals of training code that it uses to generate `notebooks/archive/CyberCast_Model_From_Scratch.ipynb` which in turn trains `cybercast_best_model.pt`. However, a clean standalone Python script is absent.
- **Evaluation Source**: The notebooks named `01_Phase_2_Model_Training.ipynb` and `02_Phase_2_Model_Evaluation.ipynb` strictly load pre-computed CSV files. They do NOT contain `torch.save`, `optimizer.step()`, or `model.fit`. They are **DOCUMENTATION ONLY**.

## 3. Phase 2.1 Analysis
- **Artifacts**: `models/archive/best_world_model.pt`, `models/archive/scaler.pkl`, `models/archive/logistic_regression.pkl`
- **Training Source**: `archive/cybercast_pipeline.py` and `archive/nb_cells.py`. A direct text search reveals that these files perform actual training operations (`loss.backward()`, `joblib.dump()`, `torch.save()`) targeting these specific filenames.
- **Evaluation Source**: `scripts/threshold_analysis.py`. This script purely loads the model using `torch.load()` to perform evaluation on the validation set. It does not train the model.
- **Notebooks**: `CyberCast_Complete.ipynb` and `CyberCast_All_Pipelines.ipynb` contain actual training code, but `01_Phase_2_1_Experiments_Training.ipynb` is strictly documentation.

## 4. Phase 2.2 Analysis
- **Artifacts**: `results/phase2_2/models/model_20_SET_B.pt`
- **Training Source**: `scripts/phase2_2_experiments.py`. This script is explicitly verified to contain `loss.backward()`, optimizer steps, and the `torch.save()` call saving models to `results/phase2_2/models/`. It controls the strict chronological splits and feature selections (SET_A, SET_B, SET_C).
- **Notebooks**: `01_Phase_2_2_Experiments_Training.ipynb` is strictly documentation.

## 5. Phase 2.3 Analysis (Current Production)
- **Artifacts**: `models/production/model_SET_R_h20.pt`, `models/production/scaler_SET_R_h20.joblib`
- **Training Source**: `scripts/phase2_3_experiments.py`. This script defines the entire pipeline from raw data aggregation, feature engineering (SET_R containing 89 features), chronological splits (Train/Val/Test), training loop, and evaluation.
- **Notebooks**: `01_Phase_2_3_Training.ipynb` and `Phase_2_3_Complete_Analysis.ipynb` are documentation/evaluation viewers. The true source of truth is the python script.

## 6. Audit Conclusions
- Do not trust notebook filenames. "Training" in a notebook name often just meant "Visualizing Training Results".
- The only way to prove a model's origin is tracing `torch.save()` in a `.py` file.
- Phase 2 and 2.1 training source code was mostly relocated into the `archive/` directory during cleanup.
- Phase 2.2 and 2.3 pipelines are clean, well-defined, and fully reproducible via `scripts/phase2_x_experiments.py`.
