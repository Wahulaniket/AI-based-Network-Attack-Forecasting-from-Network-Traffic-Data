# Model Migration Plan

| Current Path | Proposed Path | Classification | Action | Reason |
|--------------|---------------|----------------|--------|--------|
| models/best_world_model.pt | models/archive/legacy/best_world_model.pt | LEGACY | ARCHIVE | Clean up root/models directory |
| models/cybercast_best_model.pt | models/archive/legacy/cybercast_best_model.pt | LEGACY | ARCHIVE | Clean up root/models directory |
| models/phase2_1/best_model.pt | models/archive/legacy/best_model.pt | LEGACY | ARCHIVE | Clean up root/models directory |
| models/phase2_3/model_SET_A_h20.pt | models/archive/phase_2.3/model_SET_A_h20.pt | BASELINE | ARCHIVE | Preserve for reproducibility |
| models/phase2_3/model_SET_R_h10.pt | models/archive/phase_2.3/model_SET_R_h10.pt | EXPERIMENTAL | ARCHIVE | Preserve for reproducibility |
| models/phase2_3/model_SET_R_h20.pt | models/production/model_SET_R_h20.pt | PRODUCTION_FROZEN | MOVE | Isolate production artifacts |
| models/phase2_3/model_SET_R_h30.pt | models/archive/phase_2.3/model_SET_R_h30.pt | EXPERIMENTAL | ARCHIVE | Preserve for reproducibility |
| models/phase2_3/model_SET_R_h5.pt | models/archive/phase_2.3/model_SET_R_h5.pt | EXPERIMENTAL | ARCHIVE | Preserve for reproducibility |
| models/phase2_3/model_SET_R_Repro_1_h20.pt | models/archive/phase_2.3/model_SET_R_Repro_1_h20.pt | EXPERIMENTAL | ARCHIVE | Preserve for reproducibility |
| models/phase2_3/model_SET_R_Repro_2_h20.pt | models/archive/phase_2.3/model_SET_R_Repro_2_h20.pt | EXPERIMENTAL | ARCHIVE | Preserve for reproducibility |
| models/phase2_3/model_SET_R_SELECT_h20.pt | models/archive/phase_2.3/model_SET_R_SELECT_h20.pt | BASELINE | ARCHIVE | Preserve for reproducibility |
| results/phase2_2/models/model_10_SET_A.pt | models/archive/phase_2.2/model_10_SET_A.pt | EXPERIMENTAL | ARCHIVE | Preserve for reproducibility |
| results/phase2_2/models/model_10_SET_B.pt | models/archive/phase_2.2/model_10_SET_B.pt | EXPERIMENTAL | ARCHIVE | Preserve for reproducibility |
| results/phase2_2/models/model_10_SET_C.pt | models/archive/phase_2.2/model_10_SET_C.pt | EXPERIMENTAL | ARCHIVE | Preserve for reproducibility |
| results/phase2_2/models/model_20_SET_A.pt | models/archive/phase_2.2/model_20_SET_A.pt | EXPERIMENTAL | ARCHIVE | Preserve for reproducibility |
| results/phase2_2/models/model_20_SET_B.pt | models/archive/phase_2.2/model_20_SET_B.pt | EXPERIMENTAL | ARCHIVE | Preserve for reproducibility |
| results/phase2_2/models/model_20_SET_C.pt | models/archive/phase_2.2/model_20_SET_C.pt | EXPERIMENTAL | ARCHIVE | Preserve for reproducibility |
| results/phase2_2/models/model_30_SET_A.pt | models/archive/phase_2.2/model_30_SET_A.pt | EXPERIMENTAL | ARCHIVE | Preserve for reproducibility |
| results/phase2_2/models/model_30_SET_B.pt | models/archive/phase_2.2/model_30_SET_B.pt | EXPERIMENTAL | ARCHIVE | Preserve for reproducibility |
| results/phase2_2/models/model_30_SET_C.pt | models/archive/phase_2.2/model_30_SET_C.pt | EXPERIMENTAL | ARCHIVE | Preserve for reproducibility |
| results/phase2_2/models/model_5_SET_A.pt | models/archive/phase_2.2/model_5_SET_A.pt | EXPERIMENTAL | ARCHIVE | Preserve for reproducibility |
| results/phase2_2/models/model_5_SET_B.pt | models/archive/phase_2.2/model_5_SET_B.pt | EXPERIMENTAL | ARCHIVE | Preserve for reproducibility |
| results/phase2_2/models/model_5_SET_C.pt | models/archive/phase_2.2/model_5_SET_C.pt | EXPERIMENTAL | ARCHIVE | Preserve for reproducibility |
| models/phase2_3/scaler_SET_A_h20.joblib | models/archive/phase_2.3/scaler_SET_A_h20.joblib | BASELINE | ARCHIVE | Preserve for reproducibility |
| models/phase2_3/scaler_SET_R_h10.joblib | models/archive/phase_2.3/scaler_SET_R_h10.joblib | EXPERIMENTAL | ARCHIVE | Preserve for reproducibility |
| models/phase2_3/scaler_SET_R_h20.joblib | DELETE | DUPLICATE | DELETE | Identical hash exists |
| models/phase2_3/scaler_SET_R_h30.joblib | DELETE | DUPLICATE | DELETE | Identical hash exists |
| models/phase2_3/scaler_SET_R_h5.joblib | DELETE | DUPLICATE | DELETE | Identical hash exists |
| models/phase2_3/scaler_SET_R_Repro_1_h20.joblib | DELETE | DUPLICATE | DELETE | Identical hash exists |
| models/phase2_3/scaler_SET_R_Repro_2_h20.joblib | DELETE | DUPLICATE | DELETE | Identical hash exists |
| models/phase2_3/scaler_SET_R_SELECT_h20.joblib | models/archive/phase_2.3/scaler_SET_R_SELECT_h20.joblib | BASELINE | ARCHIVE | Preserve for reproducibility |
| models/feature_scaler.pkl | models/archive/legacy/feature_scaler.pkl | LEGACY | ARCHIVE | Clean up root/models directory |
| models/logistic_regression.pkl | models/archive/legacy/logistic_regression.pkl | LEGACY | ARCHIVE | Clean up root/models directory |
| models/scaler.pkl | models/archive/legacy/scaler.pkl | LEGACY | ARCHIVE | Clean up root/models directory |
