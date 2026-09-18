# Final Migration Plan

## Migration Sequence
1. **Create destination directories** (e.g. `archive/`, `models/production/`)
2. **Move documentation** into `docs/`
3. **Move research scripts** to `scripts/` or `archive/`
4. **Move experiment/baseline/legacy models** to respective `models/` subdirectories
5. **Move production model** to `models/production/`
6. **Update imports and paths** in `src/` and `dashboard/`
7. **Update configuration references**
8. **Run tests**
9. **Run dashboard build**
10. **Verify production model chain**
11. **Verify hashes**
12. **Verify no broken references**

## Complete File Map
| Current Path | Category | Pipeline | Action | Proposed Path | Risk |
|--------------|----------|----------|--------|---------------|------|
| audit_metrics.json | UNKNOWN | None | REVIEW | audit_metrics.json | LOW |
| audit_script.py | UNKNOWN | None | REVIEW | audit_script.py | LOW |
| combine_notebooks.py | NOTEBOOK | None | ARCHIVE | notebooks/archive/combine_notebooks.py | LOW |
| create_notebook.py | LEGACY | None | ARCHIVE | archive/create_notebook.py | LOW |
| create_pipeline_notebook.py | UNKNOWN | None | REVIEW | create_pipeline_notebook.py | LOW |
| cybercast_pipeline.py | LEGACY | None | ARCHIVE | archive/cybercast_pipeline.py | LOW |
| dump_nb.ps1 | TEMPORARY | None | ARCHIVE | archive/dump_nb.ps1 | LOW |
| fix.ps1 | TEMPORARY | None | ARCHIVE | archive/fix.ps1 | LOW |
| fix.py | TEMPORARY | None | ARCHIVE | archive/fix.py | LOW |
| fix23.py | TEMPORARY | None | ARCHIVE | archive/fix23.py | LOW |
| fix_memory.ps1 | TEMPORARY | None | ARCHIVE | archive/fix_memory.ps1 | LOW |
| generate_notebook.py | UNKNOWN | None | REVIEW | generate_notebook.py | LOW |
| gen_phase2.py | LEGACY | None | ARCHIVE | archive/gen_phase2.py | LOW |
| gen_phase2_1.py | LEGACY | None | ARCHIVE | archive/gen_phase2_1.py | LOW |
| gen_phase2_2.py | LEGACY | None | ARCHIVE | archive/gen_phase2_2.py | LOW |
| gen_phase2_3.py | LEGACY | None | ARCHIVE | archive/gen_phase2_3.py | LOW |
| README.md | DOCUMENTATION | None | KEEP | README.md | LOW |
| requirements.txt | CONFIG | None | KEEP | requirements.txt | LOW |
| scratch.py | TEMPORARY | None | ARCHIVE | archive/scratch.py | LOW |
| test_flows.py | TEST | None | KEEP | test_flows.py | LOW |
| test_scheduler.py | TEST | None | KEEP | test_scheduler.py | LOW |
| configs\config.py | UNKNOWN | None | REVIEW | configs/config.py | LOW |
| configs\config.yaml | UNKNOWN | None | REVIEW | configs/config.yaml | LOW |
| configs\model_config.yaml | UNKNOWN | None | REVIEW | configs/model_config.yaml | LOW |
| dashboard\app.py | UNKNOWN | None | REVIEW | dashboard/app.py | LOW |
| dashboard\ui\.oxlintrc.json | UNKNOWN | None | REVIEW | dashboard/ui/.oxlintrc.json | LOW |
| dashboard\ui\index.html | UNKNOWN | None | REVIEW | dashboard/ui/index.html | LOW |
| dashboard\ui\package-lock.json | DASHBOARD | PIPELINE F | KEEP | dashboard/ui/package-lock.json | LOW |
| dashboard\ui\package.json | DASHBOARD | PIPELINE F | KEEP | dashboard/ui/package.json | LOW |
| dashboard\ui\README.md | DOCUMENTATION | None | KEEP | dashboard/ui/README.md | LOW |
| dashboard\ui\tsconfig.app.json | UNKNOWN | None | REVIEW | dashboard/ui/tsconfig.app.json | LOW |
| dashboard\ui\tsconfig.json | UNKNOWN | None | REVIEW | dashboard/ui/tsconfig.json | LOW |
| dashboard\ui\tsconfig.node.json | UNKNOWN | None | REVIEW | dashboard/ui/tsconfig.node.json | LOW |
| dashboard\ui\vite.config.ts | UNKNOWN | None | REVIEW | dashboard/ui/vite.config.ts | LOW |
| dashboard\ui\dist\favicon.svg | UNKNOWN | None | REVIEW | dashboard/ui/dist/favicon.svg | LOW |
| dashboard\ui\dist\icons.svg | UNKNOWN | None | REVIEW | dashboard/ui/dist/icons.svg | LOW |
| dashboard\ui\dist\index.html | UNKNOWN | None | REVIEW | dashboard/ui/dist/index.html | LOW |
| dashboard\ui\dist\assets\index-C-Sye3Ls.js | UNKNOWN | None | REVIEW | dashboard/ui/dist/assets/index-C-Sye3Ls.js | LOW |
| dashboard\ui\dist\assets\index-OY4EDgWk.css | UNKNOWN | None | REVIEW | dashboard/ui/dist/assets/index-OY4EDgWk.css | LOW |
| dashboard\ui\public\favicon.svg | DASHBOARD | PIPELINE F | KEEP | dashboard/ui/public/favicon.svg | LOW |
| dashboard\ui\public\icons.svg | DASHBOARD | PIPELINE F | KEEP | dashboard/ui/public/icons.svg | LOW |
| dashboard\ui\src\App.css | DASHBOARD | PIPELINE F | KEEP | dashboard/ui/src/App.css | LOW |
| dashboard\ui\src\App.tsx | DASHBOARD | PIPELINE F | KEEP | dashboard/ui/src/App.tsx | LOW |
| dashboard\ui\src\index.css | DASHBOARD | PIPELINE F | KEEP | dashboard/ui/src/index.css | LOW |
| dashboard\ui\src\main.tsx | DASHBOARD | PIPELINE F | KEEP | dashboard/ui/src/main.tsx | LOW |
| dashboard\ui\src\assets\hero.png | DASHBOARD | PIPELINE F | KEEP | dashboard/ui/src/assets/hero.png | LOW |
| dashboard\ui\src\assets\react.svg | DASHBOARD | PIPELINE F | KEEP | dashboard/ui/src/assets/react.svg | LOW |
| dashboard\ui\src\assets\vite.svg | DASHBOARD | PIPELINE F | KEEP | dashboard/ui/src/assets/vite.svg | LOW |
| dashboard\ui\src\components\layout\Sidebar.tsx | DASHBOARD | PIPELINE F | KEEP | dashboard/ui/src/components/layout/Sidebar.tsx | LOW |
| dashboard\ui\src\components\ui\badge.tsx | DASHBOARD | PIPELINE F | KEEP | dashboard/ui/src/components/ui/badge.tsx | LOW |
| dashboard\ui\src\components\ui\card.tsx | DASHBOARD | PIPELINE F | KEEP | dashboard/ui/src/components/ui/card.tsx | LOW |
| dashboard\ui\src\hooks\useChunkedReplay.ts | DASHBOARD | PIPELINE F | KEEP | dashboard/ui/src/hooks/useChunkedReplay.ts | LOW |
| dashboard\ui\src\hooks\useCyberCastAPI.ts | DASHBOARD | PIPELINE F | KEEP | dashboard/ui/src/hooks/useCyberCastAPI.ts | LOW |
| dashboard\ui\src\hooks\useLiveLabAPI.ts | DASHBOARD | PIPELINE F | KEEP | dashboard/ui/src/hooks/useLiveLabAPI.ts | LOW |
| dashboard\ui\src\lib\utils.ts | DASHBOARD | PIPELINE F | KEEP | dashboard/ui/src/lib/utils.ts | LOW |
| dashboard\ui\src\pages\AttackIntelligence.tsx | DASHBOARD | PIPELINE F | KEEP | dashboard/ui/src/pages/AttackIntelligence.tsx | LOW |
| dashboard\ui\src\pages\AttackTimeline.tsx | DASHBOARD | PIPELINE F | KEEP | dashboard/ui/src/pages/AttackTimeline.tsx | LOW |
| dashboard\ui\src\pages\CommandCenter.tsx | DASHBOARD | PIPELINE F | KEEP | dashboard/ui/src/pages/CommandCenter.tsx | LOW |
| dashboard\ui\src\pages\DataProvenance.tsx | DASHBOARD | PIPELINE F | KEEP | dashboard/ui/src/pages/DataProvenance.tsx | LOW |
| dashboard\ui\src\pages\Explainability.tsx | DASHBOARD | PIPELINE F | KEEP | dashboard/ui/src/pages/Explainability.tsx | LOW |
| dashboard\ui\src\pages\HistoricalReplay.tsx | DASHBOARD | PIPELINE F | KEEP | dashboard/ui/src/pages/HistoricalReplay.tsx | LOW |
| dashboard\ui\src\pages\ModelPerformance.tsx | DASHBOARD | PIPELINE F | KEEP | dashboard/ui/src/pages/ModelPerformance.tsx | LOW |
| dashboard\ui\src\pages\NetworkTraffic.tsx | DASHBOARD | PIPELINE F | KEEP | dashboard/ui/src/pages/NetworkTraffic.tsx | LOW |
| data\processed\network_states_10s.parquet | DATA | None | KEEP | data/processed/network_states_10s.parquet | LOW |
| data\raw\02-14-2018.csv | DATA | None | KEEP | data/raw/02-14-2018.csv | LOW |
| data\raw\02-15-2018.csv | DATA | None | KEEP | data/raw/02-15-2018.csv | LOW |
| data\raw\02-16-2018.csv | DATA | None | KEEP | data/raw/02-16-2018.csv | LOW |
| data\raw\02-20-2018.csv | DATA | None | KEEP | data/raw/02-20-2018.csv | LOW |
| data\raw\02-21-2018.csv | DATA | None | KEEP | data/raw/02-21-2018.csv | LOW |
| data\raw\02-22-2018.csv | DATA | None | KEEP | data/raw/02-22-2018.csv | LOW |
| data\raw\02-23-2018.csv | DATA | None | KEEP | data/raw/02-23-2018.csv | LOW |
| data\raw\02-28-2018.csv | DATA | None | KEEP | data/raw/02-28-2018.csv | LOW |
| data\raw\03-01-2018.csv | DATA | None | KEEP | data/raw/03-01-2018.csv | LOW |
| data\raw\03-02-2018.csv | DATA | None | KEEP | data/raw/03-02-2018.csv | LOW |
| data\raw\archive.zip | UNKNOWN | None | REVIEW | data/raw/archive.zip | LOW |
| data\raw\ids-intrusion-csv.zip | UNKNOWN | None | REVIEW | data/raw/ids-intrusion-csv.zip | LOW |
| docs\FINAL_REPOSITORY_ARCHITECTURE.md | DOCUMENTATION | None | KEEP | docs/FINAL_REPOSITORY_ARCHITECTURE.md | LOW |
| docs\models\model_inventory.csv | UNKNOWN | None | REVIEW | docs/models/model_inventory.csv | LOW |
| docs\models\model_inventory.md | DOCUMENTATION | None | KEEP | docs/models/model_inventory.md | LOW |
| docs\models\model_migration_plan.md | DOCUMENTATION | None | KEEP | docs/models/model_migration_plan.md | LOW |
| docs\models\model_provenance.md | DOCUMENTATION | None | KEEP | docs/models/model_provenance.md | LOW |
| evaluation\metrics.py | UNKNOWN | None | REVIEW | evaluation/metrics.py | LOW |
| evaluation\__init__.py | UNKNOWN | None | REVIEW | evaluation/__init__.py | LOW |
| figures\calibration_curve.png | UNKNOWN | None | REVIEW | figures/calibration_curve.png | LOW |
| figures\confusion_matrix.png | UNKNOWN | None | REVIEW | figures/confusion_matrix.png | LOW |
| figures\feature_importance.png | UNKNOWN | None | REVIEW | figures/feature_importance.png | LOW |
| figures\phase2_1_confusion_matrix.png | UNKNOWN | None | REVIEW | figures/phase2_1_confusion_matrix.png | LOW |
| figures\phase2_1_training_curves.png | UNKNOWN | None | REVIEW | figures/phase2_1_training_curves.png | LOW |
| figures\training_curves.png | UNKNOWN | None | REVIEW | figures/training_curves.png | LOW |
| figures\validation_pr_curve.png | UNKNOWN | None | REVIEW | figures/validation_pr_curve.png | LOW |
| figures\validation_roc_curve.png | UNKNOWN | None | REVIEW | figures/validation_roc_curve.png | LOW |
| figures\phase2_3\early_warning_degradation.png | UNKNOWN | None | REVIEW | figures/phase2_3/early_warning_degradation.png | LOW |
| forecasting\risk_scoring.py | UNKNOWN | None | REVIEW | forecasting/risk_scoring.py | LOW |
| forecasting\simulator.py | UNKNOWN | None | REVIEW | forecasting/simulator.py | LOW |
| forecasting\stage_mapping.py | UNKNOWN | None | REVIEW | forecasting/stage_mapping.py | LOW |
| forecasting\__init__.py | UNKNOWN | None | REVIEW | forecasting/__init__.py | LOW |
| models\baseline.py | UNKNOWN | None | REVIEW | models/baseline.py | LOW |
| models\best_world_model.pt | MODEL | None | ARCHIVE | models/archive/best_world_model.pt | LOW |
| models\config.json | UNKNOWN | None | REVIEW | models/config.json | LOW |
| models\cybercast_best_model.pt | MODEL | None | ARCHIVE | models/archive/cybercast_best_model.pt | LOW |
| models\cybercast_best_model_config.json | UNKNOWN | None | REVIEW | models/cybercast_best_model_config.json | LOW |
| models\feature_names.json | UNKNOWN | None | REVIEW | models/feature_names.json | LOW |
| models\feature_scaler.pkl | MODEL | None | ARCHIVE | models/archive/feature_scaler.pkl | LOW |
| models\logistic_regression.pkl | MODEL | None | ARCHIVE | models/archive/logistic_regression.pkl | LOW |
| models\lstm_model.py | UNKNOWN | None | REVIEW | models/lstm_model.py | LOW |
| models\scaler.pkl | MODEL | None | ARCHIVE | models/archive/scaler.pkl | LOW |
| models\train.py | UNKNOWN | None | REVIEW | models/train.py | LOW |
| models\__init__.py | UNKNOWN | None | REVIEW | models/__init__.py | LOW |
| models\phase2_1\best_model.pt | MODEL | None | ARCHIVE | models/archive/best_model.pt | LOW |
| models\phase2_1\best_model_config.json | UNKNOWN | None | REVIEW | models/phase2_1/best_model_config.json | LOW |
| models\phase2_3\model_SET_A_h20.pt | MODEL | None | ARCHIVE | models/archive/model_SET_A_h20.pt | LOW |
| models\phase2_3\model_SET_R_h10.pt | MODEL | None | ARCHIVE | models/archive/model_SET_R_h10.pt | LOW |
| models\phase2_3\model_SET_R_h20.pt | MODEL | None | MOVE | models/production/model_SET_R_h20.pt | HIGH |
| models\phase2_3\model_SET_R_h30.pt | MODEL | None | ARCHIVE | models/archive/model_SET_R_h30.pt | LOW |
| models\phase2_3\model_SET_R_h5.pt | MODEL | None | ARCHIVE | models/archive/model_SET_R_h5.pt | LOW |
| models\phase2_3\model_SET_R_Repro_1_h20.pt | MODEL | None | ARCHIVE | models/archive/model_SET_R_Repro_1_h20.pt | LOW |
| models\phase2_3\model_SET_R_Repro_2_h20.pt | MODEL | None | ARCHIVE | models/archive/model_SET_R_Repro_2_h20.pt | LOW |
| models\phase2_3\model_SET_R_SELECT_h20.pt | MODEL | None | ARCHIVE | models/archive/model_SET_R_SELECT_h20.pt | LOW |
| models\phase2_3\scaler_SET_A_h20.joblib | MODEL | None | ARCHIVE | models/archive/scaler_SET_A_h20.joblib | LOW |
| models\phase2_3\scaler_SET_R_h10.joblib | MODEL | None | ARCHIVE | models/archive/scaler_SET_R_h10.joblib | LOW |
| models\phase2_3\scaler_SET_R_h20.joblib | MODEL | None | MOVE | models/production/scaler_SET_R_h20.joblib | HIGH |
| models\phase2_3\scaler_SET_R_h30.joblib | MODEL | None | ARCHIVE | models/archive/scaler_SET_R_h30.joblib | LOW |
| models\phase2_3\scaler_SET_R_h5.joblib | MODEL | None | ARCHIVE | models/archive/scaler_SET_R_h5.joblib | LOW |
| models\phase2_3\scaler_SET_R_Repro_1_h20.joblib | MODEL | None | ARCHIVE | models/archive/scaler_SET_R_Repro_1_h20.joblib | LOW |
| models\phase2_3\scaler_SET_R_Repro_2_h20.joblib | MODEL | None | ARCHIVE | models/archive/scaler_SET_R_Repro_2_h20.joblib | LOW |
| models\phase2_3\scaler_SET_R_SELECT_h20.joblib | MODEL | None | ARCHIVE | models/archive/scaler_SET_R_SELECT_h20.joblib | LOW |
| nootebooks\01_dataset_inspection.ipynb | NOTEBOOK | None | ARCHIVE | notebooks/archive/01_dataset_inspection.ipynb | LOW |
| nootebooks\CyberCast_Complete.ipynb | NOTEBOOK | None | ARCHIVE | notebooks/archive/CyberCast_Complete.ipynb | LOW |
| nootebooks\CyberCast_Complete_executed.ipynb | NOTEBOOK | None | ARCHIVE | notebooks/archive/CyberCast_Complete_executed.ipynb | LOW |
| nootebooks\CyberCast_Model_From_Scratch.ipynb | NOTEBOOK | None | ARCHIVE | notebooks/archive/CyberCast_Model_From_Scratch.ipynb | LOW |
| notebooks\CyberCast_All_Phases.ipynb | NOTEBOOK | None | ARCHIVE | notebooks/archive/CyberCast_All_Phases.ipynb | LOW |
| notebooks\CyberCast_All_Pipelines.ipynb | NOTEBOOK | None | ARCHIVE | notebooks/archive/CyberCast_All_Pipelines.ipynb | LOW |
| notebooks\CyberCast_Complete.ipynb | NOTEBOOK | None | ARCHIVE | notebooks/archive/CyberCast_Complete.ipynb | LOW |
| notebooks\Phase_2_3_Complete_Analysis.ipynb | NOTEBOOK | None | ARCHIVE | notebooks/archive/Phase_2_3_Complete_Analysis.ipynb | LOW |
| notebooks\Phase_2\01_Phase_2_Model_Training.ipynb | NOTEBOOK | None | ARCHIVE | notebooks/archive/01_Phase_2_Model_Training.ipynb | LOW |
| notebooks\Phase_2\02_Phase_2_Model_Evaluation.ipynb | NOTEBOOK | None | ARCHIVE | notebooks/archive/02_Phase_2_Model_Evaluation.ipynb | LOW |
| notebooks\Phase_2_1\01_Phase_2_1_Experiments_Training.ipynb | NOTEBOOK | None | ARCHIVE | notebooks/archive/01_Phase_2_1_Experiments_Training.ipynb | LOW |
| notebooks\Phase_2_1\02_Phase_2_1_Evaluation.ipynb | NOTEBOOK | None | ARCHIVE | notebooks/archive/02_Phase_2_1_Evaluation.ipynb | LOW |
| notebooks\Phase_2_2\01_Phase_2_2_Experiments_Training.ipynb | NOTEBOOK | None | ARCHIVE | notebooks/archive/01_Phase_2_2_Experiments_Training.ipynb | LOW |
| notebooks\Phase_2_2\02_Phase_2_2_Evaluation.ipynb | NOTEBOOK | None | ARCHIVE | notebooks/archive/02_Phase_2_2_Evaluation.ipynb | LOW |
| notebooks\Phase_2_3\01_Phase_2_3_Training.ipynb | NOTEBOOK | None | ARCHIVE | notebooks/archive/01_Phase_2_3_Training.ipynb | LOW |
| notebooks\Phase_2_3\02_Phase_2_3_Final_Evaluation.ipynb | NOTEBOOK | None | ARCHIVE | notebooks/archive/02_Phase_2_3_Final_Evaluation.ipynb | LOW |
| plots\global_label_distribution.png | UNKNOWN | None | REVIEW | plots/global_label_distribution.png | LOW |
| preprocessing\flow_features.py | UNKNOWN | None | REVIEW | preprocessing/flow_features.py | LOW |
| preprocessing\packet_features.py | UNKNOWN | None | REVIEW | preprocessing/packet_features.py | LOW |
| preprocessing\windows.py | UNKNOWN | None | REVIEW | preprocessing/windows.py | LOW |
| preprocessing\__init__.py | UNKNOWN | None | REVIEW | preprocessing/__init__.py | LOW |
| results\confusion_matrix.csv | RESULT | None | KEEP | results/confusion_matrix.csv | LOW |
| results\data_audit.json | RESULT | None | KEEP | results/data_audit.json | LOW |
| results\data_quality_by_file.csv | DATA | None | KEEP | results/data_quality_by_file.csv | LOW |
| results\early_warning_results.csv | RESULT | None | KEEP | results/early_warning_results.csv | LOW |
| results\early_warning_results.json | RESULT | None | KEEP | results/early_warning_results.json | LOW |
| results\false_positive_analysis.csv | RESULT | None | KEEP | results/false_positive_analysis.csv | LOW |
| results\feature_audit.csv | RESULT | None | KEEP | results/feature_audit.csv | LOW |
| results\feature_importance.csv | RESULT | None | KEEP | results/feature_importance.csv | LOW |
| results\final_test_metrics.json | TEST | None | KEEP | results/final_test_metrics.json | LOW |
| results\forecast_results.csv | RESULT | None | KEEP | results/forecast_results.csv | LOW |
| results\leakage_audit.json | RESULT | None | KEEP | results/leakage_audit.json | LOW |
| results\metrics.csv | RESULT | None | KEEP | results/metrics.csv | LOW |
| results\metrics_comparison.csv | RESULT | None | KEEP | results/metrics_comparison.csv | LOW |
| results\model_comparison.csv | RESULT | None | KEEP | results/model_comparison.csv | LOW |
| results\predictions.csv | RESULT | None | KEEP | results/predictions.csv | LOW |
| results\sanity_audit_output.txt | RESULT | None | KEEP | results/sanity_audit_output.txt | LOW |
| results\split_audit.json | RESULT | None | KEEP | results/split_audit.json | LOW |
| results\test_demonstration.csv | TEST | None | KEEP | results/test_demonstration.csv | LOW |
| results\threshold_analysis_summary.json | RESULT | None | KEEP | results/threshold_analysis_summary.json | LOW |
| results\threshold_analysis_val.csv | RESULT | None | KEEP | results/threshold_analysis_val.csv | LOW |
| results\training_history.csv | RESULT | None | KEEP | results/training_history.csv | LOW |
| results\validation_threshold_analysis.csv | RESULT | None | KEEP | results/validation_threshold_analysis.csv | LOW |
| results\inference\cicflowmeter_live_compatibility.md | DOCUMENTATION | None | KEEP | results/inference/cicflowmeter_live_compatibility.md | LOW |
| results\inference\forecasting_methodology.md | DOCUMENTATION | None | KEEP | results/inference/forecasting_methodology.md | LOW |
| results\inference\forecast_predictions.csv | RESULT | None | KEEP | results/inference/forecast_predictions.csv | LOW |
| results\inference\forecast_predictions.json | RESULT | None | KEEP | results/inference/forecast_predictions.json | LOW |
| results\inference\live_attack_validation.csv | RESULT | None | KEEP | results/inference/live_attack_validation.csv | LOW |
| results\inference\live_attack_validation.json | RESULT | None | KEEP | results/inference/live_attack_validation.json | LOW |
| results\inference\live_attack_validation_report.md | DOCUMENTATION | None | KEEP | results/inference/live_attack_validation_report.md | LOW |
| results\inference\live_feature_mapping.md | DOCUMENTATION | None | KEEP | results/inference/live_feature_mapping.md | LOW |
| results\inference\live_lab_validation.md | DOCUMENTATION | None | KEEP | results/inference/live_lab_validation.md | LOW |
| results\inference\predictions.csv | RESULT | None | KEEP | results/inference/predictions.csv | LOW |
| results\inference\live\live_predictions.csv | RESULT | None | KEEP | results/inference/live/live_predictions.csv | LOW |
| results\phase2_1\experiment_comparison.csv | RESULT | None | KEEP | results/phase2_1/experiment_comparison.csv | LOW |
| results\phase2_1\final_test_metrics.json | TEST | None | KEEP | results/phase2_1/final_test_metrics.json | LOW |
| results\phase2_1\threshold_analysis.csv | RESULT | None | KEEP | results/phase2_1/threshold_analysis.csv | LOW |
| results\phase2_2\champion_metrics.json | RESULT | None | KEEP | results/phase2_2/champion_metrics.json | LOW |
| results\phase2_2\early_warning_diagnostic.csv | RESULT | None | KEEP | results/phase2_2/early_warning_diagnostic.csv | LOW |
| results\phase2_2\phase2_2_results.csv | RESULT | None | KEEP | results/phase2_2/phase2_2_results.csv | LOW |
| results\phase2_2\models\model_10_SET_A.pt | MODEL | None | ARCHIVE | models/archive/model_10_SET_A.pt | LOW |
| results\phase2_2\models\model_10_SET_B.pt | MODEL | None | ARCHIVE | models/archive/model_10_SET_B.pt | LOW |
| results\phase2_2\models\model_10_SET_C.pt | MODEL | None | ARCHIVE | models/archive/model_10_SET_C.pt | LOW |
| results\phase2_2\models\model_20_SET_A.pt | MODEL | None | ARCHIVE | models/archive/model_20_SET_A.pt | LOW |
| results\phase2_2\models\model_20_SET_B.pt | MODEL | None | ARCHIVE | models/archive/model_20_SET_B.pt | LOW |
| results\phase2_2\models\model_20_SET_C.pt | MODEL | None | ARCHIVE | models/archive/model_20_SET_C.pt | LOW |
| results\phase2_2\models\model_30_SET_A.pt | MODEL | None | ARCHIVE | models/archive/model_30_SET_A.pt | LOW |
| results\phase2_2\models\model_30_SET_B.pt | MODEL | None | ARCHIVE | models/archive/model_30_SET_B.pt | LOW |
| results\phase2_2\models\model_30_SET_C.pt | MODEL | None | ARCHIVE | models/archive/model_30_SET_C.pt | LOW |
| results\phase2_2\models\model_5_SET_A.pt | MODEL | None | ARCHIVE | models/archive/model_5_SET_A.pt | LOW |
| results\phase2_2\models\model_5_SET_B.pt | MODEL | None | ARCHIVE | models/archive/model_5_SET_B.pt | LOW |
| results\phase2_2\models\model_5_SET_C.pt | MODEL | None | ARCHIVE | models/archive/model_5_SET_C.pt | LOW |
| results\phase2_2\plots\early_warning_degradation.png | RESULT | None | KEEP | results/phase2_2/plots/early_warning_degradation.png | LOW |
| results\phase2_3\champion_metrics.json | RESULT | None | KEEP | results/phase2_3/champion_metrics.json | LOW |
| results\phase2_3\early_warning_diagnostic.csv | RESULT | None | KEEP | results/phase2_3/early_warning_diagnostic.csv | LOW |
| results\phase2_3\feature_audit.csv | RESULT | None | KEEP | results/phase2_3/feature_audit.csv | LOW |
| results\phase2_3\feature_comparison.csv | RESULT | None | KEEP | results/phase2_3/feature_comparison.csv | LOW |
| results\phase2_3\feature_sets.json | RESULT | None | KEEP | results/phase2_3/feature_sets.json | LOW |
| results\phase2_3\history_comparison.csv | RESULT | None | KEEP | results/phase2_3/history_comparison.csv | LOW |
| results\phase2_3\model_comparison.csv | RESULT | None | KEEP | results/phase2_3/model_comparison.csv | LOW |
| results\phase2_3\pre_attack_analysis\attack_type_summary.csv | RESULT | None | KEEP | results/phase2_3/pre_attack_analysis/attack_type_summary.csv | LOW |
| results\phase2_3\pre_attack_analysis\forecasting_validation_audit.json | RESULT | None | KEEP | results/phase2_3/pre_attack_analysis/forecasting_validation_audit.json | LOW |
| results\phase2_3\pre_attack_analysis\forecasting_validation_report.md | DOCUMENTATION | None | KEEP | results/phase2_3/pre_attack_analysis/forecasting_validation_report.md | LOW |
| results\phase2_3\pre_attack_analysis\pre_attack_analysis_report.md | DOCUMENTATION | None | KEEP | results/phase2_3/pre_attack_analysis/pre_attack_analysis_report.md | LOW |
| results\phase2_3\pre_attack_analysis\pre_attack_probability_curves.csv | RESULT | None | KEEP | results/phase2_3/pre_attack_analysis/pre_attack_probability_curves.csv | LOW |
| results\phase2_3\pre_attack_analysis\representative_transitions.csv | RESULT | None | KEEP | results/phase2_3/pre_attack_analysis/representative_transitions.csv | LOW |
| results\phase2_3\pre_attack_analysis\transition_analysis.csv | RESULT | None | KEEP | results/phase2_3/pre_attack_analysis/transition_analysis.csv | LOW |
| results\phase2_3\pre_attack_analysis\validation_attack_type_summary.csv | RESULT | None | KEEP | results/phase2_3/pre_attack_analysis/validation_attack_type_summary.csv | LOW |
| results\phase2_3\pre_attack_analysis\validation_early_warning_thresholds.csv | RESULT | None | KEEP | results/phase2_3/pre_attack_analysis/validation_early_warning_thresholds.csv | LOW |
| results\phase2_3\pre_attack_analysis\validation_probability_curve.csv | RESULT | None | KEEP | results/phase2_3/pre_attack_analysis/validation_probability_curve.csv | LOW |
| results\phase2_3\pre_attack_analysis\validation_probability_curve.png | RESULT | None | KEEP | results/phase2_3/pre_attack_analysis/validation_probability_curve.png | LOW |
| results\plots\feature_importance.png | RESULT | None | KEEP | results/plots/feature_importance.png | LOW |
| results\plots\forecast_horizon.png | RESULT | None | KEEP | results/plots/forecast_horizon.png | LOW |
| results\plots\test_evaluation.png | TEST | None | KEEP | results/plots/test_evaluation.png | LOW |
| results\plots\threshold_analysis.png | RESULT | None | KEEP | results/plots/threshold_analysis.png | LOW |
| results\plots\training_history.png | RESULT | None | KEEP | results/plots/training_history.png | LOW |
| scratch\audit_script.py | TEMPORARY | None | ARCHIVE | archive/audit_script.py | LOW |
| scratch\check_dates.py | TEMPORARY | None | ARCHIVE | archive/check_dates.py | LOW |
| scratch\check_env.py | TEMPORARY | None | ARCHIVE | archive/check_env.py | LOW |
| scratch\compare.py | TEMPORARY | None | ARCHIVE | archive/compare.py | LOW |
| scratch\dump_cells.py | TEMPORARY | None | ARCHIVE | archive/dump_cells.py | LOW |
| scratch\dump_early_cells.py | TEMPORARY | None | ARCHIVE | archive/dump_early_cells.py | LOW |
| scratch\dump_key_cells.py | TEMPORARY | None | ARCHIVE | archive/dump_key_cells.py | LOW |
| scratch\dump_key_cells2.py | TEMPORARY | None | ARCHIVE | archive/dump_key_cells2.py | LOW |
| scratch\dump_lr.py | TEMPORARY | None | ARCHIVE | archive/dump_lr.py | LOW |
| scratch\dump_model_cells.py | TEMPORARY | None | ARCHIVE | archive/dump_model_cells.py | LOW |
| scratch\dump_nb.py | TEMPORARY | None | ARCHIVE | archive/dump_nb.py | LOW |
| scratch\dump_pipeline.py | TEMPORARY | None | ARCHIVE | archive/dump_pipeline.py | LOW |
| scratch\execute_and_verify.py | TEMPORARY | None | ARCHIVE | archive/execute_and_verify.py | LOW |
| scratch\extract_outputs.py | TEMPORARY | None | ARCHIVE | archive/extract_outputs.py | LOW |
| scratch\fast_scan.py | TEMPORARY | None | ARCHIVE | archive/fast_scan.py | LOW |
| scratch\find_environments.py | TEMPORARY | None | ARCHIVE | archive/find_environments.py | LOW |
| scratch\fix_and_run.py | TEMPORARY | None | ARCHIVE | archive/fix_and_run.py | LOW |
| scratch\fix_unicode.py | TEMPORARY | None | ARCHIVE | archive/fix_unicode.py | LOW |
| scratch\full_audit.py | TEMPORARY | None | ARCHIVE | archive/full_audit.py | LOW |
| scratch\full_audit_v2.py | TEMPORARY | None | ARCHIVE | archive/full_audit_v2.py | LOW |
| scratch\generate_notebook.py | TEMPORARY | None | ARCHIVE | archive/generate_notebook.py | LOW |
| scratch\inspect_interfaces.py | TEMPORARY | None | ARCHIVE | archive/inspect_interfaces.py | LOW |
| scratch\inspect_parquet.py | TEMPORARY | None | ARCHIVE | archive/inspect_parquet.py | LOW |
| scratch\inspect_timestamp.py | TEMPORARY | None | ARCHIVE | archive/inspect_timestamp.py | LOW |
| scratch\nb_cells.py | TEMPORARY | None | ARCHIVE | archive/nb_cells.py | LOW |
| scratch\nb_dump.txt | TEMPORARY | None | ARCHIVE | archive/nb_dump.txt | LOW |
| scratch\raw_csv_fast_audit.py | TEMPORARY | None | ARCHIVE | archive/raw_csv_fast_audit.py | LOW |
| scratch\raw_data_split_audit.py | TEMPORARY | None | ARCHIVE | archive/raw_data_split_audit.py | LOW |
| scratch\read_split.py | TEMPORARY | None | ARCHIVE | archive/read_split.py | LOW |
| scratch\run_check.bat | TEMPORARY | None | ARCHIVE | archive/run_check.bat | LOW |
| scratch\test_ast.py | TEST | None | KEEP | scratch/test_ast.py | LOW |
| scratch\test_chunk.csv | TEST | None | KEEP | scratch/test_chunk.csv | LOW |
| scratch\test_label.csv | TEST | None | KEEP | scratch/test_label.csv | LOW |
| scratch\test_sniff.py | TEST | None | KEEP | scratch/test_sniff.py | LOW |
| scratch\test_synthetic.py | TEST | None | KEEP | scratch/test_synthetic.py | LOW |
| scratch\test_synthetic_2.py | TEST | None | KEEP | scratch/test_synthetic_2.py | LOW |
| scratch\test_timestamp_filter.py | TEST | None | KEEP | scratch/test_timestamp_filter.py | LOW |
| scratch\test_timestamp_filter_2.py | TEST | None | KEEP | scratch/test_timestamp_filter_2.py | LOW |
| scratch\update_notebook.py | TEMPORARY | None | ARCHIVE | archive/update_notebook.py | LOW |
| scratch\validate_live_demo.py | TEMPORARY | None | ARCHIVE | archive/validate_live_demo.py | LOW |
| scratch\verify_nb.py | TEMPORARY | None | ARCHIVE | archive/verify_nb.py | LOW |
| scripts\01_dataset_inspection.py | EXPERIMENT | None | ARCHIVE | scripts/01_dataset_inspection.py | LOW |
| scripts\analyze_forecasting_validation.py | RESEARCH | None | KEEP | scripts/analyze_forecasting_validation.py | LOW |
| scripts\analyze_pre_attack.py | RESEARCH | None | KEEP | scripts/analyze_pre_attack.py | LOW |
| scripts\audit_models.py | RESEARCH | None | KEEP | scripts/audit_models.py | LOW |
| scripts\build_final_migration_plan.py | EXPERIMENT | None | ARCHIVE | scripts/build_final_migration_plan.py | LOW |
| scripts\create_nb_from_py.py | EXPERIMENT | None | ARCHIVE | scripts/create_nb_from_py.py | LOW |
| scripts\phase2_2_experiments.py | EXPERIMENT | None | ARCHIVE | scripts/phase2_2_experiments.py | LOW |
| scripts\phase2_3_experiments.py | EXPERIMENT | None | ARCHIVE | scripts/phase2_3_experiments.py | LOW |
| scripts\reproducibility_check.py | RESEARCH | None | KEEP | scripts/reproducibility_check.py | LOW |
| scripts\threshold_analysis.py | EXPERIMENT | None | ARCHIVE | scripts/threshold_analysis.py | LOW |
| scripts\live_attack_lab\attack_lab_runner.py | EXPERIMENT | None | ARCHIVE | scripts/live_attack_lab/attack_lab_runner.py | LOW |
| scripts\live_attack_lab\generate_attack_report.py | EXPERIMENT | None | ARCHIVE | scripts/live_attack_lab/generate_attack_report.py | LOW |
| src\api\main.py | PRODUCTION | PIPELINE F | KEEP | src/api/main.py | LOW |
| src\explainability\feature_attribution.py | PRODUCTION | PIPELINE E | KEEP | src/explainability/feature_attribution.py | LOW |
| src\explainability\__init__.py | PRODUCTION | PIPELINE E | KEEP | src/explainability/__init__.py | LOW |
| src\forecasting\forecaster.py | PRODUCTION | PIPELINE E | KEEP | src/forecasting/forecaster.py | LOW |
| src\forecasting\progression.py | PRODUCTION | PIPELINE E | KEEP | src/forecasting/progression.py | LOW |
| src\forecasting\schemas.py | PRODUCTION | PIPELINE E | KEEP | src/forecasting/schemas.py | LOW |
| src\forecasting\stage_mapper.py | PRODUCTION | PIPELINE E | KEEP | src/forecasting/stage_mapper.py | LOW |
| src\forecasting\__init__.py | PRODUCTION | PIPELINE E | KEEP | src/forecasting/__init__.py | LOW |
| src\inference\feature_pipeline.py | PRODUCTION | PIPELINE B/C | KEEP | src/inference/feature_pipeline.py | LOW |
| src\inference\model_loader.py | PRODUCTION | PIPELINE B/C | KEEP | src/inference/model_loader.py | LOW |
| src\inference\predictor.py | PRODUCTION | PIPELINE B/C | KEEP | src/inference/predictor.py | LOW |
| src\inference\risk_engine.py | PRODUCTION | PIPELINE B/C | KEEP | src/inference/risk_engine.py | LOW |
| src\inference\schemas.py | PRODUCTION | PIPELINE B/C | KEEP | src/inference/schemas.py | LOW |
| src\inference\windowing.py | PRODUCTION | PIPELINE B/C | KEEP | src/inference/windowing.py | LOW |
| src\inference\__init__.py | PRODUCTION | PIPELINE B/C | KEEP | src/inference/__init__.py | LOW |
| src\live\config.py | PRODUCTION | PIPELINE D | KEEP | src/live/config.py | LOW |
| src\live\flow_aggregator.py | PRODUCTION | PIPELINE D | KEEP | src/live/flow_aggregator.py | LOW |
| src\live\live_engine.py | PRODUCTION | PIPELINE D | KEEP | src/live/live_engine.py | LOW |
| src\live\npcap_capture.py | PRODUCTION | PIPELINE D | KEEP | src/live/npcap_capture.py | LOW |
| src\live\schemas.py | PRODUCTION | PIPELINE D | KEEP | src/live/schemas.py | LOW |
| src\live\__init__.py | PRODUCTION | PIPELINE D | KEEP | src/live/__init__.py | LOW |
| tests\test_dashboard_api.py | TEST | None | KEEP | tests/test_dashboard_api.py | LOW |
| tests\test_forecasting.py | TEST | None | KEEP | tests/test_forecasting.py | LOW |
| tests\test_inference.py | TEST | None | KEEP | tests/test_inference.py | LOW |
| tests\test_live_api.py | TEST | None | KEEP | tests/test_live_api.py | LOW |
| tests\test_live_cli.py | TEST | None | KEEP | tests/test_live_cli.py | LOW |
| tests\test_live_engine.py | TEST | None | KEEP | tests/test_live_engine.py | LOW |
| tests\test_live_features.py | TEST | None | KEEP | tests/test_live_features.py | LOW |
| tests\test_live_npcap.py | TEST | None | KEEP | tests/test_live_npcap.py | LOW |
| tests\test_live_prediction.py | TEST | None | KEEP | tests/test_live_prediction.py | LOW |
