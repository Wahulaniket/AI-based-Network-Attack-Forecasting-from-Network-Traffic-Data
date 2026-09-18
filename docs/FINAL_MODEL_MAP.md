# Final Model Map

## Production Chain Provenance
INPUT -> SET_R 89 features -> frozen scaler (`models/production/scaler_SET_R_h20.joblib`) -> 20x89 sequence -> Phase 2.3 LSTM (`models/production/model_SET_R_h20.pt`) -> probability -> risk -> forecast -> explainability -> ATT&CK -> API/dashboard

## Artifact Protection Status
- **Phase 2.3 production model**: PROTECTED
- **Phase 2.3 scaler**: PROTECTED
- **Phase 2.3 feature configuration**: PROTECTED
- **Phase 2.3 champion metrics**: PROTECTED
