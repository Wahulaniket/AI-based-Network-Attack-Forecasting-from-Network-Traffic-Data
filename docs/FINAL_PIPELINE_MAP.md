# Final Pipeline Map

## PIPELINE A: Offline training/evaluation
Data -> Preprocessing -> Model Training -> Evaluation -> Metrics
*Primary Source:* `notebooks/` and `scripts/analyze_*.py`

## PIPELINE B: CSV inference
CSV File -> Windowing -> Feature Scaling -> LSTM Model -> Probability
*Primary Source:* `src/inference/`

## PIPELINE C: PCAP inference
PCAP File -> CICFlowMeter -> CSV -> Pipeline B
*Primary Source:* `src/inference/`

## PIPELINE D: Live Npcap inference
Live Traffic -> Npcap Sniffer -> Feature Extractor -> Model Inference -> Live Alerts
*Primary Source:* `src/live/`

## PIPELINE E: Forecasting / ATT&CK / explainability
Inference Probability -> Pre-Attack Window Detection -> ATT&CK Mapping -> SHAP Explainer
*Primary Source:* `src/forecasting/`, `src/explainability/`

## PIPELINE F: API -> dashboard
Backend Live State -> FastAPI Websockets/REST -> React Frontend Dashboard
*Primary Source:* `src/api/`, `dashboard/`

## Pipeline Duplication Analysis
1. **Feature Extraction**: `cybercast_pipeline.py` vs `src/inference/windowing.py`
   - *Reason*: `cybercast_pipeline.py` was used for offline phase 1/2 training, while `windowing.py` is for production inference.
   - *Action*: Archive `cybercast_pipeline.py` and consolidate logic into `src/inference/windowing.py`.
