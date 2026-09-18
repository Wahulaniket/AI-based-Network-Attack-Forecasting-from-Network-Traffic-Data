# Feature Pipeline Comparison

## cybercast_pipeline.py (Offline/Training)
- **Usage**: Phase 1 & 2 offline dataset processing.
- **Normalization**: Uses standard pandas/scikit-learn scalers fit on entire dataset.
- **Status**: Legacy (Not used in live inference).

## src/inference/windowing.py (Live/Production)
- **Usage**: Pipeline B/C/D for dynamic CSV and live Npcap inference.
- **Normalization**: Real-time sliding window aggregation using loaded `scaler_SET_R_h20.joblib`.
- **Status**: Production.

**Conclusion**: Do not merge. They serve different epochs of the project. Archive cybercast_pipeline.py.