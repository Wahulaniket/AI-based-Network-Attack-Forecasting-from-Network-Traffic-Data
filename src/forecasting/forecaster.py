import os
import sys
import argparse
import pandas as pd
import json
from datetime import datetime

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from src.inference.model_loader import load_inference_artifacts
from src.inference.feature_pipeline import process_features
from src.inference.windowing import create_time_windows, create_causal_sequences
from src.explainability.feature_attribution import explain_prediction
from src.forecasting.stage_mapper import infer_attack_stage
from src.forecasting.progression import compute_risk_trend, forecast_next_stage
from src.forecasting.schemas import ForecastOutput

def run_forecasting(predictions_csv: str, raw_csv: str, lines_limit: int = None):
    print("\n" + "="*50)
    print("1. Loading Inference Artifacts and Model")
    print("="*50)
    artifacts = load_inference_artifacts(repo_root)
    model = artifacts['model']
    scaler = artifacts['scaler']
    features = artifacts['features']
    device = artifacts['device']
    
    print("\n" + "="*50)
    print("2. Loading Predictions and Raw Traffic")
    print("="*50)
    preds_df = pd.read_csv(predictions_csv)
    raw_df = pd.read_csv(raw_csv, low_memory=False, nrows=lines_limit)
    
    print(f"Loaded {len(preds_df)} predictions.")
    print(f"Loaded {len(raw_df)} raw flows.")
    
    # Reconstruct the temporal sequences exactly as inference did
    df_processed = process_features(raw_df, features)
    states = create_time_windows(df_processed, features, window_seconds=10)
    X_raw, timestamps = create_causal_sequences(states, features, seq_length=20)
    
    # Scale sequences
    batch_size, seq_length, num_features = X_raw.shape
    X_flat = X_raw.reshape(-1, num_features)
    X_scaled_flat = scaler.transform(X_flat)
    X_scaled = X_scaled_flat.reshape(batch_size, seq_length, num_features)
    
    # Map predictions to index
    timestamp_to_idx = {ts.isoformat(): i for i, ts in enumerate(timestamps)}
    
    print("\n" + "="*50)
    print("3. Generating Forecasts and Explainability")
    print("="*50)
    
    forecasts = []
    history_probs = []
    
    for i, row in preds_df.iterrows():
        ts_str = row['timestamp']
        if ts_str not in timestamp_to_idx:
            # We don't have the features for this prediction (maybe lines_limit truncated it)
            continue
            
        seq_idx = timestamp_to_idx[ts_str]
        sequence = X_scaled[seq_idx]
        
        # State idx for the current window is seq_idx + seq_length - 1
        state_idx = seq_idx + seq_length - 1
        current_features = states.iloc[state_idx]
        
        # Track history for progression
        history_probs.append(row['attack_probability'])
        delta, trend = compute_risk_trend(history_probs)
        
        # 1. Map ATT&CK stage deterministically from features
        stage_score = infer_attack_stage(current_features)
        
        # 2. Forecast next stage probabilistically
        next_stage, next_conf = forecast_next_stage(stage_score.stage, trend)
        
        # 3. Explainability (perturbation attribution)
        # We only run attribution if there is meaningful risk (e.g., > 0.01) to save compute
        if row['attack_probability'] > 0.01:
            explanation = explain_prediction(model, sequence, features, device)
        else:
            explanation = {"top_features": [], "temporal_evidence": []}
            
        # Serialize evidence dicts
        evidence_dicts = [{"feature": e.feature, "reason": e.reason} for e in stage_score.evidence]
        
        forecast = ForecastOutput(
            timestamp=ts_str,
            window_start=row['window_start'],
            window_end=row['window_end'],
            attack_probability=row['attack_probability'],
            model_threshold=row['model_threshold'],
            binary_prediction=row['binary_prediction'],
            risk_level=row['risk_level'],
            current_stage=stage_score.stage,
            stage_confidence=stage_score.confidence,
            stage_evidence=evidence_dicts,
            forecasted_next_stage=next_stage,
            forecast_confidence=next_conf,
            probability_delta=delta,
            risk_trend=trend,
            top_features=explanation['top_features'],
            temporal_evidence=explanation['temporal_evidence']
        )
        forecasts.append(forecast)
        
        if (i+1) % 100 == 0:
            print(f"Processed {i+1}/{len(preds_df)} forecasts...")
            
    print("\n" + "="*50)
    print("4. Saving Output")
    print("="*50)
    
    out_dir = os.path.join(repo_root, "results", "inference")
    
    # Save CSV
    out_csv = os.path.join(out_dir, "forecast_predictions.csv")
    forecast_df = pd.DataFrame([f.to_dict() for f in forecasts])
    forecast_df.to_csv(out_csv, index=False)
    
    # Save JSON
    out_json = os.path.join(out_dir, "forecast_predictions.json")
    with open(out_json, "w") as jf:
        json.dump([f.to_dict() for f in forecasts], jf, indent=4)
        
    print(f"[DONE] Generated {len(forecasts)} forecasts.")
    print(f"Saved to {out_csv} and {out_json}")
    
    print("\nSample Forecast JSON (First Record):")
    print(json.dumps(forecasts[0].to_dict(), indent=2))
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CyberCast Forecaster")
    parser.add_argument("--input", type=str, required=True, help="Path to predictions.csv")
    parser.add_argument("--raw-data", type=str, required=True, help="Path to raw CSV used for predictions")
    parser.add_argument("--lines", type=int, default=None, help="Number of lines to read from raw CSV")
    args = parser.parse_args()
    
    run_forecasting(args.input, args.raw_data, args.lines)
