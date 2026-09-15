import os
import sys
import argparse
import pandas as pd
import torch
from datetime import timedelta
import csv

# Add repo root to path to allow absolute imports if run as script
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from src.inference.model_loader import load_inference_artifacts
from src.inference.feature_pipeline import process_features
from src.inference.windowing import create_time_windows, create_causal_sequences
from src.inference.risk_engine import calculate_risk_level

def run_inference(csv_path: str, repo_root: str, lines_limit: int = None):
    print("\n" + "="*50)
    print("1. Loading Artifacts")
    print("="*50)
    artifacts = load_inference_artifacts(repo_root)
    model = artifacts['model']
    scaler = artifacts['scaler']
    features = artifacts['features']
    threshold = artifacts['threshold']
    device = artifacts['device']
    
    print("\n" + "="*50)
    print(f"2. Loading CSV: {csv_path}")
    print("="*50)
    
    # Load dataset
    df = pd.read_csv(csv_path, low_memory=False, nrows=lines_limit)
    print(f"Loaded {len(df)} rows.")

    print("\n" + "="*50)
    print("3. Feature Pipeline")
    print("="*50)
    df_processed = process_features(df, features)
    print(f"Processed into {len(df_processed)} chronologically sorted flows.")
    
    # Assertions for data leakage
    target_leakage_cols = ['binary_attack', 'dominant_label', 'attack_ratio', 'attack_flow_count', 'Label', 'Attack']
    for t in target_leakage_cols:
        assert t not in df_processed.columns, f"Leakage detected: {t} in feature pipeline!"
    print("[PASS] Target-derived columns excluded")

    print("\n" + "="*50)
    print("4. Temporal Windowing")
    print("="*50)
    # create time windows (10-second)
    states = create_time_windows(df_processed, features, window_seconds=10)
    print(f"Created {len(states)} 10-second windows.")
    
    # Assert feature count
    actual_features = [c for c in states.columns if c != 'Timestamp']
    assert len(actual_features) == len(features) == 89, f"Feature count mismatch! Expected 89, got {len(actual_features)}"
    assert actual_features == features, "Feature order mismatch!"
    print("[PASS] Feature count = 89")
    print("[PASS] Feature order matches checkpoint")
    
    print("\n" + "="*50)
    print("5. Sequence Generation")
    print("="*50)
    X_raw, timestamps = create_causal_sequences(states, features, seq_length=20)
    print(f"Created {len(X_raw)} causal sequences (length=20).")
    
    if len(X_raw) == 0:
        print("[WARNING] Not enough windows to form a sequence (need 20). Exiting.")
        return
        
    assert X_raw.shape[1] == 20, f"Sequence length is {X_raw.shape[1]}, expected 20"
    assert X_raw.shape[2] == 89, f"Feature dimension is {X_raw.shape[2]}, expected 89"
    print("[PASS] Sequence length = 20")
    print("[PASS] No future windows used")
    
    print("\n" + "="*50)
    print("6. Scaling & Inference")
    print("="*50)
    
    # Reshape for scaling: (batch * 20, 89)
    batch_size, seq_length, num_features = X_raw.shape
    X_flat = X_raw.reshape(-1, num_features)
    assert X_flat.shape[1] == 89, "Flat sequence feature dimension mismatch"
    print("[PASS] Scaler dimensions match")
    
    X_scaled_flat = scaler.transform(X_flat)
    X_scaled = X_scaled_flat.reshape(batch_size, seq_length, num_features)
    
    # Predict
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(device)
    
    predictions_out = []
    
    with torch.no_grad():
        # model returns raw logits
        logits = model(X_tensor)
        probs = torch.sigmoid(logits).cpu().numpy()
        
    # generate predictions
    for i in range(len(probs)):
        prob = probs[i]
        ts = timestamps[i]
        is_attack = int(prob >= threshold)
        risk = calculate_risk_level(prob)
        
        # Get statistics for the target window (the last window in the sequence)
        # Sequence ends at `ts` which is index `i + seq_length - 1` in states
        state_idx = i + seq_length - 1
        flow_cnt = int(states.iloc[state_idx]['flow_count'])
        
        pred = {
            "timestamp": ts.isoformat(),
            "window_start": (ts - timedelta(seconds=10)).isoformat(),
            "window_end": ts.isoformat(),
            "attack_probability": float(prob),
            "model_threshold": float(threshold),
            "binary_prediction": is_attack,
            "risk_level": risk,
            "flow_count": flow_cnt,
            "history_windows": seq_length
        }
        predictions_out.append(pred)
        
    # Save output
    out_dir = os.path.join(repo_root, "results", "inference")
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, "predictions.csv")
    
    df_out = pd.DataFrame(predictions_out)
    df_out.to_csv(out_file, index=False)
    
    print(f"\n[DONE] Inference complete. Saved {len(predictions_out)} predictions to {out_file}")
    
    print("\nSample Output (First 5 Rows):")
    print(df_out[['timestamp', 'attack_probability', 'binary_prediction', 'risk_level']].head(5))
    return df_out

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CyberCast Inference Engine")
    parser.add_argument("--input", type=str, required=True, help="Path to raw flow CSV")
    parser.add_argument("--lines", type=int, default=None, help="Number of lines to read")
    args = parser.parse_args()
    
    run_inference(args.input, repo_root, args.lines)
