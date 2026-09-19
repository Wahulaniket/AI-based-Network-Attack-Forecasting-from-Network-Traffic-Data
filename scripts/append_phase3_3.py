import nbformat as nbf
import numpy as np
import pandas as pd
from pathlib import Path
import subprocess

def append_phase3_3():
    nb_path = Path("notebooks/phase3/Phase3_CyberCast_WorldModel.ipynb")
    with open(nb_path, "r", encoding="utf-8") as f:
        nb = nbf.read(f, as_version=4)
        
    cells = nb['cells']
    
    cells.append(nbf.v4.new_markdown_cell("""
## [PHASE 3.3] K-STEP RECURSIVE FORECASTING (VALIDATION ONLY)

### Objective
Evaluate the temporal stability and error accumulation of the frozen V3 World Model over multiple forecast horizons ($K=1, 3, 5, 10, 20, 30$).

### Methodology
1. **Strict Causality**: For a given start state $S_t$, the model generates $\hat{S}_{t+1}$. This prediction is scaled exactly as the training data, appended to the history window, and fed recursively to predict $\hat{S}_{t+2}$, up to $\hat{S}_{t+K}$. Ground-truth states are NEVER used during generation.
2. **Primary Experiment**: Restricted to starting points where every consecutive transition up to $K$ is exactly 10 seconds.
3. **Secondary Diagnostic**: Includes all observed-state transitions, regardless of time gaps, measuring actual wall-clock elapsed time.
4. **Persistence Baseline**: $\hat{S}_{t+K} = S_t$ for all $K$.
"""))

    cells.append(nbf.v4.new_code_cell("""
K_HORIZONS = [1, 3, 5, 10, 20, 30]

# Extract Validation Timestamps and States
# idx_80 and idx_90 are defined in the earlier cells
val_indices = np.arange(idx_80, idx_90)
ts_val = S_t.index[val_indices]

# Prepare Batched Auto-regressive Generation
def batched_recursive_forecast(start_indices, K):
    # start_indices: array of indices i (relative to val_orig) representing state S_t
    # We need history [i - H + 1 : i + 1]
    
    N_starts = len(start_indices)
    batch_X_scaled = np.zeros((N_starts, H, 77), dtype=np.float32)
    
    for b_idx, i in enumerate(start_indices):
        batch_X_scaled[b_idx] = val_scaled_in[i - H + 1 : i + 1]
        
    batch_X = torch.tensor(batch_X_scaled, dtype=torch.float32).to(device)
    model.eval()
    
    with torch.no_grad():
        for k in range(1, K + 1):
            # Predict Primitives
            preds = model(batch_X).cpu().numpy()
            
            # Inverse Scale
            lstm_prim_log = preds * target_scales
            lstm_prim_orig = np.expm1(lstm_prim_log)
            
            # Deterministic Reconstruction
            S_hat = reconstruct_state(lstm_prim_orig, feature_names)
            
            # Forward Scale for next input
            S_hat_log = np.log1p(np.maximum(S_hat, 0))
            S_hat_scaled = input_scaler.transform(S_hat_log)
            
            # Update History Window
            if k < K:
                S_hat_scaled_t = torch.tensor(S_hat_scaled, dtype=torch.float32).unsqueeze(1).to(device)
                batch_X = torch.cat([batch_X[:, 1:, :], S_hat_scaled_t], dim=1)
                
    return S_hat, lstm_prim_orig

metrics_v3_recursive = {
    "Primary": {},
    "Secondary": {},
    "Error_Growth": []
}

for K in K_HORIZONS:
    # Find eligible starting points in validation set
    # i must be >= H - 1
    # i + K must be < len(val_orig)
    
    primary_starts = []
    secondary_starts = []
    wall_clocks = []
    
    for i in range(H - 1, len(val_orig) - K):
        # time delta from t to t+K
        t_start = ts_val[i]
        t_end = ts_val[i + K]
        elapsed = (t_end - t_start).total_seconds()
        
        # secondary
        secondary_starts.append(i)
        wall_clocks.append(elapsed)
        
        # primary: all K transitions are exactly 10s
        gaps = (ts_val[i:i+K+1][1:] - ts_val[i:i+K+1][:-1]).total_seconds()
        if np.all(gaps == 10.0):
            primary_starts.append(i)
            
    wall_clocks = np.array(wall_clocks)
    
    print(f"\\n--- HORIZON K={K} ---")
    print(f"Secondary Starts: {len(secondary_starts)}")
    print(f"Wall-clock stats: Median={np.median(wall_clocks):.1f}s, Mean={np.mean(wall_clocks):.1f}s, Min={np.min(wall_clocks):.1f}s, Max={np.max(wall_clocks):.1f}s")
    print(f"Primary (Exactly 10s gaps) Starts: {len(primary_starts)}")
    
    if len(primary_starts) == 0:
        print(f"No eligible primary starts for K={K}. Skipping.")
        continue
        
    # Run Primary Batched Generation
    S_hat_K, prim_preds = batched_recursive_forecast(primary_starts, K)
    
    # Ground Truth & Persistence
    Y_true = np.zeros((len(primary_starts), 77))
    Y_pers = np.zeros((len(primary_starts), 77))
    for b_idx, i in enumerate(primary_starts):
        Y_true[b_idx] = val_orig[i + K]
        Y_pers[b_idx] = val_orig[i] # S_t (persistence)
        
    # Metrics
    res_lstm = calc_metrics(Y_true, S_hat_K)
    res_pers = calc_metrics(Y_true, Y_pers)
    
    # Physical Validity
    neg_prim = (prim_preds < 0).sum()
    neg_recon = (S_hat_K < 0).sum()
    
    derived_err = np.abs(S_hat_K[:, feature_names.index('Flow Pkts/s')] - ((S_hat_K[:, feature_names.index('Tot Fwd Pkts')] + S_hat_K[:, feature_names.index('Tot Bwd Pkts')]) / 10.0))
    violations = (derived_err > 1e-5).sum()
    
    print(f"LSTM RMSE: {res_lstm['RMSE']:.1f} | Persistence RMSE: {res_pers['RMSE']:.1f}")
    print(f"Physical Validity: Neg Primitives={neg_prim}, Neg Reconstructed={neg_recon}, Violations={violations}")
    
    # Feature Metrics
    f_metrics = []
    for f in important_features:
        if f not in feature_names: continue
        f_idx = feature_names.index(f)
        yt = Y_true[:, f_idx]
        lp = S_hat_K[:, f_idx]
        pp = Y_pers[:, f_idx]
        
        lstm_rmse = np.sqrt(mean_squared_error(yt, lp))
        pers_rmse = np.sqrt(mean_squared_error(yt, pp))
        
        f_metrics.append({
            "feature": f,
            "LSTM_RMSE": float(lstm_rmse),
            "Persistence_RMSE": float(pers_rmse)
        })
        
    metrics_v3_recursive["Primary"][f"K={K}"] = {
        "Eligible_Starts": len(primary_starts),
        "Nominal_Seconds": K * 10,
        "LSTM_Global": res_lstm,
        "Persistence_Global": res_pers,
        "Physical_Validity": {
            "Negative_Primitives": int(neg_prim),
            "Negative_Reconstructed": int(neg_recon),
            "Consistency_Violations": int(violations)
        },
        "Feature_Metrics": f_metrics
    }
    
    metrics_v3_recursive["Error_Growth"].append({
        "Horizon_K": K,
        "Nominal_Seconds": K * 10,
        "Eligible_Starts": len(primary_starts),
        "Median_Elapsed": float(np.median(wall_clocks)),
        "LSTM_RMSE": float(res_lstm['RMSE']),
        "Persistence_RMSE": float(res_pers['RMSE']),
        "LSTM_NRMSE": float(res_lstm['NRMSE']),
        "Persistence_NRMSE": float(res_pers['NRMSE'])
    })

# Save JSON
with open(RESULTS_DIR / "world_model" / "phase3_3_validation_metrics.json", "w") as f:
    json.dump(metrics_v3_recursive, f, indent=4)
"""))

    cells.append(nbf.v4.new_markdown_cell("""
### Error Accumulation Over Horizons
"""))

    cells.append(nbf.v4.new_code_cell("""
import matplotlib.pyplot as plt

growth_df = pd.DataFrame(metrics_v3_recursive["Error_Growth"])
display(growth_df)

plt.figure(figsize=(10, 6))
plt.plot(growth_df["Horizon_K"], growth_df["LSTM_RMSE"], marker='o', color='red', label='LSTM V3 RMSE')
plt.plot(growth_df["Horizon_K"], growth_df["Persistence_RMSE"], marker='s', color='blue', linestyle='--', label='Persistence RMSE')
plt.title("Error Accumulation (RMSE) vs Recursive Forecast Horizon (K)")
plt.xlabel("Forecast Horizon K (Transitions)")
plt.ylabel("Global RMSE")
plt.legend()
plt.grid(True)
plt.show()
"""))

    nb['cells'] = cells
    
    with open(nb_path, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
        
    print(f"Appended Phase 3.3 to {nb_path}")

if __name__ == "__main__":
    append_phase3_3()
    nb_path = "notebooks/phase3/Phase3_CyberCast_WorldModel.ipynb"
    subprocess.run(["python", "-m", "jupyter", "nbconvert", "--to", "notebook", "--execute", "--inplace", nb_path], check=True)
    print("Notebook executed successfully.")
