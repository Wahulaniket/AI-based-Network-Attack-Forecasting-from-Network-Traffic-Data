import os
import sys
import json
import torch
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

# Setup paths
ROOT_DIR = Path("d:/working_projects/SIH/cyberCast")
os.chdir(ROOT_DIR)
sys.path.append(str(ROOT_DIR))

from src.inference.model_loader import load_inference_artifacts

def run_analysis():
    print("Loading artifacts...")
    artifacts = load_inference_artifacts(str(ROOT_DIR))
    model = artifacts['model']
    scaler = artifacts['scaler']
    features = artifacts['features']
    threshold = artifacts['threshold']
    device = artifacts['device']

    print("Loading data...")
    df = pd.read_parquet('data/processed/network_states_10s.parquet')
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df = df.sort_values('Timestamp').reset_index(drop=True)

    print("Filtering valid timestamps...")
    mask = (df['Timestamp'] >= '2018-02-14 00:00:00') & (df['Timestamp'] <= '2018-03-02 23:59:59.999999')
    df = df[mask].reset_index(drop=True)

    print("Identifying transitions...")
    if 'binary_attack' in df.columns:
        attack_col = 'binary_attack'
    else:
        raise ValueError("Cannot find attack label column")

    attack_binary = (df[attack_col] > 0).astype(int).values
    df['attack_binary'] = attack_binary

    transitions = []
    for i in range(1, len(df)):
        if attack_binary[i-1] == 0 and attack_binary[i] == 1:
            transitions.append(i) 

    def predict_at_index(idx):
        if idx < 19 or idx >= len(df):
            return None
        window_df = df.iloc[idx-19:idx+1]
        X_raw = window_df[features].values
        
        assert len(X_raw) == 20
        assert X_raw.shape == (20, 89)
        assert window_df['Timestamp'].max() <= df.loc[idx, 'Timestamp']
        
        X_scaled = scaler.transform(X_raw)
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            logit = model(X_tensor)
            prob = torch.sigmoid(logit).item()
        
        return prob

    results = []
    curve_data = []

    out_dir = ROOT_DIR / 'results' / 'phase2_3' / 'pre_attack_analysis'
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Found {len(transitions)} BENIGN -> ATTACK transitions. Processing...")

    for onset_idx in transitions:
        attack_type = "Unknown"
        if 'dominant_label' in df.columns:
            attack_type = df.loc[onset_idx, 'dominant_label']
            
        prob_at_onset = predict_at_index(onset_idx)
        prob_1_before = predict_at_index(onset_idx - 1)
        prob_5_before = predict_at_index(onset_idx - 5)
        prob_10_before = predict_at_index(onset_idx - 10)
        
        if prob_1_before is None:
            continue
            
        curve_row = {'onset_idx': onset_idx, 'attack_type': attack_type, 'onset_timestamp': df.loc[onset_idx, 'Timestamp']}
        for offset in range(-20, 1): 
            idx = onset_idx + offset
            prob = predict_at_index(idx)
            curve_row[f't{offset}'] = prob
            
        curve_data.append(curve_row)
        
        cross_before = False
        lead_time = None
        first_cross_idx = None
        for offset in range(-20, 0):
            if curve_row[f't{offset}'] is not None and curve_row[f't{offset}'] >= threshold:
                if first_cross_idx is None:
                    first_cross_idx = onset_idx + offset
                    cross_before = True
                    
        if cross_before:
            dt = (df.loc[onset_idx, 'Timestamp'] - df.loc[first_cross_idx, 'Timestamp']).total_seconds()
            lead_time = dt
            
        cross_after = False
        if not cross_before and prob_at_onset is not None and prob_at_onset >= threshold:
            cross_after = True
            
        baseline_probs = [curve_row[f't{o}'] for o in range(-20, -10) if curve_row[f't{o}'] is not None]
        baseline_prob = np.mean(baseline_probs) if baseline_probs else 0
        
        results.append({
            'transition_timestamp': df.loc[onset_idx, 'Timestamp'],
            'attack_type': attack_type,
            'prob_10_before': prob_10_before,
            'prob_5_before': prob_5_before,
            'prob_1_before': prob_1_before,  
            'prob_at_onset': prob_at_onset,
            'baseline_prob': baseline_prob,
            'prob_increase_from_baseline': prob_1_before - baseline_prob if prob_1_before else None,
            'prob_increase_immediately_before': prob_1_before - (curve_row.get('t-2') or baseline_prob) if prob_1_before else None,
            'threshold_crossed_before_attack': cross_before,
            'threshold_crossed_only_after_attack': cross_after,
            'lead_time_seconds': lead_time
        })

    results_df = pd.DataFrame(results)
    curve_df = pd.DataFrame(curve_data)

    results_df.to_csv(out_dir / 'transition_analysis.csv', index=False)
    curve_df.to_csv(out_dir / 'pre_attack_probability_curves.csv', index=False)

    summary = []
    for at, group in results_df.groupby('attack_type'):
        summary.append({
            'attack_type': at,
            'num_transitions': len(group),
            'mean_preattack_prob': group['prob_1_before'].mean(),
            'median_preattack_prob': group['prob_1_before'].median(),
            'max_preattack_prob': group['prob_1_before'].max(),
            'mean_during_attack_prob': group['prob_at_onset'].mean(),
            'median_during_attack_prob': group['prob_at_onset'].median(),
            'num_cross_before': group['threshold_crossed_before_attack'].sum(),
            'perc_cross_before': (group['threshold_crossed_before_attack'].sum() / len(group)) * 100,
            'num_cross_after_only': group['threshold_crossed_only_after_attack'].sum(),
            'num_never_cross': len(group) - group['threshold_crossed_before_attack'].sum() - group['threshold_crossed_only_after_attack'].sum()
        })
        
    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(out_dir / 'attack_type_summary.csv', index=False)

    rep_df = results_df.sort_values('prob_1_before', ascending=False).head(10)
    rep_df.to_csv(out_dir / 'representative_transitions.csv', index=False)

    print("\n--- DATA LEAKAGE VERIFICATION ---")
    if len(curve_df) > 0:
        sample_onset_idx = int(curve_df.iloc[0]['onset_idx'])
        t = sample_onset_idx - 1
        sample_df = df.iloc[t-19:t+1]
        
        print(f"Target Onset Timestamp (t+1): {df.loc[sample_onset_idx, 'Timestamp']}")
        print(f"Max History Timestamp (t): {sample_df['Timestamp'].max()}")
        assert sample_df['Timestamp'].max() < df.loc[sample_onset_idx, 'Timestamp']
        print("Verification Passed: Max history timestamp < Target timestamp")
        print("Verification Passed: Target attack traffic does not enter X.")
        print("Verification Passed: X.shape = (1, 20, 89) before model inference.\n")

    report_path = out_dir / 'pre_attack_analysis_report.md'
    total_trans = len(results_df)
    cross_before = int(results_df['threshold_crossed_before_attack'].sum()) if total_trans > 0 else 0
    cross_after = int(results_df['threshold_crossed_only_after_attack'].sum()) if total_trans > 0 else 0
    median_preattack = float(results_df['prob_1_before'].median()) if total_trans > 0 else 0
    median_postattack = float(results_df['prob_at_onset'].median()) if total_trans > 0 else 0
    lead_times = results_df['lead_time_seconds'].dropna()
    median_lead_time = float(lead_times.median()) if not lead_times.empty else None

    with open(report_path, 'w') as f:
        f.write("# CyberCast Pre-Attack Forecasting Analysis\n\n")
        f.write("## 1. Does the frozen Phase 2.3 model produce elevated probability BEFORE attack onset?\n")
        f.write(f"The analysis measured median pre-attack probability of {median_preattack:.4f} compared to median probability during attack of {median_postattack:.4f}. ")
        if cross_before > 0:
            f.write("Yes, in some cases the model produces an elevated probability crossing the threshold before attack onset.\n")
        else:
            f.write("No, the model does not produce threshold-crossing elevated probabilities before attack onset.\n")
            
        f.write(f"\n## 2. How many BENIGN -> ATTACK transitions were analyzed?\n")
        f.write(f"A total of {total_trans} valid transitions were analyzed.\n")
        
        f.write(f"\n## 3. What is the distribution of pre-attack probabilities?\n")
        if total_trans > 0:
            f.write(f"Median: {median_preattack:.4f}\nMean: {results_df['prob_1_before'].mean():.4f}\nMax: {results_df['prob_1_before'].max():.4f}\n")
        
        f.write(f"\n## 4. How many transitions crossed 0.983041 before attack onset?\n")
        f.write(f"{cross_before} transitions crossed the threshold before attack onset.\n")
        
        f.write(f"\n## 5. How many crossed only after attack onset?\n")
        f.write(f"{cross_after} transitions crossed the threshold ONLY after attack onset.\n")
        
        f.write(f"\n## 6. What is the measured lead time for valid pre-attack threshold crossings?\n")
        if lead_times.empty:
            f.write("No pre-attack threshold crossing observed.\n")
        else:
            f.write(f"Median lead time: {median_lead_time} seconds. (Range: {lead_times.min()} - {lead_times.max()} seconds).\n")
            
        f.write(f"\n## 7. Which attack types have measurable pre-attack probability signals?\n")
        types_with_signal = summary_df[summary_df['num_cross_before'] > 0]['attack_type'].tolist() if total_trans > 0 else []
        if types_with_signal:
            f.write(f"Attack types with pre-attack signals: {', '.join(types_with_signal)}\n")
        else:
            f.write("No attack types showed measurable pre-attack signals crossing the threshold.\n")
            
        f.write(f"\n## 8. Are there attack types where the model responds only after attack traffic begins?\n")
        types_after = summary_df[summary_df['num_cross_after_only'] > 0]['attack_type'].tolist() if total_trans > 0 else []
        if types_after:
            f.write(f"Yes, the following attack types primarily showed response only after onset: {', '.join(types_after)}\n")
        else:
            f.write("No.\n")
        
        f.write(f"\n## 9. Does the analysis provide evidence supporting the claim that CyberCast can forecast the next attack window?\n")
        if cross_before > 0:
            f.write("There is some evidence supporting the forecasting claim for certain transitions, but it may be weak or inconsistent depending on the attack types.\n")
        else:
            f.write("The current frozen model demonstrates live attack-response behaviour for this traffic pattern, but this analysis does not establish reliable pre-attack forecasting.\n")
            
        f.write(f"\n## 10. What limitations prevent stronger claims?\n")
        f.write("- Analysis is purely offline and dataset-bound.\n- Transition definition relies on discrete windows, true physical lead time might differ.\n- If the dataset has sequential attacks (e.g. Brute Force followed immediately by DoS), the 'benign' state in between might be brief or noisy.\n")
        
    print("\n============================================================")
    print("ANALYSIS STATUS: PASS")
    print(f"transitions analyzed: {total_trans}")
    print(f"attack types: {results_df['attack_type'].nunique() if total_trans > 0 else 0}")
    print(f"pre-attack threshold crossings: {cross_before}")
    print(f"post-attack-only crossings: {cross_after}")
    print(f"median pre-attack probability: {median_preattack:.4f}")
    print(f"median attack-window probability: {median_postattack:.4f}")
    print(f"measured pre-attack lead time: {median_lead_time} sec" if not lead_times.empty else "measured pre-attack lead time: No pre-attack threshold crossing observed.")
    if cross_before == 0:
        print("conclusion: The model primarily detects attacks after onset rather than forecasting them.")
    else:
        print("conclusion: The model shows some forecasting capabilities, detecting attacks before onset.")

if __name__ == "__main__":
    run_analysis()
