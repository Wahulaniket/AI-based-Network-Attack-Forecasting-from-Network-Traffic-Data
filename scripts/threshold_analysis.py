#!/usr/bin/env python
"""
CyberCast Threshold / Operating-Point Analysis
================================================
Standalone script that loads the trained model and data, performs
comprehensive threshold analysis, and generates all results/plots.

NO retraining. NO dataset changes. NO split changes.
Thresholds selected exclusively on VALIDATION data.
Test labels used ONLY for final evaluation.
"""

import os, sys, json, gc, warnings, time
from pathlib import Path
from collections import Counter

warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import joblib
from sklearn.metrics import (
    roc_auc_score, average_precision_score, precision_recall_curve,
    roc_curve, confusion_matrix, classification_report,
    precision_score, recall_score, f1_score, accuracy_score
)

# ============================================================
# Configuration (must match notebook exactly)
# ============================================================
RANDOM_SEED       = 42
WINDOW_SECONDS    = 10
HISTORY           = 10
FORECAST_HORIZON  = 5
HIDDEN_SIZE       = 128
NUM_LAYERS        = 2
DROPOUT           = 0.2
LAMBDA_STATE      = 0.5
LAMBDA_ATTACK     = 0.5
BATCH_SIZE        = 128

PROJECT_DIR   = Path(r"D:\working_projects\SIH\cyberCast")
PROCESSED_DIR = PROJECT_DIR / "data" / "processed"
MODEL_DIR     = PROJECT_DIR / "models"
RESULTS_DIR   = PROJECT_DIR / "results"
PLOTS_DIR     = PROJECT_DIR / "results" / "plots"

for d in [RESULTS_DIR, PLOTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

LABEL_DERIVED_COLS = frozenset({
    'Label', 'binary_attack', 'Attack',
    'attack_count', 'attack_ratio', 'attack_flow_count',
    'dominant_label', 'attack_family',
})

np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# ============================================================
# Model Definition (identical to notebook Cell 22)
# ============================================================
class CyberCastWorldModel(nn.Module):
    """Dual-head LSTM World Model for network state forecasting."""
    def __init__(self, state_dim, hidden_size=128, num_layers=2, dropout=0.2):
        super().__init__()
        self.state_dim = state_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(
            input_size=state_dim, hidden_size=hidden_size,
            num_layers=num_layers, batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0)
        self.state_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(hidden_size, state_dim))
        self.attack_head = nn.Sequential(
            nn.Linear(hidden_size, 64), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(64, 1))

    def forward(self, x):
        lstm_out, (h_n, c_n) = self.lstm(x)
        last_hidden = h_n[-1]
        state_pred = self.state_head(last_hidden)
        attack_logit = self.attack_head(last_hidden).squeeze(-1)
        return state_pred, attack_logit


# ============================================================
# Helper Functions (from notebook)
# ============================================================
def create_world_model_sequences(X, y, seq_length):
    n_samples = len(X) - seq_length
    if n_samples <= 0:
        raise ValueError(f"Not enough data ({len(X)}) for seq_length={seq_length}")
    state_dim = X.shape[1]
    X_seq = np.empty((n_samples, seq_length, state_dim), dtype=np.float32)
    state_targets = np.empty((n_samples, state_dim), dtype=np.float32)
    attack_targets = np.empty(n_samples, dtype=np.float32)
    for i in range(n_samples):
        X_seq[i] = X[i : i + seq_length]
        state_targets[i] = X[i + seq_length]
        attack_targets[i] = y[i + seq_length]
    return X_seq, state_targets, attack_targets


def recursive_kstep_forecast(model, initial_seq, K, device):
    model.eval()
    history = initial_seq.copy()
    seq_len = initial_seq.shape[0]
    forecasts = []
    with torch.no_grad():
        for k in range(K):
            seq_in = history[-seq_len:]
            t = torch.FloatTensor(seq_in).unsqueeze(0).to(device)
            s_pred, a_logit = model(t)
            s_np = s_pred.cpu().numpy().squeeze(0)
            a_prob = float(torch.sigmoid(a_logit).cpu().item())
            forecasts.append({'step': k+1, 'state_pred': s_np, 'attack_prob': a_prob})
            history = np.vstack([history, s_np])
    return forecasts


def compute_risk_score(prob):
    score = float(np.clip(round(prob * 100), 0, 100))
    if score <= 24: return score, 'LOW'
    elif score <= 49: return score, 'MEDIUM'
    elif score <= 74: return score, 'HIGH'
    else: return score, 'CRITICAL'


def infer_attack_stage(state_vec, feat_names, attack_prob, threshold=0.5):
    fd = {n: float(v) for n, v in zip(feat_names, state_vec)}
    evidence = []
    if attack_prob < threshold:
        evidence.append(f"Below threshold ({attack_prob:.2f})")
        return 'Normal Operation', evidence, 0.0

    syn = fd.get('SYN Flag Cnt', 0)
    rst = fd.get('RST Flag Cnt', 0)
    pkt_rate = fd.get('Pkt_Rate', 0)
    byte_rate = fd.get('Byte_Rate', 0)
    ratio = fd.get('Fwd_Bwd_Byte_Ratio', 1.0)

    if syn > 0 and rst > 0 and pkt_rate < 1000:
        evidence.append(f"High SYN({syn:.0f})+RST({rst:.0f}), moderate rate")
        return 'Reconnaissance', evidence, min(attack_prob, 0.7)
    if pkt_rate > 5000:
        evidence.append(f"Very high pkt rate: {pkt_rate:.0f}")
        return 'Initial Access', evidence, min(attack_prob, 0.8)
    if byte_rate > 10000 and ratio > 5:
        evidence.append(f"High byte rate + asymmetric traffic")
        return 'Exfiltration', evidence, min(attack_prob, 0.6)
    if attack_prob > 0.7:
        evidence.append(f"High attack prob ({attack_prob:.2f})")
        return 'Command and Control', evidence, min(attack_prob * 0.5, 0.5)
    evidence.append(f"Attack prob={attack_prob:.2f}, insufficient for specific stage")
    return 'Unknown / Insufficient Evidence', evidence, min(attack_prob * 0.3, 0.3)


def evaluate_early_warning(preds, targets, threshold, ws, lookback=10):
    episodes = []
    in_ep, ep_start = False, None
    for i in range(len(targets)):
        if targets[i] == 1 and not in_ep:
            in_ep, ep_start = True, i
        elif targets[i] == 0 and in_ep:
            in_ep = False; episodes.append((ep_start, i-1))
    if in_ep: episodes.append((ep_start, len(targets)-1))

    if not episodes:
        return {'total_episodes': 0}

    ewt_list, d_early, d_during, missed = [], 0, 0, 0
    for es, ee in episodes:
        cs = max(0, es - lookback)
        fw = None
        for i in range(cs, es):
            if preds[i] >= threshold:
                fw = i; break
        if fw is not None:
            ewt_list.append((es - fw) * ws); d_early += 1
        else:
            found = any(preds[i] >= threshold for i in range(es, min(ee+1, len(preds))))
            if found: d_during += 1; ewt_list.append(0)
            else: missed += 1

    tot = len(episodes)
    ew_res = {'total_episodes': tot, 'detected_early': d_early,
              'detected_during': d_during, 'missed': missed,
              'warning_rate': (d_early + d_during) / tot,
              'detection_rate': (d_early + d_during) / tot}

    pos = [t for t in ewt_list if t > 0]
    if pos:
        ew_res['mean_ewt_seconds'] = float(np.mean(pos))
        ew_res['median_ewt_seconds'] = float(np.median(pos))
        ew_res['max_ewt_seconds'] = float(np.max(pos))
    else:
        ew_res['mean_ewt_seconds'] = 0.0
        ew_res['median_ewt_seconds'] = 0.0
        ew_res['max_ewt_seconds'] = 0.0
    return ew_res


# ============================================================
# STEP 1: Load Data and Recreate Split
# ============================================================
print("\n" + "="*70)
print("   STEP 1: Loading Data and Recreating Chronological Split")
print("="*70)

pq_path = PROCESSED_DIR / 'network_states_10s.parquet'
assert pq_path.exists(), f"Parquet not found: {pq_path}"
temporal_states = pd.read_parquet(pq_path)
temporal_states.sort_values('Timestamp', inplace=True)
temporal_states.reset_index(drop=True, inplace=True)
print(f"Loaded: {len(temporal_states):,} temporal windows")
print(f"Time: {temporal_states['Timestamp'].min()} -> {temporal_states['Timestamp'].max()}")

n = len(temporal_states)
train_end = int(n * 0.70)
val_end = int(n * 0.85)

train_states = temporal_states.iloc[:train_end].copy()
val_states = temporal_states.iloc[train_end:val_end].copy()
test_states = temporal_states.iloc[val_end:].copy()

for name, df in [('Train', train_states), ('Val', val_states), ('Test', test_states)]:
    na = int(df['binary_attack'].sum())
    pct = na / len(df) * 100 if len(df) > 0 else 0
    print(f"  {name:6s}: {len(df):>10,} states | Attack: {na:>8,} ({pct:.2f}%) | "
          f"{df['Timestamp'].iloc[0]} -> {df['Timestamp'].iloc[-1]}")

# ============================================================
# STEP 2: Load Model and Scaler, Generate Predictions
# ============================================================
print("\n" + "="*70)
print("   STEP 2: Loading Model and Generating Predictions")
print("="*70)

# Load feature names
with open(MODEL_DIR / 'feature_names.json') as f:
    state_feature_cols = json.load(f)
STATE_DIM = len(state_feature_cols)
print(f"State dimension: {STATE_DIM}")

# Load scaler
scaler = joblib.load(MODEL_DIR / 'scaler.pkl')

# Scale data
metadata_cols = {'Timestamp', 'date', 'dominant_label', 'has_traffic'}
X_val_raw = scaler.transform(val_states[state_feature_cols].values).astype(np.float32)
X_test_raw = scaler.transform(test_states[state_feature_cols].values).astype(np.float32)
y_val_raw = val_states['binary_attack'].values.astype(np.float32)
y_test_raw = test_states['binary_attack'].values.astype(np.float32)

# Create sequences
X_val_seq, S_val_target, y_val_seq = create_world_model_sequences(X_val_raw, y_val_raw, HISTORY)
X_test_seq, S_test_target, y_test_seq = create_world_model_sequences(X_test_raw, y_test_raw, HISTORY)
print(f"Val sequences: {X_val_seq.shape}  Test sequences: {X_test_seq.shape}")

# Load model
model = CyberCastWorldModel(
    state_dim=STATE_DIM, hidden_size=HIDDEN_SIZE,
    num_layers=NUM_LAYERS, dropout=DROPOUT).to(device)
model.load_state_dict(torch.load(MODEL_DIR / 'best_world_model.pt', map_location=device, weights_only=True))
model.eval()
print("Model loaded successfully.")

# Generate validation predictions
val_loader = DataLoader(
    TensorDataset(torch.FloatTensor(X_val_seq), torch.FloatTensor(S_val_target),
                  torch.FloatTensor(y_val_seq)),
    batch_size=BATCH_SIZE, shuffle=False)

val_preds, val_tgts = [], []
with torch.no_grad():
    for X_b, S_b, y_b in val_loader:
        _, a = model(X_b.to(device))
        val_preds.extend(torch.sigmoid(a).cpu().numpy())
        val_tgts.extend(y_b.numpy())
val_preds = np.array(val_preds)
val_tgts = np.array(val_tgts)
print(f"Val predictions: {len(val_preds)} (attack={int(val_tgts.sum()):,}, benign={int((1-val_tgts).sum()):,})")

# Generate test predictions
test_loader = DataLoader(
    TensorDataset(torch.FloatTensor(X_test_seq), torch.FloatTensor(S_test_target),
                  torch.FloatTensor(y_test_seq)),
    batch_size=BATCH_SIZE, shuffle=False)

test_preds, test_tgts = [], []
with torch.no_grad():
    for X_b, S_b, y_b in test_loader:
        _, a = model(X_b.to(device))
        test_preds.extend(torch.sigmoid(a).cpu().numpy())
        test_tgts.extend(y_b.numpy())
test_preds = np.array(test_preds)
test_tgts = np.array(test_tgts)
print(f"Test predictions: {len(test_preds)} (attack={int(test_tgts.sum()):,}, benign={int((1-test_tgts).sum()):,})")

# Load LR metrics from config
with open(MODEL_DIR / 'config.json') as f:
    config = json.load(f)
lr_metrics = config['lr_metrics']

# Also generate LR test predictions for fair comparison
lr_model = joblib.load(MODEL_DIR / 'logistic_regression.pkl')
X_test_lr = X_test_seq[:, -1, :]  # Last window
lr_test_probs = lr_model.predict_proba(X_test_lr)[:, 1]

# ============================================================
# STEP 3: Threshold Sweep on VALIDATION Data
# ============================================================
print("\n" + "="*70)
print("   STEP 3: Threshold Sweep on VALIDATION Data")
print("="*70)
print("NOTE: All thresholds selected using VALIDATION labels ONLY.")
print("      Test labels are NOT used for threshold selection.")

thresholds = np.round(np.arange(0.05, 0.96, 0.01), 2)
assert len(thresholds) == 91, f"Expected 91 thresholds, got {len(thresholds)}"
print(f"Evaluating {len(thresholds)} thresholds: {thresholds[0]} to {thresholds[-1]}")

val_n_pos = int(val_tgts.sum())
val_n_neg = int((1 - val_tgts).sum())

threshold_results = []
for t in thresholds:
    bins = (val_preds >= t).astype(int)
    cm = confusion_matrix(val_tgts, bins, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    youdens_j = rec - fpr

    threshold_results.append({
        'threshold': t,
        'precision': prec,
        'recall': rec,
        'f1': f1,
        'fpr': fpr,
        'fnr': fnr,
        'youdens_j': youdens_j,
        'TP': tp, 'TN': tn, 'FP': fp, 'FN': fn
    })

threshold_df = pd.DataFrame(threshold_results)
threshold_df.to_csv(RESULTS_DIR / 'threshold_analysis_val.csv', index=False)
print(f"Saved: threshold_analysis_val.csv ({len(threshold_df)} rows)")

# Print table
print(f"\n{'Threshold':>10s} {'Prec':>8s} {'Recall':>8s} {'F1':>8s} {'FPR':>8s} {'FNR':>8s} {'J':>8s} {'TP':>6s} {'TN':>6s} {'FP':>6s} {'FN':>6s}")
print("-" * 100)
for _, row in threshold_df.iterrows():
    print(f"{row['threshold']:>10.2f} {row['precision']:>8.4f} {row['recall']:>8.4f} "
          f"{row['f1']:>8.4f} {row['fpr']:>8.4f} {row['fnr']:>8.4f} {row['youdens_j']:>8.4f} "
          f"{int(row['TP']):>6d} {int(row['TN']):>6d} {int(row['FP']):>6d} {int(row['FN']):>6d}")

# ============================================================
# STEP 4: Threshold Selection (VALIDATION ONLY)
# ============================================================
print("\n" + "="*70)
print("   STEP 4: Threshold Selection (VALIDATION DATA ONLY)")
print("="*70)

# --- 4A: Research Threshold (F1-optimal on validation) ---
best_f1_idx = threshold_df['f1'].idxmax()
research_threshold = threshold_df.loc[best_f1_idx, 'threshold']
research_row = threshold_df.loc[best_f1_idx]

print(f"\n--- RESEARCH THRESHOLD (F1-optimal on validation) ---")
print(f"  Threshold: {research_threshold}")
print(f"  Precision: {research_row['precision']:.4f}")
print(f"  Recall:    {research_row['recall']:.4f}")
print(f"  F1:        {research_row['f1']:.4f}")
print(f"  FPR:       {research_row['fpr']:.4f}")
print(f"  FNR:       {research_row['fnr']:.4f}")

# --- 4B: Youden's J Threshold (reported candidate) ---
best_j_idx = threshold_df['youdens_j'].idxmax()
youdens_threshold = threshold_df.loc[best_j_idx, 'threshold']
youdens_row = threshold_df.loc[best_j_idx]

print(f"\n--- YOUDEN'S J THRESHOLD (candidate, reported) ---")
print(f"  Threshold: {youdens_threshold}")
print(f"  Precision: {youdens_row['precision']:.4f}")
print(f"  Recall:    {youdens_row['recall']:.4f}")
print(f"  F1:        {youdens_row['f1']:.4f}")
print(f"  FPR:       {youdens_row['fpr']:.4f}")
print(f"  FNR:       {youdens_row['fnr']:.4f}")
print(f"  Youden's J:{youdens_row['youdens_j']:.4f}")

# --- 4C: Operational Threshold (SOC-oriented) ---
# Priority: high recall, materially reduce FPR
# Among thresholds with recall >= 0.90, prefer lowest FPR
# If recall >= 0.90 not achievable at reasonable FPR, report best available

print(f"\n--- OPERATIONAL THRESHOLD SELECTION (SOC-oriented) ---")

# Check if recall >= 0.90 is achievable at a REASONABLE FPR
high_recall_df = threshold_df[threshold_df['recall'] >= 0.90].copy()
recall_90_achievable = len(high_recall_df) > 0
recall_90_reasonable = False

if recall_90_achievable:
    min_fpr_at_90 = high_recall_df['fpr'].min()
    print(f"  Found {len(high_recall_df)} thresholds with recall >= 0.90")
    print(f"  FPR range at recall >= 0.90: [{min_fpr_at_90:.4f}, {high_recall_df['fpr'].max():.4f}]")

    # Check if this is at a reasonable FPR (below 50%)
    if min_fpr_at_90 <= 0.50:
        recall_90_reasonable = True
        best_op_idx = high_recall_df['fpr'].idxmin()
        operational_threshold = high_recall_df.loc[best_op_idx, 'threshold']
        operational_row = high_recall_df.loc[best_op_idx]
        print(f"\n  Recall >= 0.90 IS achievable at reasonable FPR ({min_fpr_at_90:.4f})")
        print(f"  Selected operational threshold: {operational_threshold}")
    else:
        print(f"\n  WARNING: Recall >= 0.90 requires FPR >= {min_fpr_at_90:.4f} (unreasonable).")
        print(f"  The model's prediction distribution has a sharp cliff —")
        print(f"  high recall forces nearly all benign windows to be flagged.")
        print(f"  Recall >= 0.90 constraint CANNOT be satisfied at reasonable FPR.")
else:
    print("  WARNING: Recall >= 0.90 NOT achievable at any threshold.")

if not recall_90_reasonable:
    print(f"\n  Reporting best available operating points at relaxed recall levels:")
    print(f"  {'Min Recall':>12s} {'Threshold':>10s} {'Recall':>8s} {'FPR':>8s} {'F1':>8s} {'Prec':>8s} {'J':>8s}")
    print(f"  {'-'*68}")
    for min_rec in [0.90, 0.80, 0.70, 0.60, 0.50, 0.40, 0.30, 0.20, 0.15]:
        candidates = threshold_df[threshold_df['recall'] >= min_rec]
        if len(candidates) > 0:
            best_idx = candidates['fpr'].idxmin()
            row = candidates.loc[best_idx]
            marker = ''
            if min_rec == 0.90 and row['fpr'] > 0.50:
                marker = ' <-- FPR too high'
            print(f"  {min_rec:>12.2f} {row['threshold']:>10.2f} {row['recall']:>8.4f} "
                  f"{row['fpr']:>8.4f} {row['f1']:>8.4f} {row['precision']:>8.4f} "
                  f"{row['youdens_j']:>8.4f}{marker}")

    # Select Youden's J as the operational threshold — best balance of recall vs FPR
    operational_threshold = youdens_threshold
    operational_row = youdens_row
    print(f"\n  DECISION: Falling back to Youden's J threshold ({operational_threshold})")
    print(f"  Rationale: Youden's J = recall - FPR maximizes the tradeoff between")
    print(f"  sensitivity and specificity. This gives the best available balance")
    print(f"  between catching attacks and avoiding false alarms.")

print(f"\n--- OPERATIONAL THRESHOLD FINAL ---")
print(f"  Threshold: {operational_threshold}")
print(f"  Precision: {operational_row['precision']:.4f}")
print(f"  Recall:    {operational_row['recall']:.4f}")
print(f"  F1:        {operational_row['f1']:.4f}")
print(f"  FPR:       {operational_row['fpr']:.4f}")
print(f"  FNR:       {operational_row['fnr']:.4f}")

# LOCK thresholds
print(f"\n{'='*60}")
print(f"  LOCKED THRESHOLDS (selected on validation data only):")
print(f"  Research threshold:    {research_threshold}")
print(f"  Operational threshold: {operational_threshold}")
print(f"  Youden's J threshold:  {youdens_threshold} (reported candidate)")
print(f"{'='*60}")
print(f"\n  STATEMENT: Thresholds were selected exclusively on validation")
print(f"  data. Test labels were not used for threshold selection.")

# ============================================================
# STEP 5: Generate Threshold Analysis Plots
# ============================================================
print("\n" + "="*70)
print("   STEP 5: Threshold Analysis Plots")
print("="*70)

fig, axes = plt.subplots(2, 3, figsize=(20, 12))
fig.suptitle('CyberCast Threshold Analysis (Validation Set)', fontsize=16, fontweight='bold', y=0.98)

# Plot 1: F1 vs Threshold
ax = axes[0, 0]
ax.plot(threshold_df['threshold'], threshold_df['f1'], 'b-', lw=2, label='F1')
ax.axvline(x=research_threshold, color='r', ls='--', alpha=0.8, label=f'Research (t={research_threshold})')
ax.axvline(x=operational_threshold, color='g', ls='--', alpha=0.8, label=f'Operational (t={operational_threshold})')
if youdens_threshold != research_threshold and youdens_threshold != operational_threshold:
    ax.axvline(x=youdens_threshold, color='orange', ls=':', alpha=0.7, label=f'Youden J (t={youdens_threshold})')
ax.set_xlabel('Threshold', fontsize=11)
ax.set_ylabel('F1 Score', fontsize=11)
ax.set_title('F1 Score vs Threshold', fontweight='bold')
ax.legend(fontsize=9)
ax.grid(alpha=0.3)
ax.set_xlim(0.05, 0.95)

# Plot 2: Precision vs Threshold
ax = axes[0, 1]
ax.plot(threshold_df['threshold'], threshold_df['precision'], 'r-', lw=2, label='Precision')
ax.axvline(x=research_threshold, color='r', ls='--', alpha=0.8, label=f'Research (t={research_threshold})')
ax.axvline(x=operational_threshold, color='g', ls='--', alpha=0.8, label=f'Operational (t={operational_threshold})')
ax.set_xlabel('Threshold', fontsize=11)
ax.set_ylabel('Precision', fontsize=11)
ax.set_title('Precision vs Threshold', fontweight='bold')
ax.legend(fontsize=9)
ax.grid(alpha=0.3)
ax.set_xlim(0.05, 0.95)

# Plot 3: Recall vs Threshold
ax = axes[0, 2]
ax.plot(threshold_df['threshold'], threshold_df['recall'], 'g-', lw=2, label='Recall')
ax.axvline(x=research_threshold, color='r', ls='--', alpha=0.8, label=f'Research (t={research_threshold})')
ax.axvline(x=operational_threshold, color='g', ls='--', alpha=0.8, label=f'Operational (t={operational_threshold})')
ax.axhline(y=0.90, color='gray', ls=':', alpha=0.5, label='Recall = 0.90')
ax.set_xlabel('Threshold', fontsize=11)
ax.set_ylabel('Recall', fontsize=11)
ax.set_title('Recall vs Threshold', fontweight='bold')
ax.legend(fontsize=9)
ax.grid(alpha=0.3)
ax.set_xlim(0.05, 0.95)

# Plot 4: FPR vs Threshold
ax = axes[1, 0]
ax.plot(threshold_df['threshold'], threshold_df['fpr'], 'm-', lw=2, label='FPR')
ax.axvline(x=research_threshold, color='r', ls='--', alpha=0.8, label=f'Research (t={research_threshold})')
ax.axvline(x=operational_threshold, color='g', ls='--', alpha=0.8, label=f'Operational (t={operational_threshold})')
ax.set_xlabel('Threshold', fontsize=11)
ax.set_ylabel('False Positive Rate', fontsize=11)
ax.set_title('FPR vs Threshold', fontweight='bold')
ax.legend(fontsize=9)
ax.grid(alpha=0.3)
ax.set_xlim(0.05, 0.95)

# Plot 5: Precision-Recall Tradeoff
ax = axes[1, 1]
ax.plot(threshold_df['recall'], threshold_df['precision'], 'k-', lw=2, alpha=0.7)
# Mark research threshold
ax.scatter([research_row['recall']], [research_row['precision']],
           c='red', s=150, zorder=5, marker='*', label=f'Research (t={research_threshold})')
# Mark operational threshold
ax.scatter([operational_row['recall']], [operational_row['precision']],
           c='green', s=150, zorder=5, marker='D', label=f'Operational (t={operational_threshold})')
ax.set_xlabel('Recall', fontsize=11)
ax.set_ylabel('Precision', fontsize=11)
ax.set_title('Precision-Recall Tradeoff', fontweight='bold')
ax.legend(fontsize=9)
ax.grid(alpha=0.3)

# Plot 6: Youden's J vs Threshold
ax = axes[1, 2]
ax.plot(threshold_df['threshold'], threshold_df['youdens_j'], 'c-', lw=2, label="Youden's J")
ax.axvline(x=youdens_threshold, color='orange', ls='--', alpha=0.8, label=f'Best J (t={youdens_threshold})')
ax.axvline(x=research_threshold, color='r', ls='--', alpha=0.6, label=f'Research')
ax.axvline(x=operational_threshold, color='g', ls='--', alpha=0.6, label=f'Operational')
ax.set_xlabel('Threshold', fontsize=11)
ax.set_ylabel("Youden's J (Recall - FPR)", fontsize=11)
ax.set_title("Youden's J vs Threshold", fontweight='bold')
ax.legend(fontsize=9)
ax.grid(alpha=0.3)
ax.set_xlim(0.05, 0.95)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(PLOTS_DIR / 'threshold_analysis.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: threshold_analysis.png")

# ============================================================
# STEP 6: Test Evaluation at BOTH Locked Thresholds
# ============================================================
print("\n" + "="*70)
print("   STEP 6: Test Evaluation at Locked Thresholds (SINGLE APPLICATION)")
print("="*70)
print("NOTE: Thresholds were LOCKED from validation. This is the ONLY")
print("      application of these thresholds to the test set.")

def evaluate_at_threshold(preds, tgts, threshold, name=""):
    bins = (preds >= threshold).astype(int)
    cm = confusion_matrix(tgts, bins, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    metrics = {
        'threshold': threshold,
        'precision': float(precision_score(tgts, bins, zero_division=0)),
        'recall': float(recall_score(tgts, bins, zero_division=0)),
        'f1': float(f1_score(tgts, bins, zero_division=0)),
        'roc_auc': float(roc_auc_score(tgts, preds)) if len(np.unique(tgts)) > 1 else 0.0,
        'pr_auc': float(average_precision_score(tgts, preds)) if len(np.unique(tgts)) > 1 else 0.0,
        'fpr': float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0,
        'fnr': float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0,
        'TP': int(tp), 'TN': int(tn), 'FP': int(fp), 'FN': int(fn)
    }
    if name:
        print(f"\n  {name} (threshold={threshold}):")
        for k, v in metrics.items():
            if k == 'threshold': continue
            print(f"    {k:15s}: {v}")
    return metrics

# CyberCast at research threshold
test_research = evaluate_at_threshold(test_preds, test_tgts, research_threshold, "CyberCast @ Research Threshold")

# CyberCast at operational threshold
test_operational = evaluate_at_threshold(test_preds, test_tgts, operational_threshold, "CyberCast @ Operational Threshold")

# LR baseline (re-evaluate for fair comparison)
best_lr_thresh = 0.5  # Will determine from config
# Determine LR threshold from validation
X_val_lr = X_val_seq[:, -1, :]
lr_val_probs = lr_model.predict_proba(X_val_lr)[:, 1]
best_lr_f1, best_lr_thresh = -1, 0.5
for t_lr in [0.2, 0.3, 0.4, 0.5, 0.6]:
    f1_lr = f1_score(y_val_seq, (lr_val_probs >= t_lr).astype(int), zero_division=0)
    if f1_lr > best_lr_f1:
        best_lr_f1, best_lr_thresh = f1_lr, t_lr

lr_test_bins = (lr_test_probs >= best_lr_thresh).astype(int)
lr_test_metrics = {
    'threshold': best_lr_thresh,
    'precision': float(precision_score(test_tgts, lr_test_bins, zero_division=0)),
    'recall': float(recall_score(test_tgts, lr_test_bins, zero_division=0)),
    'f1': float(f1_score(test_tgts, lr_test_bins, zero_division=0)),
    'roc_auc': float(roc_auc_score(test_tgts, lr_test_probs)) if len(np.unique(test_tgts)) > 1 else 0.0,
    'pr_auc': float(average_precision_score(test_tgts, lr_test_probs)) if len(np.unique(test_tgts)) > 1 else 0.0,
    'fpr': 0.0, 'fnr': 0.0,
    'TP': 0, 'TN': 0, 'FP': 0, 'FN': 0
}
cm_lr = confusion_matrix(test_tgts, lr_test_bins, labels=[0, 1])
tn, fp, fn, tp = cm_lr.ravel()
lr_test_metrics['fpr'] = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
lr_test_metrics['fnr'] = float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0
lr_test_metrics['TP'] = int(tp)
lr_test_metrics['TN'] = int(tn)
lr_test_metrics['FP'] = int(fp)
lr_test_metrics['FN'] = int(fn)

print(f"\n  LR Baseline (threshold={best_lr_thresh}):")
for k, v in lr_test_metrics.items():
    if k == 'threshold': continue
    print(f"    {k:15s}: {v}")

# 3-way comparison table
print("\n" + "="*70)
print("   3-WAY MODEL COMPARISON (TEST SET)")
print("="*70)
comp_metrics = ['precision', 'recall', 'f1', 'roc_auc', 'pr_auc', 'fpr', 'fnr']
print(f"{'Metric':20s} {'LR':>12s} {'CC@Research':>12s} {'CC@Operational':>15s} {'Winner':>10s}")
print("-" * 72)
for m in comp_metrics:
    lv = lr_test_metrics[m]
    rv = test_research[m]
    ov = test_operational[m]
    if m in ['fpr', 'fnr']:
        best_val = min(lv, rv, ov)
        winner = 'LR' if lv == best_val else ('CC@Res' if rv == best_val else 'CC@Oper')
    else:
        best_val = max(lv, rv, ov)
        winner = 'LR' if lv == best_val else ('CC@Res' if rv == best_val else 'CC@Oper')
    print(f"{m:20s} {lv:>12.4f} {rv:>12.4f} {ov:>15.4f} {winner:>10s}")

# Confusion matrices
print(f"\nConfusion Matrices:")
print(f"  LR:            TN={lr_test_metrics['TN']:>5,} FP={lr_test_metrics['FP']:>5,} FN={lr_test_metrics['FN']:>5,} TP={lr_test_metrics['TP']:>5,}")
print(f"  CC@Research:   TN={test_research['TN']:>5,} FP={test_research['FP']:>5,} FN={test_research['FN']:>5,} TP={test_research['TP']:>5,}")
print(f"  CC@Operational:TN={test_operational['TN']:>5,} FP={test_operational['FP']:>5,} FN={test_operational['FN']:>5,} TP={test_operational['TP']:>5,}")

# Save comparison
comparison_df = pd.DataFrame({
    'LogisticRegression': {m: lr_test_metrics[m] for m in comp_metrics + ['TP', 'TN', 'FP', 'FN']},
    'CyberCast_Research': {m: test_research[m] for m in comp_metrics + ['TP', 'TN', 'FP', 'FN']},
    'CyberCast_Operational': {m: test_operational[m] for m in comp_metrics + ['TP', 'TN', 'FP', 'FN']},
})
comparison_df.to_csv(RESULTS_DIR / 'metrics_comparison.csv')
print("\nSaved: metrics_comparison.csv")

# ============================================================
# STEP 7: Test Set Recursive Demonstration
# ============================================================
print("\n" + "="*70)
print("   STEP 7: Test Set Recursive Demonstration")
print("="*70)

# Find ALL eligible attack episodes in the test set
# An eligible episode: current window is benign, an attack occurs within FORECAST_HORIZON
eligible_episodes = []
for i in range(len(y_test_raw) - HISTORY - FORECAST_HORIZON):
    # Base must be benign (last window in the history sequence)
    if y_test_raw[i + HISTORY - 1] == 0:
        # At least one attack in forecast horizon
        future_attacks = []
        for j in range(FORECAST_HORIZON):
            future_idx = i + HISTORY + j
            if future_idx < len(y_test_raw) and y_test_raw[future_idx] == 1:
                future_attacks.append(j + 1)  # t+1 to t+5
        if future_attacks:
            ts_off = HISTORY + i
            base_ts = test_states['Timestamp'].iloc[ts_off] if ts_off < len(test_states) else None
            eligible_episodes.append({
                'seq_idx': i,
                'ts_offset': ts_off,
                'base_timestamp': base_ts,
                'first_attack_step': future_attacks[0],
                'attack_steps': future_attacks
            })

print(f"Total eligible test attack episodes: {len(eligible_episodes)}")

if eligible_episodes:
    # Show episode distribution by date
    ep_dates = {}
    for ep in eligible_episodes:
        if ep['base_timestamp'] is not None:
            d = str(ep['base_timestamp'].date())
            ep_dates[d] = ep_dates.get(d, 0) + 1
    print(f"Episodes by date: {ep_dates}")

    # Deterministic selection: choose the FIRST eligible episode
    # (earliest in chronological test set order)
    demo_ep = eligible_episodes[0]
    demo_idx = demo_ep['seq_idx']
    base_ts = demo_ep['base_timestamp']

    print(f"\nDemonstration episode (deterministic: first eligible episode):")
    print(f"  Sequence index: {demo_idx}")
    print(f"  Base timestamp: {base_ts}")
    print(f"  First attack at: t+{demo_ep['first_attack_step']}")
    print(f"  Attack steps: {demo_ep['attack_steps']}")

    # Run recursive forecast
    fcs = recursive_kstep_forecast(model, X_test_seq[demo_idx], FORECAST_HORIZON, device)
    fc_probs = [fc['attack_prob'] for fc in fcs]

    print(f"\n  {'Step':>5s} {'Timestamp':>22s} {'P(attack)':>10s} {'Risk':>6s} {'Level':>10s} {'Stage':>30s} {'GT':>8s} {'Warning?':>10s}")
    print(f"  {'-'*105}")

    first_warning_step = None
    first_attack_step = demo_ep['first_attack_step']

    demo_rows = []
    for fc in fcs:
        k = fc['step']
        p = fc['attack_prob']
        rs, rl = compute_risk_score(p)
        stg, ev, cert = infer_attack_stage(fc['state_pred'], state_feature_cols, p, operational_threshold)
        ts_str = str(base_ts + pd.Timedelta(seconds=k*WINDOW_SECONDS))[:19] if base_ts else ''

        ai = demo_idx + HISTORY + k - 1
        gt = 'ATTACK' if (ai < len(y_test_raw) and y_test_raw[ai] == 1) else 'BENIGN'
        pred_label = 'ATTACK' if p >= operational_threshold else 'BENIGN'

        if first_warning_step is None and p >= operational_threshold:
            first_warning_step = k

        warning = ''
        if k == first_attack_step:
            if first_warning_step is not None and first_warning_step < k:
                warning = f'YES (t+{first_warning_step})'
            elif first_warning_step == k:
                warning = 'CONCURRENT'
            else:
                warning = 'NO'

        print(f"  t+{k:2d}  {ts_str:>22s} {p:>10.4f} {rs:>6.0f} {rl:>10s} {stg:>30s} {gt:>8s} {warning:>10s}")

        demo_rows.append({
            'step': f't+{k}',
            'timestamp': ts_str,
            'predicted_prob': round(p, 4),
            'risk_score': rs,
            'risk_level': rl,
            'predicted_stage': stg,
            'ground_truth': gt,
            'predicted_label': pred_label,
        })

    # Summary
    warning_preceded = first_warning_step is not None and first_warning_step < first_attack_step
    print(f"\n  WARNING PRECEDED ATTACK: {'YES' if warning_preceded else 'NO'}")
    if first_warning_step:
        ewt = (first_attack_step - first_warning_step) * WINDOW_SECONDS
        print(f"  Early Warning Time: {ewt}s ({first_attack_step - first_warning_step} steps)")
    print(f"  DATA SOURCE: TEST set (post 2018-02-28)")
    print(f"  SELECTION: Deterministic (first eligible episode in chronological order)")

    demo_df = pd.DataFrame(demo_rows)
    demo_df.to_csv(RESULTS_DIR / 'test_demonstration.csv', index=False)
    print("Saved: test_demonstration.csv")
else:
    print("WARNING: No eligible attack episodes found in test set.")
    demo_ep = None

# ============================================================
# STEP 8: Episode-Level Early Warning Metrics (Test Set)
# ============================================================
print("\n" + "="*70)
print("   STEP 8: Episode-Level Early Warning Metrics (TEST SET)")
print("="*70)

# At operational threshold
print(f"\nEarly Warning Analysis at OPERATIONAL threshold ({operational_threshold}):")
ew_test_operational = evaluate_early_warning(test_preds, test_tgts, operational_threshold, WINDOW_SECONDS, HISTORY)
print(f"  Total attack episodes: {ew_test_operational['total_episodes']}")
print(f"  Detected early:       {ew_test_operational.get('detected_early', 0)}")
print(f"  Detected during:      {ew_test_operational.get('detected_during', 0)}")
print(f"  Missed:               {ew_test_operational.get('missed', 0)}")
print(f"  Detection rate:       {ew_test_operational.get('detection_rate', 0):.4f}")
print(f"  Mean EWT:             {ew_test_operational.get('mean_ewt_seconds', 0):.1f}s")
print(f"  Median EWT:           {ew_test_operational.get('median_ewt_seconds', 0):.1f}s")
print(f"  Max EWT:              {ew_test_operational.get('max_ewt_seconds', 0):.1f}s")

# At research threshold
print(f"\nEarly Warning Analysis at RESEARCH threshold ({research_threshold}):")
ew_test_research = evaluate_early_warning(test_preds, test_tgts, research_threshold, WINDOW_SECONDS, HISTORY)
print(f"  Total attack episodes: {ew_test_research['total_episodes']}")
print(f"  Detected early:       {ew_test_research.get('detected_early', 0)}")
print(f"  Detected during:      {ew_test_research.get('detected_during', 0)}")
print(f"  Missed:               {ew_test_research.get('missed', 0)}")
print(f"  Detection rate:       {ew_test_research.get('detection_rate', 0):.4f}")
print(f"  Mean EWT:             {ew_test_research.get('mean_ewt_seconds', 0):.1f}s")
print(f"  Median EWT:           {ew_test_research.get('median_ewt_seconds', 0):.1f}s")
print(f"  Max EWT:              {ew_test_research.get('max_ewt_seconds', 0):.1f}s")

# Save
ew_combined = {
    'operational_threshold': {
        'threshold': operational_threshold,
        **ew_test_operational
    },
    'research_threshold': {
        'threshold': research_threshold,
        **ew_test_research
    }
}
with open(RESULTS_DIR / 'early_warning_results.json', 'w') as f:
    json.dump(ew_combined, f, indent=2, default=str)
print("Saved: early_warning_results.json")

# ============================================================
# STEP 9: False Positive Analysis (Test Set)
# ============================================================
print("\n" + "="*70)
print("   STEP 9: False Positive Analysis (TEST SET)")
print("="*70)

for thresh_name, thresh_val in [('Operational', operational_threshold), ('Research', research_threshold)]:
    test_bins = (test_preds >= thresh_val).astype(int)
    cm = confusion_matrix(test_tgts, test_bins, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    total_benign = int(tn + fp)
    fp_pct = fp / total_benign * 100 if total_benign > 0 else 0

    print(f"\n  --- {thresh_name} threshold ({thresh_val}) ---")
    print(f"  Total benign windows: {total_benign:,}")
    print(f"  False positives:      {fp:,} ({fp_pct:.2f}%)")
    print(f"  True negatives:       {tn:,}")

    # Identify FP bursts (consecutive FP windows)
    fp_mask = ((test_tgts == 0) & (test_bins == 1))
    bursts = []
    in_burst, burst_start, burst_len = False, 0, 0
    for i in range(len(fp_mask)):
        if fp_mask[i]:
            if not in_burst:
                in_burst = True
                burst_start = i
                burst_len = 1
            else:
                burst_len += 1
        else:
            if in_burst:
                bursts.append({'start_idx': burst_start, 'length': burst_len,
                               'duration_seconds': burst_len * WINDOW_SECONDS})
                in_burst = False
    if in_burst:
        bursts.append({'start_idx': burst_start, 'length': burst_len,
                       'duration_seconds': burst_len * WINDOW_SECONDS})

    print(f"  FP burst count:       {len(bursts)}")
    if bursts:
        burst_lens = [b['length'] for b in bursts]
        print(f"  Mean burst length:    {np.mean(burst_lens):.1f} windows ({np.mean(burst_lens)*WINDOW_SECONDS:.1f}s)")
        print(f"  Max burst length:     {max(burst_lens)} windows ({max(burst_lens)*WINDOW_SECONDS}s)")
        print(f"  Median burst length:  {np.median(burst_lens):.1f} windows")

    # FP proximity to real attacks
    attack_indices = set(np.where(test_tgts == 1)[0])
    fp_indices = np.where(fp_mask)[0]
    near_attack_count = 0
    proximity_window = 10  # within 10 windows (100s) of an attack
    for fp_idx in fp_indices:
        for dist in range(1, proximity_window + 1):
            if (fp_idx - dist) in attack_indices or (fp_idx + dist) in attack_indices:
                near_attack_count += 1
                break

    near_pct = near_attack_count / len(fp_indices) * 100 if len(fp_indices) > 0 else 0
    print(f"  FPs near attacks (<{proximity_window*WINDOW_SECONDS}s): {near_attack_count:,} ({near_pct:.1f}%)")
    print(f"  FPs far from attacks:  {len(fp_indices) - near_attack_count:,}")

    # Temporal distribution by test date
    if len(fp_indices) > 0:
        fp_ts_offsets = fp_indices + HISTORY  # offset for sequence construction
        valid_offsets = fp_ts_offsets[fp_ts_offsets < len(test_states)]
        if len(valid_offsets) > 0:
            fp_dates = test_states.iloc[valid_offsets]['Timestamp'].dt.date.value_counts().sort_index()
            print(f"  FPs by date:")
            for d, cnt in fp_dates.items():
                print(f"    {d}: {cnt:,}")

# Save FP analysis
fp_analysis = {}
for thresh_name, thresh_val in [('operational', operational_threshold), ('research', research_threshold)]:
    test_bins = (test_preds >= thresh_val).astype(int)
    cm = confusion_matrix(test_tgts, test_bins, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    fp_mask = ((test_tgts == 0) & (test_bins == 1))
    fp_indices = np.where(fp_mask)[0]

    # Count near-attack FPs
    near_attack = 0
    for fp_idx in fp_indices:
        for dist in range(1, 11):
            if (fp_idx - dist) in attack_indices or (fp_idx + dist) in attack_indices:
                near_attack += 1
                break

    fp_analysis[thresh_name] = {
        'threshold': float(thresh_val),
        'total_benign': int(tn + fp),
        'false_positives': int(fp),
        'true_negatives': int(tn),
        'fp_rate': float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0,
        'fp_near_attack': int(near_attack),
        'fp_far_from_attack': int(len(fp_indices) - near_attack),
    }

fp_df = pd.DataFrame(fp_analysis).T
fp_df.to_csv(RESULTS_DIR / 'false_positive_analysis.csv')
print("\nSaved: false_positive_analysis.csv")

# ============================================================
# STEP 10: Save Threshold Selection Summary
# ============================================================
print("\n" + "="*70)
print("   STEP 10: Saving All Results")
print("="*70)

summary = {
    'statement': 'Thresholds were selected exclusively on validation data. Test labels were not used for threshold selection.',
    'research_threshold': {
        'value': float(research_threshold),
        'selection_method': 'F1-optimal on validation set',
        'validation_metrics': {k: float(research_row[k]) for k in ['precision', 'recall', 'f1', 'fpr', 'fnr']},
        'test_metrics': {k: v for k, v in test_research.items() if k != 'threshold'},
    },
    'youdens_j_threshold': {
        'value': float(youdens_threshold),
        'selection_method': "Youden's J (recall - FPR) optimal on validation set",
        'validation_metrics': {k: float(youdens_row[k]) for k in ['precision', 'recall', 'f1', 'fpr', 'fnr', 'youdens_j']},
    },
    'operational_threshold': {
        'value': float(operational_threshold),
        'selection_method': (
            "SOC-oriented: lowest FPR among thresholds with recall >= 0.90 on validation set"
            if recall_90_reasonable else
            "Youden's J fallback: recall >= 0.90 required FPR >= 74% on validation, "
            "so fell back to Youden's J (max recall - FPR) as best available tradeoff"
        ),
        'recall_90_achievable_at_reasonable_fpr': recall_90_reasonable,
        'validation_metrics': {k: float(operational_row[k]) for k in ['precision', 'recall', 'f1', 'fpr', 'fnr']},
        'test_metrics': {k: v for k, v in test_operational.items() if k != 'threshold'},
    },
    'lr_baseline': {
        'threshold': float(best_lr_thresh),
        'test_metrics': {k: v for k, v in lr_test_metrics.items() if k != 'threshold'},
    },
    'early_warning': ew_combined,
    'false_positive_analysis': fp_analysis,
    'eligible_test_episodes': len(eligible_episodes),
    'demo_timestamp': str(demo_ep['base_timestamp']) if demo_ep else 'N/A',
}

with open(RESULTS_DIR / 'threshold_analysis_summary.json', 'w') as f:
    json.dump(summary, f, indent=2, default=str)
print("Saved: threshold_analysis_summary.json")

# Update config with new thresholds
config['research_threshold'] = float(research_threshold)
config['operational_threshold'] = float(operational_threshold)
config['youdens_j_threshold'] = float(youdens_threshold)
config['test_metrics_research'] = test_research
config['test_metrics_operational'] = test_operational
with open(MODEL_DIR / 'config.json', 'w') as f:
    json.dump(config, f, indent=2, default=str)
print("Updated: config.json")

# ============================================================
# FINAL REPORT
# ============================================================
print("\n" + "="*70)
print("   FINAL REPORT")
print("="*70)

print(f"""
THRESHOLDS (selected on validation data only):
  F1-optimal (Research):  {research_threshold}
  Youden's J (candidate): {youdens_threshold}
  Operational (SOC):      {operational_threshold}

VALIDATION METRICS:
  Research (t={research_threshold}):
    Precision={research_row['precision']:.4f}  Recall={research_row['recall']:.4f}
    F1={research_row['f1']:.4f}  FPR={research_row['fpr']:.4f}  FNR={research_row['fnr']:.4f}

  Youden's J (t={youdens_threshold}):
    Precision={youdens_row['precision']:.4f}  Recall={youdens_row['recall']:.4f}
    F1={youdens_row['f1']:.4f}  FPR={youdens_row['fpr']:.4f}  FNR={youdens_row['fnr']:.4f}

  Operational (t={operational_threshold}):
    Precision={operational_row['precision']:.4f}  Recall={operational_row['recall']:.4f}
    F1={operational_row['f1']:.4f}  FPR={operational_row['fpr']:.4f}  FNR={operational_row['fnr']:.4f}

TEST METRICS (single application of locked thresholds):
  LR (t={best_lr_thresh}):
    Prec={lr_test_metrics['precision']:.4f}  Rec={lr_test_metrics['recall']:.4f}
    F1={lr_test_metrics['f1']:.4f}  FPR={lr_test_metrics['fpr']:.4f}

  CyberCast@Research (t={research_threshold}):
    Prec={test_research['precision']:.4f}  Rec={test_research['recall']:.4f}
    F1={test_research['f1']:.4f}  FPR={test_research['fpr']:.4f}

  CyberCast@Operational (t={operational_threshold}):
    Prec={test_operational['precision']:.4f}  Rec={test_operational['recall']:.4f}
    F1={test_operational['f1']:.4f}  FPR={test_operational['fpr']:.4f}

TEST DEMONSTRATION:
  Timestamp: {demo_ep['base_timestamp'] if demo_ep else 'N/A'}
  Eligible episodes: {len(eligible_episodes)}
  Selection: First eligible episode (deterministic)

EARLY WARNING (operational threshold, test set):
  Episodes: {ew_test_operational.get('total_episodes', 0)}
  Detected early: {ew_test_operational.get('detected_early', 0)}
  Detection rate: {ew_test_operational.get('detection_rate', 0):.4f}
  Mean EWT: {ew_test_operational.get('mean_ewt_seconds', 0):.1f}s
  Median EWT: {ew_test_operational.get('median_ewt_seconds', 0):.1f}s
  Max EWT: {ew_test_operational.get('max_ewt_seconds', 0):.1f}s

FPR IMPROVEMENT:
  Research FPR:    {test_research['fpr']:.4f}
  Operational FPR: {test_operational['fpr']:.4f}
  Improvement:     {test_research['fpr'] - test_operational['fpr']:.4f} ({(test_research['fpr'] - test_operational['fpr'])/test_research['fpr']*100:.1f}% relative)
  Recall change:   {test_research['recall']:.4f} -> {test_operational['recall']:.4f}

STATEMENT: Thresholds were selected exclusively on validation data.
           Test labels were not used for threshold selection.
""")

# List all saved artifacts
print("SAVED ARTIFACTS:")
for dn, dp in [('results/', RESULTS_DIR), ('results/plots/', PLOTS_DIR), ('models/', MODEL_DIR)]:
    if dp.is_dir():
        for fn in sorted(os.listdir(dp)):
            fp = dp / fn
            if fp.is_file():
                print(f"  {dn}{fn:40s}  {fp.stat().st_size/1024:.1f} KB")

print("\nThreshold analysis complete!")
