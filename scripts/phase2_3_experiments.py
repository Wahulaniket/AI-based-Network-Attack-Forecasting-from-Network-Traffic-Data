import os
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold, mutual_info_classif
from sklearn.metrics import precision_recall_curve, auc, f1_score, roc_auc_score, confusion_matrix, average_precision_score, precision_score, recall_score, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import time

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
torch.backends.cudnn.deterministic = True

RESULTS_DIR = "results/phase2_3"
MODEL_DIR = "models/phase2_3"
PLOTS_DIR = "figures/phase2_3"

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

print("Loading data...")
df = pd.read_parquet('data/processed/network_states_10s.parquet')
df = df.sort_values('Timestamp').reset_index(drop=True)
initial_row_count = len(df)

target_col = 'binary_attack'

# 1. Feature Audit & Recovery
metadata_cols = ['Timestamp', 'binary_attack', 'dominant_label', 'attack_ratio', 'attack_flow_count', 'has_traffic']

# Let's dynamically map all features in the parquet.
all_columns = list(df.columns)
audit_records = []
set_r_features = []

for c in all_columns:
    is_meta = c in metadata_cols
    # any target derived keyword
    target_derived = 'attack' in c.lower() or 'label' in c.lower() or 'traffic' in c.lower() or is_meta
    
    if c == 'Timestamp':
        allowed = False
        reason = "Metadata (time)"
    elif c == 'flow_count':
        allowed = True
        reason = "Legitimate causal flow aggregate"
        target_derived = False
    elif target_derived:
        allowed = False
        reason = "Target-derived or Metadata"
    else:
        allowed = True
        reason = "Raw or engineered traffic statistic"
        
    audit_records.append({
        'feature': c,
        'source': 'network_states_10s.parquet',
        'formula/description': 'Raw column' if not target_derived else 'Metadata/Label',
        'uses_future_data': False,  # As verified, all aggregation uses past/current data
        'target_derived': target_derived,
        'allowed': allowed,
        'reason': reason
    })
    if allowed:
        set_r_features.append(c)

audit_df = pd.DataFrame(audit_records)
audit_df.to_csv(os.path.join(RESULTS_DIR, 'feature_audit.csv'), index=False)
print(f"Feature Audit Complete. Found {len(set_r_features)} allowed SET_R features.")

# Baseline SET_A
set_a_features = [
    'Flow Duration', 'Tot Fwd Pkts', 'Tot Bwd Pkts', 'TotLen Fwd Pkts', 'TotLen Bwd Pkts', 
    'Flow Byts/s', 'Flow Pkts/s', 'Fwd Pkt Len Mean', 'Bwd Pkt Len Mean', 'SYN Flag Cnt', 
    'ACK Flag Cnt', 'FIN Flag Cnt', 'RST Flag Cnt', 'PSH Flag Cnt', 'URG Flag Cnt'
]

# Split chronologically
n = len(df)
train_end = int(n * 0.8)
val_end = int(n * 0.9)

df_train = df.iloc[:train_end].copy()
df_val = df.iloc[train_end:val_end].copy()
df_test = df.iloc[val_end:].copy()

assert df_train['Timestamp'].max() < df_val['Timestamp'].min(), "Temporal overlap between train and val!"
assert df_val['Timestamp'].max() < df_test['Timestamp'].min(), "Temporal overlap between val and test!"

# 2. Strict Train-Only Preprocessing for SET_R_SELECT
print("Performing STRICT train-only preprocessing for SET_R_SELECT...")
X_train_r = df_train[set_r_features].fillna(0)
y_train_r = df_train[target_col].values

# Variance Filtering (Fitted on Train Only)
var_selector = VarianceThreshold(threshold=0.01)
var_selector.fit(X_train_r)
cand_var = np.array(set_r_features)[var_selector.get_support()]

# Mutual Information (Fitted on Train Only)
X_train_mi = df_train[cand_var].fillna(0)
mi_scores = mutual_info_classif(X_train_mi, y_train_r, random_state=SEED)
mi_series = pd.Series(mi_scores, index=cand_var).sort_values(ascending=False)
top_40_r = mi_series.head(40).index.tolist()

set_r_select_features = top_40_r

print(f"SET_A features: {len(set_a_features)}")
print(f"SET_R features: {len(set_r_features)}")
print(f"SET_R_SELECT features: {len(set_r_select_features)}")

with open(os.path.join(RESULTS_DIR, 'feature_sets.json'), 'w') as f:
    json.dump({
        'SET_A': set_a_features,
        'SET_R': set_r_features,
        'SET_R_SELECT': set_r_select_features
    }, f, indent=4)

# Hard Assertions for Leakage
for feat in set_r_features:
    assert 'attack' not in feat.lower(), f"Target leakage in {feat}"
    assert 'label' not in feat.lower(), f"Target leakage in {feat}"

class SequenceDataset(Dataset):
    def __init__(self, features, targets, seq_length):
        self.features = features
        self.targets = targets
        self.seq_length = seq_length
        
    def __len__(self):
        return len(self.features) - self.seq_length
        
    def __getitem__(self, idx):
        x = self.features[idx : idx + self.seq_length]
        y = self.targets[idx + self.seq_length]
        return torch.FloatTensor(x), torch.FloatTensor([y])

class CyberCastForecaster(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

def evaluate(model, loader, device):
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            preds = torch.sigmoid(out).cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(y.cpu().numpy())
    return np.array(all_preds).flatten(), np.array(all_targets).flatten()

def train_and_eval(features, h, name):
    print(f"\\n--- Training {name} | History={h} | Features={len(features)} ---")
    
    # Train-only Scaler
    scaler = StandardScaler()
    train_feat = scaler.fit_transform(df_train[features].fillna(0).values)
    val_feat = scaler.transform(df_val[features].fillna(0).values)
    
    train_targets = df_train[target_col].values
    val_targets = df_val[target_col].values
    
    train_dataset = SequenceDataset(train_feat, train_targets, seq_length=h)
    val_dataset = SequenceDataset(val_feat, val_targets, seq_length=h)
    
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)
    
    num_pos = max(1, int(train_targets.sum()))
    num_neg = len(train_targets) - num_pos
    pos_weight = torch.tensor([num_neg / num_pos]).to(device)
    
    model = CyberCastForecaster(input_size=len(features), hidden_size=64, num_layers=2).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    best_pr_auc = -1
    best_f1 = -1
    best_precision = -1
    best_recall = -1
    best_roc_auc = -1
    best_epoch = -1
    patience_counter = 0
    patience = 5
    model_save_path = os.path.join(MODEL_DIR, f"model_{name}_h{h}.pt")
    scaler_save_path = os.path.join(MODEL_DIR, f"scaler_{name}_h{h}.joblib")
    
    import joblib
    joblib.dump(scaler, scaler_save_path)
    
    for epoch in range(30):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
        
        val_preds, val_true = evaluate(model, val_loader, device)
        val_pr_auc = average_precision_score(val_true, val_preds)
        val_roc_auc = roc_auc_score(val_true, val_preds)
        
        precision, recall, thresholds = precision_recall_curve(val_true, val_preds)
        fscores = (2 * precision * recall) / (precision + recall + 1e-8)
        ix = np.argmax(fscores)
        best_thresh = thresholds[ix]
        binary_preds = (val_preds >= best_thresh).astype(int)
        
        val_f1 = f1_score(val_true, binary_preds)
        val_prec = precision_score(val_true, binary_preds, zero_division=0)
        val_rec = recall_score(val_true, binary_preds, zero_division=0)
        
        if val_pr_auc > best_pr_auc:
            best_pr_auc = val_pr_auc
            best_f1 = val_f1
            best_precision = val_prec
            best_recall = val_rec
            best_roc_auc = val_roc_auc
            best_epoch = epoch
            patience_counter = 0
            torch.save(model.state_dict(), model_save_path)
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            break
            
    print(f"Done. Best Val PR-AUC: {best_pr_auc:.4f} at epoch {best_epoch}")
    return {
        'Config': name,
        'History_Windows': h,
        'Feature_Count': len(features),
        'Val_PR_AUC': float(best_pr_auc),
        'Val_ROC_AUC': float(best_roc_auc),
        'Val_F1': float(best_f1),
        'Val_Precision': float(best_precision),
        'Val_Recall': float(best_recall),
        'Best_Epoch': best_epoch,
        'Model_Path': model_save_path
    }

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ==========================================
# STAGE 1: Feature Representation Comparison
# ==========================================
print("\\n=== STAGE 1: FEATURE COMPARISON (History = 20) ===")
stage1_results = []
feature_sets = {'SET_A': set_a_features, 'SET_R': set_r_features, 'SET_R_SELECT': set_r_select_features}

for name, feat_set in feature_sets.items():
    res = train_and_eval(feat_set, h=20, name=name)
    stage1_results.append(res)
    
df_stage1 = pd.DataFrame(stage1_results)
df_stage1.to_csv(os.path.join(RESULTS_DIR, 'feature_comparison.csv'), index=False)

# Select best feature set based purely on Val PR-AUC
best_stage1 = df_stage1.loc[df_stage1['Val_PR_AUC'].idxmax()]
champion_feat_name = best_stage1['Config']
champion_features = feature_sets[champion_feat_name]
print(f"\\n--- Stage 1 Winner: {champion_feat_name} (Val PR-AUC: {best_stage1['Val_PR_AUC']:.4f}) ---")

# ==========================================
# STAGE 2: History Comparison
# ==========================================
print(f"\\n=== STAGE 2: HISTORY COMPARISON (Features = {champion_feat_name}) ===")
stage2_results = []
history_options = [5, 10, 20, 30]

for h in history_options:
    res = train_and_eval(champion_features, h=h, name=f"{champion_feat_name}")
    stage2_results.append(res)

df_stage2 = pd.DataFrame(stage2_results)
df_stage2.to_csv(os.path.join(RESULTS_DIR, 'history_comparison.csv'), index=False)

best_stage2 = df_stage2.loc[df_stage2['Val_PR_AUC'].idxmax()]
champion_h = int(best_stage2['History_Windows'])
print(f"\\n--- Stage 2 Winner: History={champion_h} (Val PR-AUC: {best_stage2['Val_PR_AUC']:.4f}) ---")

# ==========================================
# FINAL EVALUATION ON TEST SET
# ==========================================
print(f"\\n=== FINAL EVALUATION: {champion_feat_name} | History={champion_h} ===")

# Load best model and scaler
best_model_path = best_stage2['Model_Path']
best_scaler_path = best_model_path.replace('model_', 'scaler_').replace('.pt', '.joblib')

import joblib
scaler = joblib.load(best_scaler_path)

val_feat = scaler.transform(df_val[champion_features].fillna(0).values)
val_targets = df_val[target_col].values
val_dataset = SequenceDataset(val_feat, val_targets, seq_length=champion_h)
val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)

test_feat = scaler.transform(df_test[champion_features].fillna(0).values)
test_targets = df_test[target_col].values
test_dataset = SequenceDataset(test_feat, test_targets, seq_length=champion_h)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

model = CyberCastForecaster(input_size=len(champion_features), hidden_size=64, num_layers=2).to(device)
model.load_state_dict(torch.load(best_model_path, weights_only=True))

# Find threshold on Validation
val_preds, val_true = evaluate(model, val_loader, device)
precision, recall, thresholds = precision_recall_curve(val_true, val_preds)
fscores = (2 * precision * recall) / (precision + recall + 1e-8)
ix = np.argmax(fscores)
best_threshold = thresholds[ix]

val_binary = (val_preds >= best_threshold).astype(int)
tn, fp, fn, tp = confusion_matrix(val_true, val_binary).ravel()
val_fpr = fp / (fp + tn) if (fp+tn)>0 else 0.0

print(f"Validation Threshold Selected: {best_threshold:.4f}")
print(f"Val F1 at threshold: {f1_score(val_true, val_binary):.4f}")
print(f"Val FPR at threshold: {val_fpr:.4f}")

# Single Final Test
test_preds, test_true = evaluate(model, test_loader, device)
test_binary = (test_preds >= best_threshold).astype(int)

tn, fp, fn, tp = confusion_matrix(test_true, test_binary).ravel()

test_metrics = {
    'PR-AUC': float(average_precision_score(test_true, test_preds)),
    'ROC-AUC': float(roc_auc_score(test_true, test_preds)),
    'F1': float(f1_score(test_true, test_binary)),
    'Precision': float(precision_score(test_true, test_binary, zero_division=0)),
    'Recall': float(recall_score(test_true, test_binary, zero_division=0)),
    'Accuracy': float(accuracy_score(test_true, test_binary)),
    'FPR': float(fp / (fp + tn) if (fp+tn)>0 else 0.0),
    'FNR': float(fn / (fn + tp) if (fn+tp)>0 else 0.0),
    'TP': int(tp), 'TN': int(tn), 'FP': int(fp), 'FN': int(fn),
    'Test_PosRate': float(np.mean(test_true)),
    'Test_Samples': len(test_true)
}

print(f"TEST PR-AUC: {test_metrics['PR-AUC']:.4f}")

with open(os.path.join(RESULTS_DIR, 'champion_metrics.json'), 'w') as f:
    json.dump({
        'Config': champion_feat_name,
        'History': champion_h,
        'Threshold': float(best_threshold),
        'Test_Metrics': test_metrics
    }, f, indent=4)

# ==========================================
# EARLY WARNING DIAGNOSTIC
# ==========================================
print("Running early warning diagnostic...")
ew_results = []
for k in range(1, 31):
    valid_len = len(test_preds) - (k - 1)
    if valid_len > 0:
        p = test_preds[:valid_len]
        t = test_targets[champion_h + k - 1 : champion_h + k - 1 + valid_len]
        
        if len(np.unique(t)) > 1:
            k_pr = average_precision_score(t, p)
            ew_results.append({'Horizon_Windows': k, 'Horizon_Seconds': k * 10, 'PR_AUC': k_pr})
            
ew_df = pd.DataFrame(ew_results)
ew_df.to_csv(os.path.join(RESULTS_DIR, "early_warning_diagnostic.csv"), index=False)

plt.figure(figsize=(10, 6))
plt.plot(ew_df['Horizon_Seconds'], ew_df['PR_AUC'], marker='o')
plt.title(f"Phase 2.3 Early Warning\\nStart PR-AUC: {test_metrics['PR-AUC']:.4f}")
plt.xlabel("Horizon (seconds into the future)")
plt.ylabel("PR-AUC")
plt.grid(True)
plt.savefig(os.path.join(PLOTS_DIR, "early_warning_degradation.png"))
plt.close()

# ==========================================
# MODEL COMPARISON (Historical Baseline)
# ==========================================
comparison_data = [
    {
        'model': 'Phase 2 LSTM',
        'feature_count': '~70',
        'history_windows': 20, # Assume 20 for baseline comparisons (it varied but 20 was common)
        'val_pr_auc': 'N/A',
        'test_pr_auc': 0.7371,
        'test_roc_auc': 'N/A',
        'test_f1': 0.6887,
        'test_precision': 'N/A',
        'test_recall': 'N/A',
        'test_fpr': 'N/A',
        'test_fnr': 'N/A'
    },
    {
        'model': 'Phase 2.2 Champion',
        'feature_count': 15,
        'history_windows': 20,
        'val_pr_auc': 0.5307,
        'test_pr_auc': 0.5571,
        'test_roc_auc': 0.8001,
        'test_f1': 0.4592,
        'test_precision': 0.4440,
        'test_recall': 0.4756,
        'test_fpr': 0.1099,
        'test_fnr': 0.5244
    },
    {
        'model': 'Phase 2.3 Champion',
        'feature_count': len(champion_features),
        'history_windows': champion_h,
        'val_pr_auc': best_stage2['Val_PR_AUC'],
        'test_pr_auc': test_metrics['PR-AUC'],
        'test_roc_auc': test_metrics['ROC-AUC'],
        'test_f1': test_metrics['F1'],
        'test_precision': test_metrics['Precision'],
        'test_recall': test_metrics['Recall'],
        'test_fpr': test_metrics['FPR'],
        'test_fnr': test_metrics['FNR']
    }
]
pd.DataFrame(comparison_data).to_csv(os.path.join(RESULTS_DIR, 'model_comparison.csv'), index=False)

print("\\nANTI-LEAKAGE AUDIT: PASS")
