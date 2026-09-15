import os
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold, mutual_info_classif
from sklearn.metrics import precision_recall_curve, auc, f1_score, roc_auc_score, confusion_matrix, average_precision_score
import matplotlib.pyplot as plt
import seaborn as sns
import time

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
torch.backends.cudnn.deterministic = True

RESULTS_DIR = "results/phase2_2"
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(os.path.join(RESULTS_DIR, "models"), exist_ok=True)
os.makedirs(os.path.join(RESULTS_DIR, "plots"), exist_ok=True)

print("Loading data...")
df = pd.read_parquet('data/processed/network_states_10s.parquet')
df = df.sort_values('Timestamp').reset_index(drop=True)

target_col = 'binary_attack'
excluded_cols = ['Timestamp', 'flow_count', 'attack_flow_count', 'attack_ratio', 'binary_attack', 'dominant_label', 'has_traffic']
for c in df.columns:
    if 'attack' in c.lower() or 'label' in c.lower() or 'traffic' in c.lower():
        if c not in excluded_cols:
            excluded_cols.append(c)

# Base SET_A
set_a_features = [
    'Flow Duration', 'Tot Fwd Pkts', 'Tot Bwd Pkts', 'TotLen Fwd Pkts', 'TotLen Bwd Pkts', 
    'Flow Byts/s', 'Flow Pkts/s', 'Fwd Pkt Len Mean', 'Bwd Pkt Len Mean', 'SYN Flag Cnt', 
    'ACK Flag Cnt', 'FIN Flag Cnt', 'RST Flag Cnt', 'PSH Flag Cnt', 'URG Flag Cnt'
]

# Split chronologically
n = len(df)
train_end = int(n * 0.8)
val_end = int(n * 0.9)

df_train_initial = df.iloc[:train_end].copy()

# 1. Feature Selection (STRICTLY ON TRAIN ONLY)
print("Selecting SET_B features...")
all_candidates = [c for c in df.columns if c not in excluded_cols and c not in set_a_features and c != 'Timestamp']
X_train_cand = df_train_initial[all_candidates].fillna(0)
y_train_initial = df_train_initial[target_col].values

var_selector = VarianceThreshold(threshold=0.01)
var_selector.fit(X_train_cand)
cand_var = np.array(all_candidates)[var_selector.get_support()]

X_train_mi = df_train_initial[cand_var].fillna(0)
# Subsample for faster MI calculation if needed, but 23k is small enough
mi_scores = mutual_info_classif(X_train_mi, y_train_initial, random_state=SEED)
mi_series = pd.Series(mi_scores, index=cand_var).sort_values(ascending=False)
top_15_b = mi_series.head(15).index.tolist()
set_b_features = set_a_features + top_15_b

# 2. Causal Temporal Features
print("Calculating SET_C features...")
# Causal strictness: use diff(1) which computes current - previous
df['delta_Tot Fwd Pkts'] = df['Tot Fwd Pkts'].diff(1).fillna(0)
df['delta_Tot Bwd Pkts'] = df['Tot Bwd Pkts'].diff(1).fillna(0)
df['delta_Flow Byts/s'] = df['Flow Byts/s'].diff(1).fillna(0)
df['delta_Flow Pkts/s'] = df['Flow Pkts/s'].diff(1).fillna(0)
df['delta_Fwd Pkt Len Mean'] = df['Fwd Pkt Len Mean'].diff(1).fillna(0)

# Explicit causal assertions
assert (df['Tot Fwd Pkts'].iloc[5] - df['Tot Fwd Pkts'].iloc[4]) == df['delta_Tot Fwd Pkts'].iloc[5], "Future leakage in diff calculation!"
set_c_features = set_b_features + ['delta_Tot Fwd Pkts', 'delta_Tot Bwd Pkts', 'delta_Flow Byts/s', 'delta_Flow Pkts/s', 'delta_Fwd Pkt Len Mean']

# Assertions
for f in set_c_features:
    assert 'attack' not in f.lower(), f"Target leakage in feature {f}"
    assert 'label' not in f.lower(), f"Target leakage in feature {f}"

# Re-split after feature creation
df_train = df.iloc[:train_end].copy()
df_val = df.iloc[train_end:val_end].copy()
df_test = df.iloc[val_end:].copy()

class SequenceDataset(Dataset):
    def __init__(self, features, targets, seq_length):
        self.features = features
        self.targets = targets
        self.seq_length = seq_length
        
    def __len__(self):
        return len(self.features) - self.seq_length
        
    def __getitem__(self, idx):
        # x: indices [idx, idx+seq_length-1] (inclusive, length=seq_length)
        # y: index [idx+seq_length] (target at t+1)
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
        return out # Return logits for BCEWithLogitsLoss

def evaluate(model, loader, device):
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            preds = torch.sigmoid(out).cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(y.cpu().numpy())
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    
    # average_precision_score requires 1D arrays
    all_targets = all_targets.flatten()
    all_preds = all_preds.flatten()
    
    pr_auc = average_precision_score(all_targets, all_preds)
    return pr_auc, all_preds, all_targets

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

history_lengths = [5, 10, 20, 30]
feature_sets = {'SET_A': set_a_features, 'SET_B': set_b_features, 'SET_C': set_c_features}

experiment_results = []
best_val_pr_auc = -1
champion_config = None
champion_model_path = None

for h in history_lengths:
    for set_name, features in feature_sets.items():
        print(f"\\n--- Experiment: History={h}, Features={set_name} ---")
        
        # Scale (Fit on Train only)
        scaler = StandardScaler()
        train_features = scaler.fit_transform(df_train[features].fillna(0).values)
        val_features = scaler.transform(df_val[features].fillna(0).values)
        test_features = scaler.transform(df_test[features].fillna(0).values)
        
        train_targets = df_train[target_col].values
        val_targets = df_val[target_col].values
        test_targets = df_test[target_col].values
        
        train_dataset = SequenceDataset(train_features, train_targets, seq_length=h)
        val_dataset = SequenceDataset(val_features, val_targets, seq_length=h)
        
        train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)
        
        # Calculate pos_weight
        num_pos = max(1, int(train_targets.sum()))
        num_neg = len(train_targets) - num_pos
        pos_weight = torch.tensor([num_neg / num_pos]).to(device)
        
        model = CyberCastForecaster(input_size=len(features), hidden_size=64, num_layers=2).to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        
        patience = 5
        best_epoch_pr = -1
        patience_counter = 0
        model_save_path = os.path.join(RESULTS_DIR, "models", f"model_{h}_{set_name}.pt")
        
        for epoch in range(30):
            model.train()
            train_loss = 0
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                out = model(x)
                loss = criterion(out, y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            
            val_pr_auc, _, _ = evaluate(model, val_loader, device)
            
            if val_pr_auc > best_epoch_pr:
                best_epoch_pr = val_pr_auc
                patience_counter = 0
                torch.save(model.state_dict(), model_save_path)
            else:
                patience_counter += 1
                
            if patience_counter >= patience:
                break
                
        print(f"Finished training. Best Val PR-AUC: {best_epoch_pr:.4f}")
        
        experiment_results.append({
            'History_Length': h,
            'Feature_Set': set_name,
            'Num_Features': len(features),
            'Val_PR_AUC': best_epoch_pr
        })
        
        if best_epoch_pr > best_val_pr_auc:
            best_val_pr_auc = best_epoch_pr
            champion_config = {'History_Length': h, 'Feature_Set': set_name}
            champion_model_path = model_save_path

results_df = pd.DataFrame(experiment_results)
results_df.to_csv(os.path.join(RESULTS_DIR, "phase2_2_results.csv"), index=False)
print("Experiments completed. Results saved.")

# --- Evaluate Champion on Test Set ---
print(f"\\nEvaluating Champion Model: History={champion_config['History_Length']}, Features={champion_config['Feature_Set']}")
best_h = champion_config['History_Length']
best_features = feature_sets[champion_config['Feature_Set']]

scaler = StandardScaler()
scaler.fit(df_train[best_features].fillna(0).values)
test_features = scaler.transform(df_test[best_features].fillna(0).values)
test_targets = df_test[target_col].values

test_dataset = SequenceDataset(test_features, test_targets, seq_length=best_h)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

champion_model = CyberCastForecaster(input_size=len(best_features), hidden_size=64, num_layers=2).to(device)
champion_model.load_state_dict(torch.load(champion_model_path))

test_pr_auc, test_preds, test_true = evaluate(champion_model, test_loader, device)

# Optimal Threshold based on Validation (since test must be evaluated completely separately)
val_features_best = scaler.transform(df_val[best_features].fillna(0).values)
val_targets_best = df_val[target_col].values
val_dataset_best = SequenceDataset(val_features_best, val_targets_best, seq_length=best_h)
val_loader_best = DataLoader(val_dataset_best, batch_size=128, shuffle=False)
_, val_preds, val_true = evaluate(champion_model, val_loader_best, device)

precision, recall, thresholds = precision_recall_curve(val_true, val_preds)
fscores = (2 * precision * recall) / (precision + recall + 1e-8)
ix = np.argmax(fscores)
best_threshold = thresholds[ix]

test_preds_binary = (test_preds >= best_threshold).astype(int)
test_f1 = f1_score(test_true, test_preds_binary)
print(f"Champion Test PR-AUC: {test_pr_auc:.4f}")
print(f"Champion Test F1-Score: {test_f1:.4f}")

# --- Early Warning Diagnostic ---
# For k steps ahead in test data (k=1 to 30)
print("Running early warning diagnostic...")
ew_results = []
for k in range(1, 31):
    valid_len = len(test_preds) - (k - 1)
    if valid_len > 0:
        p = test_preds[:valid_len]
        t = test_targets[best_h + k - 1 : best_h + k - 1 + valid_len]
        
        if len(np.unique(t)) > 1:
            k_pr = average_precision_score(t, p)
            ew_results.append({'Horizon_Windows': k, 'Horizon_Seconds': k * 10, 'PR_AUC': k_pr})
            
ew_df = pd.DataFrame(ew_results)
ew_df.to_csv(os.path.join(RESULTS_DIR, "early_warning_diagnostic.csv"), index=False)

plt.figure(figsize=(10, 6))
plt.plot(ew_df['Horizon_Seconds'], ew_df['PR_AUC'], marker='o')
plt.title(f"Early Warning Degradation (Champion Model)\\nStart PR-AUC: {test_pr_auc:.4f}")
plt.xlabel("Horizon (seconds into the future)")
plt.ylabel("PR-AUC")
plt.grid(True)
plt.savefig(os.path.join(RESULTS_DIR, "plots", "early_warning_degradation.png"))
plt.close()

with open(os.path.join(RESULTS_DIR, "champion_metrics.json"), 'w') as f:
    json.dump({
        "Config": champion_config,
        "Test_PR_AUC": float(test_pr_auc),
        "Test_F1": float(test_f1),
        "Best_Threshold": float(best_threshold)
    }, f, indent=4)

print("All tasks completed.")
