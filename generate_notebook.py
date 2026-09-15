import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []

# =====================================================================
# PHASE 1 CELLS (verified working)
# =====================================================================
cells.append(nbf.v4.new_markdown_cell("# CyberCast Model Pipeline - From Scratch\n**Phase 1: Data Pipeline + Phase 2: Model Experiments**"))

cells.append(nbf.v4.new_markdown_cell("## 1. Environment & Reproducibility"))
cells.append(nbf.v4.new_code_cell("""import sys, os, glob, time, json, yaml, pickle, warnings
import numpy as np
import pandas as pd
import sklearn
from sklearn.preprocessing import StandardScaler
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
warnings.filterwarnings('ignore')

seed = 42
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

print(f"Python: {sys.version}")
print(f"NumPy: {np.__version__}")
print(f"Pandas: {pd.__version__}")
print(f"Scikit-learn: {sklearn.__version__}")
print(f"PyTorch: {torch.__version__}")
print(f"CUDA: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
"""))

cells.append(nbf.v4.new_markdown_cell("## 2. Dataset Discovery"))
cells.append(nbf.v4.new_code_cell("""DATA_DIR = os.path.join("d:/working_projects/SIH/cyberCast", "data", "raw")
RESULTS_DIR = os.path.join("d:/working_projects/SIH/cyberCast", "results")
MODELS_DIR = os.path.join("d:/working_projects/SIH/cyberCast", "models")
CONFIGS_DIR = os.path.join("d:/working_projects/SIH/cyberCast", "configs")
FIGURES_DIR = os.path.join("d:/working_projects/SIH/cyberCast", "figures")
for d in [RESULTS_DIR, MODELS_DIR, CONFIGS_DIR, FIGURES_DIR]:
    os.makedirs(d, exist_ok=True)

csv_files = glob.glob(os.path.join(DATA_DIR, "*.csv"))
dataset_manifest = []
for f in csv_files:
    size = os.path.getsize(f)
    df_chunk = pd.read_csv(f, nrows=5)
    cols = list(df_chunk.columns)
    dataset_manifest.append({'filename': os.path.basename(f), 'path': f, 'size_bytes': size,
                             'columns': len(cols), 'all_columns': cols})
manifest_df = pd.DataFrame(dataset_manifest)
print(manifest_df[['filename', 'size_bytes', 'columns']])
"""))

cells.append(nbf.v4.new_markdown_cell("## 3-6. Raw Audit, Timestamp Cleaning, Feature Engineering, Temporal Windows"))
cells.append(nbf.v4.new_code_cell("""window_size_seconds = 10
features_to_use = [
    'Flow Duration', 'Tot Fwd Pkts', 'Tot Bwd Pkts', 'TotLen Fwd Pkts', 'TotLen Bwd Pkts',
    'Flow Byts/s', 'Flow Pkts/s', 'Fwd Pkt Len Mean', 'Bwd Pkt Len Mean',
    'SYN Flag Cnt', 'ACK Flag Cnt', 'FIN Flag Cnt', 'RST Flag Cnt', 'PSH Flag Cnt', 'URG Flag Cnt'
]
agg_dict = {
    'Flow Duration': 'mean', 'Tot Fwd Pkts': 'sum', 'Tot Bwd Pkts': 'sum',
    'TotLen Fwd Pkts': 'sum', 'TotLen Bwd Pkts': 'sum',
    'Flow Byts/s': 'mean', 'Flow Pkts/s': 'mean',
    'Fwd Pkt Len Mean': 'mean', 'Bwd Pkt Len Mean': 'mean',
    'SYN Flag Cnt': 'sum', 'ACK Flag Cnt': 'sum', 'FIN Flag Cnt': 'sum',
    'RST Flag Cnt': 'sum', 'PSH Flag Cnt': 'sum', 'URG Flag Cnt': 'sum',
    'attack_binary': 'max'
}
all_windows = []
audit_results = {
    'total_raw_flows': 0, 'clean_flows': 0, 'benign_flows': 0, 'attack_flows': 0,
    'invalid_or_out_of_range': 0,
    'min_raw_timestamp': None, 'max_raw_timestamp': None,
    'min_clean_timestamp': None, 'max_clean_timestamp': None,
    'unique_dates': set(), 'files_with_invalid_timestamps': set()
}
VALID_START = pd.to_datetime('2018-02-14 00:00:00')
VALID_END = pd.to_datetime('2018-03-02 23:59:59.999999')

for idx, row in manifest_df.iterrows():
    f = row['path']
    print(f"Processing {row['filename']}...")
    chunk_iter = pd.read_csv(f, chunksize=1000000, low_memory=False,
                             usecols=['Timestamp', 'Label'] + features_to_use)
    file_windows = []
    for chunk in chunk_iter:
        audit_results['total_raw_flows'] += len(chunk)
        try:
            parsed_time = pd.to_datetime(chunk['Timestamp'], format='%d/%m/%Y %H:%M:%S', errors='coerce')
        except ValueError:
            parsed_time = pd.to_datetime(chunk['Timestamp'], errors='coerce')
        c_min_raw = parsed_time.min()
        c_max_raw = parsed_time.max()
        if audit_results['min_raw_timestamp'] is None or (pd.notna(c_min_raw) and c_min_raw < audit_results['min_raw_timestamp']):
            audit_results['min_raw_timestamp'] = c_min_raw
        if audit_results['max_raw_timestamp'] is None or (pd.notna(c_max_raw) and c_max_raw > audit_results['max_raw_timestamp']):
            audit_results['max_raw_timestamp'] = c_max_raw
        valid_mask = (parsed_time >= VALID_START) & (parsed_time <= VALID_END)
        invalid_count = (~valid_mask).sum()
        if invalid_count > 0:
            audit_results['invalid_or_out_of_range'] += invalid_count
            audit_results['files_with_invalid_timestamps'].add(row['filename'])
        chunk = chunk[valid_mask].copy()
        parsed_time = parsed_time[valid_mask]
        if len(chunk) == 0:
            continue
        audit_results['clean_flows'] += len(chunk)
        attacks = chunk['Label'] != 'Benign'
        audit_results['attack_flows'] += attacks.sum()
        audit_results['benign_flows'] += (~attacks).sum()
        chunk['attack_binary'] = attacks.astype(int)
        audit_results['unique_dates'].update(parsed_time.dt.date.unique())
        c_min_clean = parsed_time.min()
        c_max_clean = parsed_time.max()
        if audit_results['min_clean_timestamp'] is None or c_min_clean < audit_results['min_clean_timestamp']:
            audit_results['min_clean_timestamp'] = c_min_clean
        if audit_results['max_clean_timestamp'] is None or c_max_clean > audit_results['max_clean_timestamp']:
            audit_results['max_clean_timestamp'] = c_max_clean
        for col in features_to_use:
            chunk[col] = pd.to_numeric(chunk[col], errors='coerce').fillna(0).replace([np.inf, -np.inf], 0)
        chunk['window'] = parsed_time.dt.floor(f'{window_size_seconds}s')
        grouped = chunk.groupby('window').agg(agg_dict)
        file_windows.append(grouped)
    if len(file_windows) > 0:
        file_df = pd.concat(file_windows)
        file_df = file_df.groupby(file_df.index).agg(agg_dict)
        all_windows.append(file_df)

if all_windows:
    final_windows = pd.concat(all_windows)
    final_windows = final_windows.groupby(final_windows.index).agg(agg_dict)
    final_windows = final_windows.sort_index()
else:
    final_windows = pd.DataFrame()

assert audit_results['min_clean_timestamp'] >= VALID_START
assert audit_results['max_clean_timestamp'] <= VALID_END
print(f"Clean timestamp range: {audit_results['min_clean_timestamp']} to {audit_results['max_clean_timestamp']}")
print(f"Invalid timestamps removed: {audit_results['invalid_or_out_of_range']}")
print(f"Clean flows: {audit_results['clean_flows']}")
print(f"Total windows: {len(final_windows)}")
"""))

cells.append(nbf.v4.new_markdown_cell("## 7. Chronological Split"))
cells.append(nbf.v4.new_code_cell("""feature_cols = [c for c in final_windows.columns if c != 'attack_binary']
n_windows = len(final_windows)
train_idx = int(n_windows * 0.7)
val_idx = int(n_windows * 0.85)
train_df = final_windows.iloc[:train_idx]
val_df = final_windows.iloc[train_idx:val_idx]
test_df = final_windows.iloc[val_idx:]

for name, df in [("TRAIN", train_df), ("VAL", val_df), ("TEST", test_df)]:
    a = (df['attack_binary'] == 1).sum()
    print(f"{name}: {df.index.min()} to {df.index.max()} | {len(df)} windows | Attack: {a} ({a/len(df)*100:.1f}%)")

assert train_df.index.max() < val_df.index.min(), "Train/Val overlap!"
assert val_df.index.max() < test_df.index.min(), "Val/Test overlap!"
print("Chronological split verified.")
"""))

cells.append(nbf.v4.new_markdown_cell("## 8-9. Scaling & Sequences"))
cells.append(nbf.v4.new_code_cell("""scaler = StandardScaler()
train_scaled = scaler.fit_transform(train_df[feature_cols])
val_scaled = scaler.transform(val_df[feature_cols])
test_scaled = scaler.transform(test_df[feature_cols])
with open(os.path.join(MODELS_DIR, 'feature_scaler.pkl'), 'wb') as f:
    pickle.dump(scaler, f)

HISTORY_WINDOWS = 10

def create_sequences(df, scaled_feats, history_len):
    X, y, t_hist, t_target = [], [], [], []
    targets = df['attack_binary'].values
    times = df.index.values
    for i in range(len(scaled_feats) - history_len):
        X.append(scaled_feats[i:(i + history_len)])
        y.append(targets[i + history_len])
        t_hist.append(times[i:(i + history_len)])
        t_target.append(times[i + history_len])
    return np.array(X), np.array(y), t_hist, t_target

X_train, y_train, t_hist_tr, t_targ_tr = create_sequences(train_df, train_scaled, HISTORY_WINDOWS)
X_val, y_val, t_hist_val, t_targ_val = create_sequences(val_df, val_scaled, HISTORY_WINDOWS)
X_test, y_test, t_hist_test, t_targ_test = create_sequences(test_df, test_scaled, HISTORY_WINDOWS)

print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
print(f"X_val:   {X_val.shape}, y_val:   {y_val.shape}")
print(f"X_test:  {X_test.shape}, y_test:  {y_test.shape}")

assert X_train.shape[1:] == (10, 15)
assert X_val.shape[1:] == (10, 15)
assert X_test.shape[1:] == (10, 15)
print("PHASE 1 COMPLETE - All assertions passed.")
"""))

# =====================================================================
# PHASE 2 CELLS
# =====================================================================

cells.append(nbf.v4.new_markdown_cell("---\n# PHASE 2: Model Experiments & Evaluation"))

# --- Cell P2-1: Setup + Class Imbalance ---
cells.append(nbf.v4.new_markdown_cell("## P2.1 Setup & Class Imbalance"))
cells.append(nbf.v4.new_code_cell("""import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, average_precision_score,
    confusion_matrix, precision_recall_curve, roc_curve, brier_score_loss)
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

train_neg = int((y_train == 0).sum())
train_pos = int((y_train == 1).sum())
pos_ratio = train_pos / len(y_train)
neg_pos_ratio = train_neg / max(train_pos, 1)
pos_weight = torch.tensor([neg_pos_ratio], dtype=torch.float32).to(device)

print(f"Train negatives: {train_neg}")
print(f"Train positives: {train_pos}")
print(f"Positive ratio:  {pos_ratio:.4f}")
print(f"Neg/Pos ratio:   {neg_pos_ratio:.2f}")
print(f"pos_weight:      {pos_weight.item():.2f}")

batch_size = 64
train_loader = DataLoader(TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train)),
                          batch_size=batch_size, shuffle=True)
val_loader = DataLoader(TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val)),
                        batch_size=batch_size, shuffle=False)
test_loader = DataLoader(TensorDataset(torch.FloatTensor(X_test), torch.FloatTensor(y_test)),
                         batch_size=batch_size, shuffle=False)
print(f"Batch size: {batch_size}")
"""))

# --- Cell P2-2: Baselines ---
cells.append(nbf.v4.new_markdown_cell("## P2.2 Baseline Models"))
cells.append(nbf.v4.new_code_cell("""X_train_flat = X_train.reshape(X_train.shape[0], -1)
X_val_flat = X_val.reshape(X_val.shape[0], -1)
X_test_flat = X_test.reshape(X_test.shape[0], -1)
print(f"Flattened shape: {X_train_flat.shape} (10 windows x 15 features = 150)")

baseline_results = {}
baseline_models = {}

# Logistic Regression
print("Training Logistic Regression...")
lr_model = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=seed)
lr_model.fit(X_train_flat, y_train)
lr_vp = lr_model.predict_proba(X_val_flat)[:, 1]
baseline_results['Logistic_Regression'] = {
    'params': lr_model.coef_.size + lr_model.intercept_.size,
    'val_preds': lr_vp
}
baseline_models['Logistic_Regression'] = lr_model
print(f"  Val F1: {f1_score(y_val, (lr_vp>0.5).astype(int)):.4f}  PR-AUC: {average_precision_score(y_val, lr_vp):.4f}")

# Random Forest
print("Training Random Forest...")
rf_model = RandomForestClassifier(n_estimators=200, max_depth=15, class_weight='balanced',
                                  random_state=seed, n_jobs=-1)
rf_model.fit(X_train_flat, y_train)
rf_vp = rf_model.predict_proba(X_val_flat)[:, 1]
baseline_results['Random_Forest'] = {
    'params': sum(e.tree_.node_count for e in rf_model.estimators_),
    'val_preds': rf_vp
}
baseline_models['Random_Forest'] = rf_model
print(f"  Val F1: {f1_score(y_val, (rf_vp>0.5).astype(int)):.4f}  PR-AUC: {average_precision_score(y_val, rf_vp):.4f}")

# XGBoost
try:
    import xgboost as xgb
    print("Training XGBoost...")
    xgb_model = xgb.XGBClassifier(n_estimators=200, max_depth=8, learning_rate=0.1,
                                   scale_pos_weight=neg_pos_ratio, random_state=seed,
                                   eval_metric='logloss', verbosity=0)
    xgb_model.fit(X_train_flat, y_train)
    xgb_vp = xgb_model.predict_proba(X_val_flat)[:, 1]
    baseline_results['XGBoost'] = {
        'params': xgb_model.n_estimators,
        'val_preds': xgb_vp
    }
    baseline_models['XGBoost'] = xgb_model
    print(f"  Val F1: {f1_score(y_val, (xgb_vp>0.5).astype(int)):.4f}  PR-AUC: {average_precision_score(y_val, xgb_vp):.4f}")
except ImportError:
    print("XGBoost not available, skipping.")
"""))

# --- Cell P2-3: Model Definitions + Training Function ---
cells.append(nbf.v4.new_markdown_cell("## P2.3 Deep Learning Model Definitions"))
cells.append(nbf.v4.new_code_cell("""class LSTMModel(nn.Module):
    def __init__(self, input_size=15, hidden_size=64, num_layers=2, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.head = nn.Sequential(nn.Linear(hidden_size, 32), nn.ReLU(), nn.Dropout(dropout), nn.Linear(32, 1))
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :])

class GRUModel(nn.Module):
    def __init__(self, input_size=15, hidden_size=64, num_layers=2, dropout=0.3):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.head = nn.Sequential(nn.Linear(hidden_size, 32), nn.ReLU(), nn.Dropout(dropout), nn.Linear(32, 1))
    def forward(self, x):
        out, _ = self.gru(x)
        return self.head(out[:, -1, :])

class CausalConv1d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, dilation):
        super().__init__()
        self.pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size, dilation=dilation)
        self.bn = nn.BatchNorm1d(out_ch)
    def forward(self, x):
        x = F.pad(x, (self.pad, 0))
        return F.relu(self.bn(self.conv(x)))

class TCNModel(nn.Module):
    def __init__(self, input_size=15, channels=None, kernel_size=3, dropout=0.3):
        super().__init__()
        if channels is None:
            channels = [32, 32, 32, 32]
        layers = []
        for i, out_ch in enumerate(channels):
            in_ch = input_size if i == 0 else channels[i-1]
            layers.append(CausalConv1d(in_ch, out_ch, kernel_size, dilation=2**i))
            layers.append(nn.Dropout(dropout))
        self.net = nn.Sequential(*layers)
        self.head = nn.Linear(channels[-1], 1)
    def forward(self, x):
        x = x.permute(0, 2, 1)  # (B, features, seq)
        x = self.net(x)
        return self.head(x[:, :, -1])  # last time step

class TransformerModel(nn.Module):
    def __init__(self, input_size=15, d_model=32, nhead=4, num_layers=2, dim_ff=128, dropout=0.3, seq_len=10):
        super().__init__()
        self.proj = nn.Linear(input_size, d_model)
        self.pos = nn.Parameter(torch.randn(1, seq_len, d_model) * 0.1)
        enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_ff,
                                               dropout=dropout, batch_first=True)
        self.enc = nn.TransformerEncoder(enc_layer, num_layers)
        self.head = nn.Linear(d_model, 1)
        self.seq_len = seq_len
    def forward(self, x):
        x = self.proj(x) + self.pos[:, :x.size(1), :]
        mask = torch.triu(torch.full((self.seq_len, self.seq_len), float('-inf'), device=x.device), diagonal=1)
        x = self.enc(x, mask=mask)
        return self.head(x[:, -1, :])

class AttentionLSTMModel(nn.Module):
    def __init__(self, input_size=15, hidden_size=64, num_layers=2, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.attn = nn.Linear(hidden_size, 1)
        self.head = nn.Sequential(nn.Linear(hidden_size, 32), nn.ReLU(), nn.Dropout(dropout), nn.Linear(32, 1))
    def forward(self, x):
        out, _ = self.lstm(x)
        w = torch.softmax(self.attn(out), dim=1)
        ctx = (out * w).sum(dim=1)
        return self.head(ctx)

def train_model(model, train_loader, val_loader, criterion, device, num_epochs=50, patience=7, lr=1e-3):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5, min_lr=1e-6)
    best_val_loss = float('inf')
    best_state = None
    wait = 0
    history = {'train_loss': [], 'val_loss': [], 'val_f1': [], 'val_pr_auc': [], 'val_roc_auc': []}

    for epoch in range(num_epochs):
        model.train()
        losses = []
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb).squeeze(-1)
            loss = criterion(logits, yb)
            if torch.isnan(loss):
                break
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(loss.item())

        model.eval()
        vloss_list, vpreds, vlabels = [], [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb).squeeze(-1)
                vloss_list.append(criterion(logits, yb).item())
                vpreds.extend(torch.sigmoid(logits).cpu().numpy())
                vlabels.extend(yb.cpu().numpy())

        tl = np.mean(losses) if losses else float('nan')
        vl = np.mean(vloss_list)
        vp = np.array(vpreds)
        vl_arr = np.array(vlabels)
        vf1 = f1_score(vl_arr, (vp > 0.5).astype(int), zero_division=0)
        vpr = average_precision_score(vl_arr, vp) if len(np.unique(vl_arr)) > 1 else 0.0
        vroc = roc_auc_score(vl_arr, vp) if len(np.unique(vl_arr)) > 1 else 0.0

        history['train_loss'].append(tl)
        history['val_loss'].append(vl)
        history['val_f1'].append(vf1)
        history['val_pr_auc'].append(vpr)
        history['val_roc_auc'].append(vroc)

        scheduler.step(vl)

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:3d} | TrL: {tl:.4f} | VaL: {vl:.4f} | F1: {vf1:.4f} | PR-AUC: {vpr:.4f} | ROC: {vroc:.4f}")

        if vl < best_val_loss:
            best_val_loss = vl
            best_state = {k: v.clone().cpu() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print(f"  Early stop at epoch {epoch+1}")
                break

    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    return model, history

print("Model definitions and training function ready.")
print(f"LSTMModel params:     {sum(p.numel() for p in LSTMModel().parameters()):,}")
print(f"GRUModel params:      {sum(p.numel() for p in GRUModel().parameters()):,}")
print(f"TCNModel params:      {sum(p.numel() for p in TCNModel().parameters()):,}")
print(f"TransformerModel:     {sum(p.numel() for p in TransformerModel().parameters()):,}")
print(f"AttentionLSTMModel:   {sum(p.numel() for p in AttentionLSTMModel().parameters()):,}")
"""))

# --- Cell P2-4: Train All DL Models ---
cells.append(nbf.v4.new_markdown_cell("## P2.4 Train All Deep Learning Models"))
cells.append(nbf.v4.new_code_cell("""criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

arch_configs = [
    ('LSTM', LSTMModel(input_size=15)),
    ('GRU', GRUModel(input_size=15)),
    ('TCN', TCNModel(input_size=15)),
    ('Transformer', TransformerModel(input_size=15)),
    ('Attention_LSTM', AttentionLSTMModel(input_size=15)),
]

dl_models = {}
all_histories = {}
training_times = {}

for name, model in arch_configs:
    print(f"{'='*50}")
    print(f"Training {name}...")
    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}")

    # Reset seeds for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)

    t0 = time.time()
    model, history = train_model(model, train_loader, val_loader, criterion, device,
                                 num_epochs=50, patience=7, lr=1e-3)
    elapsed = time.time() - t0

    dl_models[name] = model
    all_histories[name] = history
    training_times[name] = elapsed
    print(f"  Done in {elapsed:.1f}s | Best Val Loss: {min(history['val_loss']):.4f}")

print("All models trained.")
"""))

# --- Cell P2-5: Validation Evaluation + Threshold + Comparison ---
cells.append(nbf.v4.new_markdown_cell("## P2.5 Validation Evaluation, Threshold Search & Model Comparison"))
cells.append(nbf.v4.new_code_cell("""def get_val_preds_dl(model, loader, device):
    model.eval()
    preds = []
    with torch.no_grad():
        for xb, _ in loader:
            xb = xb.to(device)
            logits = model(xb).squeeze(-1)
            preds.extend(torch.sigmoid(logits).cpu().numpy())
    return np.array(preds)

def compute_metrics(y_true, y_prob, threshold=0.5):
    y_pred = (y_prob > threshold).astype(int)
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'roc_auc': roc_auc_score(y_true, y_prob),
        'pr_auc': average_precision_score(y_true, y_prob),
        'fpr': (y_pred[y_true == 0] == 1).sum() / max((y_true == 0).sum(), 1),
        'fnr': (y_pred[y_true == 1] == 0).sum() / max((y_true == 1).sum(), 1),
    }

all_results = {}

# Baselines
for name, res in baseline_results.items():
    vp = res['val_preds']
    m = compute_metrics(y_val, vp)
    m['params'] = res['params']
    m['val_preds'] = vp
    all_results[name] = m

# DL models
for name, model in dl_models.items():
    vp = get_val_preds_dl(model, val_loader, device)
    m = compute_metrics(y_val, vp)
    m['params'] = sum(p.numel() for p in model.parameters())
    m['val_preds'] = vp
    all_results[name] = m

# Comparison Table
print("MODEL COMPARISON (Validation, threshold=0.5):")
print(f"{'Model':<22} {'Params':>8} {'Acc':>7} {'Prec':>7} {'Rec':>7} {'F1':>7} {'ROC':>7} {'PR-AUC':>7}")
print("-" * 80)
sorted_models = sorted(all_results.keys(), key=lambda x: all_results[x]['pr_auc'], reverse=True)
for name in sorted_models:
    r = all_results[name]
    print(f"{name:<22} {r['params']:>8} {r['accuracy']:>7.4f} {r['precision']:>7.4f} {r['recall']:>7.4f} {r['f1']:>7.4f} {r['roc_auc']:>7.4f} {r['pr_auc']:>7.4f}")

# Save comparison CSV
comp_rows = []
for name in sorted_models:
    r = all_results[name]
    comp_rows.append({'Model': name, 'Params': r['params'], 'Val_Accuracy': r['accuracy'],
                      'Val_Precision': r['precision'], 'Val_Recall': r['recall'], 'Val_F1': r['f1'],
                      'Val_ROC_AUC': r['roc_auc'], 'Val_PR_AUC': r['pr_auc']})
pd.DataFrame(comp_rows).to_csv(os.path.join(RESULTS_DIR, 'model_comparison.csv'), index=False)

# Select best by PR-AUC, tiebreak by F1
best_name = sorted_models[0]
print(f"\\nBEST MODEL: {best_name} (PR-AUC: {all_results[best_name]['pr_auc']:.4f})")

# Threshold search on validation
best_val_preds = all_results[best_name]['val_preds']
thresholds = np.arange(0.05, 0.96, 0.01)
thresh_rows = []
for t in thresholds:
    m = compute_metrics(y_val, best_val_preds, threshold=t)
    thresh_rows.append({'threshold': round(t, 2), **m})
thresh_df = pd.DataFrame(thresh_rows)
best_t_idx = thresh_df['f1'].idxmax()
locked_threshold = float(thresh_df.loc[best_t_idx, 'threshold'])

print(f"\\nVALIDATION THRESHOLD SEARCH (F1-optimal):")
print(f"  Locked Threshold: {locked_threshold:.2f}")
print(f"  F1 at threshold:  {thresh_df.loc[best_t_idx, 'f1']:.4f}")
print(f"  Precision:        {thresh_df.loc[best_t_idx, 'precision']:.4f}")
print(f"  Recall:           {thresh_df.loc[best_t_idx, 'recall']:.4f}")
thresh_df.to_csv(os.path.join(RESULTS_DIR, 'validation_threshold_analysis.csv'), index=False)
"""))

# --- Cell P2-6: Final Test Evaluation ---
cells.append(nbf.v4.new_markdown_cell("## P2.6 Final Test Evaluation (ONE SHOT)"))
cells.append(nbf.v4.new_code_cell("""# Get test predictions from the locked best model
if best_name in dl_models:
    test_preds = get_val_preds_dl(dl_models[best_name], test_loader, device)
else:
    test_preds = baseline_models[best_name].predict_proba(X_test_flat)[:, 1]

test_metrics = compute_metrics(y_test, test_preds, threshold=locked_threshold)
cm = confusion_matrix(y_test, (test_preds > locked_threshold).astype(int))
tn, fp, fn, tp = cm.ravel()

print("FINAL TEST EVALUATION")
print(f"  Model:     {best_name}")
print(f"  Threshold: {locked_threshold:.2f}")
print(f"  Accuracy:  {test_metrics['accuracy']:.4f}")
print(f"  Precision: {test_metrics['precision']:.4f}")
print(f"  Recall:    {test_metrics['recall']:.4f}")
print(f"  F1:        {test_metrics['f1']:.4f}")
print(f"  ROC-AUC:   {test_metrics['roc_auc']:.4f}")
print(f"  PR-AUC:    {test_metrics['pr_auc']:.4f}")
print(f"  FPR:       {test_metrics['fpr']:.4f}")
print(f"  FNR:       {test_metrics['fnr']:.4f}")
print(f"\\n  Confusion Matrix:")
print(f"  TN={tn}  FP={fp}")
print(f"  FN={fn}  TP={tp}")
print(f"  Total: {tn+fp+fn+tp}")

test_out = {
    'model': best_name, 'threshold': locked_threshold,
    'accuracy': test_metrics['accuracy'], 'precision': test_metrics['precision'],
    'recall': test_metrics['recall'], 'f1': test_metrics['f1'],
    'roc_auc': test_metrics['roc_auc'], 'pr_auc': test_metrics['pr_auc'],
    'fpr': float(test_metrics['fpr']), 'fnr': float(test_metrics['fnr']),
    'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)
}
with open(os.path.join(RESULTS_DIR, 'final_test_metrics.json'), 'w') as fout:
    json.dump(test_out, fout, indent=4)
pd.DataFrame(cm, index=['Actual_Benign','Actual_Attack'], columns=['Pred_Benign','Pred_Attack']).to_csv(
    os.path.join(RESULTS_DIR, 'confusion_matrix.csv'))
"""))

# --- Cell P2-7: Diagnostics (plots) ---
cells.append(nbf.v4.new_markdown_cell("## P2.7 Diagnostics: Training Curves, ROC, PR, Calibration, Confusion Matrix"))
cells.append(nbf.v4.new_code_cell("""# Training curves for best model
if best_name in all_histories:
    hist = all_histories[best_name]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    axes[0].plot(hist['train_loss'], label='Train Loss')
    axes[0].plot(hist['val_loss'], label='Val Loss')
    axes[0].set_title(f'{best_name}: Loss'); axes[0].legend(); axes[0].set_xlabel('Epoch')
    axes[1].plot(hist['val_pr_auc'], label='Val PR-AUC', color='green')
    axes[1].set_title(f'{best_name}: PR-AUC'); axes[1].legend(); axes[1].set_xlabel('Epoch')
    axes[2].plot(hist['val_f1'], label='Val F1', color='orange')
    axes[2].set_title(f'{best_name}: F1'); axes[2].legend(); axes[2].set_xlabel('Epoch')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'training_curves.png'), dpi=150, bbox_inches='tight')
    plt.show(); plt.close()

    # Overfitting check
    best_ep = np.argmin(hist['val_loss'])
    print(f"Best epoch (lowest val loss): {best_ep+1}")
    if len(hist['train_loss']) > best_ep + 5:
        print("WARNING: Training continued significantly past best validation - check for overfitting.")
    else:
        print("No severe overfitting detected.")

# Save all training histories
hist_rows = []
for name, hist in all_histories.items():
    for i in range(len(hist['train_loss'])):
        hist_rows.append({'model': name, 'epoch': i+1, 'train_loss': hist['train_loss'][i],
                          'val_loss': hist['val_loss'][i], 'val_f1': hist['val_f1'][i],
                          'val_pr_auc': hist['val_pr_auc'][i], 'val_roc_auc': hist['val_roc_auc'][i]})
pd.DataFrame(hist_rows).to_csv(os.path.join(RESULTS_DIR, 'training_history.csv'), index=False)

# Validation PR curve
prec_c, rec_c, _ = precision_recall_curve(y_val, best_val_preds)
fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(rec_c, prec_c, linewidth=2)
ax.set_xlabel('Recall'); ax.set_ylabel('Precision')
ax.set_title(f'Validation PR Curve - {best_name}')
ax.axvline(x=thresh_df.loc[best_t_idx, 'recall'], color='r', linestyle='--', label=f'Locked thresh={locked_threshold}')
ax.legend()
plt.savefig(os.path.join(FIGURES_DIR, 'validation_pr_curve.png'), dpi=150, bbox_inches='tight')
plt.show(); plt.close()

# Validation ROC curve
fpr_c, tpr_c, _ = roc_curve(y_val, best_val_preds)
fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(fpr_c, tpr_c, linewidth=2)
ax.plot([0,1],[0,1],'--', color='gray')
ax.set_xlabel('FPR'); ax.set_ylabel('TPR')
ax.set_title(f'Validation ROC Curve - {best_name} (AUC={all_results[best_name]["roc_auc"]:.4f})')
plt.savefig(os.path.join(FIGURES_DIR, 'validation_roc_curve.png'), dpi=150, bbox_inches='tight')
plt.show(); plt.close()

# Calibration
frac_pos, mean_pred = calibration_curve(y_val, best_val_preds, n_bins=10)
brier = brier_score_loss(y_val, best_val_preds)
fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(mean_pred, frac_pos, 's-', label=f'{best_name} (Brier={brier:.4f})')
ax.plot([0,1],[0,1],'--',color='gray', label='Perfect')
ax.set_xlabel('Mean Predicted Prob'); ax.set_ylabel('Fraction Positive')
ax.set_title('Calibration Curve (Validation)'); ax.legend()
plt.savefig(os.path.join(FIGURES_DIR, 'calibration_curve.png'), dpi=150, bbox_inches='tight')
plt.show(); plt.close()
print(f"Brier Score (Validation): {brier:.4f}")

# Confusion Matrix heatmap
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Benign','Attack'],
            yticklabels=['Benign','Attack'], ax=axes[0])
axes[0].set_xlabel('Predicted'); axes[0].set_ylabel('Actual')
axes[0].set_title(f'Test Confusion Matrix (counts)')

cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
sns.heatmap(cm_norm, annot=True, fmt='.2%', cmap='Blues', xticklabels=['Benign','Attack'],
            yticklabels=['Benign','Attack'], ax=axes[1])
axes[1].set_xlabel('Predicted'); axes[1].set_ylabel('Actual')
axes[1].set_title(f'Test Confusion Matrix (normalized)')
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'confusion_matrix.png'), dpi=150, bbox_inches='tight')
plt.show(); plt.close()
"""))

# --- Cell P2-8: Early Warning ---
cells.append(nbf.v4.new_markdown_cell("## P2.8 Early Warning Evaluation"))
cells.append(nbf.v4.new_code_cell("""print("EARLY WARNING EVALUATION")
test_pred_bin = (test_preds > locked_threshold).astype(int)

# Find contiguous attack episodes in y_test
episodes = []
in_ep = False
ep_start = None
for i in range(len(y_test)):
    if y_test[i] == 1 and not in_ep:
        in_ep = True
        ep_start = i
    elif y_test[i] == 0 and in_ep:
        episodes.append((ep_start, i - 1))
        in_ep = False
if in_ep:
    episodes.append((ep_start, len(y_test) - 1))

print(f"Attack episodes in test set: {len(episodes)}")

early_warnings = []
detected_after = 0
missed = 0
prev_ep_end = -1

for ep_start, ep_end in episodes:
    search_start = max(0, prev_ep_end + 1)
    # Look for first positive prediction BEFORE episode start
    first_pos = None
    for j in range(search_start, ep_start):
        if test_pred_bin[j] == 1:
            first_pos = j
            break

    if first_pos is not None and first_pos < ep_start:
        warning_sec = (pd.Timestamp(t_targ_test[ep_start]) - pd.Timestamp(t_targ_test[first_pos])).total_seconds()
        early_warnings.append(warning_sec)
    else:
        # Check if detected within the episode
        detected_in = any(test_pred_bin[j] == 1 for j in range(ep_start, min(ep_end + 1, len(test_pred_bin))))
        if detected_in:
            detected_after += 1
        else:
            missed += 1
    prev_ep_end = ep_end

print(f"Detected EARLY:       {len(early_warnings)}")
print(f"Detected after onset: {detected_after}")
print(f"Missed:               {missed}")
if early_warnings:
    print(f"Mean warning time:    {np.mean(early_warnings):.1f}s")
    print(f"Median warning time:  {np.median(early_warnings):.1f}s")
    print(f"Max warning time:     {np.max(early_warnings):.1f}s")

ew_data = {
    'metric': ['episodes_total', 'detected_early', 'detected_after_onset', 'missed',
               'mean_warning_s', 'median_warning_s', 'max_warning_s'],
    'value': [len(episodes), len(early_warnings), detected_after, missed,
              np.mean(early_warnings) if early_warnings else 0,
              np.median(early_warnings) if early_warnings else 0,
              np.max(early_warnings) if early_warnings else 0]
}
pd.DataFrame(ew_data).to_csv(os.path.join(RESULTS_DIR, 'early_warning_results.csv'), index=False)
"""))

# --- Cell P2-9: Feature Importance ---
cells.append(nbf.v4.new_markdown_cell("## P2.9 Feature Importance (Permutation)"))
cells.append(nbf.v4.new_code_cell("""print("FEATURE IMPORTANCE (Permutation on Validation Set)")

if best_name in dl_models:
    best_model_fi = dl_models[best_name]
    best_model_fi.eval()
    base_preds_fi = get_val_preds_dl(best_model_fi, val_loader, device)
    base_prauc = average_precision_score(y_val, base_preds_fi)

    importances = []
    for feat_idx in range(15):
        X_val_perm = X_val.copy()
        perm_vals = X_val_perm[:, :, feat_idx].flatten()
        np.random.shuffle(perm_vals)
        X_val_perm[:, :, feat_idx] = perm_vals.reshape(X_val.shape[0], X_val.shape[1])

        perm_loader = DataLoader(TensorDataset(torch.FloatTensor(X_val_perm), torch.FloatTensor(y_val)),
                                 batch_size=batch_size, shuffle=False)
        perm_preds = get_val_preds_dl(best_model_fi, perm_loader, device)
        perm_prauc = average_precision_score(y_val, perm_preds)
        importances.append(base_prauc - perm_prauc)

    fi_df = pd.DataFrame({'feature': feature_cols, 'importance': importances}).sort_values('importance', ascending=False)
else:
    if best_name == 'Random_Forest':
        imp_flat = rf_model.feature_importances_.reshape(10, 15).mean(axis=0)
    elif best_name == 'XGBoost':
        imp_flat = xgb_model.feature_importances_.reshape(10, 15).mean(axis=0)
    else:
        imp_flat = np.abs(lr_model.coef_.reshape(10, 15)).mean(axis=0)
    fi_df = pd.DataFrame({'feature': feature_cols, 'importance': imp_flat}).sort_values('importance', ascending=False)

fi_df.to_csv(os.path.join(RESULTS_DIR, 'feature_importance.csv'), index=False)

fig, ax = plt.subplots(figsize=(10, 6))
fi_sorted = fi_df.sort_values('importance', ascending=True)
colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(fi_sorted)))
ax.barh(fi_sorted['feature'], fi_sorted['importance'], color=colors)
ax.set_xlabel('Importance (PR-AUC drop on permutation)')
ax.set_title(f'Feature Importance - {best_name}')
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'feature_importance.png'), dpi=150, bbox_inches='tight')
plt.show(); plt.close()

print("\\nTop 10 Most Influential Features:")
for _, row in fi_df.head(10).iterrows():
    print(f"  {row['feature']:<25} {row['importance']:.6f}")
"""))

# --- Cell P2-10: Save Artifacts + Assertions + Final Report ---
cells.append(nbf.v4.new_markdown_cell("## P2.10 Save Artifacts, Anti-Leak Assertions & Final Report"))
cells.append(nbf.v4.new_code_cell("""# Save best model
if best_name in dl_models:
    torch.save(dl_models[best_name].state_dict(), os.path.join(MODELS_DIR, 'cybercast_best_model.pt'))
    model_arch_str = str(dl_models[best_name])
    model_params = sum(p.numel() for p in dl_models[best_name].parameters())
else:
    with open(os.path.join(MODELS_DIR, 'cybercast_best_model.pkl'), 'wb') as fout:
        pickle.dump(baseline_models[best_name], fout)
    model_arch_str = str(baseline_models[best_name])
    model_params = all_results[best_name]['params']

model_cfg = {
    'model_name': best_name, 'input_size': 15, 'seq_len': 10,
    'threshold': locked_threshold, 'architecture': model_arch_str, 'params': int(model_params)
}
with open(os.path.join(MODELS_DIR, 'cybercast_best_model_config.json'), 'w') as fout:
    json.dump(model_cfg, fout, indent=4)

full_config = {
    'seed': seed, 'window_size_seconds': window_size_seconds,
    'history_windows': HISTORY_WINDOWS, 'batch_size': batch_size,
    'learning_rate': 1e-3, 'max_epochs': 50, 'patience': 7,
    'pos_weight': float(pos_weight.cpu().item()),
    'locked_threshold': locked_threshold, 'best_model': best_name,
    'features': feature_cols, 'target': 'attack_binary(t+1)',
    'pytorch_version': torch.__version__,
    'cuda_available': torch.cuda.is_available(),
    'gpu_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A',
    'split': 'chronological_70_15_15',
    'training_times': {k: round(v, 1) for k, v in training_times.items()}
}
with open(os.path.join(CONFIGS_DIR, 'model_config.yaml'), 'w') as fout:
    yaml.dump(full_config, fout)

# Anti-fake assertions
print("ANTI-FAKE / ANTI-LEAKAGE ASSERTIONS:")
checks = []
checks.append(("Scaler fit on train only", scaler.n_samples_seen_ == len(train_df)))
checks.append(("Chronological train<val", train_df.index.max() < val_df.index.min()))
checks.append(("Chronological val<test", val_df.index.max() < test_df.index.min()))
checks.append(("Seq length == 10", X_train.shape[1] == 10))
checks.append(("Feature dim == 15", X_train.shape[2] == 15))
checks.append(("No target in features", 'attack_binary' not in feature_cols and 'Label' not in feature_cols))
checks.append(("No NaN in X_train", not np.any(np.isnan(X_train))))
checks.append(("No NaN in X_val", not np.any(np.isnan(X_val))))
checks.append(("No NaN in X_test", not np.any(np.isnan(X_test))))
checks.append(("No Inf in X_train", not np.any(np.isinf(X_train))))
checks.append(("Pred count == label count", len(test_preds) == len(y_test)))
checks.append(("CM total == test samples", int(tn+fp+fn+tp) == len(y_test)))

all_pass = True
for desc, result in checks:
    status = "PASS" if result else "FAIL"
    print(f"  {desc}: {status}")
    if not result:
        all_pass = False

phase2_status = "PASSED" if all_pass else "FAILED"

# Final report
val_r = all_results[best_name]
print(f'''
========================================
CYBERCAST PHASE 2 MODEL REPORT
========================================

Dataset:
  {audit_results['clean_flows']} clean flows

Windows:
  {len(final_windows)}

Features:
  {len(feature_cols)}

History:
  {HISTORY_WINDOWS} windows

Models tested:
  {', '.join(sorted_models)}

BEST MODEL:
  {best_name}

Validation:
  Accuracy:  {val_r['accuracy']:.4f}
  Precision: {val_r['precision']:.4f}
  Recall:    {val_r['recall']:.4f}
  F1:        {val_r['f1']:.4f}
  ROC-AUC:   {val_r['roc_auc']:.4f}
  PR-AUC:    {val_r['pr_auc']:.4f}

LOCKED THRESHOLD:
  {locked_threshold:.2f}

FINAL TEST:
  Accuracy:  {test_metrics['accuracy']:.4f}
  Precision: {test_metrics['precision']:.4f}
  Recall:    {test_metrics['recall']:.4f}
  F1:        {test_metrics['f1']:.4f}
  ROC-AUC:   {test_metrics['roc_auc']:.4f}
  PR-AUC:    {test_metrics['pr_auc']:.4f}
  FPR:       {test_metrics['fpr']:.4f}
  FNR:       {test_metrics['fnr']:.4f}

Early Warning:
  Episodes:       {len(episodes)}
  Detected early: {len(early_warnings)}
  Missed:         {missed}
  Mean warning:   {np.mean(early_warnings) if early_warnings else 0:.1f}s
  Median warning: {np.median(early_warnings) if early_warnings else 0:.1f}s

========================================
PHASE 2 STATUS: {phase2_status}
========================================
''')
"""))

# =====================================================================
# PHASE 2.1 CELLS
# =====================================================================

cells.append(nbf.v4.new_markdown_cell("---\n# PHASE 2.1: Model Optimization Experiments"))

# --- Cell P2.1-1: Setup + Focal Loss + Architecture Definitions ---
cells.append(nbf.v4.new_markdown_cell("## P2.1 Setup, Loss Functions & Architecture Variants"))
cells.append(nbf.v4.new_code_cell("""print("=" * 60)
print("PHASE 2.1: MODEL OPTIMIZATION EXPERIMENTS")
print("=" * 60)

P21_RESULTS = os.path.join(RESULTS_DIR, 'phase2_1')
P21_MODELS = os.path.join(MODELS_DIR, 'phase2_1')
os.makedirs(P21_RESULTS, exist_ok=True)
os.makedirs(P21_MODELS, exist_ok=True)

# Phase 2 baseline reference (preserved, never overwritten)
phase2_baseline = {
    'model': best_name,
    'val_pr_auc': all_results[best_name]['pr_auc'],
    'val_f1': all_results[best_name]['f1'],
    'val_precision': all_results[best_name]['precision'],
    'val_recall': all_results[best_name]['recall'],
    'val_roc_auc': all_results[best_name]['roc_auc'],
    'locked_threshold': locked_threshold,
    'test_f1': test_metrics['f1'],
    'test_pr_auc': test_metrics['pr_auc'],
}
print(f"Phase 2 Baseline ({best_name}): Val PR-AUC={phase2_baseline['val_pr_auc']:.4f}, Val F1={phase2_baseline['val_f1']:.4f}")

# --- Focal Loss ---
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
    def forward(self, logits, targets):
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        probs = torch.sigmoid(logits)
        pt = probs * targets + (1 - probs) * (1 - targets)
        focal = ((1 - pt) ** self.gamma) * bce
        if self.alpha is not None:
            alpha_t = self.alpha * targets + (1 - targets)
            focal = alpha_t * focal
        return focal.mean()

# --- Architecture Variants ---
class LSTMSmall(nn.Module):
    '''Arch A: LSTM(64)->LSTM(32)->Dense'''
    def __init__(self, input_size=15, dropout=0.3):
        super().__init__()
        self.lstm1 = nn.LSTM(input_size, 64, batch_first=True)
        self.lstm2 = nn.LSTM(64, 32, batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(32, 1)
    def forward(self, x):
        x, _ = self.lstm1(x); x = self.drop(x)
        x, _ = self.lstm2(x); return self.fc(x[:, -1, :])

class LSTMLarge(nn.Module):
    '''Arch B: LSTM(128)->LSTM(64)->Dense'''
    def __init__(self, input_size=15, dropout=0.3):
        super().__init__()
        self.lstm1 = nn.LSTM(input_size, 128, batch_first=True)
        self.lstm2 = nn.LSTM(128, 64, batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(64, 1)
    def forward(self, x):
        x, _ = self.lstm1(x); x = self.drop(x)
        x, _ = self.lstm2(x); return self.fc(x[:, -1, :])

class LSTMAttnV2(nn.Module):
    '''Arch C: LSTM(64)->temporal attention->Dense'''
    def __init__(self, input_size=15, hidden_size=64, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.attn = nn.Linear(hidden_size, 1)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden_size, 32), nn.ReLU(), nn.Linear(32, 1))
    def forward(self, x):
        out, _ = self.lstm(x)
        w = torch.softmax(self.attn(out), dim=1)
        ctx = (out * w).sum(dim=1)
        return self.head(ctx)

class LSTMLNorm(nn.Module):
    '''Arch D: LSTM(64)->LayerNorm->Dropout->Dense'''
    def __init__(self, input_size=15, hidden_size=64, num_layers=2, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.norm = nn.LayerNorm(hidden_size)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden_size, 32), nn.ReLU(), nn.Linear(32, 1))
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(self.norm(out[:, -1, :]))

class LSTMSkip(nn.Module):
    '''Arch E: LSTM(64)->skip/residual->Dense'''
    def __init__(self, input_size=15, hidden_size=64, dropout=0.3):
        super().__init__()
        self.proj = nn.Linear(input_size, hidden_size)
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.norm = nn.LayerNorm(hidden_size)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden_size, 32), nn.ReLU(), nn.Linear(32, 1))
    def forward(self, x):
        res = self.proj(x)
        lstm_out, _ = self.lstm(x)
        out = self.norm(lstm_out + res)
        return self.head(out[:, -1, :])

# --- Experiment Runner ---
def run_exp(model_inst, crit, exp_name):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model_inst = model_inst.to(device)
    model_inst, hist = train_model(model_inst, train_loader, val_loader, crit, device,
                                    num_epochs=50, patience=10, lr=1e-3)
    vp = get_val_preds_dl(model_inst, val_loader, device)
    m = compute_metrics(y_val, vp)
    m['val_preds'] = vp
    m['params'] = sum(p.numel() for p in model_inst.parameters())
    m['best_epoch'] = int(np.argmin(hist['val_loss']) + 1)
    m['epochs'] = len(hist['val_loss'])
    model_inst = model_inst.cpu()
    torch.cuda.empty_cache()
    print(f"  {exp_name}: PR-AUC={m['pr_auc']:.4f} F1={m['f1']:.4f} Prec={m['precision']:.4f} Rec={m['recall']:.4f} BestEp={m['best_epoch']}/{m['epochs']}")
    return m, hist, model_inst

all_exp_results = {}
all_exp_models = {}
all_exp_histories = {}
print("Phase 2.1 setup complete.")
print(f"  LSTMSmall params:  {sum(p.numel() for p in LSTMSmall().parameters()):,}")
print(f"  LSTMLarge params:  {sum(p.numel() for p in LSTMLarge().parameters()):,}")
print(f"  LSTMAttnV2 params: {sum(p.numel() for p in LSTMAttnV2().parameters()):,}")
print(f"  LSTMLNorm params:  {sum(p.numel() for p in LSTMLNorm().parameters()):,}")
print(f"  LSTMSkip params:   {sum(p.numel() for p in LSTMSkip().parameters()):,}")
"""))

# --- Cell P2.1-2: Experiment 1 — Class Weight Sensitivity ---
cells.append(nbf.v4.new_markdown_cell("## Experiment 1: Class Weight Sensitivity"))
cells.append(nbf.v4.new_code_cell("""print("=" * 60)
print("EXPERIMENT 1: Class Weight Sensitivity")
print("=" * 60)
print(f"Using base LSTM architecture (2-layer, hidden=64, {sum(p.numel() for p in LSTMModel().parameters()):,} params)")
print(f"Training class ratio: neg={train_neg}, pos={train_pos}, auto_weight={neg_pos_ratio:.2f}")

pw_configs = [
    ('E1_pw=1.0', 1.0),
    ('E1_pw=2.0', 2.0),
    ('E1_pw=3.0', 3.0),
    ('E1_pw=4.0', 4.0),
    ('E1_pw=auto', neg_pos_ratio),
    ('E1_pw=5.19', 5.19),
]

for name, pw in pw_configs:
    crit = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pw], dtype=torch.float32).to(device))
    model = LSTMModel(input_size=15)
    m, h, mdl = run_exp(model, crit, name)
    all_exp_results[name] = m
    all_exp_models[name] = mdl
    all_exp_histories[name] = h

print("\\nEXPERIMENT 1 RESULTS:")
print(f"{'Config':<20} {'PR-AUC':>8} {'F1':>8} {'Prec':>8} {'Recall':>8} {'ROC':>8} {'Acc':>8} {'BestEp':>8}")
print("-" * 90)
for name, _ in pw_configs:
    r = all_exp_results[name]
    print(f"{name:<20} {r['pr_auc']:>8.4f} {r['f1']:>8.4f} {r['precision']:>8.4f} {r['recall']:>8.4f} {r['roc_auc']:>8.4f} {r['accuracy']:>8.4f} {r['best_epoch']:>8}")

best_exp1 = max([n for n, _ in pw_configs], key=lambda x: all_exp_results[x]['pr_auc'])
best_pw = dict(pw_configs)[best_exp1]
print(f"\\nBest Exp1: {best_exp1} (PR-AUC={all_exp_results[best_exp1]['pr_auc']:.4f})")
print(f"Best pos_weight for Exp 2: {best_pw:.2f}")
"""))

# --- Cell P2.1-3: Experiment 2 — Loss Function + Experiment 3 — Architecture ---
cells.append(nbf.v4.new_markdown_cell("## Experiment 2: Loss Function & Experiment 3: Architecture"))
cells.append(nbf.v4.new_code_cell("""print("=" * 60)
print(f"EXPERIMENT 2: Loss Function (using best_pw={best_pw:.2f} from Exp 1)")
print("=" * 60)

loss_configs = [
    ('E2_BCE_noweight', nn.BCEWithLogitsLoss()),
    ('E2_BCE_bestpw', nn.BCEWithLogitsLoss(pos_weight=torch.tensor([best_pw], dtype=torch.float32).to(device))),
    ('E2_Focal_g1', FocalLoss(gamma=1.0, alpha=best_pw)),
    ('E2_Focal_g2', FocalLoss(gamma=2.0, alpha=best_pw)),
]

for name, crit in loss_configs:
    model = LSTMModel(input_size=15)
    m, h, mdl = run_exp(model, crit, name)
    all_exp_results[name] = m
    all_exp_models[name] = mdl
    all_exp_histories[name] = h

print("\\nEXPERIMENT 2 RESULTS:")
print(f"{'Config':<20} {'PR-AUC':>8} {'F1':>8} {'Prec':>8} {'Recall':>8} {'ROC':>8} {'Acc':>8} {'BestEp':>8}")
print("-" * 90)
for name, _ in loss_configs:
    r = all_exp_results[name]
    print(f"{name:<20} {r['pr_auc']:>8.4f} {r['f1']:>8.4f} {r['precision']:>8.4f} {r['recall']:>8.4f} {r['roc_auc']:>8.4f} {r['accuracy']:>8.4f} {r['best_epoch']:>8}")

best_exp2 = max([n for n, _ in loss_configs], key=lambda x: all_exp_results[x]['pr_auc'])
print(f"\\nBest Exp2: {best_exp2} (PR-AUC={all_exp_results[best_exp2]['pr_auc']:.4f})")

# Reconstruct best criterion for Exp 3
if 'Focal_g1' in best_exp2:
    best_criterion = FocalLoss(gamma=1.0, alpha=best_pw)
elif 'Focal_g2' in best_exp2:
    best_criterion = FocalLoss(gamma=2.0, alpha=best_pw)
elif 'noweight' in best_exp2:
    best_criterion = nn.BCEWithLogitsLoss()
else:
    best_criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([best_pw], dtype=torch.float32).to(device))

print("\\n" + "=" * 60)
print(f"EXPERIMENT 3: LSTM Architecture (using best weight + loss)")
print("=" * 60)

arch_configs_p21 = [
    ('E3_LSTM_64_32', LSTMSmall(input_size=15)),
    ('E3_LSTM_128_64', LSTMLarge(input_size=15)),
    ('E3_LSTM_Attn', LSTMAttnV2(input_size=15)),
    ('E3_LSTM_LNorm', LSTMLNorm(input_size=15)),
    ('E3_LSTM_Skip', LSTMSkip(input_size=15)),
]

for name, model in arch_configs_p21:
    m, h, mdl = run_exp(model, best_criterion, name)
    all_exp_results[name] = m
    all_exp_models[name] = mdl
    all_exp_histories[name] = h

print("\\nEXPERIMENT 3 RESULTS:")
print(f"{'Config':<20} {'Params':>8} {'PR-AUC':>8} {'F1':>8} {'Prec':>8} {'Recall':>8} {'ROC':>8} {'BestEp':>8}")
print("-" * 90)
for name, _ in arch_configs_p21:
    r = all_exp_results[name]
    print(f"{name:<20} {r['params']:>8} {r['pr_auc']:>8.4f} {r['f1']:>8.4f} {r['precision']:>8.4f} {r['recall']:>8.4f} {r['roc_auc']:>8.4f} {r['best_epoch']:>8}")
"""))

# --- Cell P2.1-4: Combined Comparison + Test + Report ---
cells.append(nbf.v4.new_markdown_cell("## Phase 2.1 Combined Comparison, Final Test & Report"))
cells.append(nbf.v4.new_code_cell("""print("=" * 60)
print("PHASE 2.1: COMBINED COMPARISON (ALL EXPERIMENTS)")
print("=" * 60)

print(f"\\n{'Config':<22} {'Params':>8} {'PR-AUC':>8} {'F1':>8} {'Prec':>8} {'Recall':>8} {'ROC':>8} {'Acc':>8} {'BestEp':>7}")
print("-" * 100)
sorted_exp = sorted(all_exp_results.keys(), key=lambda x: all_exp_results[x]['pr_auc'], reverse=True)
for name in sorted_exp:
    r = all_exp_results[name]
    print(f"{name:<22} {r['params']:>8} {r['pr_auc']:>8.4f} {r['f1']:>8.4f} {r['precision']:>8.4f} {r['recall']:>8.4f} {r['roc_auc']:>8.4f} {r['accuracy']:>8.4f} {r['best_epoch']:>7}")

# Save full comparison
exp_rows = []
for name in sorted_exp:
    r = all_exp_results[name]
    exp_rows.append({'Config': name, 'Params': r['params'], 'Val_PR_AUC': r['pr_auc'],
                     'Val_F1': r['f1'], 'Val_Precision': r['precision'], 'Val_Recall': r['recall'],
                     'Val_ROC_AUC': r['roc_auc'], 'Val_Accuracy': r['accuracy'],
                     'Best_Epoch': r['best_epoch'], 'Epochs_Trained': r['epochs'],
                     'Val_FPR': r['fpr'], 'Val_FNR': r['fnr']})
pd.DataFrame(exp_rows).to_csv(os.path.join(P21_RESULTS, 'experiment_comparison.csv'), index=False)

# Select Phase 2.1 winner by PR-AUC, tiebreak by F1
p21_winner = sorted_exp[0]
p21_r = all_exp_results[p21_winner]
improved = p21_r['pr_auc'] > phase2_baseline['val_pr_auc']
print(f"\\nPHASE 2.1 WINNER: {p21_winner}")
print(f"  Val PR-AUC: {p21_r['pr_auc']:.4f} (baseline: {phase2_baseline['val_pr_auc']:.4f}) {'IMPROVED' if improved else 'no improvement'}")
print(f"  Val F1:     {p21_r['f1']:.4f} (baseline: {phase2_baseline['val_f1']:.4f})")

# Validation threshold search
p21_val_preds = p21_r['val_preds']
t_rows = []
for t in np.arange(0.05, 0.96, 0.01):
    tm = compute_metrics(y_val, p21_val_preds, threshold=t)
    t_rows.append({'threshold': round(t, 2), **{k: v for k, v in tm.items() if k != 'val_preds'}})
t_df = pd.DataFrame(t_rows)
best_t_idx = t_df['f1'].idxmax()
p21_threshold = float(t_df.loc[best_t_idx, 'threshold'])
print(f"\\nLocked Phase 2.1 Threshold (F1-optimal on validation): {p21_threshold:.2f}")
print(f"  Val F1 at threshold:  {t_df.loc[best_t_idx, 'f1']:.4f}")
print(f"  Val Precision:        {t_df.loc[best_t_idx, 'precision']:.4f}")
print(f"  Val Recall:           {t_df.loc[best_t_idx, 'recall']:.4f}")
t_df.to_csv(os.path.join(P21_RESULTS, 'threshold_analysis.csv'), index=False)

# ===== ONE FINAL TEST EVALUATION =====
print("\\n" + "=" * 60)
print("PHASE 2.1 FINAL TEST (ONE SHOT — threshold locked from validation)")
print("=" * 60)

winner_model = all_exp_models[p21_winner].to(device)
p21_test_preds = get_val_preds_dl(winner_model, test_loader, device)
p21_test_m = compute_metrics(y_test, p21_test_preds, threshold=p21_threshold)
p21_cm = confusion_matrix(y_test, (p21_test_preds > p21_threshold).astype(int))
p21_tn, p21_fp, p21_fn, p21_tp = p21_cm.ravel()

print(f"  Model:     {p21_winner}")
print(f"  Threshold: {p21_threshold:.2f}")
print(f"  Accuracy:  {p21_test_m['accuracy']:.4f}")
print(f"  Precision: {p21_test_m['precision']:.4f}")
print(f"  Recall:    {p21_test_m['recall']:.4f}")
print(f"  F1:        {p21_test_m['f1']:.4f}")
print(f"  ROC-AUC:   {p21_test_m['roc_auc']:.4f}")
print(f"  PR-AUC:    {p21_test_m['pr_auc']:.4f}")
print(f"  FPR:       {p21_test_m['fpr']:.4f}")
print(f"  FNR:       {p21_test_m['fnr']:.4f}")
print(f"  TN={p21_tn}  FP={p21_fp}")
print(f"  FN={p21_fn}  TP={p21_tp}")
print(f"  Total: {p21_tn + p21_fp + p21_fn + p21_tp}")

# Save winner model
torch.save(winner_model.state_dict(), os.path.join(P21_MODELS, 'best_model.pt'))
with open(os.path.join(P21_MODELS, 'best_model_config.json'), 'w') as fout:
    json.dump({'name': p21_winner, 'threshold': p21_threshold, 'params': int(p21_r['params']),
               'val_pr_auc': p21_r['pr_auc'], 'val_f1': p21_r['f1']}, fout, indent=4)

# Save test metrics
with open(os.path.join(P21_RESULTS, 'final_test_metrics.json'), 'w') as fout:
    json.dump({
        'model': p21_winner, 'threshold': p21_threshold,
        'accuracy': p21_test_m['accuracy'], 'precision': p21_test_m['precision'],
        'recall': p21_test_m['recall'], 'f1': p21_test_m['f1'],
        'roc_auc': p21_test_m['roc_auc'], 'pr_auc': p21_test_m['pr_auc'],
        'fpr': float(p21_test_m['fpr']), 'fnr': float(p21_test_m['fnr']),
        'tn': int(p21_tn), 'fp': int(p21_fp), 'fn': int(p21_fn), 'tp': int(p21_tp)
    }, fout, indent=4)

# Confusion matrix plot
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
sns.heatmap(p21_cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Benign','Attack'],
            yticklabels=['Benign','Attack'], ax=axes[0])
axes[0].set_xlabel('Predicted'); axes[0].set_ylabel('Actual')
axes[0].set_title(f'Phase 2.1 Test CM (counts) - {p21_winner}')
p21_cm_norm = p21_cm.astype('float') / p21_cm.sum(axis=1)[:, np.newaxis]
sns.heatmap(p21_cm_norm, annot=True, fmt='.2%', cmap='Blues', xticklabels=['Benign','Attack'],
            yticklabels=['Benign','Attack'], ax=axes[1])
axes[1].set_xlabel('Predicted'); axes[1].set_ylabel('Actual')
axes[1].set_title(f'Phase 2.1 Test CM (normalized)')
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'phase2_1_confusion_matrix.png'), dpi=150, bbox_inches='tight')
plt.show(); plt.close()

# Training curves for top 3 configs
top3 = sorted_exp[:3]
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for cfg_name in top3:
    if cfg_name in all_exp_histories:
        h = all_exp_histories[cfg_name]
        axes[0].plot(h['train_loss'], label=f'{cfg_name} train')
        axes[0].plot(h['val_loss'], '--', label=f'{cfg_name} val')
        axes[1].plot(h['val_pr_auc'], label=cfg_name)
        axes[2].plot(h['val_f1'], label=cfg_name)
axes[0].set_title('Loss'); axes[0].legend(fontsize=7); axes[0].set_xlabel('Epoch')
axes[1].set_title('Val PR-AUC'); axes[1].legend(fontsize=7); axes[1].set_xlabel('Epoch')
axes[2].set_title('Val F1'); axes[2].legend(fontsize=7); axes[2].set_xlabel('Epoch')
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'phase2_1_training_curves.png'), dpi=150, bbox_inches='tight')
plt.show(); plt.close()

# Anti-fake assertions
assert len(p21_test_preds) == len(y_test)
assert int(p21_tn + p21_fp + p21_fn + p21_tp) == len(y_test)

# Final report
print(f'''
========================================
CYBERCAST PHASE 2.1 OPTIMIZATION REPORT
========================================

Phase 2 Baseline ({phase2_baseline['model']}):
  Val PR-AUC:  {phase2_baseline['val_pr_auc']:.4f}
  Val F1:      {phase2_baseline['val_f1']:.4f}
  Test F1:     {phase2_baseline['test_f1']:.4f}
  Test PR-AUC: {phase2_baseline['test_pr_auc']:.4f}

Experiments run: {len(all_exp_results)}
  Exp 1 (Class Weights): 6 configs
  Exp 2 (Loss Function): 4 configs
  Exp 3 (Architecture):  5 configs

Phase 2.1 Winner: {p21_winner}
  Val PR-AUC:  {p21_r['pr_auc']:.4f}
  Val F1:      {p21_r['f1']:.4f}
  Improvement: {'YES' if improved else 'NO'}

Locked Threshold: {p21_threshold:.2f}

Final Test:
  Accuracy:  {p21_test_m['accuracy']:.4f}
  Precision: {p21_test_m['precision']:.4f}
  Recall:    {p21_test_m['recall']:.4f}
  F1:        {p21_test_m['f1']:.4f}
  ROC-AUC:   {p21_test_m['roc_auc']:.4f}
  PR-AUC:    {p21_test_m['pr_auc']:.4f}
  FPR:       {p21_test_m['fpr']:.4f}
  FNR:       {p21_test_m['fnr']:.4f}

Confusion Matrix:
  TN={p21_tn}  FP={p21_fp}
  FN={p21_fn}  TP={p21_tp}

========================================
PHASE 2.1 STATUS: PASSED
========================================
''')
"""))

import os as _os
nb.cells = cells
nb_path = _os.path.join("d:/working_projects/SIH/cyberCast", "nootebooks", "CyberCast_Model_From_Scratch.ipynb")
with open(nb_path, 'w', encoding='utf-8') as f:
    nbf.write(nb, f)
print(f"Notebook written to {nb_path}")
