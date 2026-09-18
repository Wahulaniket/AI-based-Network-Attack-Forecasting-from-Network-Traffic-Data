# %% [markdown]
# # CyberCast -- AI-Based Network Attack Forecasting World Model
#
# ## Smart India Hackathon (SIH) 2026 / NTRO
#
# ---
#
# ### World Model Architecture
#
# ```
# Historical Network States  [S(t-9), ..., S(t)]
#                 |
#         Shared LSTM Backbone
#                 |
#     +-----------+-----------+
#     |                       |
# State Head             Attack Head
#     |                       |
# Predicted S(t+1)     P(attack at t+1)
#     |
# Append to history -> Predict S(t+2) -> ... -> S(t+K)
# ```
#
# ### Audit Summary (issues fixed from original notebook)
#
# | Category | Issue | Fix |
# |---|---|---|
# | Architecture | Model was a binary classifier | Dual-head LSTM: state + attack heads |
# | Architecture | No multi-task loss | MSE + BCE with configurable lambdas |
# | Architecture | K-step forecast peeked at ground truth | Genuine recursive autoregressive rollout |
# | Missing | No LR baseline | Full Logistic Regression baseline added |
# | Missing | No attack stage mapping | Behaviour-based heuristic mapping |
# | Leakage | Label-derived cols could enter state | Strict LABEL_DERIVED_COLS exclusion |
# | Paths | Google Colab / Drive paths | Local Windows pathlib paths |
# | Memory | All CSVs loaded at once | Incremental per-file processing |

# %% [markdown]
# ---
# ## Section 2 -- Configuration
#
# All hyperparameters and paths in one place.

# %%
# ============================================================
# SECTION 2 -- Configuration
# ============================================================
import os, sys, random, gc, json, time, warnings
from pathlib import Path
from collections import Counter, OrderedDict

warnings.filterwarnings('ignore')

RANDOM_SEED       = 42
WINDOW_SECONDS    = 10
HISTORY           = 10
FORECAST_HORIZON  = 5
HIDDEN_SIZE       = 128
NUM_LAYERS        = 2
DROPOUT           = 0.2
LAMBDA_STATE      = 0.5
LAMBDA_ATTACK     = 0.5
LEARNING_RATE     = 1e-3
BATCH_SIZE        = 128
MAX_EPOCHS        = 30
PATIENCE          = 5

PROJECT_DIR   = Path(r"D:\working_projects\SIH\cyberCast")
RAW_DATA_DIR  = PROJECT_DIR / "data" / "raw"
PROCESSED_DIR = PROJECT_DIR / "data" / "processed"
MODEL_DIR     = PROJECT_DIR / "models"
RESULTS_DIR   = PROJECT_DIR / "results"
PLOTS_DIR     = PROJECT_DIR / "results" / "plots"

for d in [PROCESSED_DIR, MODEL_DIR, RESULTS_DIR, PLOTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Label-derived columns that must NEVER enter model state features
LABEL_DERIVED_COLS = frozenset({
    'Label', 'binary_attack', 'Attack',
    'attack_count', 'attack_ratio', 'attack_flow_count',
    'dominant_label', 'attack_family',
})

print("Configuration set. RAW_DATA_DIR =", RAW_DATA_DIR)

# %% [markdown]
# ---
# ## Section 3 -- Environment Check

# %%
# ============================================================
# SECTION 3 -- Environment Check
# ============================================================
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score, average_precision_score, precision_recall_curve,
    roc_curve, confusion_matrix, classification_report,
    precision_score, recall_score, f1_score, accuracy_score
)

print("=" * 55)
print("       CyberCast -- Environment Info")
print("=" * 55)
print(f"  Python      : {sys.version.split()[0]}")
print(f"  pandas      : {pd.__version__}")
print(f"  numpy       : {np.__version__}")
print(f"  scikit-learn: {sklearn.__version__}")
print(f"  PyTorch     : {torch.__version__}")
print(f"  CUDA avail  : {torch.cuda.is_available()}")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\n  Device: {device}")
if torch.cuda.is_available():
    print(f"   GPU: {torch.cuda.get_device_name(0)}")

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

print(f"\n  Random seed: {RANDOM_SEED}")

# %% [markdown]
# ---
# ## Section 4 -- Dataset Discovery

# %%
# ============================================================
# SECTION 4 -- Dataset Discovery
# ============================================================
csv_files = sorted(RAW_DATA_DIR.glob("*.csv"))
if len(csv_files) == 0:
    raise FileNotFoundError(f"No CSV files found in {RAW_DATA_DIR}")

file_info = []
for fpath in csv_files:
    size_mb = fpath.stat().st_size / (1024 * 1024)
    file_info.append({'filename': fpath.name, 'path': fpath, 'size_mb': size_mb})

print(f"Found {len(file_info)} CSV files:")
for fi in file_info:
    print(f"  {fi['filename']:30s}  {fi['size_mb']:.1f} MB")

# %% [markdown]
# ---
# ## Section 5 -- Dataset Inspection
#
# Inspect every CSV: shape, columns, timestamp range, label distribution,
# missing values, infinite values, duplicate rows.

# %%
# ============================================================
# SECTION 5 -- Dataset Inspection
# ============================================================
def inspect_csv(fpath, sample_nrows=10000):
    fname = fpath.name
    size_mb = fpath.stat().st_size / (1024**2)
    print(f"\n--- {fname} ({size_mb:.1f} MB) ---")
    df = pd.read_csv(fpath, nrows=sample_nrows, low_memory=False)
    df.columns = df.columns.str.strip()
    print(f"  Shape (sample): {df.shape}")
    if 'Timestamp' in df.columns:
        ts = pd.to_datetime(df['Timestamp'], format='mixed', dayfirst=False, errors='coerce')
        print(f"  Timestamp range: {ts.min()} -> {ts.max()}")
        print(f"  Invalid timestamps: {ts.isna().sum()}")
    if 'Label' in df.columns:
        labels = df['Label'].astype(str).str.strip()
        print(f"  Label distribution:")
        for lab, cnt in labels.value_counts().items():
            print(f"    {lab:35s} {cnt:>8,}")
    n_miss = df.isnull().sum().sum()
    num_cols = df.select_dtypes(include=[np.number]).columns
    n_inf = sum(np.isinf(df[c].values).sum() for c in num_cols)
    n_dup = df.duplicated().sum()
    print(f"  Missing: {n_miss}  Inf: {n_inf}  Duplicates: {n_dup}")
    del df; gc.collect()

for fi in file_info:
    inspect_csv(fi['path'])

# Global label distribution
print("\n--- Global Label Distribution ---")
global_label_counts = Counter()
total_raw_rows = 0
for fi in file_info:
    try:
        labels = pd.read_csv(fi['path'], usecols=['Label'], dtype={'Label': str})
        labels['Label'] = labels['Label'].str.strip()
        for lab, cnt in labels['Label'].value_counts().items():
            global_label_counts[lab] += cnt
        total_raw_rows += len(labels)
        del labels; gc.collect()
    except Exception as e:
        print(f"  ERROR reading {fi['filename']}: {e}")

for lab, cnt in sorted(global_label_counts.items(), key=lambda x: -x[1]):
    pct = cnt / total_raw_rows * 100
    print(f"  {lab:40s} {cnt:>12,}  {pct:.2f}%")
print(f"  TOTAL: {total_raw_rows:,}")

benign_count = global_label_counts.get('Benign', 0)
attack_count = total_raw_rows - benign_count
print(f"\n  Benign: {benign_count:,} ({benign_count/total_raw_rows*100:.1f}%)")
print(f"  Attack: {attack_count:,} ({attack_count/total_raw_rows*100:.1f}%)")

# %% [markdown]
# ---
# ## Sections 6-10 -- Data Cleaning, Label Processing, Feature Engineering, Temporal Windows
#
# Memory-safe incremental pipeline: for each CSV, clean -> engineer features ->
# aggregate into 10-second temporal windows -> free raw DataFrame.
#
# ### Strict Label Exclusion
# `LABEL_DERIVED_COLS` contains every column derived from attack labels.
# These are NEVER included in model input state features.

# %%
# ============================================================
# SECTIONS 6-10 -- Incremental Pipeline
# ============================================================

ATTACK_FAMILY_MAP = {
    'Benign': 'Benign',
    'FTP-BruteForce': 'Brute Force', 'SSH-Bruteforce': 'Brute Force',
    'Brute Force -Web': 'Brute Force', 'Brute Force -XSS': 'Brute Force',
    'SQL Injection': 'Brute Force',
    'Bot': 'Botnet',
    'DDoS attacks-LOIC-HTTP': 'DDoS', 'DDoS attack-LOIC-UDP': 'DDoS',
    'DDoS attack-HOIC': 'DDoS', 'DDOS attack-HOIC': 'DDoS',
    'DDOS attack-LOIC-UDP': 'DDoS',
    'DoS attacks-Hulk': 'DoS', 'DoS attacks-SlowHTTPTest': 'DoS',
    'DoS attacks-GoldenEye': 'DoS', 'DoS attacks-Slowloris': 'DoS',
    'Infilteration': 'Infiltration', 'Infiltration': 'Infiltration',
}

def map_attack_family(label):
    return ATTACK_FAMILY_MAP.get(str(label).strip(), 'Other Attack')

def safe_div(a, b, fill=0.0):
    return np.where(b != 0, a / b, fill)

def engineer_flow_features(df):
    """Create derived features from CICFlowMeter columns. NEVER uses Label."""
    eng = []
    if 'Tot Fwd Pkts' in df.columns and 'Tot Bwd Pkts' in df.columns:
        df['Fwd_Bwd_Pkt_Ratio'] = safe_div(df['Tot Fwd Pkts'].values, df['Tot Bwd Pkts'].values)
        eng.append('Fwd_Bwd_Pkt_Ratio')
    if 'TotLen Fwd Pkts' in df.columns and 'TotLen Bwd Pkts' in df.columns:
        df['Fwd_Bwd_Byte_Ratio'] = safe_div(df['TotLen Fwd Pkts'].values, df['TotLen Bwd Pkts'].values)
        eng.append('Fwd_Bwd_Byte_Ratio')
    if 'Tot Fwd Pkts' in df.columns and 'Tot Bwd Pkts' in df.columns:
        df['Total_Pkts'] = df['Tot Fwd Pkts'] + df['Tot Bwd Pkts']
        eng.append('Total_Pkts')
    if 'TotLen Fwd Pkts' in df.columns and 'TotLen Bwd Pkts' in df.columns:
        df['Total_Bytes'] = df['TotLen Fwd Pkts'] + df['TotLen Bwd Pkts']
        eng.append('Total_Bytes')
    if 'Total_Pkts' in df.columns and 'Total_Bytes' in df.columns:
        df['Avg_Pkt_Size'] = safe_div(df['Total_Bytes'].values, df['Total_Pkts'].values)
        eng.append('Avg_Pkt_Size')
    if 'Total_Pkts' in df.columns and 'Flow Duration' in df.columns:
        dur = df['Flow Duration'].values / 1e6
        df['Pkts_Per_Sec'] = safe_div(df['Total_Pkts'].values, dur)
        eng.append('Pkts_Per_Sec')
    if 'Total_Bytes' in df.columns and 'Flow Duration' in df.columns:
        dur = df['Flow Duration'].values / 1e6
        df['Bytes_Per_Sec'] = safe_div(df['Total_Bytes'].values, dur)
        eng.append('Bytes_Per_Sec')
    if 'SYN Flag Cnt' in df.columns and 'FIN Flag Cnt' in df.columns:
        df['SYN_FIN_Ratio'] = safe_div(df['SYN Flag Cnt'].values, df['FIN Flag Cnt'].values + 1)
        eng.append('SYN_FIN_Ratio')
    if 'RST Flag Cnt' in df.columns and 'SYN Flag Cnt' in df.columns:
        df['RST_SYN_Ratio'] = safe_div(df['RST Flag Cnt'].values, df['SYN Flag Cnt'].values + 1)
        eng.append('RST_SYN_Ratio')
    if 'Tot Fwd Pkts' in df.columns and 'Total_Pkts' in df.columns:
        df['Fwd_Pkt_Proportion'] = safe_div(df['Tot Fwd Pkts'].values, df['Total_Pkts'].values)
        eng.append('Fwd_Pkt_Proportion')
    for col in eng:
        df[col] = df[col].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return df, eng


def clean_single_csv(fpath):
    """Clean a single CIC-IDS2018 CSV file."""
    fname = fpath.name
    size_mb = fpath.stat().st_size / (1024**2)
    print(f"\n  Cleaning: {fname} ({size_mb:.1f} MB)")

    if size_mb > 1000:
        print(f"    Large file -- loading in 500K-row chunks")
        chunks = []
        for chunk in pd.read_csv(fpath, chunksize=500_000, low_memory=False):
            chunk.columns = chunk.columns.str.strip()
            chunks.append(chunk)
        df = pd.concat(chunks, ignore_index=True)
        del chunks; gc.collect()
    else:
        df = pd.read_csv(fpath, low_memory=False)
        df.columns = df.columns.str.strip()

    initial = len(df)
    print(f"    Loaded: {initial:,} rows x {df.shape[1]} cols")

    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].astype(str).str.strip()

    assert 'Label' in df.columns, f"Label column missing in {fname}"
    assert 'Timestamp' in df.columns, f"Timestamp column missing in {fname}"

    df['Timestamp'] = pd.to_datetime(df['Timestamp'], format='mixed', dayfirst=False, errors='coerce')
    bad_ts = df['Timestamp'].isna().sum()
    if bad_ts > 0:
        print(f"    Dropped {bad_ts:,} invalid timestamps")
        df = df.dropna(subset=['Timestamp'])

    exclude = {'Timestamp', 'Label'}
    for col in df.columns:
        if col not in exclude and df[col].dtype == object:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    num_cols = df.select_dtypes(include=[np.number]).columns
    n_inf = np.isinf(df[num_cols].values).sum()
    if n_inf > 0:
        df[num_cols] = df[num_cols].replace([np.inf, -np.inf], np.nan)

    n_nan = df[num_cols].isna().sum().sum()
    if n_nan > 0:
        for col in num_cols:
            if df[col].isna().any():
                df[col] = df[col].fillna(df[col].median())

    before = len(df)
    df = df.drop_duplicates()
    dupes = before - len(df)

    df = df.sort_values('Timestamp').reset_index(drop=True)
    df['binary_attack'] = (df['Label'] != 'Benign').astype(np.int8)
    df['attack_family'] = df['Label'].map(map_attack_family)

    n_atk = df['binary_attack'].sum()
    print(f"    Final: {len(df):,} rows | Benign: {len(df)-n_atk:,} | Attack: {n_atk:,}")
    return df


SUM_COLS = frozenset([
    'Tot Fwd Pkts', 'Tot Bwd Pkts', 'TotLen Fwd Pkts', 'TotLen Bwd Pkts',
    'Fwd Header Len', 'Bwd Header Len', 'Fwd Act Data Pkts',
    'Subflow Fwd Pkts', 'Subflow Fwd Byts', 'Subflow Bwd Pkts', 'Subflow Bwd Byts',
    'Total_Pkts', 'Total_Bytes', 'Fwd PSH Flags', 'Bwd PSH Flags',
    'Fwd URG Flags', 'Bwd URG Flags',
    'FIN Flag Cnt', 'SYN Flag Cnt', 'RST Flag Cnt',
    'PSH Flag Cnt', 'ACK Flag Cnt', 'URG Flag Cnt',
    'CWE Flag Count', 'ECE Flag Cnt',
])

def create_temporal_windows(df, feature_cols, window_seconds):
    """Convert cleaned DataFrame into temporal network states."""
    df = df.set_index('Timestamp').sort_index()
    freq = f'{window_seconds}s'

    valid_feats = [c for c in feature_cols if c in df.columns and c not in LABEL_DERIVED_COLS]
    agg_dict = {col: ('sum' if col in SUM_COLS else 'mean') for col in valid_feats}

    states = df[valid_feats].resample(freq).agg(agg_dict)

    # Flow count
    first_col = valid_feats[0]
    states['flow_count'] = df[first_col].resample(freq).count().values

    # --- TARGET columns (NOT model inputs) ---
    atk_sum = df['binary_attack'].resample(freq).sum()
    atk_cnt = df['binary_attack'].resample(freq).count()
    states['attack_flow_count'] = atk_sum.values
    states['attack_ratio'] = np.where(atk_cnt.values > 0, atk_sum.values / atk_cnt.values, 0.0)
    states['binary_attack'] = (atk_sum.values > 0).astype(np.int8)

    # Dominant label
    dom = df['Label'].resample(freq).agg(lambda x: x.mode().iloc[0] if len(x) > 0 else 'Benign')
    states['dominant_label'] = dom.values

    states['has_traffic'] = (states['flow_count'] > 0).astype(np.int8)
    states = states.fillna(0.0)

    for col in states.select_dtypes(include=[np.float64]).columns:
        states[col] = states[col].astype(np.float32)

    states = states.reset_index()
    if 'index' in states.columns:
        states = states.rename(columns={'index': 'Timestamp'})
    return states


# Discover feature columns from first file
print("Discovering feature columns...")
df_probe = pd.read_csv(file_info[0]['path'], nrows=100, low_memory=False)
df_probe.columns = df_probe.columns.str.strip()
df_probe, eng_names = engineer_flow_features(df_probe)
all_numeric = df_probe.select_dtypes(include=[np.number]).columns.tolist()
base_feature_cols = [c for c in all_numeric if c not in LABEL_DERIVED_COLS]
base_feature_cols = list(dict.fromkeys(base_feature_cols))
print(f"  Base feature columns: {len(base_feature_cols)}")
del df_probe; gc.collect()


# === MAIN INCREMENTAL PIPELINE ===
print("\n" + "=" * 60)
print("   INCREMENTAL CLEANING + WINDOWING PIPELINE")
print("=" * 60)

all_windowed = []
files_processed = 0

for fi in file_info:
    try:
        df_clean = clean_single_csv(fi['path'])
        df_clean, _ = engineer_flow_features(df_clean)
        file_feats = [c for c in base_feature_cols if c in df_clean.columns]
        windowed = create_temporal_windows(df_clean, file_feats, WINDOW_SECONDS)
        n_st = len(windowed)
        n_at = int(windowed['binary_attack'].sum())
        print(f"    Windows: {n_st:,} | Attack: {n_at:,}")
        all_windowed.append(windowed)
        files_processed += 1
        del df_clean, windowed; gc.collect()
    except Exception as e:
        print(f"    FAILED: {fi['filename']}: {e}")
        import traceback; traceback.print_exc()

if files_processed == 0:
    raise RuntimeError("No files processed successfully!")

print(f"\nConcatenating {files_processed} windowed DataFrames...")
temporal_states = pd.concat(all_windowed, ignore_index=True)
del all_windowed; gc.collect()

temporal_states.sort_values('Timestamp', inplace=True)
temporal_states.reset_index(drop=True, inplace=True)

before_dedup = len(temporal_states)
temporal_states = temporal_states.drop_duplicates(subset=['Timestamp'], keep='first')
temporal_states.reset_index(drop=True, inplace=True)
deduped = before_dedup - len(temporal_states)
if deduped > 0:
    print(f"  Removed {deduped:,} duplicate timestamp windows")

n_atk_total = int(temporal_states['binary_attack'].sum())
print(f"\nCombined: {len(temporal_states):,} windows")
print(f"Time: {temporal_states['Timestamp'].min()} -> {temporal_states['Timestamp'].max()}")
print(f"Benign: {len(temporal_states)-n_atk_total:,} | Attack: {n_atk_total:,}")

pq_path = PROCESSED_DIR / f'network_states_{WINDOW_SECONDS}s.parquet'
temporal_states.to_parquet(pq_path, index=False)
print(f"Saved: {pq_path}")

# %% [markdown]
# ---
# ## Section 7 -- Label / Attack-Family / Binary Mapping

# %%
# Label mapping summary
print("Label -> Attack Family -> Binary:")
print(f"{'Original Label':40s} {'Family':20s} {'Binary':>6s}")
print("-" * 68)
for lab in sorted(global_label_counts.keys()):
    fam = map_attack_family(lab)
    b = 0 if lab == 'Benign' else 1
    print(f"{lab:40s} {fam:20s} {b:>6d}")
print("\nNote: Attack families are dataset-specific groupings, NOT MITRE ATT&CK stages.")

# %% [markdown]
# ---
# ## Section 11 -- Chronological Train / Validation / Test Split

# %%
# ============================================================
# SECTION 11 -- Chronological Split
# ============================================================
temporal_states['date'] = temporal_states['Timestamp'].dt.date
date_summary = temporal_states.groupby('date').agg(
    n_windows=('binary_attack', 'count'),
    n_attack=('binary_attack', 'sum'),
).reset_index()
date_summary['attack_pct'] = (date_summary['n_attack'] / date_summary['n_windows'] * 100).round(2)

print("Date Coverage:")
for _, row in date_summary.iterrows():
    print(f"  {row['date']}  windows={row['n_windows']:,}  attack={row['n_attack']:,}  ({row['attack_pct']}%)")

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

assert train_states['Timestamp'].max() < val_states['Timestamp'].min(), "Train/Val overlap!"
assert val_states['Timestamp'].max() < test_states['Timestamp'].min(), "Val/Test overlap!"
print("No temporal overlap between splits.")

# %% [markdown]
# ---
# ## Section 12 -- Scaling (fitted on train only)

# %%
# ============================================================
# SECTION 12 -- Scaling
# ============================================================
metadata_cols = {'Timestamp', 'date', 'dominant_label', 'has_traffic'}
all_exclude = LABEL_DERIVED_COLS | metadata_cols

state_feature_cols = [c for c in temporal_states.columns
                      if c not in all_exclude
                      and temporal_states[c].dtype in [np.float32, np.float64,
                                                       np.int8, np.int16, np.int32, np.int64]]
if 'flow_count' in temporal_states.columns and 'flow_count' not in state_feature_cols:
    state_feature_cols.append('flow_count')

# STRICT CHECK: no label-derived column
for c in state_feature_cols:
    assert c not in LABEL_DERIVED_COLS, f"LEAKAGE: '{c}' is label-derived!"
print(f"State feature columns ({len(state_feature_cols)}): no label-derived features.")
STATE_DIM = len(state_feature_cols)
print(f"State dimension: {STATE_DIM}")

scaler = StandardScaler()
scaler.fit(train_states[state_feature_cols].values)
print("Scaler fitted on training data ONLY.")

X_train_raw = scaler.transform(train_states[state_feature_cols].values).astype(np.float32)
X_val_raw = scaler.transform(val_states[state_feature_cols].values).astype(np.float32)
X_test_raw = scaler.transform(test_states[state_feature_cols].values).astype(np.float32)

y_train_raw = train_states['binary_attack'].values.astype(np.float32)
y_val_raw = val_states['binary_attack'].values.astype(np.float32)
y_test_raw = test_states['binary_attack'].values.astype(np.float32)

for name, X in [('Train', X_train_raw), ('Val', X_val_raw), ('Test', X_test_raw)]:
    assert not np.isnan(X).any(), f"NaN in {name}"
    assert not np.isinf(X).any(), f"Inf in {name}"
print(f"X_train: {X_train_raw.shape}  X_val: {X_val_raw.shape}  X_test: {X_test_raw.shape}")

scaler_path = MODEL_DIR / 'scaler.pkl'
joblib.dump(scaler, scaler_path)
print(f"Saved scaler: {scaler_path}")

# %% [markdown]
# ---
# ## Section 13 -- Sequence Construction

# %%
# ============================================================
# SECTION 13 -- Sequence Construction
# ============================================================
def create_world_model_sequences(X, y, seq_length):
    """Create sequences: input [S(t-L+1)..S(t)], state target S(t+1), attack target y(t+1)."""
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

print(f"Creating sequences (L={HISTORY})...")
X_train_seq, S_train_target, y_train_seq = create_world_model_sequences(X_train_raw, y_train_raw, HISTORY)
X_val_seq, S_val_target, y_val_seq = create_world_model_sequences(X_val_raw, y_val_raw, HISTORY)
X_test_seq, S_test_target, y_test_seq = create_world_model_sequences(X_test_raw, y_test_raw, HISTORY)

print(f"  X_train_seq: {X_train_seq.shape}  S_target: {S_train_target.shape}  y: {y_train_seq.shape}")
print(f"  X_val_seq:   {X_val_seq.shape}")
print(f"  X_test_seq:  {X_test_seq.shape}")
for name, y in [('Train', y_train_seq), ('Val', y_val_seq), ('Test', y_test_seq)]:
    na = int(y.sum())
    print(f"  {name}: Benign={len(y)-na:,}  Attack={na:,} ({na/len(y)*100:.2f}%)")

# %% [markdown]
# ---
# ## Section 14 -- Logistic Regression Baseline

# %%
# ============================================================
# SECTION 14 -- Logistic Regression Baseline
# ============================================================
print("Training Logistic Regression baseline...")
X_train_lr = X_train_seq[:, -1, :]
X_val_lr = X_val_seq[:, -1, :]
X_test_lr = X_test_seq[:, -1, :]

lr_model = LogisticRegression(class_weight='balanced', max_iter=1000,
                               random_state=RANDOM_SEED, solver='lbfgs')
lr_model.fit(X_train_lr, y_train_seq)

lr_val_probs = lr_model.predict_proba(X_val_lr)[:, 1]
lr_test_probs = lr_model.predict_proba(X_test_lr)[:, 1]

best_lr_f1, best_lr_thresh = -1, 0.5
for t in [0.2, 0.3, 0.4, 0.5, 0.6]:
    f1 = f1_score(y_val_seq, (lr_val_probs >= t).astype(int), zero_division=0)
    if f1 > best_lr_f1:
        best_lr_f1, best_lr_thresh = f1, t
print(f"  Best LR threshold (val): {best_lr_thresh}  F1={best_lr_f1:.4f}")

lr_test_bin = (lr_test_probs >= best_lr_thresh).astype(int)
lr_metrics = {
    'precision': float(precision_score(y_test_seq, lr_test_bin, zero_division=0)),
    'recall': float(recall_score(y_test_seq, lr_test_bin, zero_division=0)),
    'f1': float(f1_score(y_test_seq, lr_test_bin, zero_division=0)),
    'roc_auc': float(roc_auc_score(y_test_seq, lr_test_probs)) if len(np.unique(y_test_seq)) > 1 else 0.0,
    'pr_auc': float(average_precision_score(y_test_seq, lr_test_probs)) if len(np.unique(y_test_seq)) > 1 else 0.0,
}
cm_lr = confusion_matrix(y_test_seq, lr_test_bin)
tn, fp, fn, tp = cm_lr.ravel() if cm_lr.shape == (2,2) else (0,0,0,0)
lr_metrics['fpr'] = float(fp / (fp+tn)) if (fp+tn) > 0 else 0.0
lr_metrics['fnr'] = float(fn / (fn+tp)) if (fn+tp) > 0 else 0.0

print("  LR Test Results:")
for k, v in lr_metrics.items():
    print(f"    {k:15s}: {v:.4f}")

joblib.dump(lr_model, MODEL_DIR / 'logistic_regression.pkl')
print(f"  Saved: logistic_regression.pkl")

# %% [markdown]
# ---
# ## Section 15 -- CyberCast World Model (Dual-Head LSTM)
#
# ```
# Input (batch, seq_len, state_dim)
#     |
# LSTM backbone (2 layers, 128 hidden)
#     |
#     +-- State Head --> S(t+1) prediction
#     +-- Attack Head --> P(attack at t+1)
# ```

# %%
# ============================================================
# SECTION 15 -- CyberCast World Model
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

model = CyberCastWorldModel(
    state_dim=STATE_DIM, hidden_size=HIDDEN_SIZE,
    num_layers=NUM_LAYERS, dropout=DROPOUT).to(device)

print("CyberCast World Model:")
print(model)
total_params = sum(p.numel() for p in model.parameters())
print(f"\nParameters: {total_params:,}  |  State dim: {STATE_DIM}  |  Device: {device}")

# %% [markdown]
# ---
# ## Section 16 -- Model Training (Multi-Task Loss)
#
# `total_loss = lambda_state * MSE(state) + lambda_attack * BCE(attack)`
#
# Early stopping on best validation total loss.

# %%
# ============================================================
# SECTION 16 -- Training
# ============================================================
pin_mem = torch.cuda.is_available()
train_loader = DataLoader(
    TensorDataset(torch.FloatTensor(X_train_seq), torch.FloatTensor(S_train_target),
                  torch.FloatTensor(y_train_seq)),
    batch_size=BATCH_SIZE, shuffle=False, pin_memory=pin_mem)
val_loader = DataLoader(
    TensorDataset(torch.FloatTensor(X_val_seq), torch.FloatTensor(S_val_target),
                  torch.FloatTensor(y_val_seq)),
    batch_size=BATCH_SIZE, shuffle=False, pin_memory=pin_mem)
test_loader = DataLoader(
    TensorDataset(torch.FloatTensor(X_test_seq), torch.FloatTensor(S_test_target),
                  torch.FloatTensor(y_test_seq)),
    batch_size=BATCH_SIZE, shuffle=False, pin_memory=pin_mem)

n_pos = int(y_train_seq.sum())
n_neg = len(y_train_seq) - n_pos
assert n_pos > 0, "No attack samples in training!"
pos_weight_val = n_neg / n_pos
pos_weight_t = torch.tensor([pos_weight_val], dtype=torch.float32).to(device)
print(f"pos_weight = {pos_weight_val:.2f}  (neg={n_neg:,}  pos={n_pos:,})")

state_criterion = nn.MSELoss()
attack_criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_t)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
best_model_path = MODEL_DIR / 'best_world_model.pt'

history = {
    'train_total_loss': [], 'train_state_loss': [], 'train_attack_loss': [],
    'val_total_loss': [], 'val_state_loss': [], 'val_attack_loss': [],
    'val_roc_auc': [], 'val_pr_auc': [],
}

best_val_loss = float('inf')
best_epoch = 0
patience_counter = 0

print(f"\nTraining (epochs={MAX_EPOCHS}, patience={PATIENCE}, lr={LEARNING_RATE})")
print(f"  lambda_state={LAMBDA_STATE}  lambda_attack={LAMBDA_ATTACK}")
training_start = time.time()

for epoch in range(1, MAX_EPOCHS + 1):
    model.train()
    tr_t, tr_s, tr_a = [], [], []
    for X_b, S_b, y_b in train_loader:
        X_b, S_b, y_b = X_b.to(device), S_b.to(device), y_b.to(device)
        optimizer.zero_grad()
        s_pred, a_logit = model(X_b)
        ls = state_criterion(s_pred, S_b)
        la = attack_criterion(a_logit, y_b)
        loss = LAMBDA_STATE * ls + LAMBDA_ATTACK * la
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        tr_t.append(loss.item()); tr_s.append(ls.item()); tr_a.append(la.item())

    model.eval()
    vl_t, vl_s, vl_a = [], [], []
    vl_preds, vl_tgts = [], []
    with torch.no_grad():
        for X_b, S_b, y_b in val_loader:
            X_b, S_b, y_b = X_b.to(device), S_b.to(device), y_b.to(device)
            s_pred, a_logit = model(X_b)
            ls = state_criterion(s_pred, S_b)
            la = attack_criterion(a_logit, y_b)
            loss = LAMBDA_STATE * ls + LAMBDA_ATTACK * la
            vl_t.append(loss.item()); vl_s.append(ls.item()); vl_a.append(la.item())
            vl_preds.extend(torch.sigmoid(a_logit).cpu().numpy())
            vl_tgts.extend(y_b.cpu().numpy())

    m = lambda x: float(np.mean(x))
    vp = np.array(vl_preds); vt = np.array(vl_tgts)
    try: vroc = roc_auc_score(vt, vp)
    except: vroc = 0.0
    try: vpr = average_precision_score(vt, vp)
    except: vpr = 0.0

    history['train_total_loss'].append(m(tr_t))
    history['train_state_loss'].append(m(tr_s))
    history['train_attack_loss'].append(m(tr_a))
    history['val_total_loss'].append(m(vl_t))
    history['val_state_loss'].append(m(vl_s))
    history['val_attack_loss'].append(m(vl_a))
    history['val_roc_auc'].append(vroc)
    history['val_pr_auc'].append(vpr)

    cur_vl = m(vl_t)
    status = ""
    if cur_vl < best_val_loss:
        best_val_loss = cur_vl
        best_epoch = epoch
        patience_counter = 0
        status = "* BEST"
        torch.save(model.state_dict(), best_model_path)
    else:
        patience_counter += 1
        status = f"wait {patience_counter}/{PATIENCE}"

    print(f"  Ep {epoch:3d}: TrLoss={m(tr_t):.5f} (st={m(tr_s):.5f} at={m(tr_a):.5f}) "
          f"VlLoss={cur_vl:.5f} ROC={vroc:.4f} PR={vpr:.4f}  {status}")

    if patience_counter >= PATIENCE:
        print(f"  Early stopping. Best epoch: {best_epoch}")
        break

training_time = time.time() - training_start
print(f"Training time: {training_time:.1f}s  Best val loss: {best_val_loss:.5f} (ep {best_epoch})")

hist_df = pd.DataFrame(history)
hist_df.to_csv(RESULTS_DIR / 'training_history.csv', index_label='epoch')
print("Saved: training_history.csv")

# %% [markdown]
# ---
# ## Section 17 -- Training Curves

# %%
fig, axes = plt.subplots(1, 3, figsize=(20, 5))
axes[0].plot(history['train_total_loss'], label='Train', lw=2)
axes[0].plot(history['val_total_loss'], label='Val', lw=2)
axes[0].axvline(x=best_epoch-1, color='g', ls='--', alpha=0.7, label=f'Best ({best_epoch})')
axes[0].set_title('Total Loss'); axes[0].legend(); axes[0].grid(alpha=0.3)

axes[1].plot(history['train_state_loss'], label='Train State', lw=2)
axes[1].plot(history['val_state_loss'], label='Val State', lw=2)
axes[1].set_title('State Reconstruction (MSE)'); axes[1].legend(); axes[1].grid(alpha=0.3)

axes[2].plot(history['train_attack_loss'], label='Train Attack', lw=2)
axes[2].plot(history['val_attack_loss'], label='Val Attack', lw=2)
axes[2].set_title('Attack Prediction (BCE)'); axes[2].legend(); axes[2].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(PLOTS_DIR / 'training_history.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: training_history.png")

# %% [markdown]
# ---
# ## Section 18 -- One-Step Evaluation

# %%
# ============================================================
# SECTION 18 -- One-Step Evaluation
# ============================================================
model.load_state_dict(torch.load(best_model_path, map_location=device, weights_only=True))
print(f"Loaded best model (epoch {best_epoch})")

model.eval()
val_preds, val_tgts = [], []
with torch.no_grad():
    for X_b, S_b, y_b in val_loader:
        _, a = model(X_b.to(device))
        val_preds.extend(torch.sigmoid(a).cpu().numpy())
        val_tgts.extend(y_b.numpy())
val_preds = np.array(val_preds)
val_tgts = np.array(val_tgts)

best_wm_f1, best_wm_thresh = -1, 0.5
print("Threshold selection (validation):")
for t in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]:
    f1 = f1_score(val_tgts, (val_preds >= t).astype(int), zero_division=0)
    rec = recall_score(val_tgts, (val_preds >= t).astype(int), zero_division=0)
    marker = ''
    if rec >= 0.3 and f1 > best_wm_f1:
        best_wm_f1, best_wm_thresh = f1, t
        marker = ' <-- BEST'
    print(f"  t={t:.1f}  F1={f1:.4f}  Recall={rec:.4f}{marker}")
print(f"Selected threshold: {best_wm_thresh}")

# Test evaluation
test_preds, test_tgts = [], []
with torch.no_grad():
    for X_b, S_b, y_b in test_loader:
        _, a = model(X_b.to(device))
        test_preds.extend(torch.sigmoid(a).cpu().numpy())
        test_tgts.extend(y_b.numpy())
test_preds = np.array(test_preds)
test_tgts = np.array(test_tgts)
test_bins = (test_preds >= best_wm_thresh).astype(int)

wm_metrics = {
    'precision': float(precision_score(test_tgts, test_bins, zero_division=0)),
    'recall': float(recall_score(test_tgts, test_bins, zero_division=0)),
    'f1': float(f1_score(test_tgts, test_bins, zero_division=0)),
    'roc_auc': float(roc_auc_score(test_tgts, test_preds)) if len(np.unique(test_tgts)) > 1 else 0.0,
    'pr_auc': float(average_precision_score(test_tgts, test_preds)) if len(np.unique(test_tgts)) > 1 else 0.0,
}
cm = confusion_matrix(test_tgts, test_bins)
tn, fp, fn, tp = cm.ravel() if cm.shape == (2,2) else (0,0,0,0)
wm_metrics['fpr'] = float(fp/(fp+tn)) if (fp+tn) > 0 else 0.0
wm_metrics['fnr'] = float(fn/(fn+tp)) if (fn+tp) > 0 else 0.0
wm_metrics['threshold'] = best_wm_thresh

print("\nCyberCast World Model -- TEST Results:")
for k, v in wm_metrics.items():
    print(f"  {k:15s}: {v:.4f}")
print(f"\nConfusion Matrix: TN={tn:,} FP={fp:,} FN={fn:,} TP={tp:,}")

pred_df = pd.DataFrame({'actual': test_tgts, 'predicted_prob': test_preds, 'predicted_label': test_bins})
pred_df.to_csv(RESULTS_DIR / 'predictions.csv', index=False)

# %%
# Plots
fig, axes = plt.subplots(1, 3, figsize=(20, 6))
fpr_c, tpr_c, _ = roc_curve(test_tgts, test_preds)
axes[0].plot(fpr_c, tpr_c, lw=2); axes[0].plot([0,1],[0,1],'k--',alpha=0.3)
axes[0].set_title(f"ROC (AUC={wm_metrics['roc_auc']:.4f})"); axes[0].grid(alpha=0.3)

prec_c, rec_c, _ = precision_recall_curve(test_tgts, test_preds)
axes[1].plot(rec_c, prec_c, lw=2, color='r')
axes[1].set_title(f"PR (AUC={wm_metrics['pr_auc']:.4f})"); axes[1].grid(alpha=0.3)

sns.heatmap(cm, annot=True, fmt=',', cmap='Blues',
            xticklabels=['Benign','Attack'], yticklabels=['Benign','Attack'],
            ax=axes[2], cbar=False)
axes[2].set_title('Confusion Matrix')

plt.tight_layout()
plt.savefig(PLOTS_DIR / 'test_evaluation.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: test_evaluation.png")

# %% [markdown]
# ---
# ## Section 19 -- True Recursive K-Step Forecasting
#
# Genuine autoregressive rollout: predict S(t+1), append PREDICTED state
# (NOT ground truth), predict S(t+2), etc.

# %%
# ============================================================
# SECTION 19 -- Recursive K-Step Forecasting
# ============================================================
def recursive_kstep_forecast(model, initial_seq, K, device):
    """Genuine recursive K-step forecast using predicted states."""
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

print(f"Recursive {FORECAST_HORIZON}-step forecasting...")

n_test_seqs = len(X_test_seq)
n_samples = min(2000, n_test_seqs)
sample_idx = np.linspace(0, n_test_seqs - 1, n_samples, dtype=int)

horizon_probs = {k: [] for k in range(1, FORECAST_HORIZON + 1)}
horizon_actuals = {k: [] for k in range(1, FORECAST_HORIZON + 1)}

for idx in sample_idx:
    fcs = recursive_kstep_forecast(model, X_test_seq[idx], FORECAST_HORIZON, device)
    for fc in fcs:
        k = fc['step']
        horizon_probs[k].append(fc['attack_prob'])
        ti = idx + HISTORY + k - 1
        horizon_actuals[k].append(y_test_raw[ti] if ti < len(y_test_raw) else np.nan)

forecast_results = {}
print(f"\n  {'Step':>5s} {'Horizon':>8s} {'ROC':>8s} {'PR':>8s} {'F1':>8s}")
for k in range(1, FORECAST_HORIZON + 1):
    p = np.array(horizon_probs[k])
    a = np.array(horizon_actuals[k])
    v = ~np.isnan(a)
    if v.sum() == 0 or len(np.unique(a[v])) < 2:
        continue
    pv, av = p[v], a[v]
    bv = (pv >= best_wm_thresh).astype(int)
    roc = roc_auc_score(av, pv)
    pr = average_precision_score(av, pv)
    f1 = f1_score(av, bv, zero_division=0)
    prec = precision_score(av, bv, zero_division=0)
    rec = recall_score(av, bv, zero_division=0)
    forecast_results[k] = {'horizon_seconds': k*WINDOW_SECONDS,
                            'roc_auc': roc, 'pr_auc': pr, 'f1': f1,
                            'precision': prec, 'recall': rec}
    print(f"  t+{k:2d}  {k*WINDOW_SECONDS:5d}s   {roc:.4f}  {pr:.4f}  {f1:.4f}")

if forecast_results:
    pd.DataFrame(forecast_results).T.to_csv(RESULTS_DIR / 'forecast_results.csv', index_label='step')
    print("Saved: forecast_results.csv")

if len(forecast_results) > 1:
    fig, ax = plt.subplots(figsize=(10, 6))
    ks = sorted(forecast_results.keys())
    ss = [forecast_results[k]['horizon_seconds'] for k in ks]
    ax.plot(ss, [forecast_results[k]['roc_auc'] for k in ks], 'o-', label='ROC-AUC', lw=2)
    ax.plot(ss, [forecast_results[k]['pr_auc'] for k in ks], 's-', label='PR-AUC', lw=2)
    ax.plot(ss, [forecast_results[k]['f1'] for k in ks], '^-', label='F1', lw=2)
    ax.set_xlabel('Forecast Horizon (s)'); ax.set_ylabel('Score')
    ax.set_title('Recursive Forecast Performance vs Horizon', fontweight='bold')
    ax.legend(); ax.grid(alpha=0.3); ax.set_ylim(0, 1.05)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'forecast_horizon.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Saved: forecast_horizon.png")

# %% [markdown]
# ---
# ## Section 20 -- Risk Scoring

# %%
def compute_risk_score(prob):
    score = float(np.clip(round(prob * 100), 0, 100))
    if score <= 24: return score, 'LOW'
    elif score <= 49: return score, 'MEDIUM'
    elif score <= 74: return score, 'HIGH'
    else: return score, 'CRITICAL'

def compute_kstep_risk(probs, K):
    w = np.array([K - i for i in range(len(probs))], dtype=np.float32)
    score = float(np.clip(np.average(probs, weights=w) * 100, 0, 100))
    if score <= 24: return score, 'LOW'
    elif score <= 49: return score, 'MEDIUM'
    elif score <= 74: return score, 'HIGH'
    else: return score, 'CRITICAL'

print("Risk Score Mapping:")
for p in [0.05, 0.30, 0.55, 0.82, 0.97]:
    s, l = compute_risk_score(p)
    print(f"  P={p:.2f} -> Score={s:.0f} -> {l}")

# %% [markdown]
# ---
# ## Section 21 -- Attack Stage Mapping (Behaviour-Based)
#
# NOT MITRE ATT&CK ground-truth labels. Heuristic inference from observable features.

# %%
def infer_attack_stage(state_vec, feat_names, attack_prob, threshold=0.5):
    """Infer attack stage from predicted network state (heuristic)."""
    fd = {n: float(v) for n, v in zip(feat_names, state_vec)}
    evidence = []
    if attack_prob < threshold:
        return 'Normal Operations', ['No attack indicators'], 0.0

    syn = fd.get('SYN Flag Cnt', 0)
    rst = fd.get('RST Flag Cnt', 0)
    pkt_rate = fd.get('Pkts_Per_Sec', fd.get('Flow Pkts/s', 0))
    byte_rate = fd.get('Bytes_Per_Sec', fd.get('Flow Byts/s', 0))
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

print("Attack stage mapping defined (behaviour-based heuristics).")

# %% [markdown]
# ---
# ## Section 22 -- Explainability (Permutation Importance)

# %%
def permutation_importance_wm(model, X, y, feat_names, device, n_repeats=3, bs=256):
    """Permutation importance for World Model attack head."""
    model.eval()
    def pred(X):
        ps = []
        with torch.no_grad():
            for i in range(0, len(X), bs):
                _, a = model(torch.FloatTensor(X[i:i+bs]).to(device))
                ps.extend(torch.sigmoid(a).cpu().numpy())
        return np.array(ps)

    base = pred(X)
    try: base_score = average_precision_score(y, base)
    except: base_score = 0.5

    imps = {}
    for fi in range(X.shape[2]):
        drops = []
        for _ in range(n_repeats):
            Xp = X.copy()
            pi = np.random.permutation(len(Xp))
            Xp[:, :, fi] = X[pi, :, fi]
            pp = pred(Xp)
            try: ps = average_precision_score(y, pp)
            except: ps = 0.5
            drops.append(base_score - ps)
        fn = feat_names[fi] if fi < len(feat_names) else f'feat_{fi}'
        imps[fn] = {'mean_drop': float(np.mean(drops)), 'std_drop': float(np.std(drops))}

    return sorted(imps.items(), key=lambda x: -x[1]['mean_drop'])

n_exp = min(2000, len(X_test_seq))
sorted_importances = permutation_importance_wm(
    model, X_test_seq[:n_exp], y_test_seq[:n_exp], state_feature_cols, device, n_repeats=3)

print("Top features by importance:")
for f, v in sorted_importances[:15]:
    print(f"  {f:40s}  drop={v['mean_drop']:.6f}")

imp_df = pd.DataFrame([{'feature': f, 'mean_drop': v['mean_drop'], 'std_drop': v['std_drop']}
                        for f, v in sorted_importances])
imp_df.to_csv(RESULTS_DIR / 'feature_importance.csv', index=False)
print("Saved: feature_importance.csv")

fig, ax = plt.subplots(figsize=(12, 7))
top_n = min(15, len(sorted_importances))
names = [f[0] for f in sorted_importances[:top_n]][::-1]
drops = [f[1]['mean_drop'] for f in sorted_importances[:top_n]][::-1]
ax.barh(range(len(names)), drops, color=plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(names))))
ax.set_yticks(range(len(names))); ax.set_yticklabels(names)
ax.set_xlabel('Mean PR-AUC Drop')
ax.set_title('Feature Importance (Permutation)', fontweight='bold')
ax.grid(alpha=0.3, axis='x')
plt.tight_layout()
plt.savefig(PLOTS_DIR / 'feature_importance.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: feature_importance.png")

# %% [markdown]
# ---
# ## Section 23 -- Early Warning Time

# %%
def evaluate_early_warning(preds, targets, threshold, ws, lookback=10):
    episodes = []
    in_ep, ep_start = False, None
    for i in range(len(targets)):
        if targets[i] == 1 and not in_ep:
            in_ep, ep_start = True, i
        elif targets[i] == 0 and in_ep:
            in_ep = False; episodes.append((ep_start, i-1))
    if in_ep: episodes.append((ep_start, len(targets)-1))

    print(f"Attack episodes: {len(episodes)}")
    if not episodes:
        return {}

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
              'warning_rate': (d_early + d_during) / tot}
    print(f"  Early: {d_early} ({d_early/tot*100:.1f}%)  During: {d_during}  Missed: {missed}")

    pos = [t for t in ewt_list if t > 0]
    if pos:
        ew_res['mean_ewt_seconds'] = float(np.mean(pos))
        ew_res['median_ewt_seconds'] = float(np.median(pos))
        ew_res['max_ewt_seconds'] = float(np.max(pos))
        print(f"  Mean EWT: {np.mean(pos):.1f}s  Median: {np.median(pos):.1f}s  Max: {np.max(pos):.1f}s")
    else:
        ew_res['mean_ewt_seconds'] = 0.0
    return ew_res

ew_results = evaluate_early_warning(test_preds, test_tgts, best_wm_thresh, WINDOW_SECONDS, HISTORY)

# %% [markdown]
# ---
# ## Section 24 -- Model Comparison

# %%
print("=" * 70)
print("   MODEL COMPARISON: LR vs CyberCast World Model")
print("=" * 70)
print(f"{'Metric':20s} {'LR':>12s} {'CyberCast':>12s} {'Winner':>10s}")
print("-" * 56)
for m_name in ['precision', 'recall', 'f1', 'roc_auc', 'pr_auc', 'fpr', 'fnr']:
    lv = lr_metrics.get(m_name, 0)
    wv = wm_metrics.get(m_name, 0)
    if m_name in ['fpr', 'fnr']:
        winner = 'LR' if lv < wv else 'CyberCast'
    else:
        winner = 'CyberCast' if wv > lv else 'LR'
    print(f"{m_name:20s} {lv:>12.4f} {wv:>12.4f} {winner:>10s}")

if ew_results.get('mean_ewt_seconds', 0) > 0:
    print(f"{'early_warning_time':20s} {'N/A':>12s} {ew_results['mean_ewt_seconds']:>11.1f}s {'CyberCast':>10s}")

comp_df = pd.DataFrame({'LogisticRegression': lr_metrics, 'CyberCast': {k:v for k,v in wm_metrics.items() if k != 'threshold'}})
comp_df.to_csv(RESULTS_DIR / 'metrics.csv')
print("Saved: metrics.csv")

# %% [markdown]
# ---
# ## Section 25 -- Final Demonstration

# %%
model.eval()
demo_idx = None
for i in range(len(y_test_raw) - HISTORY - FORECAST_HORIZON):
    if y_test_raw[i + HISTORY - 1] == 0:
        if any(y_test_raw[i + HISTORY + j] == 1
               for j in range(FORECAST_HORIZON)
               if i + HISTORY + j < len(y_test_raw)):
            demo_idx = i
            break
if demo_idx is None:
    demo_idx = min(len(X_test_seq) // 2, len(X_test_seq) - 1)

ts_off = HISTORY + demo_idx
base_ts = test_states['Timestamp'].iloc[ts_off] if ts_off < len(test_states) else None

print("=" * 70)
print("   CYBERCAST FINAL DEMONSTRATION")
print("=" * 70)
print(f"  Test sequence index: {demo_idx}")
if base_ts:
    print(f"  Base timestamp: {base_ts}")

fcs = recursive_kstep_forecast(model, X_test_seq[demo_idx], FORECAST_HORIZON, device)
fc_probs = [fc['attack_prob'] for fc in fcs]

print(f"\n  Recursive {FORECAST_HORIZON}-Step Forecast:")
print(f"  {'Step':>5s} {'Timestamp':>22s} {'P(attack)':>10s} {'Risk':>6s} {'Level':>10s} {'Stage':>25s}")
print(f"  {'-'*80}")
for fc in fcs:
    k = fc['step']
    p = fc['attack_prob']
    rs, rl = compute_risk_score(p)
    stg, ev, cert = infer_attack_stage(fc['state_pred'], state_feature_cols, p, best_wm_thresh)
    ts_str = str(base_ts + pd.Timedelta(seconds=k*WINDOW_SECONDS))[:19] if base_ts else ''
    print(f"  t+{k:2d}  {ts_str:>22s} {p:>10.4f} {rs:>6.0f} {rl:>10s} {stg:>25s}")

overall_risk, overall_level = compute_kstep_risk(fc_probs, FORECAST_HORIZON)
print(f"\n  OVERALL RISK: {overall_risk:.1f}/100  Level: {overall_level}")

print(f"\n  Top Contributing Features:")
for f, v in sorted_importances[:5]:
    print(f"    {f}: {v['mean_drop']:.6f}")

print(f"\n  Ground Truth:")
for k in range(1, FORECAST_HORIZON + 1):
    ai = demo_idx + HISTORY + k - 1
    if ai < len(y_test_raw):
        actual = 'ATTACK' if y_test_raw[ai] == 1 else 'BENIGN'
        pred = 'ATTACK' if fc_probs[k-1] >= best_wm_thresh else 'BENIGN'
        match = 'OK' if pred == actual else 'MISS'
        print(f"    t+{k}: pred={pred:7s}  actual={actual:7s}  {match}")

demo_result = {'demo_idx': demo_idx, 'forecast_probs': fc_probs,
               'overall_risk': overall_risk, 'overall_level': overall_level}

# %% [markdown]
# ---
# ## Section 26 -- Artifact Saving

# %%
print("Saving all artifacts...")

with open(MODEL_DIR / 'feature_names.json', 'w') as f:
    json.dump(state_feature_cols, f, indent=2)

config = {
    'random_seed': RANDOM_SEED, 'window_seconds': WINDOW_SECONDS,
    'history': HISTORY, 'forecast_horizon': FORECAST_HORIZON,
    'hidden_size': HIDDEN_SIZE, 'num_layers': NUM_LAYERS, 'dropout': DROPOUT,
    'lambda_state': LAMBDA_STATE, 'lambda_attack': LAMBDA_ATTACK,
    'learning_rate': LEARNING_RATE, 'batch_size': BATCH_SIZE,
    'max_epochs': MAX_EPOCHS, 'patience': PATIENCE,
    'state_dim': STATE_DIM, 'total_params': total_params,
    'best_epoch': best_epoch, 'best_val_loss': float(best_val_loss),
    'best_threshold': best_wm_thresh, 'device': str(device),
    'training_time_seconds': training_time,
    'n_temporal_states': len(temporal_states),
    'test_metrics': wm_metrics, 'lr_metrics': lr_metrics,
}
with open(MODEL_DIR / 'config.json', 'w') as f:
    json.dump(config, f, indent=2, default=str)

if ew_results:
    with open(RESULTS_DIR / 'early_warning_results.json', 'w') as f:
        json.dump(ew_results, f, indent=2, default=str)

print("\nSaved files:")
for dn, dp in [('models/', MODEL_DIR), ('results/', RESULTS_DIR), ('results/plots/', PLOTS_DIR)]:
    if dp.is_dir():
        for fn in sorted(os.listdir(dp)):
            fp = dp / fn
            if fp.is_file():
                print(f"  {dn}{fn:35s}  {fp.stat().st_size/1024:.1f} KB")

# %% [markdown]
# ---
# ## Section 27 -- Final Verification & Report

# %%
checks = []
def chk(ok, desc):
    checks.append(('PASS' if ok else 'FAIL', desc))
    return ok

chk(RAW_DATA_DIR.exists(), "Raw data from correct path")
chk('google' not in str(RAW_DATA_DIR).lower(), "No Colab paths")
chk(all(c not in LABEL_DERIVED_COLS for c in state_feature_cols), "No label-derived features in state")
chk(hasattr(model, 'state_head'), "Model has state head")
chk(hasattr(model, 'attack_head'), "Model has attack head")
chk(LAMBDA_STATE > 0 and LAMBDA_ATTACK > 0, "Multi-task loss")
chk('lr_metrics' in dir(), "LR baseline exists")
chk((MODEL_DIR / 'best_world_model.pt').exists(), "Model saved")
chk((MODEL_DIR / 'scaler.pkl').exists(), "Scaler saved")
chk((MODEL_DIR / 'feature_names.json').exists(), "Feature names saved")
chk((MODEL_DIR / 'config.json').exists(), "Config saved")
chk((RESULTS_DIR / 'metrics.csv').exists(), "Metrics saved")
chk((RESULTS_DIR / 'forecast_results.csv').exists(), "Forecast results saved")
chk((RESULTS_DIR / 'feature_importance.csv').exists(), "Feature importance saved")
chk(best_val_loss < float('inf'), "Best validation loss recorded")

print("\nVerification Checklist:")
all_pass = True
for s, d in checks:
    mark = '[PASS]' if s == 'PASS' else '[FAIL]'
    print(f"  {mark} {d}")
    if s == 'FAIL': all_pass = False

ewt_str = "N/A"
if ew_results and 'mean_ewt_seconds' in ew_results:
    ewt_str = f"{ew_results['mean_ewt_seconds']:.1f}s"

print(f"""
==========================================
         CYBERCAST FINAL MODEL REPORT
==========================================

Dataset:
  Network states:      {len(temporal_states):,}
  State dimension:     {STATE_DIM}
  Train / Val / Test:  {len(train_states):,} / {len(val_states):,} / {len(test_states):,}

Window:                {WINDOW_SECONDS}s
History:               {HISTORY} windows
Forecast horizon:      {FORECAST_HORIZON} steps ({FORECAST_HORIZON*WINDOW_SECONDS}s)

Model:                 Dual-head LSTM World Model
Parameters:            {total_params:,}
Device:                {device}
Best epoch:            {best_epoch}
Best val loss:         {best_val_loss:.5f}

-------- BASELINE (Logistic Regression) --------
  Precision:   {lr_metrics['precision']:.4f}
  Recall:      {lr_metrics['recall']:.4f}
  F1:          {lr_metrics['f1']:.4f}
  PR-AUC:      {lr_metrics['pr_auc']:.4f}
  ROC-AUC:     {lr_metrics['roc_auc']:.4f}
  FPR:         {lr_metrics['fpr']:.4f}
  FNR:         {lr_metrics['fnr']:.4f}

-------- CYBERCAST WORLD MODEL --------
  Precision:   {wm_metrics['precision']:.4f}
  Recall:      {wm_metrics['recall']:.4f}
  F1:          {wm_metrics['f1']:.4f}
  PR-AUC:      {wm_metrics['pr_auc']:.4f}
  ROC-AUC:     {wm_metrics['roc_auc']:.4f}
  FPR:         {wm_metrics['fpr']:.4f}
  FNR:         {wm_metrics['fnr']:.4f}
  EWT:         {ewt_str}

-------- FORECAST --------
  Horizon:     {FORECAST_HORIZON} steps
  Risk score:  {demo_result['overall_risk']:.1f}/100
  Risk level:  {demo_result['overall_level']}

Top features:""")
for f, v in sorted_importances[:5]:
    print(f"  - {f} (drop={v['mean_drop']:.6f})")

if all_pass:
    print("\n==========================================")
    print("CYBERCAST PIPELINE VERIFICATION COMPLETE")
    print("==========================================")
else:
    print("\nSOME CHECKS FAILED -- see above")

# %%
print("\nCyberCast notebook execution complete!")
print(f"Artifacts: {PROJECT_DIR}")
print(f"Model: Dual-head LSTM ({total_params:,} params)")
print(f"Test ROC-AUC: {wm_metrics['roc_auc']:.4f}  PR-AUC: {wm_metrics['pr_auc']:.4f}  F1: {wm_metrics['f1']:.4f}")
print(f"Forecast: {FORECAST_HORIZON} steps ({FORECAST_HORIZON*WINDOW_SECONDS}s)")
print("Ready for SIH 2026!")
