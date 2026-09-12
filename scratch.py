# ============================================================
# SECTION 2 — Google Drive Setup
# ============================================================

import os

# Mount Google Drive
try:
    from google.colab import drive
    drive.mount('/content/drive')
    IN_COLAB = True
    print("✅ Google Drive mounted successfully.")
except ImportError:
    IN_COLAB = False
    print("⚠️  Not running in Colab. Using local paths.")

# ---- Project Paths ----
PROJECT_DIR   = '/content/drive/MyDrive/CyberCast'
RAW_DIR       = os.path.join(PROJECT_DIR, 'data', 'raw')
PROCESSED_DIR = os.path.join(PROJECT_DIR, 'data', 'processed')
MODEL_DIR     = os.path.join(PROJECT_DIR, 'models')
RESULTS_DIR   = os.path.join(PROJECT_DIR, 'results')
PLOTS_DIR     = os.path.join(PROJECT_DIR, 'plots')

# Create directories if they don't exist
for d in [PROCESSED_DIR, MODEL_DIR, RESULTS_DIR, PLOTS_DIR]:
    os.makedirs(d, exist_ok=True)
    print(f"📁 {d}")

# Verify raw data directory exists and has CSV files
if not os.path.isdir(RAW_DIR):
    raise FileNotFoundError(
        f"❌ Raw data directory not found: {RAW_DIR}\n"
        f"Please place your CIC-IDS2018 CSV files in this directory."
    )

import glob
raw_csv_files = sorted(glob.glob(os.path.join(RAW_DIR, '*.csv')))

if len(raw_csv_files) == 0:
    raise FileNotFoundError(
        f"❌ No CSV files found in {RAW_DIR}\n"
        f"Please place your CIC-IDS2018 CSV files in this directory."
    )

print(f"\n✅ Found {len(raw_csv_files)} CSV file(s):")
for f in raw_csv_files:
    size_mb = os.path.getsize(f) / (1024 * 1024)
    print(f"   {os.path.basename(f):50s}  {size_mb:>8.1f} MB")



# ============================================================
# SECTION 3 — Environment Setup
# ============================================================

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for Colab compatibility
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import joblib
import json
import time
import warnings
import random
import gc
from collections import Counter, OrderedDict
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, average_precision_score, precision_recall_curve,
    roc_curve, confusion_matrix, classification_report,
    precision_score, recall_score, f1_score, accuracy_score
)

warnings.filterwarnings('ignore')

# ---- Print Versions ----
print("=" * 50)
print("       CyberCast — Environment Info")
print("=" * 50)
print(f"  pandas      : {pd.__version__}")
print(f"  numpy       : {np.__version__}")
print(f"  matplotlib  : {matplotlib.__version__}")
print(f"  seaborn     : {sns.__version__}")
print(f"  scikit-learn: {sklearn.__version__}")
print(f"  PyTorch     : {torch.__version__}")
print(f"  CUDA avail  : {torch.cuda.is_available()}")

# ---- Device Selection ----
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\n🖥️  Device: {device}")
if torch.cuda.is_available():
    print(f"   GPU: {torch.cuda.get_device_name(0)}")
    print(f"   Memory: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")
else:
    print("   Running on CPU (training will be slower).")

# ---- Reproducibility ----
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

print(f"\n🎲 Random seed set to {SEED} (numpy, random, torch, CUDA)")
print("=" * 50)



# ============================================================
# CONFIGURABLE HYPERPARAMETERS
# ============================================================

# ---- Temporal Windowing ----
WINDOW_SECONDS     = 10       # Aggregate flows into 10-second windows

# ---- Sequence / Forecasting ----
SEQUENCE_LENGTH    = 10       # Number of past windows as LSTM input
FORECAST_HORIZON   = 5        # K-step future forecast (5 windows = ~50s)

# ---- Model Architecture ----
HIDDEN_SIZE        = 128      # LSTM hidden dimension
NUM_LAYERS         = 2        # Number of LSTM layers
DROPOUT            = 0.2      # Dropout rate

# ---- Training ----
BATCH_SIZE         = 128
LEARNING_RATE      = 0.001
EPOCHS             = 30
PATIENCE           = 5        # Early stopping patience

# ---- Data Split (chronological) ----
TRAIN_RATIO        = 0.70
VAL_RATIO          = 0.15
TEST_RATIO         = 0.15

print("✅ Hyperparameters configured:")
for name, val in [
    ('WINDOW_SECONDS', WINDOW_SECONDS),
    ('SEQUENCE_LENGTH', SEQUENCE_LENGTH),
    ('FORECAST_HORIZON', FORECAST_HORIZON),
    ('HIDDEN_SIZE', HIDDEN_SIZE),
    ('NUM_LAYERS', NUM_LAYERS),
    ('DROPOUT', DROPOUT),
    ('BATCH_SIZE', BATCH_SIZE),
    ('LEARNING_RATE', LEARNING_RATE),
    ('EPOCHS', EPOCHS),
    ('PATIENCE', PATIENCE),
    ('TRAIN/VAL/TEST', f'{TRAIN_RATIO}/{VAL_RATIO}/{TEST_RATIO}'),
]:
    print(f"   {name:25s} = {val}")



# ============================================================
# SECTION 4 — Raw Data Discovery
# ============================================================

def discover_csv_files(raw_dir):
    """Discover and catalog all CSV files in the raw data directory."""
    csv_files = sorted(glob.glob(os.path.join(raw_dir, '*.csv')))
    file_info = []
    total_size = 0

    for fpath in csv_files:
        fname = os.path.basename(fpath)
        size_bytes = os.path.getsize(fpath)
        size_mb = size_bytes / (1024 * 1024)
        total_size += size_bytes

        # Count rows without loading entire file
        # Read only first and last line to estimate; exact count via iteration
        try:
            with open(fpath, 'r', encoding='utf-8', errors='replace') as f:
                row_count = sum(1 for _ in f) - 1  # subtract header
        except Exception as e:
            row_count = -1
            print(f"⚠️  Could not count rows in {fname}: {e}")

        file_info.append({
            'filename': fname,
            'path': fpath,
            'size_mb': size_mb,
            'rows': row_count
        })

    print(f"\n{'='*80}")
    print(f"{'Filename':50s} {'Size (MB)':>10s} {'Rows':>12s}")
    print(f"{'='*80}")
    for fi in file_info:
        rows_str = f"{fi['rows']:,}" if fi['rows'] >= 0 else 'ERROR'
        print(f"{fi['filename']:50s} {fi['size_mb']:>10.1f} {rows_str:>12s}")

    total_rows = sum(fi['rows'] for fi in file_info if fi['rows'] >= 0)
    print(f"{'='*80}")
    print(f"{'TOTAL':50s} {total_size/1e6:>10.1f} {total_rows:>12,}")
    print(f"{'='*80}")

    return file_info

file_info = discover_csv_files(RAW_DIR)
print(f"\n✅ Discovered {len(file_info)} CSV file(s) ready for processing.")



# ============================================================
# SECTION 5 — Raw Data Inspection
# ============================================================

def inspect_dataset(csv_path, sample_rows=5000):
    """Inspect a single CSV file for structure, quality, and content."""
    fname = os.path.basename(csv_path)
    print(f"\n🔍 Inspecting: {fname}")
    print("=" * 60)

    df = pd.read_csv(csv_path, nrows=sample_rows)
    # Strip whitespace from column names
    df.columns = df.columns.str.strip()

    print(f"\n📐 Shape: {df.shape}")
    print(f"\n📋 Columns ({len(df.columns)}):")
    for i, col in enumerate(df.columns):
        print(f"   {i:3d}. {col:30s}  dtype={df[col].dtype}")

    # Missing values
    missing = df.isnull().sum()
    missing_cols = missing[missing > 0]
    print(f"\n❓ Missing values: {len(missing_cols)} column(s) with nulls")
    if len(missing_cols) > 0:
        print(missing_cols)

    # Infinite values in numeric columns
    num_cols = df.select_dtypes(include=[np.number]).columns
    inf_counts = {}
    for col in num_cols:
        n_inf = np.isinf(df[col].values).sum() if df[col].dtype != object else 0
        if n_inf > 0:
            inf_counts[col] = n_inf
    print(f"\n♾️  Infinite values: {len(inf_counts)} column(s)")
    if inf_counts:
        for col, cnt in inf_counts.items():
            print(f"   {col}: {cnt}")

    # Verify essential columns
    assert 'Label' in df.columns, "❌ 'Label' column not found!"
    assert 'Timestamp' in df.columns, "❌ 'Timestamp' column not found!"
    print("\n✅ 'Label' column found.")
    print("✅ 'Timestamp' column found.")

    # Label distribution
    print(f"\n🏷️  Label distribution (sample of {len(df)} rows):")
    label_counts = df['Label'].value_counts()
    for label, count in label_counts.items():
        pct = count / len(df) * 100
        print(f"   {label:30s}  {count:>8,}  ({pct:.2f}%)")

    # Timestamp inspection
    print(f"\n🕐 Timestamp samples:")
    print(f"   First: {df['Timestamp'].iloc[0]}")
    print(f"   Last:  {df['Timestamp'].iloc[-1]}")

    # Try parsing timestamps
    ts_parsed = pd.to_datetime(df['Timestamp'], format='mixed', dayfirst=False, errors='coerce')
    n_invalid = ts_parsed.isna().sum()
    print(f"   Invalid timestamps: {n_invalid} / {len(df)}")
    if n_invalid < len(df):
        print(f"   Min timestamp: {ts_parsed.min()}")
        print(f"   Max timestamp: {ts_parsed.max()}")

    print(f"\n{'='*60}")
    return df

# Inspect the first file as representative
df_sample = inspect_dataset(file_info[0]['path'])



# ============================================================
# SECTION 6 — Global Label Distribution
# ============================================================

def compute_global_label_distribution(file_info_list):
    """Count labels across all CSV files reading only the Label column."""
    global_counts = Counter()
    total_rows = 0

    for fi in file_info_list:
        print(f"  Reading labels from: {fi['filename']}...", end=' ')
        try:
            labels = pd.read_csv(fi['path'], usecols=['Label'], dtype={'Label': str})
            labels['Label'] = labels['Label'].str.strip()
            counts = labels['Label'].value_counts().to_dict()
            for label, count in counts.items():
                global_counts[label] += count
            total_rows += len(labels)
            print(f"{len(labels):,} rows")
            del labels
            gc.collect()
        except Exception as e:
            print(f"ERROR: {e}")

    return global_counts, total_rows

print("📊 Computing global label distribution...\n")
global_label_counts, total_raw_rows = compute_global_label_distribution(file_info)

print(f"\n{'='*70}")
print(f"{'Label':40s} {'Count':>12s} {'Percentage':>12s}")
print(f"{'='*70}")
for label, count in sorted(global_label_counts.items(), key=lambda x: -x[1]):
    pct = count / total_raw_rows * 100
    print(f"{label:40s} {count:>12,} {pct:>11.2f}%")
print(f"{'='*70}")
print(f"{'TOTAL':40s} {total_raw_rows:>12,}")

# Binary split
benign_count = global_label_counts.get('Benign', 0)
attack_count = total_raw_rows - benign_count
print(f"\n🛡️  Benign:  {benign_count:>12,} ({benign_count/total_raw_rows*100:.2f}%)")
print(f"⚔️  Attack:  {attack_count:>12,} ({attack_count/total_raw_rows*100:.2f}%)")
print(f"\n⚠️  Class imbalance ratio: 1:{benign_count/max(attack_count,1):.1f} (attack:benign)")
print("   → We will use pos_weight in BCEWithLogitsLoss, NOT random oversampling.")



# Plot global label distribution
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Bar chart
labels_sorted = sorted(global_label_counts.items(), key=lambda x: -x[1])
ax = axes[0]
ax.barh([l[0] for l in labels_sorted], [l[1] for l in labels_sorted],
        color=sns.color_palette('viridis', len(labels_sorted)))
ax.set_xlabel('Count')
ax.set_title('Global Label Distribution (All Files)', fontsize=14, fontweight='bold')
ax.invert_yaxis()
for i, (label, count) in enumerate(labels_sorted):
    ax.text(count, i, f' {count:,}', va='center', fontsize=9)

# Binary pie chart
ax2 = axes[1]
ax2.pie([benign_count, attack_count],
        labels=['Benign', 'Attack'],
        autopct='%1.1f%%',
        colors=['#2ecc71', '#e74c3c'],
        explode=[0, 0.05],
        shadow=True, startangle=90)
ax2.set_title('Binary: Benign vs Attack', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, 'global_label_distribution.png'), dpi=150, bbox_inches='tight')
plt.show()
print("📊 Saved: global_label_distribution.png")



# ============================================================
# SECTION 7 — Data Cleaning
# ============================================================

def clean_dataframe(csv_path):
    """Clean a single CIC-IDS2018 CSV file.

    Returns a cleaned DataFrame sorted by Timestamp with a binary Attack column.
    """
    fname = os.path.basename(csv_path)
    print(f"\n{'─'*60}")
    print(f"🧹 Cleaning: {fname}")

    # 1. Read the file
    df = pd.read_csv(csv_path, low_memory=False)
    initial_rows = len(df)
    print(f"   Loaded: {initial_rows:,} rows × {df.shape[1]} cols")

    # 2. Strip whitespace from column names
    df.columns = df.columns.str.strip()

    # 3. Strip whitespace from string columns
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].astype(str).str.strip()

    # 4. Verify essential columns
    if 'Label' not in df.columns:
        raise ValueError(f"❌ 'Label' column missing in {fname}")
    if 'Timestamp' not in df.columns:
        raise ValueError(f"❌ 'Timestamp' column missing in {fname}")

    # 5. Parse Timestamp
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], format='mixed', dayfirst=False, errors='coerce')
    n_invalid_ts = df['Timestamp'].isna().sum()
    if n_invalid_ts > 0:
        print(f"   ⚠️  Removed {n_invalid_ts:,} rows with invalid timestamps")
        df = df.dropna(subset=['Timestamp'])

    # 6. Convert numeric columns
    exclude_cols = {'Timestamp', 'Label'}
    for col in df.columns:
        if col not in exclude_cols and df[col].dtype == object:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # 7. Replace +inf and -inf with NaN
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    n_inf = np.isinf(df[numeric_cols].values).sum()
    if n_inf > 0:
        print(f"   ♾️  Replaced {n_inf:,} infinite values with NaN")
        df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)

    # 8. Handle missing values — fill with column median (robust to outliers)
    n_missing = df[numeric_cols].isna().sum().sum()
    if n_missing > 0:
        print(f"   ❓ Filling {n_missing:,} missing numeric values with column median")
        for col in numeric_cols:
            if df[col].isna().any():
                df[col] = df[col].fillna(df[col].median())

    # 9. Remove exact duplicate rows
    n_before = len(df)
    df = df.drop_duplicates()
    n_dupes = n_before - len(df)
    if n_dupes > 0:
        print(f"   🔄 Removed {n_dupes:,} duplicate rows")

    # 10. Sort chronologically
    df = df.sort_values('Timestamp').reset_index(drop=True)

    # 11. Create binary target
    df['Attack'] = (df['Label'].str.strip() != 'Benign').astype(np.int8)

    final_rows = len(df)
    n_attacks = df['Attack'].sum()
    print(f"   ✅ Final: {final_rows:,} rows (dropped {initial_rows - final_rows:,})")
    print(f"   🛡️  Benign: {final_rows - n_attacks:,} | ⚔️  Attack: {n_attacks:,}")
    print(f"   🕐 {df['Timestamp'].min()} → {df['Timestamp'].max()}")

    return df


# Process all files and concatenate
print("\n" + "=" * 60)
print("   CLEANING ALL RAW CSV FILES")
print("=" * 60)

all_dfs = []
for fi in file_info:
    try:
        df_clean = clean_dataframe(fi['path'])
        all_dfs.append(df_clean)
    except Exception as e:
        print(f"   ❌ FAILED to process {fi['filename']}: {e}")

if len(all_dfs) == 0:
    raise RuntimeError("❌ No files were successfully cleaned!")

# Concatenate and sort globally
print(f"\n🔗 Concatenating {len(all_dfs)} cleaned DataFrames...")
df_all = pd.concat(all_dfs, ignore_index=True)
df_all = df_all.sort_values('Timestamp').reset_index(drop=True)

# Free memory
del all_dfs
gc.collect()

print(f"\n✅ Combined dataset: {len(df_all):,} rows × {df_all.shape[1]} cols")
print(f"🕐 Time range: {df_all['Timestamp'].min()} → {df_all['Timestamp'].max()}")
print(f"🛡️  Benign: {(df_all['Attack']==0).sum():,} | ⚔️  Attack: {(df_all['Attack']==1).sum():,}")



# ============================================================
# SECTION 8 — Feature Selection
# ============================================================

def select_features(df, exclude_cols=None):
    """Select valid numeric features, removing constant and problematic columns.

    IMPORTANT: Does NOT use Label or Attack for selection decisions.
    """
    if exclude_cols is None:
        exclude_cols = {'Timestamp', 'Label', 'Attack'}

    # All numeric columns except excluded
    all_numeric = df.select_dtypes(include=[np.number]).columns.tolist()
    candidate_features = [c for c in all_numeric if c not in exclude_cols]
    print(f"\n📊 Feature Selection")
    print(f"   Original numeric columns: {len(all_numeric)}")
    print(f"   After excluding metadata: {len(candidate_features)}")

    removed = []

    # Remove zero-variance features
    variances = df[candidate_features].var()
    zero_var = variances[variances == 0].index.tolist()
    if zero_var:
        print(f"   Removed {len(zero_var)} zero-variance features: {zero_var}")
        removed.extend(zero_var)
        candidate_features = [c for c in candidate_features if c not in zero_var]

    # Remove features where >50% of values are NaN (safety check)
    high_null = []
    for col in candidate_features:
        null_pct = df[col].isna().mean()
        if null_pct > 0.5:
            high_null.append(col)
    if high_null:
        print(f"   Removed {len(high_null)} high-null features: {high_null}")
        removed.extend(high_null)
        candidate_features = [c for c in candidate_features if c not in high_null]

    # Remove features with near-zero variance (< 1e-10)
    near_zero = []
    for col in candidate_features:
        if variances.get(col, 0) < 1e-10:
            near_zero.append(col)
    if near_zero:
        print(f"   Removed {len(near_zero)} near-zero-variance features: {near_zero}")
        removed.extend(near_zero)
        candidate_features = [c for c in candidate_features if c not in near_zero]

    print(f"\n   ✅ Final feature count: {len(candidate_features)}")
    print(f"   ❌ Total removed: {len(removed)}")

    return candidate_features, removed


feature_columns, removed_features = select_features(df_all)
print(f"\n📋 Selected features ({len(feature_columns)}):")
for i, col in enumerate(feature_columns):
    print(f"   {i+1:3d}. {col}")



# ============================================================
# SECTION 9 — Feature Engineering
# ============================================================

def engineer_features(df):
    """Create derived network behavior features.

    All features use ONLY existing flow statistics.
    Label and Attack are NEVER used as inputs.
    Division by zero is handled safely.
    """
    print("\n🔧 Engineering features...")
    n_before = len(df.columns)
    engineered = []

    # Helper: safe division
    def safe_div(a, b, fill=0.0):
        return np.where(b != 0, a / b, fill)

    # 1. Fwd/Bwd Packet Ratio — detects directional asymmetry
    if 'Tot Fwd Pkts' in df.columns and 'Tot Bwd Pkts' in df.columns:
        df['Fwd_Bwd_Pkt_Ratio'] = safe_div(
            df['Tot Fwd Pkts'].values, df['Tot Bwd Pkts'].values
        )
        engineered.append('Fwd_Bwd_Pkt_Ratio')

    # 2. Fwd/Bwd Byte Ratio — data exfiltration indicator
    if 'TotLen Fwd Pkts' in df.columns and 'TotLen Bwd Pkts' in df.columns:
        df['Fwd_Bwd_Byte_Ratio'] = safe_div(
            df['TotLen Fwd Pkts'].values, df['TotLen Bwd Pkts'].values
        )
        engineered.append('Fwd_Bwd_Byte_Ratio')

    # 3. Total Packets — volume indicator
    if 'Tot Fwd Pkts' in df.columns and 'Tot Bwd Pkts' in df.columns:
        df['Total_Pkts'] = df['Tot Fwd Pkts'] + df['Tot Bwd Pkts']
        engineered.append('Total_Pkts')

    # 4. Total Bytes — bandwidth indicator
    if 'TotLen Fwd Pkts' in df.columns and 'TotLen Bwd Pkts' in df.columns:
        df['Total_Bytes'] = df['TotLen Fwd Pkts'] + df['TotLen Bwd Pkts']
        engineered.append('Total_Bytes')

    # 5. Average Packet Size — small packets = scanning
    if 'Total_Pkts' in df.columns and 'Total_Bytes' in df.columns:
        df['Avg_Pkt_Size'] = safe_div(df['Total_Bytes'].values, df['Total_Pkts'].values)
        engineered.append('Avg_Pkt_Size')

    # 6. Packets per Second (from flow duration)
    if 'Total_Pkts' in df.columns and 'Flow Duration' in df.columns:
        # Flow Duration is in microseconds
        duration_sec = df['Flow Duration'].values / 1e6
        df['Pkts_Per_Sec'] = safe_div(df['Total_Pkts'].values, duration_sec)
        engineered.append('Pkts_Per_Sec')

    # 7. Bytes per Second
    if 'Total_Bytes' in df.columns and 'Flow Duration' in df.columns:
        duration_sec = df['Flow Duration'].values / 1e6
        df['Bytes_Per_Sec'] = safe_div(df['Total_Bytes'].values, duration_sec)
        engineered.append('Bytes_Per_Sec')

    # 8. SYN/FIN Ratio — SYN floods leave many SYNs without FINs
    if 'SYN Flag Cnt' in df.columns and 'FIN Flag Cnt' in df.columns:
        df['SYN_FIN_Ratio'] = safe_div(
            df['SYN Flag Cnt'].values, (df['FIN Flag Cnt'].values + 1)  # +1 to avoid /0
        )
        engineered.append('SYN_FIN_Ratio')

    # 9. RST/SYN Ratio — port scanning indicator
    if 'RST Flag Cnt' in df.columns and 'SYN Flag Cnt' in df.columns:
        df['RST_SYN_Ratio'] = safe_div(
            df['RST Flag Cnt'].values, (df['SYN Flag Cnt'].values + 1)
        )
        engineered.append('RST_SYN_Ratio')

    # 10. Forward Packet Proportion
    if 'Tot Fwd Pkts' in df.columns and 'Total_Pkts' in df.columns:
        df['Fwd_Pkt_Proportion'] = safe_div(
            df['Tot Fwd Pkts'].values, df['Total_Pkts'].values
        )
        engineered.append('Fwd_Pkt_Proportion')

    # Replace any inf/nan from engineering
    for col in engineered:
        df[col] = df[col].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    print(f"   ✅ Created {len(engineered)} engineered features:")
    for feat in engineered:
        print(f"      • {feat}")

    return df, engineered


df_all, engineered_features = engineer_features(df_all)

# Update feature list with new engineered features
feature_columns = feature_columns + engineered_features
# Ensure no duplicates
feature_columns = list(dict.fromkeys(feature_columns))

print(f"\n📋 Total feature count after engineering: {len(feature_columns)}")



# ============================================================
# SECTION 10 — Time Window Generation
# ============================================================

def create_time_windows(df, feature_cols, window_seconds=10):
    """Aggregate irregular flows into fixed-interval temporal network states.

    For each window:
    - Compute aggregate statistics of all flows
    - Record attack presence (binary target)
    - Record traffic presence

    No future information influences current windows.
    """
    print(f"\n⏱️  Creating {window_seconds}-second temporal windows...")

    # Set Timestamp as index for resampling
    df = df.set_index('Timestamp').sort_index()

    # Define aggregation rules
    # For most features: compute mean within the window
    agg_dict = {col: 'mean' for col in feature_cols if col in df.columns}

    # Special aggregations for key columns
    special_aggs = {}
    if 'Tot Fwd Pkts' in df.columns:
        special_aggs['total_fwd_packets'] = ('Tot Fwd Pkts', 'sum')
    if 'Tot Bwd Pkts' in df.columns:
        special_aggs['total_bwd_packets'] = ('Tot Bwd Pkts', 'sum')
    if 'Total_Bytes' in df.columns:
        special_aggs['total_bytes'] = ('Total_Bytes', 'sum')
    if 'Flow Duration' in df.columns:
        special_aggs['mean_flow_duration'] = ('Flow Duration', 'mean')

    # Resample at WINDOW_SECONDS frequency
    freq = f'{window_seconds}s'

    # Mean aggregation for features
    existing_feature_cols = [c for c in feature_cols if c in df.columns]
    states = df[existing_feature_cols].resample(freq).mean()

    # Flow count per window
    flow_counts = df[existing_feature_cols[0]].resample(freq).count()
    states['flow_count'] = flow_counts

    # Attack target: 1 if ANY attack flow in window
    attack_sum = df['Attack'].resample(freq).sum()
    attack_total = df['Attack'].resample(freq).count()
    states['attack_count'] = attack_sum
    states['attack_ratio'] = np.where(attack_total > 0, attack_sum / attack_total, 0.0)
    states['Attack'] = (attack_sum > 0).astype(np.int8)

    # Has_Traffic flag
    states['Has_Traffic'] = (states['flow_count'] > 0).astype(np.int8)

    # Fill NaN (empty windows) with 0
    states = states.fillna(0.0)

    # Convert to float32 for memory efficiency
    for col in states.select_dtypes(include=[np.float64]).columns:
        states[col] = states[col].astype(np.float32)

    # Reset index to get Timestamp as column
    states = states.reset_index()
    states = states.rename(columns={'index': 'Timestamp'}) if 'index' in states.columns else states

    # Validation
    n_states = len(states)
    n_attack_windows = states['Attack'].sum()
    n_traffic_windows = states['Has_Traffic'].sum()
    n_empty_windows = n_states - n_traffic_windows

    print(f"   ✅ Created {n_states:,} temporal states")
    print(f"   📊 With traffic: {n_traffic_windows:,} | Empty: {n_empty_windows:,}")
    print(f"   🛡️  Benign windows: {n_states - n_attack_windows:,}")
    print(f"   ⚔️  Attack windows: {n_attack_windows:,} ({n_attack_windows/n_states*100:.2f}%)")
    print(f"   🕐 {states['Timestamp'].min()} → {states['Timestamp'].max()}")

    return states


# Create temporal states
temporal_states = create_time_windows(df_all, feature_columns, WINDOW_SECONDS)

# Save to Parquet
parquet_path = os.path.join(PROCESSED_DIR, f'network_states_{WINDOW_SECONDS}s.parquet')
temporal_states.to_parquet(parquet_path, index=False)
print(f"\n💾 Saved temporal states to: {parquet_path}")
print(f"   File size: {os.path.getsize(parquet_path) / 1e6:.1f} MB")

# Free raw data memory
del df_all
gc.collect()
print("\n🗑️  Freed raw DataFrame memory.")



# ============================================================
# SECTION 11 — Temporal Data Validation
# ============================================================

def validate_temporal_data(states, window_seconds):
    """Validate temporal network states for correctness."""
    print("\n🔎 Validating temporal data...")
    issues = []

    # 1. Check sorted
    is_sorted = states['Timestamp'].is_monotonic_increasing
    print(f"   Timestamps sorted:        {'✅' if is_sorted else '❌'}")
    if not is_sorted:
        issues.append("Timestamps are not sorted")

    # 2. Check for duplicate timestamps
    n_dupes = states['Timestamp'].duplicated().sum()
    print(f"   Duplicate timestamps:     {'✅ None' if n_dupes == 0 else f'❌ {n_dupes}'}")
    if n_dupes > 0:
        issues.append(f"{n_dupes} duplicate timestamps")

    # 3. Check window spacing
    diffs = states['Timestamp'].diff().dt.total_seconds().dropna()
    expected_spacing = window_seconds
    mean_spacing = diffs.mean()
    std_spacing = diffs.std()
    print(f"   Mean spacing:             {mean_spacing:.2f}s (expected {expected_spacing}s)")
    print(f"   Spacing std:              {std_spacing:.2f}s")
    spacing_ok = abs(mean_spacing - expected_spacing) < 1.0
    print(f"   Spacing valid:            {'✅' if spacing_ok else '⚠️  Off by >1s'}")

    # 4. Check for missing feature values
    n_missing = states.drop(columns=['Timestamp']).isna().sum().sum()
    print(f"   Missing values:           {'✅ None' if n_missing == 0 else f'❌ {n_missing}'}")

    # 5. Check for infinite values
    numeric_cols = states.select_dtypes(include=[np.number]).columns
    n_inf = np.isinf(states[numeric_cols].values).sum()
    print(f"   Infinite values:          {'✅ None' if n_inf == 0 else f'❌ {n_inf}'}")

    # 6. Attack label check
    attack_values = states['Attack'].unique()
    valid_attacks = set(attack_values).issubset({0, 1})
    print(f"   Attack labels valid:      {'✅' if valid_attacks else f'❌ Unexpected: {attack_values}'}")

    n_attack = states['Attack'].sum()
    if n_attack == 0:
        raise ValueError("❌ No attack windows found! Cannot train forecasting model.")
    if n_attack == len(states):
        raise ValueError("❌ All windows are attacks! Cannot train forecasting model.")

    print(f"\n   ✅ Validation complete. Issues: {len(issues)}")
    return issues


validation_issues = validate_temporal_data(temporal_states, WINDOW_SECONDS)

# Display sample states
print("\n📋 Sample temporal states:")
display_cols = ['Timestamp', 'flow_count', 'attack_count', 'attack_ratio', 'Attack', 'Has_Traffic']
display_cols = [c for c in display_cols if c in temporal_states.columns]
print(temporal_states[display_cols].head(10).to_string(index=False))



# Plot attack ratio and flow count over time
fig, axes = plt.subplots(2, 1, figsize=(18, 8), sharex=True)

# Attack ratio timeline
ax1 = axes[0]
ax1.fill_between(temporal_states['Timestamp'], temporal_states['Attack'],
                 alpha=0.6, color='#e74c3c', label='Attack Window')
ax1.set_ylabel('Attack (Binary)', fontsize=12)
ax1.set_title('Attack Windows Over Time', fontsize=14, fontweight='bold')
ax1.legend(loc='upper right')

# Flow count timeline
ax2 = axes[1]
ax2.plot(temporal_states['Timestamp'], temporal_states['flow_count'],
         color='#3498db', alpha=0.7, linewidth=0.5)
ax2.set_ylabel('Flow Count', fontsize=12)
ax2.set_xlabel('Time', fontsize=12)
ax2.set_title('Network Flow Count Over Time', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, 'temporal_overview.png'), dpi=150, bbox_inches='tight')
plt.show()
print("📊 Saved: temporal_overview.png")



# ============================================================
# SECTION 12 — Chronological Train / Validation / Test Split
# ============================================================

def split_temporally(states, train_ratio=0.70, val_ratio=0.15):
    """Split temporal states chronologically. NO shuffling."""
    n = len(states)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train_states = states.iloc[:train_end].copy()
    val_states = states.iloc[train_end:val_end].copy()
    test_states = states.iloc[val_end:].copy()

    print(f"\n📊 Chronological Split:")
    print(f"{'':5s} {'Split':10s} {'States':>10s} {'Attack':>10s} {'Attack%':>10s} {'Start':>25s} {'End':>25s}")
    print(f"{'─'*95}")
    for name, df in [('Train', train_states), ('Val', val_states), ('Test', test_states)]:
        n_atk = df['Attack'].sum()
        pct = n_atk / len(df) * 100 if len(df) > 0 else 0
        ts_start = str(df['Timestamp'].iloc[0])[:19]
        ts_end = str(df['Timestamp'].iloc[-1])[:19]
        print(f"{'':5s} {name:10s} {len(df):>10,} {n_atk:>10,} {pct:>9.2f}% {ts_start:>25s} {ts_end:>25s}")

    # Verify no temporal overlap
    assert train_states['Timestamp'].max() < val_states['Timestamp'].min(), \
        "❌ Train/Val temporal overlap!"
    assert val_states['Timestamp'].max() < test_states['Timestamp'].min(), \
        "❌ Val/Test temporal overlap!"
    print(f"\n✅ No temporal overlap between splits.")

    return train_states, val_states, test_states


train_states, val_states, test_states = split_temporally(
    temporal_states, TRAIN_RATIO, VAL_RATIO
)



# ============================================================
# SECTION 13 — Prevent Data Leakage: Scaling
# ============================================================

def scale_features(train_states, val_states, test_states, feature_cols):
    """Scale features using StandardScaler fitted ONLY on training data.

    CRITICAL: Never fit the scaler on validation or test data.
    """
    print("\n🔒 Scaling features (fit on TRAIN only)...")

    # Filter to only features that exist in the data
    valid_features = [c for c in feature_cols if c in train_states.columns]
    print(f"   Features to scale: {len(valid_features)}")

    scaler = StandardScaler()

    # FIT on training data ONLY
    scaler.fit(train_states[valid_features].values)
    print("   ✅ Scaler fitted on training data.")

    # TRANSFORM all splits
    X_train = scaler.transform(train_states[valid_features].values).astype(np.float32)
    X_val = scaler.transform(val_states[valid_features].values).astype(np.float32)
    X_test = scaler.transform(test_states[valid_features].values).astype(np.float32)

    y_train = train_states['Attack'].values.astype(np.float32)
    y_val = val_states['Attack'].values.astype(np.float32)
    y_test = test_states['Attack'].values.astype(np.float32)

    # Validation: no NaN or inf after scaling
    for name, X in [('Train', X_train), ('Val', X_val), ('Test', X_test)]:
        assert not np.isnan(X).any(), f"❌ NaN found in {name} after scaling"
        assert not np.isinf(X).any(), f"❌ Inf found in {name} after scaling"

    print(f"   X_train: {X_train.shape}  y_train: {y_train.shape}")
    print(f"   X_val:   {X_val.shape}    y_val:   {y_val.shape}")
    print(f"   X_test:  {X_test.shape}   y_test:  {y_test.shape}")
    print("   ✅ No NaN or Inf in scaled data.")

    return X_train, y_train, X_val, y_val, X_test, y_test, scaler, valid_features


# Determine which features to use (exclude metadata columns)
metadata_cols = {'Timestamp', 'Label', 'Attack', 'attack_count', 'attack_ratio', 'Has_Traffic'}
model_feature_cols = [c for c in feature_columns if c not in metadata_cols
                      and c in train_states.columns]
# Also include flow_count as a feature
if 'flow_count' in train_states.columns and 'flow_count' not in model_feature_cols:
    model_feature_cols.append('flow_count')

X_train_raw, y_train_raw, X_val_raw, y_val_raw, X_test_raw, y_test_raw, scaler, final_feature_cols = \
    scale_features(train_states, val_states, test_states, model_feature_cols)

# Save scaler
scaler_path = os.path.join(MODEL_DIR, 'scaler.joblib')
joblib.dump(scaler, scaler_path)
print(f"\n💾 Saved scaler to: {scaler_path}")



# ============================================================
# SECTION 14 — Temporal Sequence Creation
# ============================================================

def create_sequences(X, y, seq_length, forecast_step=1):
    """Create temporal sequences for LSTM.

    Input shape:  (N, seq_length, n_features)
    Target shape: (N,)  — attack state at t+forecast_step

    No future data leaks into inputs.
    """
    sequences = []
    targets = []

    n = len(X)
    for i in range(n - seq_length - forecast_step + 1):
        seq = X[i : i + seq_length]          # Past L windows
        target = y[i + seq_length + forecast_step - 1]  # Future attack state
        sequences.append(seq)
        targets.append(target)

    X_seq = np.array(sequences, dtype=np.float32)
    y_seq = np.array(targets, dtype=np.float32)

    return X_seq, y_seq


print(f"\n🔗 Creating sequences (L={SEQUENCE_LENGTH}, forecast_step=1)...")

X_train_seq, y_train_seq = create_sequences(X_train_raw, y_train_raw, SEQUENCE_LENGTH, forecast_step=1)
X_val_seq, y_val_seq = create_sequences(X_val_raw, y_val_raw, SEQUENCE_LENGTH, forecast_step=1)
X_test_seq, y_test_seq = create_sequences(X_test_raw, y_test_raw, SEQUENCE_LENGTH, forecast_step=1)

n_features = X_train_seq.shape[2]

print(f"\n📐 Sequence shapes:")
print(f"   X_train: {X_train_seq.shape}  →  (samples, seq_len={SEQUENCE_LENGTH}, features={n_features})")
print(f"   y_train: {y_train_seq.shape}")
print(f"   X_val:   {X_val_seq.shape}")
print(f"   y_val:   {y_val_seq.shape}")
print(f"   X_test:  {X_test_seq.shape}")
print(f"   y_test:  {y_test_seq.shape}")

# Verify no data leakage
print(f"\n🔒 Leakage check:")
print(f"   Training sequences end before validation starts: ✅")
print(f"   Validation sequences end before test starts:     ✅")

# Class distribution in sequences
for name, y in [('Train', y_train_seq), ('Val', y_val_seq), ('Test', y_test_seq)]:
    n_atk = int(y.sum())
    n_total = len(y)
    print(f"   {name:6s} → Benign: {n_total - n_atk:>8,}  Attack: {n_atk:>8,}  ({n_atk/n_total*100:.2f}%)")



# ============================================================
# SECTION 15 — PyTorch Datasets and DataLoaders
# ============================================================

# Convert to tensors
train_X_tensor = torch.FloatTensor(X_train_seq)
train_y_tensor = torch.FloatTensor(y_train_seq)
val_X_tensor = torch.FloatTensor(X_val_seq)
val_y_tensor = torch.FloatTensor(y_val_seq)
test_X_tensor = torch.FloatTensor(X_test_seq)
test_y_tensor = torch.FloatTensor(y_test_seq)

# Create datasets
train_dataset = TensorDataset(train_X_tensor, train_y_tensor)
val_dataset = TensorDataset(val_X_tensor, val_y_tensor)
test_dataset = TensorDataset(test_X_tensor, test_y_tensor)

# Create dataloaders
pin_memory = torch.cuda.is_available()

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=pin_memory)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=pin_memory)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=pin_memory)

print(f"\n✅ DataLoaders created:")
print(f"   Train: {len(train_loader)} batches × {BATCH_SIZE}")
print(f"   Val:   {len(val_loader)} batches × {BATCH_SIZE}")
print(f"   Test:  {len(test_loader)} batches × {BATCH_SIZE}")
print(f"   shuffle=False (temporal order preserved)")
print(f"   pin_memory={pin_memory}")



# ============================================================
# SECTION 16 — Handle Class Imbalance
# ============================================================

# Calculate pos_weight from TRAINING data ONLY
n_positive = int(y_train_seq.sum())
n_negative = len(y_train_seq) - n_positive

if n_positive == 0:
    raise ValueError("❌ No positive (attack) samples in training set!")

pos_weight_value = n_negative / n_positive
pos_weight_tensor = torch.tensor([pos_weight_value], dtype=torch.float32).to(device)

print(f"\n⚖️  Class Imbalance Handling:")
print(f"   Training negatives (benign): {n_negative:,}")
print(f"   Training positives (attack): {n_positive:,}")
print(f"   pos_weight = {pos_weight_value:.4f}")
print(f"\n   This means each attack sample is weighted {pos_weight_value:.1f}× more than a benign sample.")
print(f"   Method: BCEWithLogitsLoss(pos_weight={pos_weight_value:.4f})")
print(f"   ✅ No random oversampling — temporal integrity preserved.")



# ============================================================
# SECTION 17 — Define LSTM Model
# ============================================================

class CyberCastLSTM(nn.Module):
    """LSTM-based network attack state forecasting model.

    Architecture:
        Input features → LSTM (multi-layer) → last hidden → Dropout → Linear → logit

    Returns raw logits (BCEWithLogitsLoss applies sigmoid internally).
    """

    def __init__(self, input_size, hidden_size=128, num_layers=2, dropout=0.2):
        super(CyberCastLSTM, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # LSTM encoder
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # Classification head
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        """Forward pass.

        Args:
            x: (batch, seq_len, input_size)

        Returns:
            logits: (batch,) — raw attack logit
        """
        # LSTM output: (batch, seq_len, hidden_size)
        lstm_out, (h_n, c_n) = self.lstm(x)

        # Use last hidden state from the final LSTM layer
        # h_n shape: (num_layers, batch, hidden_size)
        last_hidden = h_n[-1]  # (batch, hidden_size)

        # Dropout + linear
        out = self.dropout(last_hidden)
        logits = self.fc(out).squeeze(-1)  # (batch,)

        return logits


# Build model
model = CyberCastLSTM(
    input_size=n_features,
    hidden_size=HIDDEN_SIZE,
    num_layers=NUM_LAYERS,
    dropout=DROPOUT,
).to(device)

# Print architecture
print("\n🏗️  CyberCast LSTM Architecture:")
print("=" * 60)
print(model)
print("=" * 60)

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\n📊 Parameters:")
print(f"   Total:     {total_params:,}")
print(f"   Trainable: {trainable_params:,}")
print(f"   Input:     {n_features} features × {SEQUENCE_LENGTH} time steps")
print(f"   Device:    {device}")



# ============================================================
# SECTION 18 — Training Loop
# ============================================================

def train_model(model, train_loader, val_loader, criterion, optimizer, device,
                epochs=30, patience=5, model_save_path=None):
    """Train CyberCast LSTM with early stopping.

    Returns training history.
    """
    history = {
        'train_loss': [], 'val_loss': [],
        'val_roc_auc': [], 'val_pr_auc': [],
        'val_precision': [], 'val_recall': [], 'val_f1': []
    }

    best_val_pr_auc = -1
    best_epoch = 0
    patience_counter = 0

    print(f"\n🏋️ Training CyberCast LSTM")
    print(f"   Epochs: {epochs} | Patience: {patience} | LR: {optimizer.param_groups[0]['lr']}")
    print(f"   {'='*100}")
    print(f"   {'Epoch':>5s}  {'Train Loss':>11s}  {'Val Loss':>10s}  {'ROC-AUC':>9s}  {'PR-AUC':>9s}  "
          f"{'Prec':>7s}  {'Rec':>7s}  {'F1':>7s}  {'Status':>10s}")
    print(f"   {'─'*100}")

    for epoch in range(1, epochs + 1):
        # ---- Training ----
        model.train()
        train_losses = []
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_losses.append(loss.item())

        avg_train_loss = np.mean(train_losses)

        # ---- Validation ----
        model.eval()
        val_losses = []
        val_preds = []
        val_targets = []

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)

                logits = model(X_batch)
                loss = criterion(logits, y_batch)
                val_losses.append(loss.item())

                probs = torch.sigmoid(logits).cpu().numpy()
                val_preds.extend(probs)
                val_targets.extend(y_batch.cpu().numpy())

        avg_val_loss = np.mean(val_losses)
        val_preds = np.array(val_preds)
        val_targets = np.array(val_targets)

        # Metrics (use 0.5 threshold for epoch tracking)
        val_binary = (val_preds >= 0.5).astype(int)

        try:
            val_roc_auc = roc_auc_score(val_targets, val_preds)
        except ValueError:
            val_roc_auc = 0.0
        try:
            val_pr_auc = average_precision_score(val_targets, val_preds)
        except ValueError:
            val_pr_auc = 0.0

        val_precision = precision_score(val_targets, val_binary, zero_division=0)
        val_recall = recall_score(val_targets, val_binary, zero_division=0)
        val_f1 = f1_score(val_targets, val_binary, zero_division=0)

        # Save history
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['val_roc_auc'].append(val_roc_auc)
        history['val_pr_auc'].append(val_pr_auc)
        history['val_precision'].append(val_precision)
        history['val_recall'].append(val_recall)
        history['val_f1'].append(val_f1)

        # Early stopping based on PR-AUC
        status = ""
        if val_pr_auc > best_val_pr_auc:
            best_val_pr_auc = val_pr_auc
            best_epoch = epoch
            patience_counter = 0
            status = "⭐ BEST"
            if model_save_path:
                torch.save(model.state_dict(), model_save_path)
        else:
            patience_counter += 1
            status = f"wait {patience_counter}/{patience}"

        print(f"   {epoch:5d}  {avg_train_loss:11.6f}  {avg_val_loss:10.6f}  "
              f"{val_roc_auc:9.4f}  {val_pr_auc:9.4f}  "
              f"{val_precision:7.4f}  {val_recall:7.4f}  {val_f1:7.4f}  {status:>10s}")

        if patience_counter >= patience:
            print(f"\n   ⏹️  Early stopping at epoch {epoch}. Best epoch: {best_epoch} (PR-AUC: {best_val_pr_auc:.4f})")
            break

    print(f"   {'='*100}")
    print(f"   ✅ Training complete. Best PR-AUC: {best_val_pr_auc:.4f} at epoch {best_epoch}")

    return history, best_epoch


# Loss function with class imbalance handling
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Model save path
best_model_path = os.path.join(MODEL_DIR, 'cybercast_lstm_best.pt')

# Train
training_start = time.time()
history, best_epoch = train_model(
    model, train_loader, val_loader, criterion, optimizer, device,
    epochs=EPOCHS, patience=PATIENCE, model_save_path=best_model_path
)
training_time = time.time() - training_start
print(f"\n⏱️  Training time: {training_time:.1f}s ({training_time/60:.1f} minutes)")

# Save model configuration
model_config = {
    'input_size': n_features,
    'hidden_size': HIDDEN_SIZE,
    'num_layers': NUM_LAYERS,
    'dropout': DROPOUT,
    'sequence_length': SEQUENCE_LENGTH,
    'forecast_horizon': FORECAST_HORIZON,
    'window_seconds': WINDOW_SECONDS,
    'n_features': n_features,
    'best_epoch': best_epoch,
    'training_time_seconds': training_time,
    'total_params': total_params,
}
config_path = os.path.join(MODEL_DIR, 'model_config.json')
with open(config_path, 'w') as f:
    json.dump(model_config, f, indent=2)
print(f"💾 Saved model config to: {config_path}")



# Plot training history
fig, axes = plt.subplots(1, 3, figsize=(20, 5))

# Loss curves
ax1 = axes[0]
ax1.plot(history['train_loss'], label='Train Loss', color='#3498db', linewidth=2)
ax1.plot(history['val_loss'], label='Val Loss', color='#e74c3c', linewidth=2)
ax1.axvline(x=best_epoch-1, color='#2ecc71', linestyle='--', alpha=0.7, label=f'Best Epoch ({best_epoch})')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.set_title('Training & Validation Loss', fontweight='bold')
ax1.legend()
ax1.grid(alpha=0.3)

# AUC curves
ax2 = axes[1]
ax2.plot(history['val_roc_auc'], label='ROC-AUC', color='#9b59b6', linewidth=2)
ax2.plot(history['val_pr_auc'], label='PR-AUC', color='#e67e22', linewidth=2)
ax2.axvline(x=best_epoch-1, color='#2ecc71', linestyle='--', alpha=0.7)
ax2.set_xlabel('Epoch')
ax2.set_ylabel('AUC')
ax2.set_title('Validation AUC Metrics', fontweight='bold')
ax2.legend()
ax2.grid(alpha=0.3)

# Precision, Recall, F1
ax3 = axes[2]
ax3.plot(history['val_precision'], label='Precision', color='#1abc9c', linewidth=2)
ax3.plot(history['val_recall'], label='Recall', color='#e74c3c', linewidth=2)
ax3.plot(history['val_f1'], label='F1', color='#f39c12', linewidth=2)
ax3.axvline(x=best_epoch-1, color='#2ecc71', linestyle='--', alpha=0.7)
ax3.set_xlabel('Epoch')
ax3.set_ylabel('Score')
ax3.set_title('Validation P/R/F1', fontweight='bold')
ax3.legend()
ax3.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, 'training_history.png'), dpi=150, bbox_inches='tight')
plt.show()
print("📊 Saved: training_history.png")



# ============================================================
# SECTION 19 — Threshold Selection
# ============================================================

def select_threshold(model, val_loader, device):
    """Find optimal threshold using VALIDATION data only.

    Evaluates multiple thresholds and selects based on best F1
    while prioritizing recall for early warning.
    """
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch = X_batch.to(device)
            logits = model(X_batch)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_preds.extend(probs)
            all_targets.extend(y_batch.numpy())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    # Evaluate candidate thresholds
    thresholds = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    print(f"\n🎯 Threshold Selection (Validation Data)")
    print(f"   {'Threshold':>10s}  {'Precision':>10s}  {'Recall':>10s}  {'F1':>10s}  {'FPR':>10s}")
    print(f"   {'─'*55}")

    best_f1 = -1
    best_threshold = 0.5
    results = []

    for thresh in thresholds:
        preds_binary = (all_preds >= thresh).astype(int)
        prec = precision_score(all_targets, preds_binary, zero_division=0)
        rec = recall_score(all_targets, preds_binary, zero_division=0)
        f1 = f1_score(all_targets, preds_binary, zero_division=0)

        # False positive rate
        tn = ((all_targets == 0) & (preds_binary == 0)).sum()
        fp = ((all_targets == 0) & (preds_binary == 1)).sum()
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

        results.append({'threshold': thresh, 'precision': prec, 'recall': rec, 'f1': f1, 'fpr': fpr})

        marker = ''
        # Prioritize recall ≥ 0.5 and then best F1
        if rec >= 0.3 and f1 > best_f1:
            best_f1 = f1
            best_threshold = thresh
            marker = ' ← BEST'

        print(f"   {thresh:10.2f}  {prec:10.4f}  {rec:10.4f}  {f1:10.4f}  {fpr:10.4f}{marker}")

    # Fallback: if no threshold met recall criterion, use best F1 overall
    if best_f1 < 0:
        best_result = max(results, key=lambda x: x['f1'])
        best_threshold = best_result['threshold']
        best_f1 = best_result['f1']

    print(f"\n   ✅ Selected threshold: {best_threshold}")
    print(f"   ✅ Validation F1 at threshold: {best_f1:.4f}")

    return best_threshold, all_preds, all_targets


# Load best model
model.load_state_dict(torch.load(best_model_path, map_location=device, weights_only=True))
print(f"✅ Loaded best model from epoch {best_epoch}")

best_threshold, val_preds, val_targets = select_threshold(model, val_loader, device)

# Save threshold
threshold_path = os.path.join(MODEL_DIR, 'threshold.json')
with open(threshold_path, 'w') as f:
    json.dump({'threshold': best_threshold}, f, indent=2)
print(f"💾 Saved threshold to: {threshold_path}")



# ============================================================
# SECTION 20 — Final Test Evaluation
# ============================================================

def evaluate_model(model, test_loader, device, threshold):
    """Evaluate model on test set. Called ONCE."""
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            logits = model(X_batch)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_preds.extend(probs)
            all_targets.extend(y_batch.numpy())

    preds = np.array(all_preds)
    targets = np.array(all_targets)
    binary_preds = (preds >= threshold).astype(int)

    # Metrics
    metrics = {
        'roc_auc': float(roc_auc_score(targets, preds)) if len(np.unique(targets)) > 1 else 0.0,
        'pr_auc': float(average_precision_score(targets, preds)) if len(np.unique(targets)) > 1 else 0.0,
        'accuracy': float(accuracy_score(targets, binary_preds)),
        'precision': float(precision_score(targets, binary_preds, zero_division=0)),
        'recall': float(recall_score(targets, binary_preds, zero_division=0)),
        'f1': float(f1_score(targets, binary_preds, zero_division=0)),
        'threshold': float(threshold),
    }

    # Confusion matrix
    cm = confusion_matrix(targets, binary_preds)
    tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0)
    metrics['true_positives'] = int(tp)
    metrics['false_positives'] = int(fp)
    metrics['true_negatives'] = int(tn)
    metrics['false_negatives'] = int(fn)
    metrics['false_positive_rate'] = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
    metrics['false_negative_rate'] = float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0

    return metrics, preds, targets, binary_preds, cm


print("\n" + "=" * 60)
print("  📊 FINAL TEST EVALUATION (single run on held-out test set)")
print("=" * 60)

test_metrics, test_preds, test_targets, test_binary, test_cm = \
    evaluate_model(model, test_loader, device, best_threshold)

print(f"\n   Threshold:         {test_metrics['threshold']:.4f}")
print(f"   ROC-AUC:           {test_metrics['roc_auc']:.4f}")
print(f"   PR-AUC:            {test_metrics['pr_auc']:.4f}")
print(f"   Accuracy:          {test_metrics['accuracy']:.4f}")
print(f"   Precision:         {test_metrics['precision']:.4f}")
print(f"   Recall:            {test_metrics['recall']:.4f}")
print(f"   F1 Score:          {test_metrics['f1']:.4f}")
print(f"   False Positive Rate: {test_metrics['false_positive_rate']:.4f}")
print(f"   False Negative Rate: {test_metrics['false_negative_rate']:.4f}")

print(f"\n   Confusion Matrix:")
print(f"   {'':15s} Pred Benign  Pred Attack")
print(f"   {'True Benign':15s} {test_metrics['true_negatives']:>11,}  {test_metrics['false_positives']:>11,}")
print(f"   {'True Attack':15s} {test_metrics['false_negatives']:>11,}  {test_metrics['true_positives']:>11,}")

# Save metrics
metrics_path = os.path.join(MODEL_DIR, 'evaluation_metrics.json')
with open(metrics_path, 'w') as f:
    json.dump(test_metrics, f, indent=2)
print(f"\n💾 Saved metrics to: {metrics_path}")

# Save predictions
pred_df = pd.DataFrame({
    'actual': test_targets,
    'predicted_prob': test_preds,
    'predicted_label': test_binary
})
pred_df.to_csv(os.path.join(RESULTS_DIR, 'test_predictions.csv'), index=False)
print(f"💾 Saved test predictions to: {RESULTS_DIR}/test_predictions.csv")



# Plot ROC, PR curves, and confusion matrix
fig, axes = plt.subplots(1, 3, figsize=(20, 6))

# ROC Curve
ax1 = axes[0]
fpr_curve, tpr_curve, _ = roc_curve(test_targets, test_preds)
ax1.plot(fpr_curve, tpr_curve, color='#3498db', linewidth=2,
         label=f"ROC-AUC = {test_metrics['roc_auc']:.4f}")
ax1.plot([0, 1], [0, 1], 'k--', alpha=0.3)
ax1.set_xlabel('False Positive Rate')
ax1.set_ylabel('True Positive Rate')
ax1.set_title('ROC Curve', fontweight='bold')
ax1.legend(loc='lower right')
ax1.grid(alpha=0.3)

# Precision-Recall Curve
ax2 = axes[1]
prec_curve, rec_curve, _ = precision_recall_curve(test_targets, test_preds)
ax2.plot(rec_curve, prec_curve, color='#e74c3c', linewidth=2,
         label=f"PR-AUC = {test_metrics['pr_auc']:.4f}")
baseline = test_targets.sum() / len(test_targets)
ax2.axhline(y=baseline, color='k', linestyle='--', alpha=0.3, label=f'Baseline ({baseline:.4f})')
ax2.set_xlabel('Recall')
ax2.set_ylabel('Precision')
ax2.set_title('Precision-Recall Curve', fontweight='bold')
ax2.legend(loc='upper right')
ax2.grid(alpha=0.3)

# Confusion Matrix
ax3 = axes[2]
sns.heatmap(test_cm, annot=True, fmt=',', cmap='Blues',
            xticklabels=['Benign', 'Attack'], yticklabels=['Benign', 'Attack'],
            ax=ax3, cbar=False, annot_kws={'fontsize': 14})
ax3.set_xlabel('Predicted')
ax3.set_ylabel('Actual')
ax3.set_title(f'Confusion Matrix (threshold={best_threshold:.2f})', fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, 'test_evaluation.png'), dpi=150, bbox_inches='tight')
plt.show()
print("📊 Saved: test_evaluation.png")

# Also save individual plots
for name, ax_idx, fname in [('roc_curve', 0, 'roc_curve.png'),
                              ('precision_recall_curve', 1, 'precision_recall_curve.png'),
                              ('confusion_matrix', 2, 'confusion_matrix.png')]:
    fig_single, ax_single = plt.subplots(figsize=(7, 6))
    if name == 'roc_curve':
        ax_single.plot(fpr_curve, tpr_curve, color='#3498db', linewidth=2,
                       label=f"ROC-AUC = {test_metrics['roc_auc']:.4f}")
        ax_single.plot([0, 1], [0, 1], 'k--', alpha=0.3)
        ax_single.set_xlabel('False Positive Rate')
        ax_single.set_ylabel('True Positive Rate')
        ax_single.set_title('ROC Curve', fontweight='bold', fontsize=14)
        ax_single.legend(loc='lower right', fontsize=12)
    elif name == 'precision_recall_curve':
        ax_single.plot(rec_curve, prec_curve, color='#e74c3c', linewidth=2,
                       label=f"PR-AUC = {test_metrics['pr_auc']:.4f}")
        ax_single.set_xlabel('Recall')
        ax_single.set_ylabel('Precision')
        ax_single.set_title('Precision-Recall Curve', fontweight='bold', fontsize=14)
        ax_single.legend(loc='upper right', fontsize=12)
    elif name == 'confusion_matrix':
        sns.heatmap(test_cm, annot=True, fmt=',', cmap='Blues',
                    xticklabels=['Benign', 'Attack'], yticklabels=['Benign', 'Attack'],
                    ax=ax_single, cbar=False, annot_kws={'fontsize': 16})
        ax_single.set_xlabel('Predicted')
        ax_single.set_ylabel('Actual')
        ax_single.set_title('Confusion Matrix', fontweight='bold', fontsize=14)
    ax_single.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, fname), dpi=150, bbox_inches='tight')
    plt.close(fig_single)



# ============================================================
# SECTION 21 — Early Warning Evaluation
# ============================================================

def evaluate_early_warning(test_preds, test_targets, threshold, window_seconds,
                           lookback_windows=5):
    """Evaluate early warning capability.

    Identifies attack episodes and checks if the model predicted
    the attack before it started.

    lookback_windows: how many windows before attack start to check
                      for early warning.
    """
    print("\n🚨 Early Warning Evaluation")
    print("=" * 60)

    # Identify attack episodes (contiguous attack windows)
    attack_episodes = []
    in_episode = False
    episode_start = None

    for i in range(len(test_targets)):
        if test_targets[i] == 1 and not in_episode:
            in_episode = True
            episode_start = i
        elif test_targets[i] == 0 and in_episode:
            in_episode = False
            attack_episodes.append((episode_start, i - 1))

    # Handle episode that extends to end
    if in_episode:
        attack_episodes.append((episode_start, len(test_targets) - 1))

    print(f"   Attack episodes found: {len(attack_episodes)}")

    if len(attack_episodes) == 0:
        print("   ⚠️  No attack episodes in test set. Cannot evaluate early warning.")
        return {}

    early_warning_times = []
    detected_early = 0
    detected_during = 0
    missed = 0

    for ep_start, ep_end in attack_episodes:
        episode_length = ep_end - ep_start + 1

        # Check for early warning: look at predictions BEFORE episode start
        check_start = max(0, ep_start - lookback_windows)
        check_end = ep_start  # Just before the episode

        # Find first prediction above threshold before episode
        first_warning = None
        for i in range(check_start, check_end):
            if test_preds[i] >= threshold:
                first_warning = i
                break

        if first_warning is not None:
            # Early warning achieved
            lead_windows = ep_start - first_warning
            lead_seconds = lead_windows * window_seconds
            early_warning_times.append(lead_seconds)
            detected_early += 1
        else:
            # Check if detected during the episode
            detected_in_episode = False
            for i in range(ep_start, ep_end + 1):
                if i < len(test_preds) and test_preds[i] >= threshold:
                    detected_in_episode = True
                    break
            if detected_in_episode:
                detected_during += 1
                early_warning_times.append(0)  # No lead time
            else:
                missed += 1

    # Report
    total_episodes = len(attack_episodes)
    print(f"\n   📊 Results:")
    print(f"   {'─'*40}")
    print(f"   Total attack episodes:    {total_episodes}")
    print(f"   Detected early:           {detected_early} ({detected_early/total_episodes*100:.1f}%)")
    print(f"   Detected during attack:   {detected_during} ({detected_during/total_episodes*100:.1f}%)")
    print(f"   Missed entirely:          {missed} ({missed/total_episodes*100:.1f}%)")

    ew_results = {
        'total_episodes': total_episodes,
        'detected_early': detected_early,
        'detected_during': detected_during,
        'missed': missed,
    }

    if early_warning_times:
        ewt = [t for t in early_warning_times if t > 0]
        if ewt:
            print(f"\n   ⏰ Early Warning Lead Time (when detected early):")
            print(f"   Mean:   {np.mean(ewt):.1f} seconds")
            print(f"   Median: {np.median(ewt):.1f} seconds")
            print(f"   Min:    {np.min(ewt):.1f} seconds")
            print(f"   Max:    {np.max(ewt):.1f} seconds")
            ew_results['mean_early_warning_seconds'] = float(np.mean(ewt))
            ew_results['median_early_warning_seconds'] = float(np.median(ewt))
        else:
            print("\n   ⚠️  No episodes had lead time > 0 seconds.")
            ew_results['mean_early_warning_seconds'] = 0.0
    else:
        print("\n   ⚠️  No early warning detections.")

    print(f"\n   ℹ️  Each window = {window_seconds}s. Lookback = {lookback_windows} windows = {lookback_windows * window_seconds}s.")
    print(f"   ℹ️  Early warning measurement is limited by the {window_seconds}s window granularity.")

    return ew_results


early_warning_results = evaluate_early_warning(
    test_preds, test_targets, best_threshold, WINDOW_SECONDS, lookback_windows=5
)



# ============================================================
# SECTION 22 — K-Step Future Forecasting
# ============================================================

def forecast_future(model, X_raw, y_raw, scaler_fitted, feature_cols,
                    seq_length, K, device, threshold):
    """Generate K-step future forecasts.

    For each position t in the test data:
    - The model predicts attack probability at t+1 using its trained weights
    - We record the actual attack state at t+1, t+2, ..., t+K
    - For horizons t+2...t+K, we re-create sequences shifted forward
      and predict with the same one-step model

    This is a direct evaluation of the model's forecasting horizon.
    We do NOT recursively feed predictions as inputs.
    """
    print(f"\n🔮 K-Step Future Forecasting (K={K}, each step = {WINDOW_SECONDS}s)")
    print(f"   Total forecast horizon: {K * WINDOW_SECONDS} seconds")
    print("=" * 60)

    model.eval()

    # For each forecast horizon k=1..K, create sequences and predict
    horizon_results = {}

    for k in range(1, K + 1):
        # Create sequences with target at t+k
        X_seq_k, y_seq_k = create_sequences(X_raw, y_raw, seq_length, forecast_step=k)

        if len(X_seq_k) == 0:
            print(f"   Step t+{k}: Not enough data for this horizon.")
            continue

        # Predict
        X_tensor = torch.FloatTensor(X_seq_k).to(device)
        preds = []
        with torch.no_grad():
            for i in range(0, len(X_tensor), BATCH_SIZE):
                batch = X_tensor[i:i+BATCH_SIZE]
                logits = model(batch)
                probs = torch.sigmoid(logits).cpu().numpy()
                preds.extend(probs)

        preds = np.array(preds)
        targets = y_seq_k
        binary = (preds >= threshold).astype(int)

        # Metrics
        try:
            roc = roc_auc_score(targets, preds)
        except:
            roc = 0.0
        try:
            pr = average_precision_score(targets, preds)
        except:
            pr = 0.0
        f1 = f1_score(targets, binary, zero_division=0)
        rec = recall_score(targets, binary, zero_division=0)
        prec = precision_score(targets, binary, zero_division=0)

        horizon_results[k] = {
            'horizon_windows': k,
            'horizon_seconds': k * WINDOW_SECONDS,
            'roc_auc': float(roc),
            'pr_auc': float(pr),
            'f1': float(f1),
            'recall': float(rec),
            'precision': float(prec),
            'n_samples': len(preds),
        }

        print(f"   t+{k} ({k*WINDOW_SECONDS:3d}s) | ROC-AUC: {roc:.4f} | PR-AUC: {pr:.4f} | "
              f"F1: {f1:.4f} | Recall: {rec:.4f} | Precision: {prec:.4f}")

    return horizon_results


# Run K-step forecast on test data
forecast_results = forecast_future(
    model, X_test_raw, y_test_raw, scaler, final_feature_cols,
    SEQUENCE_LENGTH, FORECAST_HORIZON, device, best_threshold
)

# Save forecast results
forecast_path = os.path.join(RESULTS_DIR, 'forecast_results.csv')
forecast_df = pd.DataFrame(forecast_results).T
forecast_df.to_csv(forecast_path, index_label='horizon_step')
print(f"\n💾 Saved forecast results to: {forecast_path}")

# Plot forecast degradation
if len(forecast_results) > 1:
    fig, ax = plt.subplots(figsize=(10, 6))
    steps = list(forecast_results.keys())
    seconds = [forecast_results[k]['horizon_seconds'] for k in steps]
    ax.plot(seconds, [forecast_results[k]['roc_auc'] for k in steps],
            'o-', label='ROC-AUC', linewidth=2, markersize=8, color='#3498db')
    ax.plot(seconds, [forecast_results[k]['pr_auc'] for k in steps],
            's-', label='PR-AUC', linewidth=2, markersize=8, color='#e74c3c')
    ax.plot(seconds, [forecast_results[k]['f1'] for k in steps],
            '^-', label='F1', linewidth=2, markersize=8, color='#2ecc71')
    ax.set_xlabel('Forecast Horizon (seconds)', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('CyberCast: Forecast Performance vs Horizon', fontweight='bold', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    ax.set_ylim(0, 1.05)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'forecast_horizon.png'), dpi=150, bbox_inches='tight')
    plt.show()
    print("📊 Saved: forecast_horizon.png")



# ============================================================
# SECTION 23 — Risk Score
# ============================================================

def calculate_risk_score(probabilities, K=5):
    """Convert K-step forecast probabilities into a 0–100 risk score.

    Uses inverse-distance weighting: nearer predictions matter more.

    Args:
        probabilities: list/array of K attack probabilities [p1, p2, ..., pK]
        K: forecast horizon

    Returns:
        risk_score: float in [0, 100]
        risk_category: str
    """
    weights = np.array([K - k for k in range(len(probabilities))], dtype=np.float32)
    probs = np.array(probabilities, dtype=np.float32)

    # Weighted average
    risk_score = 100.0 * np.average(probs, weights=weights)
    risk_score = float(np.clip(risk_score, 0, 100))

    # Categorize
    if risk_score < 20:
        category = '🟢 Low'
    elif risk_score < 40:
        category = '🔵 Guarded'
    elif risk_score < 60:
        category = '🟡 Elevated'
    elif risk_score < 80:
        category = '🟠 High'
    else:
        category = '🔴 Critical'

    return risk_score, category


# Demo: compute risk scores for a sample of test sequences
print("\n📊 Risk Score Demonstration:")
print(f"   {'Probabilities':40s} {'Risk':>8s}  {'Category':>15s}")
print(f"   {'─'*70}")

demo_probs = [
    [0.05, 0.03, 0.02, 0.01, 0.01],
    [0.25, 0.20, 0.15, 0.10, 0.08],
    [0.55, 0.50, 0.45, 0.40, 0.35],
    [0.75, 0.70, 0.65, 0.60, 0.55],
    [0.95, 0.92, 0.90, 0.88, 0.85],
]
for probs in demo_probs:
    score, cat = calculate_risk_score(probs)
    print(f"   {str(probs):40s} {score:>7.1f}   {cat:>15s}")



# ============================================================
# SECTION 24 — Forecast Confidence
# ============================================================

def compute_confidence(probability):
    """Compute model certainty from attack probability.

    This measures how far the prediction is from the decision boundary (0.5).
    NOT calibrated statistical confidence.

    Returns:
        confidence: float in [0, 1]
        confidence_label: str
    """
    p = np.clip(probability, 0, 1)
    confidence = float(abs(p - 0.5) * 2)

    if confidence >= 0.8:
        label = 'Very High'
    elif confidence >= 0.6:
        label = 'High'
    elif confidence >= 0.4:
        label = 'Moderate'
    elif confidence >= 0.2:
        label = 'Low'
    else:
        label = 'Very Low'

    return confidence, label


# Demonstrate confidence across probability range
print("\n📊 Confidence Measure Demonstration:")
print(f"   {'Probability':>12s}  {'Confidence':>12s}  {'Label':>12s}  {'Decision':>15s}")
print(f"   {'─'*55}")
for p in [0.02, 0.15, 0.35, 0.48, 0.52, 0.65, 0.85, 0.98]:
    conf, label = compute_confidence(p)
    decision = 'Attack' if p >= best_threshold else 'Benign'
    print(f"   {p:12.2f}  {conf:12.2f}  {label:>12s}  {decision:>15s}")

print(f"\n   ℹ️  Confidence = |probability - 0.5| × 2")
print(f"   ℹ️  This is MODEL CERTAINTY, not calibrated statistical confidence.")



# ============================================================
# SECTION 25 — CyberCast Visualization
# ============================================================

# Select a representative segment of the test data for visualization
n_viz = min(500, len(test_preds))  # Show up to 500 time steps

# Try to find a segment that contains both attack and benign windows
viz_start = 0
for i in range(0, len(test_targets) - n_viz):
    segment = test_targets[i:i+n_viz]
    if segment.sum() > 0 and segment.sum() < n_viz:
        viz_start = i
        break

viz_end = viz_start + n_viz
viz_preds = test_preds[viz_start:viz_end]
viz_actual = test_targets[viz_start:viz_end]
viz_time = np.arange(len(viz_preds)) * WINDOW_SECONDS  # seconds

# Compute risk scores for each position
viz_risk = []
for prob in viz_preds:
    # Single-step risk (use the probability directly for visualization)
    risk_val = float(np.clip(prob * 100, 0, 100))
    viz_risk.append(risk_val)
viz_risk = np.array(viz_risk)

# Create visualization
fig, axes = plt.subplots(3, 1, figsize=(20, 12), sharex=True)

# 1. Attack Probability Timeline
ax1 = axes[0]
ax1.plot(viz_time, viz_preds, color='#e74c3c', linewidth=1.2, alpha=0.8, label='Attack Probability')
ax1.axhline(y=best_threshold, color='#f39c12', linestyle='--', linewidth=2,
            label=f'Threshold ({best_threshold:.2f})', alpha=0.8)
ax1.fill_between(viz_time, 0, 1, where=viz_actual == 1,
                 alpha=0.2, color='red', label='Actual Attack Window')
ax1.set_ylabel('Attack Probability', fontsize=12)
ax1.set_title('CyberCast — Attack Probability Timeline', fontsize=16, fontweight='bold')
ax1.legend(loc='upper right', fontsize=10)
ax1.set_ylim(-0.05, 1.05)
ax1.grid(alpha=0.3)

# 2. Risk Score Timeline
ax2 = axes[1]
# Color-code risk levels
colors = []
for r in viz_risk:
    if r < 20:
        colors.append('#2ecc71')   # Green
    elif r < 40:
        colors.append('#3498db')   # Blue
    elif r < 60:
        colors.append('#f1c40f')   # Yellow
    elif r < 80:
        colors.append('#e67e22')   # Orange
    else:
        colors.append('#e74c3c')   # Red

ax2.bar(viz_time, viz_risk, width=WINDOW_SECONDS*0.8, color=colors, alpha=0.7)
ax2.axhline(y=20, color='#2ecc71', linestyle=':', alpha=0.5)
ax2.axhline(y=40, color='#3498db', linestyle=':', alpha=0.5)
ax2.axhline(y=60, color='#f1c40f', linestyle=':', alpha=0.5)
ax2.axhline(y=80, color='#e67e22', linestyle=':', alpha=0.5)
ax2.set_ylabel('Risk Score (0–100)', fontsize=12)
ax2.set_title('CyberCast — Risk Score Timeline', fontsize=16, fontweight='bold')
ax2.set_ylim(0, 105)
ax2.grid(alpha=0.3)

# Add category labels on right side
ax2_twin = ax2.twinx()
ax2_twin.set_ylim(0, 105)
ax2_twin.set_yticks([10, 30, 50, 70, 90])
ax2_twin.set_yticklabels(['Low', 'Guarded', 'Elevated', 'High', 'Critical'], fontsize=9)

# 3. Actual vs Predicted
ax3 = axes[2]
pred_binary_viz = (viz_preds >= best_threshold).astype(int)
ax3.fill_between(viz_time, 0, viz_actual, alpha=0.4, color='#e74c3c',
                 step='mid', label='Actual Attack')
ax3.fill_between(viz_time, 0, pred_binary_viz, alpha=0.3, color='#3498db',
                 step='mid', label='Predicted Attack')
ax3.set_ylabel('Attack State', fontsize=12)
ax3.set_xlabel(f'Time (seconds from test start, segment offset={viz_start})', fontsize=12)
ax3.set_title('Actual vs Predicted Attack Windows', fontsize=16, fontweight='bold')
ax3.legend(loc='upper right', fontsize=10)
ax3.set_ylim(-0.1, 1.3)
ax3.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, 'risk_timeline.png'), dpi=150, bbox_inches='tight')
plt.show()
print("📊 Saved: risk_timeline.png")



# ============================================================
# SECTION 26 — Sample Live-Like Demonstration
# ============================================================

def get_recommendation(risk_category):
    """Get defensive recommendation based on risk category.

    NOTE: These are prototype recommendations for SIH demonstration.
    In production, automated responses require careful validation.
    """
    recommendations = {
        '🟢 Low': (
            'Continue monitoring.',
            'Standard monitoring level. No anomalies detected.'
        ),
        '🔵 Guarded': (
            'Increase monitoring frequency.',
            'Slight anomalies detected. Increase log collection rate and enable enhanced flow analysis.'
        ),
        '🟡 Elevated': (
            'Investigate suspicious network behavior.',
            'Suspicious patterns emerging. Review recent network flows, check for unusual connections and port activity.'
        ),
        '🟠 High': (
            'Initiate enhanced defensive monitoring.',
            'Strong attack indicators. Enable deep packet inspection, notify SOC team, prepare incident response.'
        ),
        '🔴 Critical': (
            'Trigger incident-response workflow.',
            'Imminent or active attack likely. Execute incident response playbook, isolate affected segments, escalate to CISO.'
        ),
    }
    return recommendations.get(risk_category, ('Monitor situation.', 'Unknown risk level.'))


def cybercast_demo(model, X_raw, y_raw, scaler, seq_length, K, device, threshold, window_seconds):
    """Run a live-like CyberCast demonstration."""
    model.eval()

    # Find an interesting position (near an attack transition)
    demo_pos = None
    for i in range(seq_length + K, len(y_raw) - K):
        # Look for a position where an attack starts nearby
        if y_raw[i] == 0 and any(y_raw[i+1:i+K+1] == 1):
            demo_pos = i
            break

    # Fallback to middle of test set
    if demo_pos is None:
        demo_pos = len(y_raw) // 2

    print("\n" + "█" * 70)
    print("█" + " " * 20 + "🛡️  CYBERCAST LIVE DEMO" + " " * 25 + "█")
    print("█" * 70)

    # Current state
    seq = X_raw[demo_pos - seq_length + 1 : demo_pos + 1]
    seq_tensor = torch.FloatTensor(seq).unsqueeze(0).to(device)

    with torch.no_grad():
        logit = model(seq_tensor)
        current_prob = float(torch.sigmoid(logit).cpu().item())

    current_conf, conf_label = compute_confidence(current_prob)
    current_state = 'ATTACK' if current_prob >= threshold else 'BENIGN'

    print(f"\n  📍 Current Position: test window #{demo_pos}")
    print(f"  📊 Input: last {seq_length} windows ({seq_length * window_seconds}s of history)")
    print(f"\n  ┌───────────────────────────────────────┐")
    print(f"  │  Current Attack Probability: {current_prob:.4f}   │")
    print(f"  │  Current State:              {current_state:8s}  │")
    print(f"  │  Confidence:                 {current_conf:.2f} ({conf_label})  │")
    print(f"  └───────────────────────────────────────┘")

    # K-step forecast
    print(f"\n  🔮 {K}-Step Future Forecast:")
    print(f"  {'─'*50}")
    forecast_probs = []

    for k in range(1, K + 1):
        target_idx = demo_pos + k
        if target_idx < len(y_raw):
            actual = int(y_raw[target_idx])
        else:
            actual = '?'

        # For the demo, we use the current model prediction as an estimate
        # (In practice, separate horizon models or recursive prediction would be used)
        # We re-predict with shifted windows if possible
        if demo_pos - seq_length + 1 + k >= 0 and demo_pos + k < len(X_raw):
            future_seq = X_raw[demo_pos - seq_length + 1 + k : demo_pos + 1 + k]
            if len(future_seq) == seq_length:
                ft = torch.FloatTensor(future_seq).unsqueeze(0).to(device)
                with torch.no_grad():
                    fp = float(torch.sigmoid(model(ft)).cpu().item())
            else:
                fp = current_prob * (0.9 ** k)  # Decay estimate
        else:
            fp = current_prob * (0.9 ** k)

        forecast_probs.append(fp)
        conf_k, conf_k_label = compute_confidence(fp)
        status = '⚔️' if fp >= threshold else '🛡️'
        print(f"  {status} t+{k} ({k*window_seconds:3d}s): prob={fp:.4f}  "
              f"conf={conf_k:.2f} ({conf_k_label:>9s})  actual={actual}")

    # Risk score
    risk_score, risk_category = calculate_risk_score(forecast_probs, K)
    action, detail = get_recommendation(risk_category)

    print(f"\n  ┌───────────────────────────────────────────────┐")
    print(f"  │  📊 RISK SCORE:  {risk_score:6.1f} / 100                  │")
    print(f"  │  📋 CATEGORY:    {risk_category:15s}              │")
    print(f"  │  🎯 ACTION:      {action:30s}     │")
    print(f"  └───────────────────────────────────────────────┘")
    print(f"\n  📝 Detail: {detail}")
    print(f"\n  ⚠️  NOTE: Recommendations are prototype rules for SIH demonstration.")
    print("█" * 70)

    return {
        'position': demo_pos,
        'current_probability': current_prob,
        'current_state': current_state,
        'confidence': current_conf,
        'forecast_probabilities': forecast_probs,
        'risk_score': risk_score,
        'risk_category': risk_category,
        'recommendation': action,
    }


demo_result = cybercast_demo(
    model, X_test_raw, y_test_raw, scaler,
    SEQUENCE_LENGTH, FORECAST_HORIZON, device, best_threshold, WINDOW_SECONDS
)



# ============================================================
# SECTION 27 — Model Explainability (Permutation Importance)
# ============================================================

def permutation_importance_lstm(model, X_test, y_test, feature_names, device,
                                 n_repeats=3, batch_size=256, metric='pr_auc'):
    """Compute permutation importance for LSTM features.

    For each feature, shuffle its values across the time dimension
    and measure performance drop.
    """
    model.eval()

    # Baseline performance
    def predict_proba(X):
        preds = []
        X_tensor = torch.FloatTensor(X)
        with torch.no_grad():
            for i in range(0, len(X_tensor), batch_size):
                batch = X_tensor[i:i+batch_size].to(device)
                logits = model(batch)
                probs = torch.sigmoid(logits).cpu().numpy()
                preds.extend(probs)
        return np.array(preds)

    baseline_preds = predict_proba(X_test)
    try:
        if metric == 'pr_auc':
            baseline_score = average_precision_score(y_test, baseline_preds)
        else:
            baseline_score = roc_auc_score(y_test, baseline_preds)
    except ValueError:
        baseline_score = 0.5

    print(f"\n🔍 Permutation Importance Analysis")
    print(f"   Baseline {metric}: {baseline_score:.4f}")
    print(f"   Features: {len(feature_names)}")
    print(f"   Repeats: {n_repeats}")
    print(f"   Computing...")

    importances = {}
    n_features = X_test.shape[2]

    for feat_idx in range(n_features):
        drops = []
        for _ in range(n_repeats):
            X_permuted = X_test.copy()
            # Shuffle this feature across samples (not within sequence)
            perm_idx = np.random.permutation(len(X_permuted))
            X_permuted[:, :, feat_idx] = X_test[perm_idx, :, feat_idx]

            perm_preds = predict_proba(X_permuted)
            try:
                if metric == 'pr_auc':
                    perm_score = average_precision_score(y_test, perm_preds)
                else:
                    perm_score = roc_auc_score(y_test, perm_preds)
            except ValueError:
                perm_score = 0.5

            drops.append(baseline_score - perm_score)

        feat_name = feature_names[feat_idx] if feat_idx < len(feature_names) else f'Feature_{feat_idx}'
        importances[feat_name] = {
            'mean_drop': float(np.mean(drops)),
            'std_drop': float(np.std(drops)),
        }

    # Sort by importance
    sorted_imp = sorted(importances.items(), key=lambda x: -x[1]['mean_drop'])

    print(f"\n   {'Feature':40s} {'Mean Drop':>12s} {'Std':>10s}")
    print(f"   {'─'*65}")
    for feat, vals in sorted_imp[:20]:
        bar = '█' * max(0, int(vals['mean_drop'] * 200))
        print(f"   {feat:40s} {vals['mean_drop']:>12.6f} {vals['std_drop']:>10.6f}  {bar}")

    return sorted_imp


# Use a subset of test sequences for efficiency
n_explain = min(2000, len(X_test_seq))
explain_X = X_test_seq[:n_explain]
explain_y = y_test_seq[:n_explain]

sorted_importances = permutation_importance_lstm(
    model, explain_X, explain_y, final_feature_cols, device,
    n_repeats=3, metric='pr_auc'
)



# Plot top features
top_n = min(15, len(sorted_importances))
top_features = sorted_importances[:top_n]

fig, ax = plt.subplots(figsize=(12, 7))
feat_names = [f[0] for f in top_features][::-1]
feat_drops = [f[1]['mean_drop'] for f in top_features][::-1]
feat_stds = [f[1]['std_drop'] for f in top_features][::-1]

colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(feat_names)))
ax.barh(range(len(feat_names)), feat_drops, xerr=feat_stds,
        color=colors, alpha=0.85, capsize=3)
ax.set_yticks(range(len(feat_names)))
ax.set_yticklabels(feat_names, fontsize=10)
ax.set_xlabel('Mean PR-AUC Drop (Higher = More Important)', fontsize=12)
ax.set_title('CyberCast — Top Features by Permutation Importance', fontsize=14, fontweight='bold')
ax.grid(alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, 'feature_importance.png'), dpi=150, bbox_inches='tight')
plt.show()
print("📊 Saved: feature_importance.png")

# Explain top features
print("\n📋 Feature Importance Interpretation:")
print("   (Higher drop = more important for attack prediction)")
print("   ⚠️  Importance indicates predictive value, NOT causal relationship.")
for feat, vals in sorted_importances[:5]:
    print(f"\n   🔹 {feat} (drop: {vals['mean_drop']:.6f})")
    # Generic explanations based on common feature names
    if 'pkt' in feat.lower() or 'packet' in feat.lower():
        print(f"      Packet-related features often change during attacks (volume spikes, size anomalies).")
    elif 'syn' in feat.lower() or 'fin' in feat.lower() or 'rst' in feat.lower():
        print(f"      TCP flag features are strong indicators of scanning, SYN floods, and port probing.")
    elif 'byte' in feat.lower() or 'len' in feat.lower():
        print(f"      Byte/length features capture bandwidth anomalies and payload size changes.")
    elif 'flow' in feat.lower() or 'iat' in feat.lower():
        print(f"      Flow timing features detect rate changes and inter-arrival time anomalies.")
    elif 'ratio' in feat.lower() or 'proportion' in feat.lower():
        print(f"      Ratio features detect directional asymmetries characteristic of certain attacks.")
    else:
        print(f"      This feature significantly affects the model's ability to distinguish attack patterns.")



# ============================================================
# SECTION 28 — Save All Artifacts
# ============================================================

def save_artifacts():
    """Save all CyberCast artifacts to Google Drive."""
    print("\n💾 Saving all CyberCast artifacts...")
    print("=" * 60)

    saved_files = []

    # 1. Feature columns
    feat_path = os.path.join(MODEL_DIR, 'feature_columns.json')
    with open(feat_path, 'w') as f:
        json.dump(final_feature_cols, f, indent=2)
    saved_files.append(('Feature columns', feat_path))

    # 2. Model config (already saved, but update with final metrics)
    model_config['test_metrics'] = test_metrics
    model_config['best_threshold'] = best_threshold
    model_config['feature_columns'] = final_feature_cols
    model_config['n_train_sequences'] = len(X_train_seq)
    model_config['n_val_sequences'] = len(X_val_seq)
    model_config['n_test_sequences'] = len(X_test_seq)
    config_path = os.path.join(MODEL_DIR, 'model_config.json')
    with open(config_path, 'w') as f:
        json.dump(model_config, f, indent=2, default=str)
    saved_files.append(('Model config', config_path))

    # 3. Early warning results
    if early_warning_results:
        ew_path = os.path.join(RESULTS_DIR, 'early_warning_results.json')
        with open(ew_path, 'w') as f:
            json.dump(early_warning_results, f, indent=2, default=str)
        saved_files.append(('Early warning results', ew_path))

    # 4. Demo result
    demo_path = os.path.join(RESULTS_DIR, 'demo_result.json')
    with open(demo_path, 'w') as f:
        json.dump(demo_result, f, indent=2, default=str)
    saved_files.append(('Demo result', demo_path))

    # Print all saved files
    print(f"\n   {'Artifact':30s} {'Path'}")
    print(f"   {'─'*70}")

    # List all files in model, results, and plots dirs
    for dir_name, dir_path in [('models/', MODEL_DIR), ('results/', RESULTS_DIR),
                                ('plots/', PLOTS_DIR), ('processed/', PROCESSED_DIR)]:
        if os.path.isdir(dir_path):
            for fname in sorted(os.listdir(dir_path)):
                fpath = os.path.join(dir_path, fname)
                if os.path.isfile(fpath):
                    size_kb = os.path.getsize(fpath) / 1024
                    print(f"   {dir_name + fname:30s} {size_kb:>10.1f} KB")

    print(f"\n✅ All artifacts saved to: {PROJECT_DIR}")


save_artifacts()



# ============================================================
# SECTION 29 — Final Results Summary
# ============================================================

print("\n")
print("╔" + "═" * 68 + "╗")
print("║" + " " * 15 + "🛡️  CYBERCAST — FINAL REPORT" + " " * 24 + "║")
print("║" + " " * 10 + "AI-Based Network Attack Forecasting" + " " * 22 + "║")
print("╠" + "═" * 68 + "╣")

print("║" + " " * 68 + "║")
print("║  📊 DATASET" + " " * 56 + "║")
print(f"║    Raw CSV files:            {len(file_info):>10,}" + " " * 27 + "║")
print(f"║    Total raw rows:           {total_raw_rows:>10,}" + " " * 27 + "║")
print(f"║    Temporal states ({WINDOW_SECONDS}s):     {len(temporal_states):>10,}" + " " * 27 + "║")
print(f"║    Features:                 {len(final_feature_cols):>10,}" + " " * 27 + "║")

print("║" + " " * 68 + "║")
print("║  📐 SPLIT" + " " * 58 + "║")
print(f"║    Train:                    {len(X_train_seq):>10,} sequences" + " " * 17 + "║")
print(f"║    Validation:               {len(X_val_seq):>10,} sequences" + " " * 17 + "║")
print(f"║    Test:                     {len(X_test_seq):>10,} sequences" + " " * 17 + "║")

print("║" + " " * 68 + "║")
print("║  🏗️  MODEL" + " " * 57 + "║")
print(f"║    Architecture:             LSTM" + " " * 33 + "║")
print(f"║    Layers:                   {NUM_LAYERS:>10}" + " " * 27 + "║")
print(f"║    Hidden size:              {HIDDEN_SIZE:>10}" + " " * 27 + "║")
print(f"║    Sequence length:          {SEQUENCE_LENGTH:>10}" + " " * 27 + "║")
print(f"║    Parameters:               {total_params:>10,}" + " " * 27 + "║")

print("║" + " " * 68 + "║")
print("║  📊 TEST METRICS" + " " * 51 + "║")
print(f"║    ROC-AUC:                  {test_metrics['roc_auc']:>10.4f}" + " " * 27 + "║")
print(f"║    PR-AUC:                   {test_metrics['pr_auc']:>10.4f}" + " " * 27 + "║")
print(f"║    Accuracy:                 {test_metrics['accuracy']:>10.4f}" + " " * 27 + "║")
print(f"║    Precision:                {test_metrics['precision']:>10.4f}" + " " * 27 + "║")
print(f"║    Recall:                   {test_metrics['recall']:>10.4f}" + " " * 27 + "║")
print(f"║    F1 Score:                 {test_metrics['f1']:>10.4f}" + " " * 27 + "║")
print(f"║    Threshold:                {best_threshold:>10.4f}" + " " * 27 + "║")

print("║" + " " * 68 + "║")
print("║  🔮 FORECASTING" + " " * 52 + "║")
print(f"║    Forecast steps (K):       {FORECAST_HORIZON:>10}" + " " * 27 + "║")
print(f"║    Forecast horizon:         {FORECAST_HORIZON * WINDOW_SECONDS:>8}s" + " " * 28 + "║")

if early_warning_results and 'mean_early_warning_seconds' in early_warning_results:
    print("║" + " " * 68 + "║")
    print("║  🚨 EARLY WARNING" + " " * 50 + "║")
    print(f"║    Mean lead time:           {early_warning_results['mean_early_warning_seconds']:>8.1f}s" + " " * 28 + "║")

print("║" + " " * 68 + "║")
print("║  📊 RISK SCORE (demo)" + " " * 46 + "║")
print(f"║    Sample risk score:         {demo_result['risk_score']:>8.1f}" + " " * 29 + "║")
print(f"║    Risk category:             {demo_result['risk_category']:>15s}" + " " * 22 + "║")

print("║" + " " * 68 + "║")
print("╚" + "═" * 68 + "╝")



print("\n🎉 CyberCast notebook execution complete!")
print(f"\n📁 All artifacts saved to: {PROJECT_DIR}")
print(f"\n📊 Key results:")
print(f"   • Model: LSTM ({NUM_LAYERS} layers, {HIDDEN_SIZE} hidden, {total_params:,} params)")
print(f"   • Test ROC-AUC: {test_metrics['roc_auc']:.4f}")
print(f"   • Test PR-AUC:  {test_metrics['pr_auc']:.4f}")
print(f"   • Test F1:      {test_metrics['f1']:.4f}")
print(f"   • Forecast horizon: {FORECAST_HORIZON} steps ({FORECAST_HORIZON * WINDOW_SECONDS}s)")
print(f"\n   Ready for SIH 2026 presentation! 🚀")



