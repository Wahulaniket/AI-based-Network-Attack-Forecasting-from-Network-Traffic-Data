import pandas as pd
import numpy as np
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# 1. Load Phase 2 Features
with open("results/phase2_3/feature_sets.json", "r") as f:
    phase2_features = json.load(f)
SET_R = phase2_features.get("SET_R", [])
engineered = ['Fwd_Bwd_Pkt_Ratio', 'Fwd_Bwd_Byte_Ratio', 'Total_Pkts', 'Total_Bytes', 'Avg_Pkt_Size', 'Pkts_Per_Sec', 'Bytes_Per_Sec', 'SYN_FIN_Ratio', 'RST_SYN_Ratio', 'Fwd_Pkt_Proportion', 'flow_count']
raw_features_needed = [c for c in SET_R if c not in engineered and c not in ["Dst Port", "Protocol"]]

# 2. Read One Chunk
file_path = "data/raw/02-16-2018.csv"
print(f"Reading chunk from {file_path}...")
chunk = next(pd.read_csv(file_path, chunksize=100000, low_memory=False))
chunk.columns = chunk.columns.str.strip()

print(f"\n--- CHUNK STATS ---")
print(f"Input row count: {len(chunk)}")

# 3. Repeated Header Detection
header_mask = (chunk['Timestamp'] == 'Timestamp')
repeated_headers = header_mask.sum()
print(f"\n--- DATA CLEANING ---")
print(f"Repeated header rows removed: {repeated_headers}")
chunk = chunk[~header_mask]

# 4. Handle Infinity & Zero Duration & Meaningless Per-Flow Rates/Means
drop_cols = ['Flow Byts/s', 'Flow Pkts/s', 'Fwd Pkts/s', 'Bwd Pkts/s', 
             'Fwd Pkt Len Mean', 'Bwd Pkt Len Mean', 'Pkt Len Mean', 'Pkt Size Avg', 'Down/Up Ratio']
for col in drop_cols:
    if col in raw_features_needed:
        raw_features_needed.remove(col)
        
zero_duration = (chunk['Flow Duration'] == '0') | (chunk['Flow Duration'] == 0)
print(f"Zero-duration flows detected: {zero_duration.sum()} (Retaining flows, dropping per-flow Inf rates)")
chunk.drop(columns=[c for c in drop_cols if c in chunk.columns], inplace=True, errors='ignore')

# 5. Type Casting
for col in raw_features_needed:
    if col in chunk.columns:
        chunk[col] = pd.to_numeric(chunk[col], errors='coerce')

numeric_cols = chunk.select_dtypes(include=[np.number]).columns.tolist()
inf_replaced = 0
for col in numeric_cols:
    inf_mask = np.isinf(chunk[col])
    inf_count = inf_mask.sum()
    if inf_count > 0:
        inf_replaced += inf_count
    chunk[col] = chunk[col].replace([np.inf, -np.inf], np.nan)
print(f"Remaining unexpected Inf values replaced with NaN: {inf_replaced}")

# 6. Timestamp Audit
print("\n--- TIMESTAMP AUDIT ---")
VALID_START = pd.to_datetime('2018-02-14 00:00:00')
VALID_END = pd.to_datetime('2018-03-02 23:59:59')
parsed_time = pd.to_datetime(chunk['Timestamp'], format='%d/%m/%Y %H:%M:%S', errors='coerce')
mask_nat = parsed_time.isna()
if mask_nat.any():
    parsed_time[mask_nat] = pd.to_datetime(chunk.loc[mask_nat, 'Timestamp'], format='mixed', dayfirst=False, errors='coerce')

chunk['Timestamp'] = parsed_time
invalid_ts_mask = chunk['Timestamp'].isna() | (chunk['Timestamp'] < VALID_START) | (chunk['Timestamp'] > VALID_END)
print(f"Invalid timestamp rows removed: {invalid_ts_mask.sum()}")
chunk = chunk[~invalid_ts_mask]

# Labels
chunk['binary_attack'] = 0
if 'Label' in chunk.columns:
    chunk['binary_attack'] = chunk['Label'].apply(lambda x: 0 if x == 'Benign' else 1)

nan_mask = chunk[raw_features_needed].isna().any(axis=1) if set(raw_features_needed).issubset(chunk.columns) else pd.Series(False, index=chunk.index)
print(f"Genuinely invalid/NaN numeric rows dropped: {nan_mask.sum()}")
chunk = chunk[~nan_mask]

# 7. Grouping
print("\n--- AGGREGATION & STATE CONSTRUCTION ---")
WINDOW_SIZE = '10s'
WINDOW_SECONDS = 10
chunk.set_index('Timestamp', inplace=True)
chunk.sort_index(inplace=True)

# Assign proper aggregation rules
agg_dict = {}
for col in raw_features_needed:
    if col not in chunk.columns:
        continue
    
    # A. ADDITIVE
    if 'Pkts' in col and not 'Len' in col and not 's' in col: # count
        agg_dict[col] = 'sum'
    elif 'Byts' in col and not 'Len' in col and not 's' in col and not 'Win' in col and not 'Avg' in col: # bytes
        agg_dict[col] = 'sum'
    elif 'Flag' in col or 'Flags' in col: # TCP Flags
        agg_dict[col] = 'sum'
    elif 'Header' in col or 'Act Data Pkts' in col:
        agg_dict[col] = 'sum'
        
    # B. TRUE WINDOW STATISTICS
    elif 'Max' in col:
        agg_dict[col] = 'max'
    elif 'Min' in col:
        agg_dict[col] = 'min'
        
    # C. FLOW-LEVEL DESCRIPTIVE STATISTICS
    elif 'Duration' in col:
        agg_dict[col] = 'mean'
    elif 'Std' in col or 'Var' in col or 'Mean' in col:
        agg_dict[col] = 'mean'
    else:
        agg_dict[col] = 'mean'

# Create the state dataframe
state_df = chunk.resample(WINDOW_SIZE).agg(agg_dict)
state_df['flow_count'] = chunk.resample(WINDOW_SIZE).size()
state_df = state_df[state_df['flow_count'] > 0]

# D. DERIVED FEATURES (Mathematically exact at window level with Zero-Denominator Policy)
t_bytes = state_df.get('TotLen Fwd Pkts', 0) + state_df.get('TotLen Bwd Pkts', 0)
t_pkts = state_df.get('Tot Fwd Pkts', 0) + state_df.get('Tot Bwd Pkts', 0)
t_fwd_pkts = state_df.get('Tot Fwd Pkts', 0)
t_bwd_pkts = state_df.get('Tot Bwd Pkts', 0)
t_fwd_bytes = state_df.get('TotLen Fwd Pkts', 0)
t_bwd_bytes = state_df.get('TotLen Bwd Pkts', 0)

state_df['Flow Byts/s'] = t_bytes / WINDOW_SECONDS
state_df['Flow Pkts/s'] = t_pkts / WINDOW_SECONDS
state_df['Fwd Pkts/s'] = t_fwd_pkts / WINDOW_SECONDS
state_df['Bwd Pkts/s'] = t_bwd_pkts / WINDOW_SECONDS

# Zero-denominator policy: Sentinel value 0.0 means "no traffic observed in this window"
state_df['Fwd Pkt Len Mean'] = np.where(t_fwd_pkts == 0, 0.0, t_fwd_bytes / t_fwd_pkts)
state_df['Bwd Pkt Len Mean'] = np.where(t_bwd_pkts == 0, 0.0, t_bwd_bytes / t_bwd_pkts)
state_df['Pkt Len Mean'] = np.where(t_pkts == 0, 0.0, t_bytes / t_pkts)
state_df['Pkt Size Avg'] = state_df['Pkt Len Mean']
state_df['Down/Up Ratio'] = np.where(t_fwd_pkts == 0, 0.0, t_bwd_pkts / t_fwd_pkts)

# Separate targets
target_columns = ['binary_attack', 'Label', 'dominant_label', 'attack_ratio', 'attack_flow_count']
labels_df = pd.DataFrame(index=state_df.index)
if 'binary_attack' in chunk.columns:
    labels_df['binary_attack'] = chunk.resample(WINDOW_SIZE)['binary_attack'].sum()
    labels_df = labels_df.loc[state_df.index]

print(f"Output window count: {len(state_df)}")

print("\n--- SAMPLE OUTPUT STATE (S_t) ---")
if len(state_df) > 0:
    print(state_df.iloc[0].to_dict())
else:
    print("No valid states generated.")

print("\n--- SAMPLE TARGETS (Y_t) ---")
if len(labels_df) > 0:
    print(labels_df.iloc[0].to_dict())

print("\n--- ASSERTIONS ---")
state_features = list(state_df.columns)

assert set(target_columns).isdisjoint(set(state_features)), "Target leakage detected via set disjoint check!"
for target in target_columns:
    assert target not in state_df.columns, f"Target {target} leaked into state_df!"

for col in state_features:
    assert pd.api.types.is_numeric_dtype(state_df[col]), f"Column {col} is not numeric, dtype is {state_df[col].dtype}"
    assert not np.isinf(state_df[col]).any(), f"Column {col} contains Inf"

assert 'Timestamp' not in state_features, "Timestamp used as feature"

print("All assertions passed!")

print(f"\nTotal State Features: {len(state_features)}")
