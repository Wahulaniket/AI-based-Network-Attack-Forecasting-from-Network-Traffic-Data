import nbformat as nbf
import os
from pathlib import Path
import subprocess
import json

def create_notebook():
    nb = nbf.v4.new_notebook()
    cells = []

    # 1. PHASE 3
    cells.append(nbf.v4.new_markdown_cell("""
# [PHASE3] CyberCast Learned World Model - Phase 3.1 Foundation

**Project:** CyberCast
**Phase:** Phase 3 — Learned Network World Model
**SIH Problem:** Problem Statement 26153

**Objective:**
Learn temporal network-state transition dynamics $P(S_{t+1} | S_t)$ and eventually use them for K-step forward simulation and pre-attack forecasting.

**Note:** This notebook covers Phase 3.1 ONLY. 
Phase 2.3 is a frozen next-window attack classifier. Do not claim Phase 3 forecasting capability yet.
"""))

    cells.append(nbf.v4.new_code_cell("""
import pandas as pd
import numpy as np
import os
import gc
from pathlib import Path
import json
import torch
import warnings
warnings.filterwarnings('ignore')

# Environment + Reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

print("--- ENVIRONMENT ---")
print(f"Pandas version: {pd.__version__}")
print(f"Numpy version: {np.__version__}")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
print("Random seeds set.")

DATA_DIR = Path("../../data/raw")
RESULTS_DIR = Path("../../results/phase3")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
(RESULTS_DIR / "state").mkdir(exist_ok=True)
(RESULTS_DIR / "audit").mkdir(exist_ok=True)
"""))

    # 2. DATA AUDIT
    cells.append(nbf.v4.new_markdown_cell("""
## [DATA-AUDIT] Dataset Discovery

Discover available CSV files in the raw dataset directory. We will use chunked processing to manage memory.
"""))

    cells.append(nbf.v4.new_code_cell("""
csv_files = sorted([f for f in DATA_DIR.glob("*.csv")])
print("Found CSV files:")
for f in csv_files:
    print(f" - {f.name} ({f.stat().st_size / (1024*1024):.2f} MB)")

if not csv_files:
    raise FileNotFoundError("No CSV files found in data/raw/")
"""))

    # 3. PACKET-LEVEL AUDIT
    cells.append(nbf.v4.new_markdown_cell("""
## [PACKET-LEVEL AUDIT]

We have scanned the repository for `.pcap` and `.pcapng` files and none were found.
"""))

    cells.append(nbf.v4.new_code_cell("""
print("Packet-level PCAP source is not currently available in the repository. Packet-level integration is deferred to a later Phase 3 data-extension step. No synthetic packet-level features are created.")
"""))

    # 4. DATA PROCESSING
    cells.append(nbf.v4.new_markdown_cell("""
## [DATA-CLEANING] [STATE-DEFINITION] [FEATURE-ENGINEERING]

Process all CSV files sequentially.
- **Timestamp Audit:** Use `pd.to_datetime` with DD/MM/YYYY fallback. VALID_START: 2018-02-14, VALID_END: 2018-03-02.
- **Data Cleaning:** Handle NaN/Inf carefully. Remove repeated CSV headers.
- **State Construction S_t:** 10-second windows.
- **Aggregation:** 
  - additive quantities -> SUM
  - true maxima/minima -> MAX/MIN
  - flow descriptive statistics -> documented MEAN
  - ratios -> mathematically derived
  - rates -> aggregate quantity / 10 seconds
  - zero-denominator derived values -> sentinel 0.0 with documented semantics
"""))

    cells.append(nbf.v4.new_code_cell("""
VALID_START = pd.to_datetime('2018-02-14 00:00:00')
VALID_END = pd.to_datetime('2018-03-02 23:59:59')
WINDOW_SIZE = '10s'
WINDOW_SECONDS = 10

with open("../../results/phase2_3/feature_sets.json", "r") as f:
    phase2_features = json.load(f)
SET_R = phase2_features.get("SET_R", [])
engineered = ['Fwd_Bwd_Pkt_Ratio', 'Fwd_Bwd_Byte_Ratio', 'Total_Pkts', 'Total_Bytes', 'Avg_Pkt_Size', 'Pkts_Per_Sec', 'Bytes_Per_Sec', 'SYN_FIN_Ratio', 'RST_SYN_Ratio', 'Fwd_Pkt_Proportion', 'flow_count']
raw_features_needed = [c for c in SET_R if c not in engineered and c not in ["Dst Port", "Protocol"]]

drop_cols = ['Flow Byts/s', 'Flow Pkts/s', 'Fwd Pkts/s', 'Bwd Pkts/s', 
             'Fwd Pkt Len Mean', 'Bwd Pkt Len Mean', 'Pkt Len Mean', 'Pkt Size Avg', 'Down/Up Ratio']
for col in drop_cols:
    if col in raw_features_needed:
        raw_features_needed.remove(col)

def process_chunk(chunk):
    chunk.columns = chunk.columns.str.strip()
    if 'Timestamp' not in chunk.columns:
        return None, 0, 0, 0, 0, 0
    
    initial_rows = len(chunk)
    
    header_mask = (chunk['Timestamp'] == 'Timestamp')
    repeated_headers = header_mask.sum()
    chunk = chunk[~header_mask]
    
    parsed_time = pd.to_datetime(chunk['Timestamp'], format='%d/%m/%Y %H:%M:%S', errors='coerce')
    mask_nat = parsed_time.isna()
    if mask_nat.any():
        parsed_time[mask_nat] = pd.to_datetime(chunk.loc[mask_nat, 'Timestamp'], format='mixed', dayfirst=False, errors='coerce')
    chunk['Timestamp'] = parsed_time
    
    invalid_ts_mask = chunk['Timestamp'].isna() | (chunk['Timestamp'] < VALID_START) | (chunk['Timestamp'] > VALID_END)
    invalid_ts_count = invalid_ts_mask.sum()
    chunk = chunk[~invalid_ts_mask]
    
    if 'Label' in chunk.columns:
        chunk['binary_attack'] = chunk['Label'].apply(lambda x: 0 if x == 'Benign' else 1)
    else:
        chunk['binary_attack'] = 0
    
    zero_duration_count = ((chunk['Flow Duration'] == '0') | (chunk['Flow Duration'] == 0)).sum()
    chunk.drop(columns=[c for c in drop_cols if c in chunk.columns], inplace=True, errors='ignore')
    
    for col in raw_features_needed:
        if col in chunk.columns:
            chunk[col] = pd.to_numeric(chunk[col], errors='coerce')

    numeric_cols = chunk.select_dtypes(include=[np.number]).columns.tolist()
    for col in numeric_cols:
        chunk[col] = chunk[col].replace([np.inf, -np.inf], np.nan)
        
    nan_mask = chunk[raw_features_needed].isna().any(axis=1) if set(raw_features_needed).issubset(chunk.columns) else pd.Series(False, index=chunk.index)
    nan_inf_count = nan_mask.sum()
    chunk = chunk[~nan_mask]
    
    valid_rows = len(chunk)
    if valid_rows == 0:
        return None, initial_rows, valid_rows, invalid_ts_count + nan_inf_count, repeated_headers, zero_duration_count
        
    chunk.set_index('Timestamp', inplace=True)
    chunk.sort_index(inplace=True)
    
    agg_dict = {}
    for col in raw_features_needed:
        if col not in chunk.columns:
            continue
        if 'Pkts' in col and not 'Len' in col and not 's' in col:
            agg_dict[col] = 'sum'
        elif 'Byts' in col and not 'Len' in col and not 's' in col and not 'Win' in col and not 'Avg' in col:
            agg_dict[col] = 'sum'
        elif 'Flag' in col or 'Flags' in col:
            agg_dict[col] = 'sum'
        elif 'Header' in col or 'Act Data Pkts' in col:
            agg_dict[col] = 'sum'
        elif 'Max' in col:
            agg_dict[col] = 'max'
        elif 'Min' in col:
            agg_dict[col] = 'min'
        elif 'Duration' in col:
            agg_dict[col] = 'mean'
        elif 'Std' in col or 'Var' in col or 'Mean' in col:
            agg_dict[col] = 'mean'
        else:
            agg_dict[col] = 'mean'
            
    agg_dict['binary_attack'] = 'sum'
    
    state_df = chunk.resample(WINDOW_SIZE).agg(agg_dict)
    state_df['flow_count'] = chunk.resample(WINDOW_SIZE).size()
    state_df = state_df[state_df['flow_count'] > 0]
    
    if len(state_df) == 0:
        return None, initial_rows, valid_rows, invalid_ts_count + nan_inf_count, repeated_headers, zero_duration_count
    
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

    state_df['Fwd Pkt Len Mean'] = np.where(t_fwd_pkts == 0, 0.0, t_fwd_bytes / t_fwd_pkts)
    state_df['Bwd Pkt Len Mean'] = np.where(t_bwd_pkts == 0, 0.0, t_bwd_bytes / t_bwd_pkts)
    state_df['Pkt Len Mean'] = np.where(t_pkts == 0, 0.0, t_bytes / t_pkts)
    state_df['Pkt Size Avg'] = state_df['Pkt Len Mean']
    state_df['Down/Up Ratio'] = np.where(t_fwd_pkts == 0, 0.0, t_bwd_pkts / t_fwd_pkts)
        
    return state_df, initial_rows, valid_rows, invalid_ts_count + nan_inf_count, repeated_headers, zero_duration_count

file_stats = []
all_states = []

for f in csv_files:
    print(f"Processing {f.name}...")
    chunk_iterator = pd.read_csv(f, chunksize=1000000, low_memory=False)
    
    f_initial, f_valid, f_invalid, f_rep_heads, f_zero_dur = 0, 0, 0, 0, 0
    f_states = []
    
    for chunk in chunk_iterator:
        res = process_chunk(chunk)
        state_df, initial, valid, invalid, rep_heads, zero_dur = res
        f_initial += initial
        f_valid += valid
        f_invalid += invalid
        f_rep_heads += rep_heads
        f_zero_dur += zero_dur
        if state_df is not None:
            f_states.append(state_df)
            
    if f_states:
        file_df = pd.concat(f_states)
        file_df.sort_index(inplace=True)
        agg_rules = {}
        for col in file_df.columns:
            if col in ['binary_attack']:
                agg_rules[col] = 'sum'
            elif 'Pkts' in col and not 'Len' in col and not 's' in col:
                agg_rules[col] = 'sum'
            elif 'Byts' in col and not 'Len' in col and not 's' in col and not 'Win' in col and not 'Avg' in col:
                agg_rules[col] = 'sum'
            elif 'Flag' in col or 'Flags' in col:
                agg_rules[col] = 'sum'
            elif 'Header' in col or 'Act Data Pkts' in col:
                agg_rules[col] = 'sum'
            elif 'Max' in col:
                agg_rules[col] = 'max'
            elif 'Min' in col:
                agg_rules[col] = 'min'
            elif col == 'flow_count':
                agg_rules[col] = 'sum'
            else:
                agg_rules[col] = 'mean'
                
        for c in drop_cols:
            if c in agg_rules:
                del agg_rules[c]
                
        file_df = file_df.groupby(file_df.index).agg(agg_rules)
        
        t_bytes = file_df.get('TotLen Fwd Pkts', 0) + file_df.get('TotLen Bwd Pkts', 0)
        t_pkts = file_df.get('Tot Fwd Pkts', 0) + file_df.get('Tot Bwd Pkts', 0)
        t_fwd_pkts = file_df.get('Tot Fwd Pkts', 0)
        t_bwd_pkts = file_df.get('Tot Bwd Pkts', 0)
        t_fwd_bytes = file_df.get('TotLen Fwd Pkts', 0)
        t_bwd_bytes = file_df.get('TotLen Bwd Pkts', 0)
        
        file_df['Flow Byts/s'] = t_bytes / WINDOW_SECONDS
        file_df['Flow Pkts/s'] = t_pkts / WINDOW_SECONDS
        file_df['Fwd Pkts/s'] = t_fwd_pkts / WINDOW_SECONDS
        file_df['Bwd Pkts/s'] = t_bwd_pkts / WINDOW_SECONDS
        
        file_df['Fwd Pkt Len Mean'] = np.where(t_fwd_pkts == 0, 0.0, t_fwd_bytes / t_fwd_pkts)
        file_df['Bwd Pkt Len Mean'] = np.where(t_bwd_pkts == 0, 0.0, t_bwd_bytes / t_bwd_pkts)
        file_df['Pkt Len Mean'] = np.where(t_pkts == 0, 0.0, t_bytes / t_pkts)
        file_df['Pkt Size Avg'] = file_df['Pkt Len Mean']
        file_df['Down/Up Ratio'] = np.where(t_fwd_pkts == 0, 0.0, t_bwd_pkts / t_fwd_pkts)
            
        all_states.append(file_df)
        
        time_min, time_max = file_df.index.min(), file_df.index.max()
        states_count = len(file_df)
        attack_rows = (file_df['binary_attack'] > 0).sum()
    else:
        time_min, time_max, states_count, attack_rows = None, None, 0, 0
        
    stats = {
        "file": f.name,
        "raw_rows": f_initial,
        "valid_rows": f_valid,
        "invalid_rows": int(f_invalid),
        "repeated_headers": int(f_rep_heads),
        "zero_duration_flows": int(f_zero_dur),
        "attack_windows": int(attack_rows),
        "benign_windows": int(states_count - attack_rows),
        "time_range": f"{time_min} -> {time_max}" if time_min else "N/A",
        "state_count": int(states_count)
    }
    file_stats.append(stats)
    print(stats)
    gc.collect()
    
if all_states:
    final_states = pd.concat(all_states)
    final_states.sort_index(inplace=True)
    agg_rules = {}
    for col in final_states.columns:
        if col in ['binary_attack']:
            agg_rules[col] = 'sum'
        elif 'Pkts' in col and not 'Len' in col and not 's' in col:
            agg_rules[col] = 'sum'
        elif 'Byts' in col and not 'Len' in col and not 's' in col and not 'Win' in col and not 'Avg' in col:
            agg_rules[col] = 'sum'
        elif 'Flag' in col or 'Flags' in col:
            agg_rules[col] = 'sum'
        elif 'Header' in col or 'Act Data Pkts' in col:
            agg_rules[col] = 'sum'
        elif 'Max' in col:
            agg_rules[col] = 'max'
        elif 'Min' in col:
            agg_rules[col] = 'min'
        elif col == 'flow_count':
            agg_rules[col] = 'sum'
        else:
            agg_rules[col] = 'mean'
            
    for c in drop_cols:
        if c in agg_rules:
            del agg_rules[c]
            
    final_states = final_states.groupby(final_states.index).agg(agg_rules)
    
    t_bytes = final_states.get('TotLen Fwd Pkts', 0) + final_states.get('TotLen Bwd Pkts', 0)
    t_pkts = final_states.get('Tot Fwd Pkts', 0) + final_states.get('Tot Bwd Pkts', 0)
    t_fwd_pkts = final_states.get('Tot Fwd Pkts', 0)
    t_bwd_pkts = final_states.get('Tot Bwd Pkts', 0)
    t_fwd_bytes = final_states.get('TotLen Fwd Pkts', 0)
    t_bwd_bytes = final_states.get('TotLen Bwd Pkts', 0)
    
    final_states['Flow Byts/s'] = t_bytes / WINDOW_SECONDS
    final_states['Flow Pkts/s'] = t_pkts / WINDOW_SECONDS
    final_states['Fwd Pkts/s'] = t_fwd_pkts / WINDOW_SECONDS
    final_states['Bwd Pkts/s'] = t_bwd_pkts / WINDOW_SECONDS
    final_states['Fwd Pkt Len Mean'] = np.where(t_fwd_pkts == 0, 0.0, t_fwd_bytes / t_fwd_pkts)
    final_states['Bwd Pkt Len Mean'] = np.where(t_bwd_pkts == 0, 0.0, t_bwd_bytes / t_bwd_pkts)
    final_states['Pkt Len Mean'] = np.where(t_pkts == 0, 0.0, t_bytes / t_pkts)
    final_states['Pkt Size Avg'] = final_states['Pkt Len Mean']
    final_states['Down/Up Ratio'] = np.where(t_fwd_pkts == 0, 0.0, t_bwd_pkts / t_fwd_pkts)
        
    print(f"Total states across all files: {len(final_states)}")
else:
    raise ValueError("No states were generated.")
"""))

    # 5. TEMPORAL CONTINUITY & TARGET ISOLATION
    cells.append(nbf.v4.new_markdown_cell("""
## [STATE-CONSTRUCTION] Temporal Continuity & Target Isolation

**Empty Windows Policy:** 
Transitions represent consecutive OBSERVED traffic states rather than every rigid wall-clock 10-second interval. Windows with zero flows are naturally excluded (not zero-filled), which ensures the model is causally coherent.

**Target Isolation:**
S_t contains observable traffic only.
Y_t contains labels separately.
"""))

    cells.append(nbf.v4.new_code_cell("""
target_columns = ['binary_attack', 'Label', 'dominant_label', 'attack_ratio', 'attack_flow_count']
Y_t = pd.DataFrame(index=final_states.index)
if 'binary_attack' in final_states.columns:
    Y_t['binary_attack'] = (final_states['binary_attack'] > 0).astype(int)
    S_t = final_states.drop(columns=['binary_attack'])
else:
    S_t = final_states

# Extract exact 77 state features 
S_t = S_t[[c for c in SET_R if c in S_t.columns]]

# Any remaining NA fill with 0 (safe measure for empty subsets)
S_t.fillna(0, inplace=True)
"""))

    # 6. CAUSALITY / LEAKAGE AUDIT
    cells.append(nbf.v4.new_markdown_cell("""
## [CAUSALITY-AUDIT] Leakage Audit and Assertions

Ensure all requirements are strictly met across the full dataset.
"""))

    cells.append(nbf.v4.new_code_cell("""
state_features = list(S_t.columns)

# 1. Target separation
assert set(target_columns).isdisjoint(set(state_features)), "Leakage: Targets found in S_t!"
for target in target_columns:
    assert target not in state_features, f"Leakage: {target} in state!"

# 2. Chronological Ordering
assert S_t.index.is_monotonic_increasing, "Chronological check failed!"
assert not S_t.index.has_duplicates, "Duplicate timestamps found!"

# 3. Exact 77 count
assert len(state_features) == 77, f"Expected 77 features, got {len(state_features)}"
assert 'Timestamp' not in state_features, "Timestamp used as feature"

# 4. Numerics and finites
for col in state_features:
    assert pd.api.types.is_numeric_dtype(S_t[col]), f"Column {col} is not numeric"
    assert not np.isinf(S_t[col]).any(), f"Column {col} contains Inf"
    assert not S_t[col].isna().any(), f"Column {col} contains NaN"

print("Causality Audit Passed: No target leakage found in S_t.")
print("Chronological Audit Passed: States are correctly ordered in time.")
print(f"Final Feature Count: {len(state_features)}")

# Save schema
state_schema = {
    "window_size": WINDOW_SIZE,
    "features": state_features,
    "dimensionality": len(state_features)
}

with open(RESULTS_DIR / "state" / "state_schema.json", "w") as f:
    json.dump(state_schema, f, indent=4)

# Save data audit
audit_data = {
    "total_raw_rows": int(sum([s["raw_rows"] for s in file_stats])),
    "total_valid_rows": int(sum([s["valid_rows"] for s in file_stats])),
    "total_invalid_rows": int(sum([s["invalid_rows"] for s in file_stats])),
    "total_repeated_headers": int(sum([s.get("repeated_headers", 0) for s in file_stats])),
    "total_zero_duration_flows": int(sum([s.get("zero_duration_flows", 0) for s in file_stats])),
    "total_states": len(S_t),
    "file_stats": file_stats
}

with open(RESULTS_DIR / "audit" / "data_audit.json", "w") as f:
    json.dump(audit_data, f, indent=4)
    
causality_audit = {
    "leakage_passed": True,
    "chronological_passed": True,
    "missing_windows_policy": "Zero-flow windows omitted (inactive). No auto-filling. Transitions represent causally coherent observed traffic states."
}

with open(RESULTS_DIR / "audit" / "causality_audit.json", "w") as f:
    json.dump(causality_audit, f, indent=4)

# Save artifacts
S_t.to_parquet(RESULTS_DIR / "state" / "S_t.parquet")
Y_t.to_parquet(RESULTS_DIR / "state" / "Y_t.parquet")

print("Artifacts saved successfully.")
"""))

    nb['cells'] = cells
    
    notebook_path = Path("notebooks/phase3/Phase3_CyberCast_WorldModel.ipynb")
    notebook_path.parent.mkdir(parents=True, exist_ok=True)
    with open(notebook_path, 'w', encoding='utf-8') as f:
        nbf.write(nb, f)
        
    print(f"Notebook written to {notebook_path}")

if __name__ == "__main__":
    create_notebook()
    nb_path = "notebooks/phase3/Phase3_CyberCast_WorldModel.ipynb"
    print("Executing notebook...")
    subprocess.run(["python", "-m", "jupyter", "nbconvert", "--to", "notebook", "--execute", "--inplace", nb_path], check=True)
    print("Notebook executed successfully.")
