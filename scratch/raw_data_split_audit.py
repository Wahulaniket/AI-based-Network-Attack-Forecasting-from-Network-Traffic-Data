import os
import glob
import pandas as pd
import numpy as np

RAW_DATA_DIR = r"d:\working_projects\SIH\cyberCast\data\raw"
csv_files = glob.glob(os.path.join(RAW_DATA_DIR, "*.csv"))

print("================================================================================")
print("RAW DATASET AUDIT (RE-COMPUTING FROM RAW CSVs)")
print("================================================================================")

total_raw_flows = 0
benign_flows = 0
attack_flows = 0
invalid_timestamps_count = 0

file_stats = []

for f in sorted(csv_files):
    fname = os.path.basename(f)
    print(f"Processing {fname}...")
    df = pd.read_csv(f, usecols=['Timestamp', 'Label'], low_memory=False)
    n_total = len(df)
    total_raw_flows += n_total
    
    # Label counts
    labels = df['Label'].astype(str).str.upper()
    n_benign = (labels == 'BENIGN').sum()
    n_attack = n_total - n_benign
    benign_flows += n_benign
    attack_flows += n_attack
    
    # Parse timestamp
    dt = pd.to_datetime(df['Timestamp'], format='mixed', dayfirst=True, errors='coerce')
    n_invalid = dt.isna().sum()
    invalid_timestamps_count += n_invalid
    
    valid_dt = dt.dropna()
    min_t = valid_dt.min() if len(valid_dt) > 0 else None
    max_t = valid_dt.max() if len(valid_dt) > 0 else None
    
    file_stats.append({
        'file': fname, 'total': n_total, 'benign': n_benign, 'attack': n_attack,
        'invalid_ts': n_invalid, 'min_ts': str(min_t), 'max_ts': str(max_t)
    })

print("\n--- RAW DATASET SUMMARY ---")
print(f"Total Raw Flows                : {total_raw_flows:,}")
print(f"Benign Flows                   : {benign_flows:,}")
print(f"Attack Flows                   : {attack_flows:,}")
print(f"Attack Percentage              : {attack_flows / total_raw_flows * 100:.2f}%")
print(f"Invalid/Out-of-range Timestamps: {invalid_timestamps_count:,}")

# Check min and max across all files
df_file_stats = pd.DataFrame(file_stats)
valid_mins = df_file_stats['min_ts'].dropna()
valid_maxs = df_file_stats['max_ts'].dropna()
print(f"Overall Minimum Timestamp      : {valid_mins.min()}")
print(f"Overall Maximum Timestamp      : {valid_maxs.max()}")

print("\nPer File Statistics:")
print(df_file_stats.to_string(index=False))

# Now let's check split window statistics from pipeline or scratch.py
print("\n--- SPLIT WINDOW & SEQUENCE AUDIT ---")
# Let's check how the splits were defined in scratch.py or scripts/threshold_analysis.py
with open(r"d:\working_projects\SIH\cyberCast\scratch.py", 'r', encoding='utf-8') as f:
    scratch_code = f.read()

# Search for train/val/test split definitions
for line in scratch_code.split('\n'):
    if 'train' in line.lower() and ('split' in line.lower() or 'mask' in line.lower() or 'date' in line.lower() or '2018' in line):
        print(f"  [scratch.py] {line.strip()}")
