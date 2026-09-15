import os
import glob
import pandas as pd

RAW_DATA_DIR = r"d:\working_projects\SIH\cyberCast\data\raw"
csv_files = glob.glob(os.path.join(RAW_DATA_DIR, "*.csv"))

total_raw_flows = 0
benign_flows = 0
attack_flows = 0
invalid_timestamps_count = 0
all_min_ts = []
all_max_ts = []

print("Auditing raw CSV files in chunks...")

for f in sorted(csv_files):
    fname = os.path.basename(f)
    print(f"Reading {fname}...")
    file_total = 0
    file_benign = 0
    file_attack = 0
    file_invalid = 0
    
    # Read chunk by chunk
    for chunk in pd.read_csv(f, usecols=['Timestamp', 'Label'], chunksize=500000, low_memory=False):
        file_total += len(chunk)
        
        lbl = chunk['Label'].astype(str).str.upper()
        b_cnt = (lbl == 'BENIGN').sum()
        a_cnt = len(chunk) - b_cnt
        file_benign += b_cnt
        file_attack += a_cnt
        
        # Parse timestamp
        dt = pd.to_datetime(chunk['Timestamp'], format='mixed', dayfirst=True, errors='coerce')
        inv = dt.isna().sum()
        file_invalid += inv
        
        v_dt = dt.dropna()
        if len(v_dt) > 0:
            all_min_ts.append(v_dt.min())
            all_max_ts.append(v_dt.max())
            
    total_raw_flows += file_total
    benign_flows += file_benign
    attack_flows += file_attack
    invalid_timestamps_count += file_invalid
    print(f"  {fname}: Total={file_total:,}, Benign={file_benign:,}, Attack={file_attack:,}, Invalid TS={file_invalid:,}")

print("\n================================================================================")
print("RAW DATASET AUDIT RESULTS")
print("================================================================================")
print(f"Total Raw Flows                : {total_raw_flows:,}")
print(f"Benign Flows                   : {benign_flows:,}")
print(f"Attack Flows                   : {attack_flows:,}")
print(f"Attack Percentage              : {attack_flows / total_raw_flows * 100:.2f}%")
print(f"Invalid/Out-of-range Timestamps: {invalid_timestamps_count:,}")
print(f"Final Minimum Timestamp        : {min(all_min_ts)}")
print(f"Final Maximum Timestamp        : {max(all_max_ts)}")
