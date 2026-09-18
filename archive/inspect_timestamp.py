import pandas as pd
import glob
from pathlib import Path

RAW_DIR = Path('data/raw')
files_to_check = ['02-14-2018.csv', '02-22-2018.csv', '02-23-2018.csv']

for fname in files_to_check:
    fpath = RAW_DIR / fname
    if not fpath.exists():
        print(f"File not found: {fpath}")
        continue
    
    print(f"\nInspecting {fname}...")
    for chunk in pd.read_csv(fpath, chunksize=100000, low_memory=False):
        chunk.columns = chunk.columns.str.strip()
        if 'Timestamp' not in chunk.columns:
            continue
            
        raw_ts = chunk['Timestamp']
        # Parse it the same way the pipeline does
        parsed = pd.to_datetime(raw_ts, format='mixed', dayfirst=True, errors='coerce')
        
        # Check for 1970 dates
        is_1970 = parsed.dt.year == 1970
        if is_1970.any():
            bad_rows = chunk[is_1970]
            print(f"Found {len(bad_rows)} rows parsing to 1970 in {fname}.")
            print("Sample raw values:")
            for idx, row in bad_rows.head(5).iterrows():
                print(f"  Row {idx}: Raw={repr(row['Timestamp'])} ({type(row['Timestamp'])}) -> Parsed={parsed[idx]}")
            break # Just need a sample from each file
