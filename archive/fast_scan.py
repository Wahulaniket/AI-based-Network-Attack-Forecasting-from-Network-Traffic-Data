import pandas as pd
import glob
from pathlib import Path
import re

RAW_DIR = Path('data/raw')
csv_files = sorted(glob.glob(str(RAW_DIR / '*.csv')))

print(f"Fast scanning {len(csv_files)} files for invalid timestamps...")

# Regex for valid DD/MM/YYYY
date_pattern = re.compile(r'^\d{2}/\d{2}/\d{4}')

total_1970 = 0
total_invalid = 0

for fpath in csv_files:
    fname = Path(fpath).name
    print(f"\nInspecting {fname}...")
    file_1970 = 0
    file_invalid = 0
    for chunk in pd.read_csv(fpath, chunksize=500000, low_memory=False, usecols=['Timestamp']):
        ts = chunk['Timestamp'].astype(str).str.strip()
        
        # Check for 1970 strings directly
        is_1970 = ts.str.contains('1970')
        file_1970 += is_1970.sum()
        
        # Check for unparseable dates or non-2018 dates
        is_2018 = ts.str.contains('2018')
        # Basically if it's not 2018 and not 1970 and not empty, it might be invalid
        is_invalid = ~(is_2018 | is_1970 | ts.isna() | (ts == 'nan') | (ts == 'Timestamp'))
        file_invalid += is_invalid.sum()
        
    print(f"  Out-of-range (1970): {file_1970}")
    print(f"  Invalid/Other: {file_invalid}")
    total_1970 += file_1970
    total_invalid += file_invalid

print(f"\nTOTAL OUT-OF-RANGE (1970): {total_1970}")
print(f"TOTAL INVALID: {total_invalid}")
