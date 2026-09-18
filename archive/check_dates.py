import pandas as pd
import glob
import os

files = glob.glob(r'D:\working_projects\SIH\cyberCast\data\raw\*.csv')
print('Checking raw timestamps...')
for f in files:
    fname = os.path.basename(f)
    try:
        df = pd.read_csv(f, usecols=['Timestamp'], low_memory=False)
        raw_head = df['Timestamp'].head(2).tolist()
        
        # This is how the original notebook parses: 
        # pd.to_datetime(df['Timestamp'], format='mixed', dayfirst=True)
        # We also want to see what happens with default parsing vs the one in nb_cells
        dt = pd.to_datetime(df['Timestamp'], format='mixed', dayfirst=True)
        
        # Find if it produced January
        jan_mask = dt.dt.month == 1
        has_jan = jan_mask.any()
        
        min_dt, max_dt = dt.min(), dt.max()
        
        print(f'{fname}: Parsed [{min_dt} -> {max_dt}]')
        
        if has_jan:
            print(f'  !!! Found January dates in {fname}')
            jan_df = df[jan_mask]
            print('  First 10 raw January rows:')
            print(jan_df['Timestamp'].head(10).tolist())
            print('  Last 10 raw January rows:')
            print(jan_df['Timestamp'].tail(10).tolist())
            
    except Exception as e:
        print(f'{fname}: ERROR {e}')
