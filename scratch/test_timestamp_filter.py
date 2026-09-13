import os
import sys
import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cybercast_pipeline import clean_chunk

def test_no_1970_corruption():
    print("Test: Regression test for 1970 timestamp corruption")
    
    # Simulate both valid CIC-IDS2018 dates and corrupted 1970 dates
    df = pd.DataFrame({
        'Timestamp': [
            '14/02/2018 10:00:00',
            '01/03/2018 08:17:11',
            '05/01/1970 03:01:17',
            '12/01/1970 09:44:12',
            '28/02/2018 23:59:59'
        ],
        'Label': ['Benign', 'Benign', 'DDoS', 'Benign', 'Benign'],
        'Tot Fwd Pkts': [1, 2, 3, 4, 5]
    })
    
    # clean_chunk should drop the 1970 dates or any date outside Feb 14 - Mar 2 2018
    cleaned = clean_chunk(df)
    
    ts_list = cleaned['Timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S').tolist()
    
    # Ensure no 1970 date is present
    assert not any('1970' in ts for ts in ts_list), "1970 date slipped through!"
    
    # Ensure valid dates are kept
    assert '2018-02-14 10:00:00' in ts_list
    assert '2018-03-01 08:17:11' in ts_list
    assert '2018-02-28 23:59:59' in ts_list
    assert len(ts_list) == 3, f"Expected 3 valid rows, got {len(ts_list)}"
    
    print("PASS: Corrupted dates successfully filtered.")

if __name__ == '__main__':
    test_no_1970_corruption()
