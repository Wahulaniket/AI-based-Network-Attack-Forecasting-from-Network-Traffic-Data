import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from collections import Counter
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cybercast_pipeline import (
    clean_chunk, aggregate_window_group, process_file_to_windows,
    CyberCastWorldModel, recursive_forecast, SUM_COLUMNS
)

def test_timestamp_parsing():
    print("Test 1: DD/MM timestamp parsing")
    df = pd.DataFrame({'Timestamp': ['01/03/2018 08:17:11', '14/02/2018 10:00:00'], 'Label': ['Benign', 'Benign']})
    df['Flow Duration'] = [10, 20]
    cleaned = clean_chunk(df)
    ts = cleaned['Timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S').tolist()
    assert ts == ['2018-03-01 08:17:11', '2018-02-14 10:00:00']
    print("PASS")

def test_chunk_boundaries():
    print("Test 2: 10-second window boundary crossing")
    # Simulate a chunk that ends right in the middle of a 10s window, and the next chunk continues it
    df = pd.DataFrame({
        'Timestamp': [
            '14/02/2018 10:00:02', 
            '14/02/2018 10:00:08', 
            '14/02/2018 10:00:15',
            '14/02/2018 10:00:25'
        ],
        'Label': ['Benign']*4,
        'Tot Fwd Pkts': [1, 2, 3, 4]
    })
    df.to_csv('scratch/test_chunk.csv', index=False)
    # 10:00:00 window should have 3 pkts (from 02, 08).
    # 10:00:10 window should have 3 pkts (from 15)
    # 10:00:20 window should have 4 pkts (from 25)
    windows = process_file_to_windows('scratch/test_chunk.csv', ['Tot Fwd Pkts'], chunk_size=2, window_seconds=10)
    assert len(windows) == 3
    assert windows[0]['Tot Fwd Pkts'] == 3
    assert windows[1]['Tot Fwd Pkts'] == 3
    assert windows[2]['Tot Fwd Pkts'] == 4
    print("PASS")

def test_label_construction():
    print("Test 3: label construction")
    df = pd.DataFrame({'Timestamp': ['14/02/2018 10:00:00', '14/02/2018 10:00:05'], 'Label': ['Benign', 'DDoS'], 'Tot Fwd Pkts': [1, 1]})
    df.to_csv('scratch/test_label.csv', index=False)
    windows = process_file_to_windows('scratch/test_label.csv', ['Tot Fwd Pkts'], chunk_size=10, window_seconds=10)
    assert len(windows) == 1
    assert windows[0]['attack_count'] == 1
    assert windows[0]['binary_attack'] == 1
    print("PASS")

def test_recursive_forecast():
    print("Test 7: recursive forecast behavior")
    model = CyberCastWorldModel(input_size=2, hidden_size=4, num_layers=1)
    states = np.random.randn(20, 2).astype(np.float32)
    attacks = np.zeros(20)
    res = recursive_forecast(model, states, attacks, seq_length=5, K=3, device=torch.device('cpu'), batch_size=2)
    assert len(res) == 3 # K=3
    assert len(res[1]['preds']) == 13 # 20 - 5 - 3 + 1 = 13.
    print("PASS")

if __name__ == '__main__':
    test_timestamp_parsing()
    test_chunk_boundaries()
    test_label_construction()
    test_recursive_forecast()
    print("ALL SYNTHETIC TESTS PASSED")
