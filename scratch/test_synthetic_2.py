import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# Re-define functions from cybercast_pipeline to avoid running the full script
SUM_COLUMNS = {
    'Tot Fwd Pkts', 'Tot Bwd Pkts',
    'TotLen Fwd Pkts', 'TotLen Bwd Pkts',
    'Fwd Header Len', 'Bwd Header Len',
    'Fwd Act Data Pkts',
    'Subflow Fwd Pkts', 'Subflow Fwd Byts', 'Subflow Bwd Pkts', 'Subflow Bwd Byts',
    'FIN Flag Cnt', 'SYN Flag Cnt', 'RST Flag Cnt', 'PSH Flag Cnt',
    'ACK Flag Cnt', 'URG Flag Cnt', 'CWE Flag Count', 'ECE Flag Cnt',
    'Fwd PSH Flags', 'Bwd PSH Flags', 'Fwd URG Flags', 'Bwd URG Flags',
    'Fwd IAT Tot', 'Bwd IAT Tot',
}

def clean_chunk(chunk):
    chunk.columns = chunk.columns.str.strip()
    for col in chunk.select_dtypes(include=['object']).columns:
        if col not in ('Timestamp', 'Label'):
            chunk[col] = pd.to_numeric(chunk[col], errors='coerce')
    chunk['Timestamp'] = pd.to_datetime(
        chunk['Timestamp'], format='mixed', dayfirst=True, errors='coerce'
    )
    chunk = chunk.dropna(subset=['Timestamp'])
    chunk['Attack'] = (chunk['Label'].astype(str).str.strip() != 'Benign').astype(np.int8)
    num_cols = chunk.select_dtypes(include=[np.number]).columns
    chunk[num_cols] = chunk[num_cols].replace([np.inf, -np.inf], np.nan)
    for col in num_cols:
        if chunk[col].isna().any():
            med = chunk[col].median()
            chunk[col] = chunk[col].fillna(med if pd.notna(med) else 0.0)
    return chunk

def aggregate_window_group(group, feature_cols):
    row = {}
    for col in feature_cols:
        if col not in group.columns:
            row[col] = 0.0
            continue
        vals = group[col].values
        if col in SUM_COLUMNS:
            row[col] = float(np.nansum(vals))
        else:
            row[col] = float(np.nanmean(vals)) if len(vals) > 0 else 0.0
    row['flow_count'] = len(group)
    row['attack_count'] = int(group['Attack'].sum())
    row['attack_ratio'] = row['attack_count'] / max(row['flow_count'], 1)
    row['binary_attack'] = int(row['attack_count'] > 0)
    return row

def process_file_to_windows(csv_path, feature_cols, chunk_size=500000, window_seconds=10):
    all_window_rows = []
    carry_over = None
    for chunk in pd.read_csv(csv_path, chunksize=chunk_size, low_memory=False):
        chunk = clean_chunk(chunk)
        if len(chunk) == 0: continue
        if carry_over is not None and len(carry_over) > 0:
            chunk = pd.concat([carry_over, chunk], ignore_index=True)
            carry_over = None
        chunk = chunk.sort_values('Timestamp').reset_index(drop=True)
        chunk['window_start'] = chunk['Timestamp'].dt.floor(f'{window_seconds}s')
        unique_windows = chunk['window_start'].unique()
        if len(unique_windows) <= 1:
            carry_over = chunk.drop(columns=['window_start'])
            continue
        last_win = unique_windows[-1]
        carry_over = chunk[chunk['window_start'] == last_win].drop(columns=['window_start']).copy()
        complete = chunk[chunk['window_start'] != last_win]
        for win_ts, grp in complete.groupby('window_start'):
            row = aggregate_window_group(grp, feature_cols)
            row['Timestamp'] = win_ts
            all_window_rows.append(row)
    if carry_over is not None and len(carry_over) > 0:
        carry_over['window_start'] = carry_over['Timestamp'].dt.floor(f'{window_seconds}s')
        for win_ts, grp in carry_over.groupby('window_start'):
            row = aggregate_window_group(grp, feature_cols)
            row['Timestamp'] = win_ts
            all_window_rows.append(row)
    return all_window_rows

class CyberCastWorldModel(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size, hidden_size=hidden_size,
            num_layers=num_layers, batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.dropout = nn.Dropout(dropout)
        self.state_head = nn.Linear(hidden_size, input_size)
        self.attack_head = nn.Linear(hidden_size, 1)

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        h = self.dropout(h_n[-1])
        pred_state = self.state_head(h)
        attack_logit = self.attack_head(h).squeeze(-1)
        return pred_state, attack_logit

def recursive_forecast(model, states, attacks, seq_length, K, device, batch_size=256):
    model.eval()
    n = len(states) - seq_length - K + 1
    results = {k: {'preds': [], 'targets': []} for k in range(1, K+1)}
    with torch.no_grad():
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            bs = end - start
            seq_buf = np.zeros((bs, seq_length, states.shape[1]), dtype=np.float32)
            for j in range(bs):
                seq_buf[j] = states[start+j : start+j+seq_length]
            for k in range(1, K+1):
                seq_tensor = torch.from_numpy(seq_buf).to(device)
                pred_state, attack_logit = model(seq_tensor)
                probs = torch.sigmoid(attack_logit).cpu().numpy().flatten()
                pred_s = pred_state.cpu().numpy()
                for j in range(bs):
                    tidx = start + j + seq_length + k - 1
                    if tidx < len(attacks):
                        results[k]['preds'].append(float(probs[j]))
                        results[k]['targets'].append(float(attacks[tidx]))
                seq_buf[:, :-1, :] = seq_buf[:, 1:, :]
                seq_buf[:, -1, :] = pred_s
    return results

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
    df = pd.DataFrame({
        'Timestamp': ['14/02/2018 10:00:02', '14/02/2018 10:00:08', '14/02/2018 10:00:15', '14/02/2018 10:00:25'],
        'Label': ['Benign']*4,
        'Tot Fwd Pkts': [1, 2, 3, 4]
    })
    df.to_csv('scratch/test_chunk.csv', index=False)
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
    assert len(res) == 3 
    assert len(res[1]['preds']) == 13 
    print("PASS")

if __name__ == '__main__':
    test_timestamp_parsing()
    test_chunk_boundaries()
    test_label_construction()
    test_recursive_forecast()
    print("ALL SYNTHETIC TESTS PASSED")
