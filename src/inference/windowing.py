import pandas as pd
import numpy as np
from typing import List, Tuple

def create_time_windows(df: pd.DataFrame, expected_features: List[str], window_seconds: int = 10) -> pd.DataFrame:
    """Aggregate irregular flows into fixed-interval temporal network states.
    
    Uses mean aggregation and adds `flow_count`.
    Does NOT compute target attributes (Attack) to prevent leakage.
    Returns exactly the expected features.
    """
    if df.empty:
        return pd.DataFrame(columns=['Timestamp'] + expected_features)
        
    df = df.set_index('Timestamp').sort_index()

    # Resample at WINDOW_SECONDS frequency
    freq = f'{window_seconds}s'

    # Mean aggregation for all required features
    # Ensure all expected features (except flow_count) exist before resampling
    existing_cols = [c for c in expected_features if c in df.columns and c != 'flow_count']
    
    # Fill missing columns with 0.0 to guarantee correct output shape
    for col in expected_features:
        if col not in df.columns and col != 'flow_count':
            df[col] = 0.0
            existing_cols.append(col)

    states = df[existing_cols].resample(freq).mean()

    # Flow count per window
    if len(existing_cols) > 0:
        flow_counts = df[existing_cols[0]].resample(freq).count()
    else:
        flow_counts = pd.Series(0, index=states.index)
        
    states['flow_count'] = flow_counts

    # Fill NaN (empty windows) with 0
    states = states.fillna(0.0)

    # Convert to float32 for memory efficiency
    for col in states.select_dtypes(include=[np.float64]).columns:
        states[col] = states[col].astype(np.float32)

    states = states.reset_index()
    
    # Ensure exact column ordering (Timestamp + expected_features)
    final_cols = ['Timestamp'] + expected_features
    # Any missing feature must be 0
    for col in final_cols:
        if col not in states.columns:
            states[col] = 0.0
            
    return states[final_cols]

def create_causal_sequences(
    states: pd.DataFrame, 
    expected_features: List[str], 
    seq_length: int = 20
) -> Tuple[np.ndarray, List[pd.Timestamp]]:
    """Creates strictly causal rolling windows of size `seq_length`.
    
    For a given index t, the sequence is [t - seq_length + 1, ..., t].
    Only states that have `seq_length` historical windows are yielded.
    
    Returns:
        X: numpy array of shape (num_sequences, seq_length, num_features)
        timestamps: List of the timestamp of the LAST window in the sequence (time t)
    """
    if len(states) < seq_length:
        return np.empty((0, seq_length, len(expected_features))), []

    # Ensure ordering
    states = states.sort_values('Timestamp').reset_index(drop=True)
    
    feature_matrix = states[expected_features].values
    timestamps = states['Timestamp'].tolist()
    
    X = []
    out_timestamps = []
    
    for i in range(seq_length - 1, len(states)):
        # Strictly slice up to and including the current window `i`. 
        # i.e., [i - seq_length + 1 : i + 1]
        seq = feature_matrix[i - seq_length + 1 : i + 1]
        X.append(seq)
        out_timestamps.append(timestamps[i])
        
    return np.array(X), out_timestamps
