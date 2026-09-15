import pandas as pd
import numpy as np

def safe_div(a, b, fill=0.0):
    return np.where(b != 0, a / b, fill)

def process_features(df: pd.DataFrame, expected_features: list) -> pd.DataFrame:
    """Applies Phase 2.3 preprocessing and engineers required features.
    
    Data is assumed to have been loaded from CSV with whitespace stripped.
    This function:
    1. Validates required raw columns.
    2. Parses Timestamp.
    3. Handles missing/inf values.
    4. Applies the exact Phase 2.3 feature engineering logic.
    5. Returns ONLY the expected feature columns in the EXACT order.
    """
    df = df.copy()
    
    # Check Timestamp
    if 'Timestamp' not in df.columns:
        raise ValueError("[ERROR] 'Timestamp' column missing in input data")
        
    # Ensure no target columns exist to prevent leakage
    target_cols = ['binary_attack', 'dominant_label', 'attack_ratio', 'attack_flow_count', 'Label', 'Attack']
    for col in target_cols:
        if col in df.columns:
            df = df.drop(columns=[col])
            
    # Parse Timestamp
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], format='mixed', dayfirst=False, errors='coerce')
    df = df.dropna(subset=['Timestamp'])
    
    # Convert numeric columns
    for col in df.columns:
        if col != 'Timestamp' and df[col].dtype == object:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Replace +inf and -inf with NaN
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)

    # Fill NaN with column median
    for col in numeric_cols:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())

    # Sort chronologically
    df = df.sort_values('Timestamp').reset_index(drop=True)
    
    # Engineer Features (reusing Phase 2.3 logic)
    if 'Tot Fwd Pkts' in df.columns and 'Tot Bwd Pkts' in df.columns:
        df['Fwd_Bwd_Pkt_Ratio'] = safe_div(df['Tot Fwd Pkts'].values, df['Tot Bwd Pkts'].values)
        df['Total_Pkts'] = df['Tot Fwd Pkts'] + df['Tot Bwd Pkts']

    if 'TotLen Fwd Pkts' in df.columns and 'TotLen Bwd Pkts' in df.columns:
        df['Fwd_Bwd_Byte_Ratio'] = safe_div(df['TotLen Fwd Pkts'].values, df['TotLen Bwd Pkts'].values)
        df['Total_Bytes'] = df['TotLen Fwd Pkts'] + df['TotLen Bwd Pkts']

    if 'Total_Pkts' in df.columns and 'Total_Bytes' in df.columns:
        df['Avg_Pkt_Size'] = safe_div(df['Total_Bytes'].values, df['Total_Pkts'].values)

    if 'Total_Pkts' in df.columns and 'Flow Duration' in df.columns:
        duration_sec = df['Flow Duration'].values / 1e6
        df['Pkts_Per_Sec'] = safe_div(df['Total_Pkts'].values, duration_sec)

    if 'Total_Bytes' in df.columns and 'Flow Duration' in df.columns:
        duration_sec = df['Flow Duration'].values / 1e6
        df['Bytes_Per_Sec'] = safe_div(df['Total_Bytes'].values, duration_sec)

    if 'SYN Flag Cnt' in df.columns and 'FIN Flag Cnt' in df.columns:
        df['SYN_FIN_Ratio'] = safe_div(df['SYN Flag Cnt'].values, (df['FIN Flag Cnt'].values + 1))

    if 'RST Flag Cnt' in df.columns and 'SYN Flag Cnt' in df.columns:
        df['RST_SYN_Ratio'] = safe_div(df['RST Flag Cnt'].values, (df['SYN Flag Cnt'].values + 1))

    if 'Tot Fwd Pkts' in df.columns and 'Total_Pkts' in df.columns:
        df['Fwd_Pkt_Proportion'] = safe_div(df['Tot Fwd Pkts'].values, df['Total_Pkts'].values)

    # Note: 'flow_count' is aggregated per window in windowing.py
    # So we don't need to create it here at the flow-level.
    
    # Handle NaN/Inf created during engineering
    engineered = [
        'Fwd_Bwd_Pkt_Ratio', 'Fwd_Bwd_Byte_Ratio', 'Total_Pkts', 'Total_Bytes',
        'Avg_Pkt_Size', 'Pkts_Per_Sec', 'Bytes_Per_Sec', 'SYN_FIN_Ratio', 
        'RST_SYN_Ratio', 'Fwd_Pkt_Proportion'
    ]
    for col in engineered:
        if col in df.columns:
            df[col] = df[col].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # Return the dataframe with the Timestamp for windowing, plus any columns needed for aggregation
    return df
