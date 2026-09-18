import pandas as pd
import json

path = r'D:\working_projects\SIH\cyberCast\data\processed\network_states_10s.parquet'
df = pd.read_parquet(path)

print(f'1. Windows: {len(df)}')
print(f'2. Start: {df["Timestamp"].min()} End: {df["Timestamp"].max()}')
print(f'3. Ordered: {df["Timestamp"].is_monotonic_increasing}')
print(f'4. Interval (median seconds): {df["Timestamp"].diff().median().total_seconds()}')

if 'flow_count' in df.columns:
    print(f'5. Empty windows: {(df["flow_count"]==0).sum()}')
else:
    print('5. Empty windows: flow_count column missing')

n_atk = df['binary_attack'].sum() if 'binary_attack' in df.columns else 0
print(f'6. Attack: {n_atk}')
print(f'7. Benign: {len(df) - n_atk}')

cols = list(df.columns)
print(f'8. Total cols: {len(cols)}')
print(f'9. Features: {cols[:5]} ... {cols[-5:]}')

leaks = ['Label', 'binary_attack', 'attack_flow_count', 'attack_ratio', 'dominant_label', 'attack_family']
found_leaks = [c for c in cols if c in leaks]
print(f'10. Leakage cols in df: {found_leaks}')

print("Inspecting training/scaling setup from config...")
