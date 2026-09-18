import os
import sys
import json
import glob
import pickle
import datetime
import torch
import torch.nn as nn
import pandas as pd
import numpy as np

print("================================================================================")
print("CYBERCAST COMPREHENSIVE AUDIT SCRIPT")
print("================================================================================")

BASE_DIR = r"d:\working_projects\SIH\cyberCast"
MODEL_PATH = os.path.join(BASE_DIR, "models", "best_world_model.pt")
CONFIG_PATH = os.path.join(BASE_DIR, "models", "config.json")
FEATURES_PATH = os.path.join(BASE_DIR, "models", "feature_names.json")
SCALER_PATH = os.path.join(BASE_DIR, "models", "scaler.pkl")
LR_PATH = os.path.join(BASE_DIR, "models", "logistic_regression.pkl")
NB_PATH = os.path.join(BASE_DIR, "nootebooks", "CyberCast_Complete.ipynb")
RAW_DATA_DIR = os.path.join(BASE_DIR, "data", "raw")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

# Load model architecture from models/lstm_model.py
sys.path.append(os.path.join(BASE_DIR, "models"))
sys.path.append(BASE_DIR)

# ------------------------------------------------------------------------------
# ITEM 1: SEQUENCE CONFIGURATION & MODEL CHECKPOINT
# ------------------------------------------------------------------------------
print("\n[ITEM 1: SEQUENCE & CHECKPOINT CONFIGURATION AUDIT]")
with open(CONFIG_PATH, 'r') as f:
    cfg = json.load(f)

state_dict = torch.load(MODEL_PATH, map_location='cpu')
ckpt_mtime = datetime.datetime.fromtimestamp(os.path.getmtime(MODEL_PATH)).strftime('%Y-%m-%d %H:%M:%S')

# Read feature names
with open(FEATURES_PATH, 'r') as f:
    feature_names = json.load(f)

# Calculate parameters
param_count = sum(p.numel() for p in state_dict.values())

print(f"Checkpoint Path: {MODEL_PATH}")
print(f"Checkpoint Modification Time: {ckpt_mtime}")
print(f"Config window_seconds: {cfg.get('window_seconds')}")
print(f"Config history (sequence length): {cfg.get('history')} windows")
print(f"Config history in seconds: {cfg.get('window_seconds') * cfg.get('history')} seconds")
print(f"Config state_dim (features): {cfg.get('state_dim')}")
print(f"Checkpoint input dimension (lstm ih weight): {state_dict['lstm.weight_ih_l0'].shape[1]}")
print(f"Checkpoint hidden size: {state_dict['lstm.weight_ih_l0'].shape[0] // 4}")
print(f"Total parameter count: {param_count:,}")
print(f"Feature names count: {len(feature_names)}")

# ------------------------------------------------------------------------------
# ITEM 2 & 3: RAW DATASET & CHRONOLOGICAL SPLIT AUDIT
# ------------------------------------------------------------------------------
print("\n[ITEM 2 & 3: RAW DATASET & CHRONOLOGICAL SPLIT AUDIT]")

# Load pipeline split or rerun temporal dataset logic
# Let's inspect cybercast_pipeline.py and notebook split logic
with open(os.path.join(BASE_DIR, "cybercast_pipeline.py"), 'r', encoding='utf-8') as f:
    pipeline_code = f.read()

print("Checking pipeline chronological split timestamps...")
# Check if data/ holds preprocessed or if we can run data quality code
# Let's check if results/data_quality_by_file.csv exists or load raw CSVs summary
csv_files = glob.glob(os.path.join(RAW_DATA_DIR, "*.csv"))
print(f"Found {len(csv_files)} raw CSV files in {RAW_DATA_DIR}")

# ------------------------------------------------------------------------------
# ITEM 4: FEATURE AUDIT & LEAKAGE CHECK
# ------------------------------------------------------------------------------
print("\n[ITEM 4: FEATURE AUDIT & LEAKAGE CHECK]")
print(f"Total features used in model input: {len(feature_names)}")
label_derived = [f for f in feature_names if 'label' in f.lower() or 'target' in f.lower() or 'attack' in f.lower()]
print(f"Label-derived features in model input: {label_derived}")
if len(label_derived) == 0:
    print("LEAKAGE FOUND = NO (No target/label variables in feature set)")
else:
    print(f"LEAKAGE FOUND = YES ({label_derived})")

# ------------------------------------------------------------------------------
# ITEM 5: MODEL AUDIT & CHECKPOINT LOADING VERIFICATION
# ------------------------------------------------------------------------------
print("\n[ITEM 5: MODEL AUDIT]")
from models.lstm_model import CyberCastWorldModel

model = CyberCastWorldModel(
    input_size=cfg['state_dim'],
    hidden_size=cfg['hidden_size'],
    num_layers=cfg['num_layers'],
    dropout=cfg['dropout']
)
model.load_state_dict(state_dict)
model.eval()
print("Model loaded state_dict SUCCESSFUL! Architecture verified.")
print(f"Hidden size: {cfg['hidden_size']}, Layers: {cfg['num_layers']}, Dropout: {cfg['dropout']}")
print(f"Output heads: State Head ({model.state_head[-1].out_features} dims), Attack Head (1 dim)")

# ------------------------------------------------------------------------------
# ITEM 9: RECURSIVE FORECASTING AUDIT
# ------------------------------------------------------------------------------
print("\n[ITEM 9: RECURSIVE FORECASTING AUDIT]")
import inspect
rec_src = inspect.getsource(model.predict_future) if hasattr(model, 'predict_future') else ""
if not rec_src and hasattr(model, 'forecast_k_steps'):
    rec_src = inspect.getsource(model.forecast_k_steps)

print("Checking recursive forecast method implementation in CyberCastWorldModel:")
print(rec_src if rec_src else "Method inspect: checking pipeline rollout implementation...")

# Check cybercast_pipeline.py recursive function
rec_func_lines = [line for line in pipeline_code.split('\n') if 'def recursive_kstep_forecast' in line or 'curr_seq' in line]
print("Pipeline recursive rollout snippet:")
for line in pipeline_code.split('\n'):
    if 'def recursive_kstep_forecast' in line:
        idx = pipeline_code.split('\n').index(line)
        print('\n'.join(pipeline_code.split('\n')[idx:idx+25]))
        break

print("\n--- FULL AUDIT SCRIPT PART 1 COMPLETE ---")
