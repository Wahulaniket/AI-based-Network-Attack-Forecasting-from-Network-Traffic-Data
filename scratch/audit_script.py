import os
import sys
import json
import time
import datetime
import pickle
import torch
import pandas as pd
import numpy as np

print("="*80)
print("CYBERCAST IMPLEMENTATION AUDIT SCRIPT")
print("="*80)

# Paths
BASE_DIR = r"d:\working_projects\SIH\cyberCast"
MODEL_PATH = os.path.join(BASE_DIR, "models", "best_world_model.pt")
CONFIG_PATH = os.path.join(BASE_DIR, "models", "config.json")
FEATURES_PATH = os.path.join(BASE_DIR, "models", "feature_names.json")
SCALER_PATH = os.path.join(BASE_DIR, "models", "scaler.pkl")
LR_PATH = os.path.join(BASE_DIR, "models", "logistic_regression.pkl")
NB_PATH = os.path.join(BASE_DIR, "nootebooks", "CyberCast_Complete.ipynb")
PIPELINE_PATH = os.path.join(BASE_DIR, "cybercast_pipeline.py")

# 1. SEQUENCE & CHECKPOINT AUDIT
print("\n--- 1. CHECKPOINT & SEQUENCE CONFIGURATION AUDIT ---")
if os.path.exists(MODEL_PATH):
    mtime = os.path.getmtime(MODEL_PATH)
    mtime_str = datetime.datetime.fromtimestamp(mtime).strftime('%Y-%m-%d %H:%M:%S')
    size_bytes = os.path.getsize(MODEL_PATH)
    print(f"Checkpoint Path: {MODEL_PATH}")
    print(f"Modification Time: {mtime_str}")
    print(f"File Size: {size_bytes:,} bytes")
    
    state_dict = torch.load(MODEL_PATH, map_location='cpu')
    print("\nState Dict Keys & Shapes:")
    total_params = 0
    for k, v in state_dict.items():
        print(f"  {k:35s}: shape {list(v.shape)}, dtype {v.dtype}, numel {v.numel():,}")
        total_params += v.numel()
    print(f"Total Parameter Count in Checkpoint: {total_params:,}")
else:
    print(f"ERROR: Checkpoint missing at {MODEL_PATH}")

if os.path.exists(CONFIG_PATH):
    with open(CONFIG_PATH, 'r') as f:
        cfg = json.load(f)
    print("\nConfig JSON Contents:")
    for k, v in cfg.items():
        print(f"  {k}: {v}")
else:
    print(f"ERROR: Config JSON missing at {CONFIG_PATH}")

# Check feature names
if os.path.exists(FEATURES_PATH):
    with open(FEATURES_PATH, 'r') as f:
        fnames = json.load(f)
    print(f"\nFeature Names Count in JSON: {len(fnames)}")
else:
    print(f"ERROR: Feature names JSON missing at {FEATURES_PATH}")

print("\n--- AUDIT COMPLETE FOR STEP 1 ---")
