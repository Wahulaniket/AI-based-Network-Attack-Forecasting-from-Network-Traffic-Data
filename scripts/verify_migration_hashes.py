import os
import json
import hashlib
from pathlib import Path
import traceback

ROOT_DIR = Path("d:/working_projects/SIH/cyberCast")
os.chdir(ROOT_DIR)

def get_hash(filepath):
    if not os.path.exists(filepath):
        return None
    h = hashlib.sha256()
    with open(filepath, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b""):
            h.update(chunk)
    return h.hexdigest()

def verify():
    # 1. Check hashes
    with open(ROOT_DIR / 'docs' / 'frozen_artifact_hashes_pre_cleanup.json', 'r') as f:
        pre_hashes = json.load(f)
        
    # Map old paths to new paths
    new_paths = {
        "models/phase2_3/model_SET_R_h20.pt": "models/production/model_SET_R_h20.pt",
        "models/phase2_3/scaler_SET_R_h20.joblib": "models/production/scaler_SET_R_h20.joblib",
        "results/phase2_3/feature_sets.json": "results/phase2_3/feature_sets.json",
        "results/phase2_3/champion_metrics.json": "results/phase2_3/champion_metrics.json"
    }
    
    post_hashes = {}
    mismatches = []
    
    for old_path, expected_hash in pre_hashes.items():
        new_path = new_paths[old_path]
        full_path = ROOT_DIR / new_path
        h = get_hash(full_path)
        post_hashes[new_path] = h
        if h != expected_hash:
            mismatches.append(f"{new_path} hash changed! Expected: {expected_hash}, Got: {h}")
            
    with open(ROOT_DIR / 'docs' / 'frozen_artifact_hashes_post_cleanup.json', 'w') as f:
        json.dump(post_hashes, f, indent=4)
        
    print("Hash Verification:")
    if mismatches:
        for m in mismatches:
            print("ERROR: ", m)
    else:
        print("PASS: All protected artifact hashes are identical to pre-cleanup state.")

    # 2. Smoke Test Loading
    print("\nSmoke Test Loading:")
    try:
        import torch
        import joblib
        
        # Load Model
        model_path = ROOT_DIR / new_paths["models/phase2_3/model_SET_R_h20.pt"]
        # In a real environment, we'd load the weights via the class. We'll just check torch.load
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=True)
        print("PASS: Model loaded via torch.")
        
        # Load Scaler
        scaler_path = ROOT_DIR / new_paths["models/phase2_3/scaler_SET_R_h20.joblib"]
        scaler = joblib.load(scaler_path)
        print("PASS: Scaler loaded via joblib.")
        
        # Load JSON
        with open(ROOT_DIR / new_paths["results/phase2_3/feature_sets.json"], 'r') as f:
            fs = json.load(f)
            assert 'SET_R' in fs, "SET_R missing in feature sets"
        print("PASS: Feature config loaded.")
        
    except Exception as e:
        print("ERROR loading protected artifacts:")
        traceback.print_exc()

if __name__ == '__main__':
    verify()
