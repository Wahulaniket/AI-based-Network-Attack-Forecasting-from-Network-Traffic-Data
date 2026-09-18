import os
import glob
import json
import hashlib
import re
from pathlib import Path

ROOT_DIR = Path("d:/working_projects/SIH/cyberCast")
os.chdir(ROOT_DIR)

def get_hash(filepath):
    h = hashlib.sha256()
    with open(filepath, 'rb') as f:
        # read in chunks to avoid memory issues for large files
        for chunk in iter(lambda: f.read(4096), b""):
            h.update(chunk)
    return h.hexdigest()

def find_references(filename):
    refs = {'src': False, 'dashboard': False, 'notebooks': False, 'scripts': False, 'root': False}
    # basic grep over all files
    extensions = ['.py', '.ipynb', '.ts', '.tsx', '.json', '.txt', '.md']
    for root, dirs, files in os.walk(ROOT_DIR):
        if '.git' in root or '__pycache__' in root or 'node_modules' in root:
            continue
        rel_root = Path(root).relative_to(ROOT_DIR).parts
        for f in files:
            if not any(f.endswith(ext) for ext in extensions):
                continue
            fpath = os.path.join(root, f)
            try:
                with open(fpath, 'r', encoding='utf-8') as file:
                    content = file.read()
                    if filename in content:
                        if len(rel_root) > 0:
                            if rel_root[0] == 'src': refs['src'] = True
                            elif rel_root[0] == 'dashboard': refs['dashboard'] = True
                            elif rel_root[0] in ['notebooks', 'nootebooks']: refs['notebooks'] = True
                            elif rel_root[0] == 'scripts': refs['scripts'] = True
                            else: refs['root'] = True
                        else:
                            refs['root'] = True
            except:
                pass
    return refs

def audit():
    patterns = ['*.pt', '*.pth', '*.joblib', '*.pkl', '*.onnx', '*.safetensors', '*.h5', '*.keras']
    artifacts = []
    
    for p in patterns:
        for fpath in ROOT_DIR.rglob(p):
            if '.git' in fpath.parts or '__pycache__' in fpath.parts:
                continue
            artifacts.append(fpath)
            
    inventory = []
    seen_hashes = {}
    
    for fpath in artifacts:
        filename = fpath.name
        rel_path = fpath.relative_to(ROOT_DIR).as_posix()
        size = os.path.getsize(fpath)
        sha = get_hash(fpath)
        
        refs = find_references(filename)
        
        # Classification logic
        classification = "UNKNOWN"
        if sha in seen_hashes:
            classification = "DUPLICATE"
        elif "model_SET_R_h20.pt" in filename or "scaler_SET_R_h20.joblib" in filename:
            classification = "PRODUCTION_FROZEN"
        elif "phase2_3" in rel_path and ("SET_A" in filename or "SELECT" in filename):
            classification = "BASELINE"
        elif "phase2_3" in rel_path and ("Repro" in filename or "h10" in filename or "h30" in filename or "h5" in filename):
            classification = "EXPERIMENTAL"
        elif "phase2_2" in rel_path:
            classification = "EXPERIMENTAL"
        elif "phase2_1" in rel_path:
            classification = "LEGACY"
        elif "results" not in rel_path and "models" in rel_path:
            if "best_world_model" in filename or "cybercast_best" in filename or "logistic_regression" in filename or "feature_scaler" in filename or filename == "scaler.pkl":
                if any(refs.values()):
                    classification = "LEGACY"
                else:
                    classification = "ORPHAN"
                    
        if classification != "DUPLICATE":
            seen_hashes[sha] = rel_path
            
        # Parse configurations from filename for phase2 models
        feature_set = "Unknown"
        history_length = "Unknown"
        phase = "Unknown"
        
        if "SET_A" in filename: feature_set = "SET_A"
        elif "SET_B" in filename: feature_set = "SET_B"
        elif "SET_C" in filename: feature_set = "SET_C"
        elif "SET_R_SELECT" in filename: feature_set = "SET_R_SELECT"
        elif "SET_R" in filename: feature_set = "SET_R"
        
        h_match = re.search(r'h(\d+)', filename)
        if h_match: history_length = h_match.group(1)
        elif "_10_" in filename: history_length = "10"
        elif "_20_" in filename: history_length = "20"
        elif "_30_" in filename: history_length = "30"
        elif "_5_" in filename: history_length = "5"
        
        if "phase2_3" in rel_path: phase = "Phase 2.3"
        elif "phase2_2" in rel_path: phase = "Phase 2.2"
        elif "phase2_1" in rel_path: phase = "Phase 2.1"
        else: phase = "Phase 1 / Legacy"
        
        inventory.append({
            'current_path': rel_path,
            'filename': filename,
            'file_size': size,
            'SHA256': sha,
            'architecture': 'LSTM' if '.pt' in filename else 'Scaler' if '.joblib' in filename or '.pkl' in filename else 'Unknown',
            'phase': phase,
            'experiment': classification,
            'feature_set': feature_set,
            'history_length': history_length,
            'referenced_production': refs['src'],
            'referenced_dashboard': refs['dashboard'],
            'referenced_notebooks': refs['notebooks'],
            'classification': classification
        })
        
    os.makedirs(ROOT_DIR / 'docs' / 'models', exist_ok=True)
    
    import csv
    with open(ROOT_DIR / 'docs' / 'models' / 'model_inventory.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=inventory[0].keys())
        writer.writeheader()
        writer.writerows(inventory)
        
    with open(ROOT_DIR / 'docs' / 'models' / 'model_inventory.md', 'w') as f:
        f.write("# Model Inventory\n\n")
        f.write("| Filename | Phase | Classification | Feature Set | History | SHA256 |\n")
        f.write("|----------|-------|----------------|-------------|---------|--------|\n")
        for item in inventory:
            f.write(f"| {item['filename']} | {item['phase']} | {item['classification']} | {item['feature_set']} | {item['history_length']} | {item['SHA256'][:8]}... |\n")
            
    with open(ROOT_DIR / 'docs' / 'models' / 'model_provenance.md', 'w') as f:
        f.write("# Model Provenance\n\n")
        f.write("## Overview\nThis document maps the evolution of models in the CyberCast repository.\n\n")
        f.write("## Production Frozen\n")
        for item in inventory:
            if item['classification'] == 'PRODUCTION_FROZEN':
                f.write(f"- `{item['current_path']}`: The final Phase 2.3 LSTM champion.\n")
        f.write("\n## Baselines\nModels retained for scientific comparison (e.g. SET_A, SET_R_SELECT).\n")
        for item in inventory:
            if item['classification'] == 'BASELINE':
                f.write(f"- `{item['current_path']}`\n")
        f.write("\n## Experimental\nModels trained during architecture/hyperparameter search (Phase 2.2 and Phase 2.3 history/reproducibility experiments).\n")
        f.write("These should be archived, not deleted, to preserve scientific reproducibility.\n")
        
        f.write("\n## Legacy & Orphans\nOld models from Phase 1/2.1 or models with no references in code. Can be safely archived to a legacy folder.\n")
        
    with open(ROOT_DIR / 'docs' / 'models' / 'model_migration_plan.md', 'w') as f:
        f.write("# Model Migration Plan\n\n")
        f.write("| Current Path | Proposed Path | Classification | Action | Reason |\n")
        f.write("|--------------|---------------|----------------|--------|--------|\n")
        for item in inventory:
            if item['classification'] == 'PRODUCTION_FROZEN':
                prop = f"models/production/{item['filename']}"
                action = "MOVE"
                reason = "Isolate production artifacts"
            elif item['classification'] in ['EXPERIMENTAL', 'BASELINE']:
                prop = f"models/archive/{item['phase'].replace(' ','_').lower()}/{item['filename']}"
                action = "ARCHIVE"
                reason = "Preserve for reproducibility"
            elif item['classification'] == 'DUPLICATE':
                prop = "DELETE"
                action = "DELETE"
                reason = "Identical hash exists"
            else:
                prop = f"models/archive/legacy/{item['filename']}"
                action = "ARCHIVE"
                reason = "Clean up root/models directory"
            f.write(f"| {item['current_path']} | {prop} | {item['classification']} | {action} | {reason} |\n")
            
    # Final Report to stdout
    prod_count = sum(1 for x in inventory if x['classification'] == 'PRODUCTION_FROZEN')
    exp_count = sum(1 for x in inventory if x['classification'] == 'EXPERIMENTAL')
    base_count = sum(1 for x in inventory if x['classification'] == 'BASELINE')
    dup_count = sum(1 for x in inventory if x['classification'] == 'DUPLICATE')
    orphan_count = sum(1 for x in inventory if x['classification'] == 'ORPHAN')
    legacy_count = sum(1 for x in inventory if x['classification'] == 'LEGACY')
    
    print("==================================================")
    print("FINAL REPORT")
    print("==================================================")
    print(f"Total model artifacts: {len(inventory)}")
    print(f"Production models: {prod_count}")
    print(f"Experimental models: {exp_count}")
    print(f"Baselines: {base_count}")
    print(f"Legacy models: {legacy_count}")
    print(f"Duplicates: {dup_count}")
    print(f"Orphans: {orphan_count}")
    print("Recommended canonical production model: models/production/model_SET_R_h20.pt")
    
if __name__ == '__main__':
    audit()
