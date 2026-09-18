import os
import sys
import json
import hashlib
import csv
from pathlib import Path
import re
import shutil

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

def parse_migration_plan():
    plan_path = ROOT_DIR / "docs" / "FINAL_MIGRATION_PLAN.md"
    files = []
    if not plan_path.exists():
        print("MIGRATION PLAN NOT FOUND")
        sys.exit(1)
        
    with open(plan_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
    in_table = False
    for line in lines:
        if line.startswith("| Current Path | Category"):
            in_table = True
            continue
        if in_table and line.startswith("|---"):
            continue
        if in_table and line.startswith("|"):
            parts = [p.strip() for p in line.split("|")]
            if len(parts) >= 7:
                current = parts[1]
                category = parts[2]
                pipeline = parts[3]
                action = parts[4]
                proposed = parts[5]
                risk = parts[6]
                files.append({
                    'current': current,
                    'category': category,
                    'action': action,
                    'proposed': proposed,
                    'risk': risk
                })
    return files

def find_references(mapping):
    # mapping is old_path (normalized to /) -> new_path
    extensions = ['.py', '.ipynb', '.ts', '.tsx', '.md', '.json', '.yaml']
    manifest = []
    
    for root, dirs, files in os.walk(ROOT_DIR):
        if '.git' in root or 'node_modules' in root or '__pycache__' in root or 'dist' in root:
            continue
            
        for f in files:
            if not any(f.endswith(ext) for ext in extensions):
                continue
                
            fpath = os.path.join(root, f)
            rel_path = os.path.relpath(fpath, ROOT_DIR).replace('\\', '/')
            
            try:
                with open(fpath, 'r', encoding='utf-8') as file:
                    lines = file.readlines()
                    
                for i, line in enumerate(lines):
                    for old_p, new_p in mapping.items():
                        # We also check filename only if it's unique enough, but full path is safer
                        if old_p in line or os.path.basename(old_p) in line:
                            # determine type
                            ref_type = "documentation-only reference"
                            if rel_path.endswith('.py') or rel_path.endswith('.ts') or rel_path.endswith('.tsx'):
                                ref_type = "executable dependency"
                            elif rel_path.endswith('.ipynb'):
                                ref_type = "notebook dependency"
                            elif rel_path.endswith('.json') or rel_path.endswith('.yaml'):
                                ref_type = "configuration dependency"
                            if 'test' in rel_path.lower():
                                ref_type = "test dependency"
                            if rel_path.startswith('docs/') or rel_path.startswith('results/'):
                                ref_type = "documentation-only reference"
                                if 'archive' in rel_path or 'legacy' in rel_path:
                                    ref_type = "archived reference"
                                    
                            action = "UPDATE"
                            if "documentation-only" in ref_type or "archived" in ref_type:
                                action = "IGNORE"
                                
                            manifest.append({
                                'file': rel_path,
                                'line/location': f"L{i+1}",
                                'old_reference': old_p,
                                'new_reference': new_p,
                                'reference_type': ref_type,
                                'reason': "Path changed in migration",
                                'action': action
                            })
            except Exception as e:
                pass
    return manifest

def main():
    dry_run = "--dry-run" in sys.argv
    print(f"Starting Migration Script (Dry Run: {dry_run})")
    
    os.makedirs(ROOT_DIR / 'docs' / 'migration', exist_ok=True)
    
    # 1. Parse Plan
    files = parse_migration_plan()
    
    # 2. Hash protected artifacts
    protected = [
        "models/production/model_SET_R_h20.pt",
        "models/production/scaler_SET_R_h20.joblib",
        "results/phase2_3/feature_sets.json",
        "results/phase2_3/champion_metrics.json"
    ]
    
    pre_hashes = {}
    for p in protected:
        h = get_hash(ROOT_DIR / p)
        pre_hashes[p] = h
        
    with open(ROOT_DIR / 'docs' / 'frozen_artifact_hashes_pre_cleanup.json', 'w') as f:
        json.dump(pre_hashes, f, indent=4)
        
    # 3. Create Pre-cleanup Manifest & Rollback Manifest
    pre_cleanup = []
    rollback = []
    mapping = {}
    
    for item in files:
        if item['action'] in ['MOVE', 'ARCHIVE']:
            curr = ROOT_DIR / item['current']
            if curr.exists():
                h = get_hash(curr)
                sz = os.path.getsize(curr)
                pre_cleanup.append({
                    'file_path': item['current'],
                    'file_size': sz,
                    'sha256': h,
                    'timestamp': os.path.getmtime(curr)
                })
                rollback.append({
                    'original_path': item['current'],
                    'new_path': item['proposed'],
                    'sha256': h,
                    'file_size': sz
                })
                mapping[item['current'].replace('\\', '/')] = item['proposed'].replace('\\', '/')
                
    with open(ROOT_DIR / 'docs' / 'migration' / 'pre_cleanup_manifest.json', 'w') as f:
        json.dump(pre_cleanup, f, indent=4)
        
    with open(ROOT_DIR / 'docs' / 'migration' / 'rollback_manifest.json', 'w') as f:
        json.dump(rollback, f, indent=4)
        
    # 4. Review Files
    review_files = [f for f in files if f['action'] == 'REVIEW']
    with open(ROOT_DIR / 'docs' / 'review_required_files.md', 'w', encoding='utf-8') as f:
        f.write("# Review Required Files\n\n")
        f.write("| PATH | TYPE | WHY UNCLASSIFIED | DEPENDENCIES | RECOMMENDED ACTION |\n")
        f.write("|------|------|------------------|--------------|--------------------|\n")
        for r in review_files:
            f.write(f"| {r['current']} | {r['category']} | Unmatched by core pipeline regex | TBD | Leave untouched |\n")
            
    # 5. Find References
    manifest = find_references(mapping)
    with open(ROOT_DIR / 'docs' / 'migration' / 'reference_update_manifest.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['file', 'line/location', 'old_reference', 'new_reference', 'reference_type', 'reason', 'action'])
        writer.writeheader()
        writer.writerows(manifest)
        
    # 6. Feature Pipeline Comparison
    os.makedirs(ROOT_DIR / 'docs' / 'pipelines', exist_ok=True)
    with open(ROOT_DIR / 'docs' / 'pipelines' / 'feature_pipeline_comparison.md', 'w', encoding='utf-8') as f:
        f.write("# Feature Pipeline Comparison\n\n")
        f.write("## archive/archive\\cybercast_pipeline.py (Offline/Training)\n")
        f.write("- **Usage**: Phase 1 & 2 offline dataset processing.\n")
        f.write("- **Normalization**: Uses standard pandas/scikit-learn scalers fit on entire dataset.\n")
        f.write("- **Status**: Legacy (Not used in live inference).\n\n")
        f.write("## src/inference/windowing.py (Live/Production)\n")
        f.write("- **Usage**: Pipeline B/C/D for dynamic CSV and live Npcap inference.\n")
        f.write("- **Normalization**: Real-time sliding window aggregation using loaded `scaler_SET_R_h20.joblib`.\n")
        f.write("- **Status**: Production.\n\n")
        f.write("**Conclusion**: Do not merge. They serve different epochs of the project. Archive archive/archive\\cybercast_pipeline.py.")
        
    # 7. Generate Dry Run Report
    with open(ROOT_DIR / 'docs' / 'migration' / 'dry_run_report.md', 'w', encoding='utf-8') as f:
        f.write("# Dry Run Report\n\n")
        f.write("## Proposed Moves\n")
        f.write("```text\n")
        for item in files:
            if item['action'] in ['MOVE', 'ARCHIVE']:
                f.write(f"{item['action']:7} | {item['current']} -> {item['proposed']}\n")
        f.write("```\n")
        f.write(f"\n## Reference Updates Identified: {len([m for m in manifest if m['action'] == 'UPDATE'])}\n")
        f.write("Check `reference_update_manifest.csv` for details.\n")
        
    if dry_run:
        print("Dry run complete. Check docs/migration/dry_run_report.md for details.")
        return
        
    print("EXECUTING MIGRATION...")
    
    # Execute Moves
    moved_count = 0
    for item in files:
        if item['action'] in ['MOVE', 'ARCHIVE']:
            src = ROOT_DIR / item['current']
            dst = ROOT_DIR / item['proposed']
            if src.exists():
                os.makedirs(dst.parent, exist_ok=True)
                shutil.move(str(src), str(dst))
                moved_count += 1
                
    # Execute Updates
    updated_count = 0
    updated_files = set()
    for m in manifest:
        if m['action'] == 'UPDATE':
            fpath = ROOT_DIR / m['file']
            if fpath.exists():
                updated_files.add(fpath)
                
    for fpath in updated_files:
        try:
            with open(fpath, 'r', encoding='utf-8') as f:
                content = f.read()
            # Replace old paths with new paths
            for old_p, new_p in mapping.items():
                content = content.replace(old_p, new_p)
                # also handle posix vs windows slash in imports occasionally if needed
                content = content.replace(old_p.replace('/', '\\\\'), new_p.replace('/', '\\\\'))
                
            with open(fpath, 'w', encoding='utf-8') as f:
                f.write(content)
            updated_count += 1
        except Exception as e:
            print(f"Failed to update references in {fpath}: {e}")
            
    print(f"Migration Complete: Moved {moved_count} files, Updated {updated_count} files with new references.")

if __name__ == "__main__":
    main()
