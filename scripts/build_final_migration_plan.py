import os
import re
from pathlib import Path

ROOT_DIR = Path("d:/working_projects/SIH/cyberCast")
os.chdir(ROOT_DIR)

def classify_file(path_str):
    p = path_str.lower()
    
    # Ignore specific paths completely for classification summary if needed, but we must list them
    if '.git' in p or 'node_modules' in p or '__pycache__' in p or '.pytest_cache' in p:
        return None, None, None
        
    category = "UNKNOWN"
    action = "REVIEW"
    pipeline = "None"
    
    if "dashboard" in p and ("src" in p or "public" in p or "package" in p):
        category = "DASHBOARD"
        action = "KEEP"
        pipeline = "PIPELINE F"
    elif "test" in p and (p.endswith('.py') or "test" in p.split('\\')[-1]):
        category = "TEST"
        action = "KEEP"
    elif p.endswith(".md"):
        category = "DOCUMENTATION"
        action = "KEEP"
    elif "data" in p and (p.endswith('.csv') or p.endswith('.parquet') or p.endswith('.pcap')):
        category = "DATA"
        action = "KEEP"
    elif "models" in p and (p.endswith('.pt') or p.endswith('.joblib') or p.endswith('.pkl')):
        category = "MODEL"
        if "model_set_r_h20" in p or "scaler_set_r_h20" in p:
            action = "MOVE"
        else:
            action = "ARCHIVE"
    elif "notebooks" in p or "nootebooks" in p or p.endswith(".ipynb"):
        category = "NOTEBOOK"
        action = "ARCHIVE"
    elif "results" in p:
        category = "RESULT"
        action = "KEEP"
    elif "scratch" in p or "dump" in p or p.endswith(".ps1") or "fix" in p.split('\\')[-1]:
        category = "TEMPORARY"
        action = "ARCHIVE"
    elif "src" in p:
        if "live" in p:
            category = "PRODUCTION"
            pipeline = "PIPELINE D"
            action = "KEEP"
        elif "inference" in p:
            category = "PRODUCTION"
            pipeline = "PIPELINE B/C"
            action = "KEEP"
        elif "api" in p:
            category = "PRODUCTION"
            pipeline = "PIPELINE F"
            action = "KEEP"
        elif "forecasting" in p or "explainability" in p:
            category = "PRODUCTION"
            pipeline = "PIPELINE E"
            action = "KEEP"
        else:
            category = "RESEARCH"
            action = "MOVE"
    elif "scripts" in p:
        if "audit" in p or "analyze" in p or "reproducibility" in p:
            category = "RESEARCH"
            action = "KEEP"
        else:
            category = "EXPERIMENT"
            action = "ARCHIVE"
    elif p.endswith("requirements.txt") or p.endswith(".json") and "configs" in p:
        category = "CONFIG"
        action = "KEEP"
    elif "archive/archive\\cybercast_pipeline.py" in p or "create_notebook" in p or "gen_phase" in p:
        category = "LEGACY"
        action = "ARCHIVE"
        
    return category, action, pipeline

def build_map():
    all_files = []
    for root, dirs, files in os.walk(ROOT_DIR):
        for f in files:
            full_path = os.path.join(root, f)
            rel_path = os.path.relpath(full_path, ROOT_DIR)
            cat, act, pip = classify_file(rel_path)
            if cat:
                all_files.append({
                    'path': rel_path,
                    'category': cat,
                    'action': act,
                    'pipeline': pip,
                    'proposed_path': 'TBD',
                    'dependencies': 'TBD',
                    'risk': 'LOW'
                })
                
    # Refine proposed paths
    for f in all_files:
        p = f['path'].replace('\\', '/')
        if f['category'] == 'MODEL':
            if "model_SET_R_h20" in p or "scaler_SET_R_h20" in p:
                f['proposed_path'] = f"models/production/{os.path.basename(p)}"
                f['risk'] = 'HIGH'
            else:
                f['proposed_path'] = f"models/archive/{os.path.basename(p)}"
        elif f['category'] == 'TEMPORARY' or f['category'] == 'LEGACY':
            f['proposed_path'] = f"archive/{os.path.basename(p)}"
        elif f['category'] == 'NOTEBOOK':
            f['proposed_path'] = f"notebooks/archive/{os.path.basename(p)}"
        else:
            f['proposed_path'] = p
            
    os.makedirs(ROOT_DIR / 'docs', exist_ok=True)
    
    # docs/FINAL_REPOSITORY_ARCHITECTURE.md
    with open(ROOT_DIR / 'docs' / 'FINAL_REPOSITORY_ARCHITECTURE.md', 'w', encoding='utf-8') as f:
        f.write("# Final Repository Architecture\n\n")
        f.write("```text\n")
        f.write("cyberCast/\n")
        f.write("├── archive/          # Legacy, duplicates, and temporary scripts\n")
        f.write("├── assets/           # Static assets, images for docs\n")
        f.write("├── configs/          # Configuration files\n")
        f.write("├── dashboard/        # React + FastAPI dashboard\n")
        f.write("├── data/             # PCAP, CSV, and Parquet data files\n")
        f.write("├── docs/             # Markdown documentation\n")
        f.write("├── models/           \n")
        f.write("│   ├── production/   # Frozen Phase 2.3 champion and scaler\n")
        f.write("│   ├── baselines/    # Baseline scientific models\n")
        f.write("│   ├── experiments/  # Hyperparameter and feature set experiments\n")
        f.write("│   └── legacy/       # Old Phase 1 / Phase 2.1 models\n")
        f.write("├── notebooks/        # Jupyter notebooks (Research)\n")
        f.write("├── results/          # JSON metrics, CSV reports, analysis artifacts\n")
        f.write("├── scripts/          # Standalone python scripts for execution\n")
        f.write("├── src/              # Production Source Code\n")
        f.write("│   ├── api/          # FastAPI backend\n")
        f.write("│   ├── explainability/ # SHAP/LIME logic\n")
        f.write("│   ├── forecasting/  # Pre-attack evaluation logic\n")
        f.write("│   ├── inference/    # Offline/CSV/Model loading logic\n")
        f.write("│   └── live/         # Npcap live capture logic\n")
        f.write("└── tests/            # Pytest suite\n")
        f.write("```\n")
        f.write("\n## Root Directory Policy\n")
        f.write("Only the following files are allowed at the repository root:\n")
        f.write("- README.md\n- requirements.txt\n- .gitignore\n")
        
    # docs/FINAL_MIGRATION_PLAN.md
    with open(ROOT_DIR / 'docs' / 'FINAL_MIGRATION_PLAN.md', 'w', encoding='utf-8') as f:
        f.write("# Final Migration Plan\n\n")
        f.write("## Migration Sequence\n")
        f.write("1. **Create destination directories** (e.g. `archive/`, `models/production/`)\n")
        f.write("2. **Move documentation** into `docs/`\n")
        f.write("3. **Move research scripts** to `scripts/` or `archive/`\n")
        f.write("4. **Move experiment/baseline/legacy models** to respective `models/` subdirectories\n")
        f.write("5. **Move production model** to `models/production/`\n")
        f.write("6. **Update imports and paths** in `src/` and `dashboard/`\n")
        f.write("7. **Update configuration references**\n")
        f.write("8. **Run tests**\n")
        f.write("9. **Run dashboard build**\n")
        f.write("10. **Verify production model chain**\n")
        f.write("11. **Verify hashes**\n")
        f.write("12. **Verify no broken references**\n\n")
        
        f.write("## Complete File Map\n")
        f.write("| Current Path | Category | Pipeline | Action | Proposed Path | Risk |\n")
        f.write("|--------------|----------|----------|--------|---------------|------|\n")
        for fi in all_files:
            f.write(f"| {fi['path']} | {fi['category']} | {fi['pipeline']} | {fi['action']} | {fi['proposed_path']} | {fi['risk']} |\n")

    # docs/FINAL_PIPELINE_MAP.md
    with open(ROOT_DIR / 'docs' / 'FINAL_PIPELINE_MAP.md', 'w', encoding='utf-8') as f:
        f.write("# Final Pipeline Map\n\n")
        f.write("## PIPELINE A: Offline training/evaluation\n")
        f.write("Data -> Preprocessing -> Model Training -> Evaluation -> Metrics\n")
        f.write("*Primary Source:* `notebooks/` and `scripts/analyze_*.py`\n\n")
        
        f.write("## PIPELINE B: CSV inference\n")
        f.write("CSV File -> Windowing -> Feature Scaling -> LSTM Model -> Probability\n")
        f.write("*Primary Source:* `src/inference/`\n\n")
        
        f.write("## PIPELINE C: PCAP inference\n")
        f.write("PCAP File -> CICFlowMeter -> CSV -> Pipeline B\n")
        f.write("*Primary Source:* `src/inference/`\n\n")
        
        f.write("## PIPELINE D: Live Npcap inference\n")
        f.write("Live Traffic -> Npcap Sniffer -> Feature Extractor -> Model Inference -> Live Alerts\n")
        f.write("*Primary Source:* `src/live/`\n\n")
        
        f.write("## PIPELINE E: Forecasting / ATT&CK / explainability\n")
        f.write("Inference Probability -> Pre-Attack Window Detection -> ATT&CK Mapping -> SHAP Explainer\n")
        f.write("*Primary Source:* `src/forecasting/`, `src/explainability/`\n\n")
        
        f.write("## PIPELINE F: API -> dashboard\n")
        f.write("Backend Live State -> FastAPI Websockets/REST -> React Frontend Dashboard\n")
        f.write("*Primary Source:* `src/api/`, `dashboard/`\n\n")
        
        f.write("## Pipeline Duplication Analysis\n")
        f.write("1. **Feature Extraction**: `archive/archive\\cybercast_pipeline.py` vs `src/inference/windowing.py`\n")
        f.write("   - *Reason*: `archive/archive\\cybercast_pipeline.py` was used for offline phase 1/2 training, while `windowing.py` is for production inference.\n")
        f.write("   - *Action*: Archive `archive/archive\\cybercast_pipeline.py` and consolidate logic into `src/inference/windowing.py`.\n")
        
    # docs/FINAL_MODEL_MAP.md
    with open(ROOT_DIR / 'docs' / 'FINAL_MODEL_MAP.md', 'w', encoding='utf-8') as f:
        f.write("# Final Model Map\n\n")
        f.write("## Production Chain Provenance\n")
        f.write("INPUT -> SET_R 89 features -> frozen scaler (`models/production/scaler_SET_R_h20.joblib`) -> 20x89 sequence -> Phase 2.3 LSTM (`models/production/model_SET_R_h20.pt`) -> probability -> risk -> forecast -> explainability -> ATT&CK -> API/dashboard\n\n")
        f.write("## Artifact Protection Status\n")
        f.write("- **Phase 2.3 production model**: PROTECTED\n")
        f.write("- **Phase 2.3 scaler**: PROTECTED\n")
        f.write("- **Phase 2.3 feature configuration**: PROTECTED\n")
        f.write("- **Phase 2.3 champion metrics**: PROTECTED\n")
        
    # Console Report
    total = len(all_files)
    prod_files = sum(1 for f in all_files if f['category'] == 'PRODUCTION')
    res_files = sum(1 for f in all_files if f['category'] == 'RESEARCH')
    exp_files = sum(1 for f in all_files if f['category'] == 'EXPERIMENT')
    base_files = sum(1 for f in all_files if f['category'] == 'BASELINE')
    leg_files = sum(1 for f in all_files if f['category'] == 'LEGACY')
    dup_files = sum(1 for f in all_files if f['category'] == 'DUPLICATE')
    temp_files = sum(1 for f in all_files if f['category'] == 'TEMPORARY')
    man_rev = sum(1 for f in all_files if f['action'] == 'REVIEW')
    
    print("==================================================")
    print("MIGRATION PLAN GENERATED")
    print("==================================================")
    print(f"total files reviewed: {total}")
    print(f"production files: {prod_files}")
    print(f"research files: {res_files}")
    print(f"experiment files: {exp_files}")
    print(f"baseline files: {base_files}")
    print(f"legacy files: {leg_files}")
    print(f"duplicates: {dup_files}")
    print(f"temporary files: {temp_files}")
    print(f"files requiring manual review: {man_rev}")
    print(f"pipeline duplication: 1 identified (Feature extraction in archive/archive\\cybercast_pipeline.py vs src/)")
    print(f"proposed production pipeline: Unified src/ pipeline powering Pipeline D, E, F")
    print(f"proposed model structure: models/production/ isolated from models/archive/")
    print(f"proposed repository structure: Clean top-level SRC, DASHBOARD, DOCS, MODELS, ARCHIVE")

if __name__ == "__main__":
    build_map()
