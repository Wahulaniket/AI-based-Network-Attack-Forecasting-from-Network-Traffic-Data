import pandas as pd
import numpy as np
import json
from pathlib import Path

def validate_dependencies():
    DATA_PATH = Path("results/phase3/state/S_t.parquet")
    RESULTS_DIR = Path("results/phase3/world_model")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    print("Loading S_t...")
    S_t = pd.read_parquet(DATA_PATH)
    
    # Define candidates
    candidates = {}
    
    # Total packets
    tot_pkts = S_t['Tot Fwd Pkts'] + S_t['Tot Bwd Pkts']
    tot_bytes = S_t['TotLen Fwd Pkts'] + S_t['TotLen Bwd Pkts']
    
    # Rates
    candidates['Flow Pkts/s'] = tot_pkts / 10.0
    candidates['Fwd Pkts/s'] = S_t['Tot Fwd Pkts'] / 10.0
    candidates['Bwd Pkts/s'] = S_t['Tot Bwd Pkts'] / 10.0
    candidates['Flow Byts/s'] = tot_bytes / 10.0
    
    # Means
    candidates['Fwd Pkt Len Mean'] = np.where(S_t['Tot Fwd Pkts'] > 0, S_t['TotLen Fwd Pkts'] / S_t['Tot Fwd Pkts'], 0.0)
    candidates['Bwd Pkt Len Mean'] = np.where(S_t['Tot Bwd Pkts'] > 0, S_t['TotLen Bwd Pkts'] / S_t['Tot Bwd Pkts'], 0.0)
    candidates['Pkt Len Mean'] = np.where(tot_pkts > 0, tot_bytes / tot_pkts, 0.0)
    
    # Ratio
    candidates['Down/Up Ratio'] = np.where(S_t['Tot Fwd Pkts'] > 0, S_t['Tot Bwd Pkts'] / S_t['Tot Fwd Pkts'], 0.0)
    
    # Synonyms
    candidates['Fwd Seg Size Avg'] = candidates['Fwd Pkt Len Mean']
    candidates['Bwd Seg Size Avg'] = candidates['Bwd Pkt Len Mean']
    candidates['Pkt Size Avg'] = candidates['Pkt Len Mean']
    
    candidates['Subflow Fwd Pkts'] = S_t['Tot Fwd Pkts']
    candidates['Subflow Fwd Byts'] = S_t['TotLen Fwd Pkts']
    candidates['Subflow Bwd Pkts'] = S_t['Tot Bwd Pkts']
    candidates['Subflow Bwd Byts'] = S_t['TotLen Bwd Pkts']
    
    validation_report = {}
    valid_derived_features = []
    
    for feat, pred_val in candidates.items():
        if feat not in S_t.columns:
            continue
            
        actual = S_t[feat].values
        pred = pred_val.values if isinstance(pred_val, pd.Series) else pred_val
        
        diff = np.abs(actual - pred)
        mae = np.mean(diff)
        max_err = np.max(diff)
        
        # Consider a match if difference is due to float precision
        is_match = diff < 1e-5
        match_pct = np.mean(is_match) * 100.0
        
        validation_report[feat] = {
            "MAE": float(mae),
            "Max_Error": float(max_err),
            "Exact_Match_Pct": float(match_pct),
            "Validated": bool(match_pct > 99.9)
        }
        
        if validation_report[feat]["Validated"]:
            valid_derived_features.append(feat)
            
    with open(RESULTS_DIR / "dependency_validation.json", "w") as f:
        json.dump(validation_report, f, indent=4)
        
    print(f"Validation complete. Valid derived features: {len(valid_derived_features)}")
    
    # Now build the full state dependency map
    with open("results/phase3/state/state_schema.json", "r") as f:
        schema = json.load(f)
        
    all_features = schema['features']
    
    dependency_map = {}
    for feat in all_features:
        if feat in valid_derived_features:
            cat = "DETERMINISTICALLY DERIVED"
            # Define simple formula mapping for V3 implementation
            if feat == 'Flow Pkts/s': comp = ['Tot Fwd Pkts', 'Tot Bwd Pkts']; formula = "(Tot Fwd Pkts + Tot Bwd Pkts) / 10.0"
            elif feat == 'Fwd Pkts/s': comp = ['Tot Fwd Pkts']; formula = "Tot Fwd Pkts / 10.0"
            elif feat == 'Bwd Pkts/s': comp = ['Tot Bwd Pkts']; formula = "Tot Bwd Pkts / 10.0"
            elif feat == 'Flow Byts/s': comp = ['TotLen Fwd Pkts', 'TotLen Bwd Pkts']; formula = "(TotLen Fwd Pkts + TotLen Bwd Pkts) / 10.0"
            elif feat == 'Fwd Pkt Len Mean': comp = ['TotLen Fwd Pkts', 'Tot Fwd Pkts']; formula = "TotLen Fwd Pkts / Tot Fwd Pkts"
            elif feat == 'Bwd Pkt Len Mean': comp = ['TotLen Bwd Pkts', 'Tot Bwd Pkts']; formula = "TotLen Bwd Pkts / Tot Bwd Pkts"
            elif feat == 'Pkt Len Mean': comp = ['TotLen Fwd Pkts', 'TotLen Bwd Pkts', 'Tot Fwd Pkts', 'Tot Bwd Pkts']; formula = "(TotLen Fwd Pkts + TotLen Bwd Pkts) / (Tot Fwd Pkts + Tot Bwd Pkts)"
            elif feat == 'Down/Up Ratio': comp = ['Tot Bwd Pkts', 'Tot Fwd Pkts']; formula = "Tot Bwd Pkts / Tot Fwd Pkts"
            elif feat == 'Fwd Seg Size Avg': comp = ['TotLen Fwd Pkts', 'Tot Fwd Pkts']; formula = "TotLen Fwd Pkts / Tot Fwd Pkts"
            elif feat == 'Bwd Seg Size Avg': comp = ['TotLen Bwd Pkts', 'Tot Bwd Pkts']; formula = "TotLen Bwd Pkts / Tot Bwd Pkts"
            elif feat == 'Pkt Size Avg': comp = ['TotLen Fwd Pkts', 'TotLen Bwd Pkts', 'Tot Fwd Pkts', 'Tot Bwd Pkts']; formula = "(TotLen Fwd Pkts + TotLen Bwd Pkts) / (Tot Fwd Pkts + Tot Bwd Pkts)"
            elif feat == 'Subflow Fwd Pkts': comp = ['Tot Fwd Pkts']; formula = "Tot Fwd Pkts"
            elif feat == 'Subflow Fwd Byts': comp = ['TotLen Fwd Pkts']; formula = "TotLen Fwd Pkts"
            elif feat == 'Subflow Bwd Pkts': comp = ['Tot Bwd Pkts']; formula = "Tot Bwd Pkts"
            elif feat == 'Subflow Bwd Byts': comp = ['TotLen Bwd Pkts']; formula = "TotLen Bwd Pkts"
            else: comp = []; formula = ""
            
            dependency_map[feat] = {
                "category": cat,
                "components": comp,
                "formula": formula,
                "non_negativity_constraint": True,
                "zero_denominator_policy": "0.0 sentinel",
                "predicted_directly": False
            }
        else:
            dependency_map[feat] = {
                "category": "PRIMITIVE / FLOW-LEVEL DESCRIPTIVE",
                "components": [],
                "formula": "",
                "non_negativity_constraint": True,
                "zero_denominator_policy": "N/A",
                "predicted_directly": True
            }
            
    with open(RESULTS_DIR / "state_dependency_map.json", "w") as f:
        json.dump(dependency_map, f, indent=4)
        
    print("Dependency map generated.")

if __name__ == '__main__':
    validate_dependencies()
