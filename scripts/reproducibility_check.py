import os
import torch
import numpy as np

# Set deterministic behavior
def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# We will import the necessary components from phase2_3_experiments
import phase2_3_experiments as p23

def run_reproducibility():
    print("Running Reproducibility Check 1 (Seed=42)...")
    set_seed(42)
    res1 = p23.train_and_eval(p23.set_r_features, h=20, name="SET_R_Repro_1")
    
    print("\\nRunning Reproducibility Check 2 (Seed=42)...")
    set_seed(42)
    res2 = p23.train_and_eval(p23.set_r_features, h=20, name="SET_R_Repro_2")
    
    print(f"\\n--- Reproducibility Results ---")
    print(f"Run 1 Val PR-AUC: {res1['Val_PR_AUC']:.4f} (Best Epoch: {res1['Best_Epoch']})")
    print(f"Run 2 Val PR-AUC: {res2['Val_PR_AUC']:.4f} (Best Epoch: {res2['Best_Epoch']})")

if __name__ == "__main__":
    run_reproducibility()
