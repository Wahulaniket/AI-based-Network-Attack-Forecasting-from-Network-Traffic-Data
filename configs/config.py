import os
from pathlib import Path
import torch

# Directory structure
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
LABELS_DIR = DATA_DIR / "labels"

MODELS_DIR = PROJECT_ROOT / "models"
EVALUATION_DIR = PROJECT_ROOT / "evaluation"

# Ensure directories exist
for directory in [PROCESSED_DATA_DIR, LABELS_DIR, MODELS_DIR, EVALUATION_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Hardware Config
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dataset Config
# Because the exact filename can vary, we will define a pattern or dynamically search in Notebook 1
# but we can set defaults here.
CIC_IDS_2018_GLOB = "*.csv"

# Preprocessing Config
BINARY_LABEL_MAPPING = {
    "Benign": 0,
    "Attack": 1
}

# Time Window Config
WINDOW_SIZE = 10  # Seconds
AVAILABLE_WINDOW_SIZES = [5, 10, 30]

# Model Config
SEQUENCE_LENGTH = 10
FORECAST_HORIZON = 5  # K-step
HIDDEN_SIZE = 128
NUM_LAYERS = 2
DROPOUT = 0.2

# Training Config
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
EPOCHS = 20
RANDOM_SEED = 42

# Ensure Reproducibility Helper
def set_seed(seed=RANDOM_SEED):
    import random
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# Attack Stage Mapping Configuration
ATTACK_STAGE_MAPPING = {
    "Reconnaissance": ["PortScan", "Infiltration"],
    "Initial Access": ["FTP-Patator", "SSH-Patator", "Brute Force -Web", "Brute Force -XSS"],
    "Execution": ["SQL Injection"],
    "Persistence": [],
    "Privilege Escalation": [],
    "Credential Access": [],
    "Lateral Movement": [],
    "Command and Control": ["Bot"],
    "Exfiltration": ["DoS attacks-Hulk", "DoS attacks-SlowHTTPTest", "DoS attacks-GoldenEye", "DoS attacks-Slowloris", "DDOS attack-LOIC-UDP", "DDOS attack-HOIC"]
}
