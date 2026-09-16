import os
import json

# Define the paths for the frozen artifacts
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
PHASE2_3_RESULTS_DIR = os.path.join(REPO_ROOT, "results", "phase2_3")
PHASE2_3_MODELS_DIR = os.path.join(REPO_ROOT, "models", "phase2_3")

# We dynamically load the true configurations from the champion metrics so they are not hardcoded
_champion_metrics_path = os.path.join(PHASE2_3_RESULTS_DIR, "champion_metrics.json")
try:
    with open(_champion_metrics_path, "r") as f:
        _metrics = json.load(f)
        FROZEN_THRESHOLD = float(_metrics.get("Threshold", 0.9830410480499268))
        FROZEN_HISTORY_WINDOWS = int(_metrics.get("History", 20))
        FROZEN_FEATURE_SET = _metrics.get("Config", "SET_R")
except Exception as e:
    FROZEN_THRESHOLD = 0.9830410480499268
    FROZEN_HISTORY_WINDOWS = 20
    FROZEN_FEATURE_SET = "SET_R"

# Constants
WINDOW_SECONDS = 10
CONTEXT_SECONDS = WINDOW_SECONDS * FROZEN_HISTORY_WINDOWS
EXPECTED_FEATURE_COUNT = 89

# Environment overrides for live setup
LIVE_INTERFACE_NAME = os.getenv("CYBERCAST_LIVE_INTERFACE_NAME", "Wi-Fi")
LIVE_INTERFACE_INDEX = int(os.getenv("CYBERCAST_LIVE_INTERFACE_INDEX", "21"))
LIVE_HOST_IP = os.getenv("CYBERCAST_LIVE_HOST", "10.246.189.7")
LIVE_MODE_ENABLED = os.getenv("CYBERCAST_LIVE_MODE", "true").lower() == "true"
