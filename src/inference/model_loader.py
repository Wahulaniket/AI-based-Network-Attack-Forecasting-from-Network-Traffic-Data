import os
import json
import torch
import torch.nn as nn
import joblib

class CyberCastLSTM(nn.Module):
    """LSTM-based network attack state forecasting model (Phase 2.3).
    
    Architecture:
        Input features → LSTM (multi-layer) → last hidden → Dropout → Linear → logit
    """

    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.2):
        super(CyberCastLSTM, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # LSTM encoder
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # Classification head
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        """Forward pass.
        Args:
            x: (batch, seq_len, input_size)
        Returns:
            logits: (batch,)
        """
        lstm_out, (h_n, c_n) = self.lstm(x)
        last_hidden = h_n[-1]  # (batch, hidden_size)
        out = self.dropout(last_hidden)
        logits = self.fc(out).squeeze(-1)  # (batch,)
        return logits

def load_inference_artifacts(repo_root: str):
    """Safely loads all frozen artifacts for Phase 2.3 inference."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Paths
    model_path = os.path.join(repo_root, 'models', 'production', 'model_SET_R_h20.pt')
    scaler_path = os.path.join(repo_root, 'models', 'production', 'scaler_SET_R_h20.joblib')
    metrics_path = os.path.join(repo_root, 'results', 'phase2_3', 'champion_metrics.json')
    features_path = os.path.join(repo_root, 'results', 'phase2_3', 'feature_sets.json')

    # Verify existence
    for p in [model_path, scaler_path, metrics_path, features_path]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"[ERROR] Missing required artifact: {p}")
            
    # Load feature names
    with open(features_path, 'r') as f:
        feature_sets = json.load(f)
    
    if 'SET_R' not in feature_sets:
        raise ValueError("[ERROR] SET_R not found in feature_sets.json")
    
    features = feature_sets['SET_R']
    if len(features) != 89:
        raise ValueError(f"[ERROR] Expected 89 features for SET_R, got {len(features)}")
    
    # Load metrics for threshold
    with open(metrics_path, 'r') as f:
        metrics = json.load(f)
    
    threshold = metrics.get('Threshold')
    if threshold is None:
        print("[WARNING] Threshold not found in champion_metrics.json, using fallback 0.9830")
        threshold = 0.9830410480499268
        
    # Load Scaler
    scaler = joblib.load(scaler_path)
    if scaler.n_features_in_ != 89:
        raise ValueError(f"[ERROR] Scaler expects {scaler.n_features_in_} features, but SET_R has 89.")

    # Load Model
    model = CyberCastLSTM(input_size=89, hidden_size=64, num_layers=2, dropout=0.2)
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    # Checkpoint might be the state_dict or a dict containing state_dict
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    elif isinstance(checkpoint, dict):
        model.load_state_dict(checkpoint)
    else:
        raise ValueError("[ERROR] Unrecognized checkpoint format.")
        
    model.to(device)
    model.eval()

    print("CyberCast Inference Engine")
    print("--------------------------")
    print("Model: Phase 2.3 SET_R LSTM")
    print("Features: 89")
    print("History: 20 windows")
    print("Window size: 10 seconds")
    print("Temporal context: 200 seconds")
    print(f"Threshold: {threshold}")
    print(f"Device: {device}")

    return {
        "model": model,
        "scaler": scaler,
        "features": features,
        "threshold": threshold,
        "device": device
    }
