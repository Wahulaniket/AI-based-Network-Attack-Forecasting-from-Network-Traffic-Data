import torch
import numpy as np
from typing import List, Dict, Any

def explain_prediction(model, sequence: np.ndarray, feature_names: List[str], device: torch.device) -> Dict[str, Any]:
    """Computes perturbation-based feature and temporal attribution for a single causal sequence.
    
    Args:
        model: The loaded CyberCastLSTM model.
        sequence: A scaled numpy array of shape (seq_length, num_features).
        feature_names: List of feature names matching the 89 features.
        device: torch device.
        
    Returns:
        Dict containing top contributing features and temporal windows.
    """
    model.eval()
    
    # 1. Baseline prediction
    seq_tensor = torch.tensor(sequence, dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        baseline_logit = model(seq_tensor)
        baseline_prob = torch.sigmoid(baseline_logit).item()
        
    seq_length, num_features = sequence.shape
    
    # 2. Feature Attribution (masking each feature across all time steps)
    feature_importances = []
    for f_idx in range(num_features):
        masked_seq = sequence.copy()
        # Since data is standard scaled, 0.0 is the mean value.
        masked_seq[:, f_idx] = 0.0
        
        m_tensor = torch.tensor(masked_seq, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            m_prob = torch.sigmoid(model(m_tensor)).item()
            
        # Importance = baseline - masked. 
        # If dropping the feature lowers the risk (m_prob < baseline), the feature increased the risk.
        importance = baseline_prob - m_prob
        feature_importances.append({
            "feature": feature_names[f_idx],
            "importance": float(importance),
            "direction": "increases_risk" if importance > 0 else "decreases_risk"
        })
        
    # Sort by absolute importance
    feature_importances.sort(key=lambda x: abs(x["importance"]), reverse=True)
    
    # 3. Temporal Attribution (masking each time step across all features)
    temporal_importances = []
    for t_idx in range(seq_length):
        masked_seq = sequence.copy()
        masked_seq[t_idx, :] = 0.0
        
        m_tensor = torch.tensor(masked_seq, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            m_prob = torch.sigmoid(model(m_tensor)).item()
            
        importance = baseline_prob - m_prob
        # Window index: t_idx (0 is oldest, 19 is most recent)
        # We store it relative to the current time t (t-19 to t)
        offset_from_current = t_idx - (seq_length - 1) # 0 for current, -19 for oldest
        
        temporal_importances.append({
            "window_offset": offset_from_current,
            "importance": float(importance)
        })
        
    temporal_importances.sort(key=lambda x: abs(x["importance"]), reverse=True)
    
    return {
        "top_features": feature_importances[:5],
        "temporal_evidence": temporal_importances[:3]
    }
