from typing import List, Tuple, Optional
import pandas as pd

def compute_risk_trend(history_probs: List[float]) -> Tuple[Optional[float], str]:
    """Computes the risk trend (delta and direction) strictly from causal history."""
    if len(history_probs) < 2:
        return None, "stable"
        
    current = history_probs[-1]
    previous = history_probs[-2]
    delta = current - previous
    
    if delta > 0.05:
        trend = "escalating"
    elif delta < -0.05:
        trend = "de-escalating"
    else:
        trend = "stable"
        
    return delta, trend

def forecast_next_stage(current_stage: str, risk_trend: str) -> Tuple[Optional[str], str]:
    """Probabilistically suggests the next logical ATT&CK stage based on progression.
    
    This is a HEURISTIC NEXT-STAGE FORECAST. Does NOT assert this as ground truth
    and does not implement a learned transition model (P(S[t+1] | S[t])).
    """
    if current_stage == "UNKNOWN / INSUFFICIENT EVIDENCE":
        return None, "insufficient_evidence"
        
    if risk_trend == "de-escalating":
        return None, "low"
        
    progression_map = {
        "Reconnaissance": "Initial Access",
        "Initial Access": "Discovery",
        "Discovery": "Lateral Movement",
        "Lateral Movement": "Command and Control",
        "Command and Control": "Exfiltration / Impact",
        "Exfiltration": "Impact",
        "Impact": None # Terminal Stage
    }
    
    next_stage = progression_map.get(current_stage, None)
    if next_stage is None:
        confidence = "low"
    else:
        confidence = "low" if risk_trend == "stable" else "medium"
    
    return next_stage, confidence
