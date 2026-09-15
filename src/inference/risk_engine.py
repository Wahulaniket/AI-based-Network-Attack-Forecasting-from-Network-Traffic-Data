from typing import Tuple

def calculate_risk_level(probability: float) -> str:
    """Deterministic risk classification around the model probability.
    
    This is a presentation-layer classification strictly mapped to ranges:
    LOW       < 0.25
    MEDIUM    0.25-0.50
    HIGH      0.50-0.75
    CRITICAL  >= 0.75
    
    This is strictly decoupled from the model threshold (which is evaluated separately for the binary prediction).
    """
    if probability < 0.25:
        return "LOW"
    elif probability < 0.50:
        return "MEDIUM"
    elif probability < 0.75:
        return "HIGH"
    else:
        return "CRITICAL"
