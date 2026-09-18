from typing import Tuple

def calculate_risk_level(probability: float, threshold: float = 0.9830410480499268) -> str:
    """Deterministic risk classification around model probability and decision threshold.
    
    A probability below the model threshold must not be labeled HIGH or CRITICAL.
    
    Mapping relative to threshold:
    LOW       < threshold * 0.5
    MEDIUM    threshold * 0.5 <= prob < threshold
    HIGH      threshold <= prob < threshold + (1.0 - threshold) * 0.5
    CRITICAL  >= threshold + (1.0 - threshold) * 0.5
    """
    if probability < threshold * 0.5:
        return "LOW"
    elif probability < threshold:
        return "MEDIUM"
    elif probability < threshold + (1.0 - threshold) * 0.5:
        return "HIGH"
    else:
        return "CRITICAL"
