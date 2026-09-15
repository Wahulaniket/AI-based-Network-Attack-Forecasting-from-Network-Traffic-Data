from dataclasses import dataclass
from typing import List, Dict, Any, Optional

@dataclass
class StageEvidence:
    feature: str
    reason: str

@dataclass
class StageScore:
    stage: str
    score: float
    confidence: str
    evidence: List[StageEvidence]
    
@dataclass
class Explanation:
    top_features: List[Dict[str, Any]]
    temporal_evidence: List[Dict[str, Any]]

@dataclass
class ForecastOutput:
    timestamp: str
    window_start: str
    window_end: str
    
    attack_probability: float
    model_threshold: float
    binary_prediction: int
    risk_level: str
    
    current_stage: str
    stage_confidence: str
    stage_evidence: List[Dict[str, str]]
    
    forecasted_next_stage: Optional[str]
    forecast_confidence: str
    
    probability_delta: Optional[float]
    risk_trend: str
    
    top_features: List[Dict[str, Any]]
    temporal_evidence: List[Dict[str, Any]]
    
    forecast_method: str = "heuristic"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "window_start": self.window_start,
            "window_end": self.window_end,
            "attack_probability": self.attack_probability,
            "model_threshold": self.model_threshold,
            "binary_prediction": self.binary_prediction,
            "risk_level": self.risk_level,
            "current_stage": self.current_stage,
            "stage_confidence": self.stage_confidence,
            "stage_evidence": self.stage_evidence,
            "forecasted_next_stage": self.forecasted_next_stage,
            "forecast_confidence": self.forecast_confidence,
            "forecast_method": self.forecast_method,
            "probability_delta": self.probability_delta,
            "risk_trend": self.risk_trend,
            "top_features": self.top_features,
            "temporal_evidence": self.temporal_evidence
        }
