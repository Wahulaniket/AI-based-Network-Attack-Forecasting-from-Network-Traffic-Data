from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from datetime import datetime

@dataclass
class WindowStatistics:
    flow_count: int
    packet_count: int
    total_bytes: int

@dataclass
class PredictionOutput:
    timestamp: datetime
    window_start: datetime
    window_end: datetime
    attack_probability: float
    model_threshold: float
    binary_prediction: int
    risk_level: str
    history_windows: int
    feature_count: int
    feature_names: List[str]
    statistics: Optional[WindowStatistics] = None

    def to_dict(self) -> Dict[str, Any]:
        data = {
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "window_start": self.window_start.isoformat() if self.window_start else None,
            "window_end": self.window_end.isoformat() if self.window_end else None,
            "attack_probability": self.attack_probability,
            "model_threshold": self.model_threshold,
            "binary_prediction": self.binary_prediction,
            "risk_level": self.risk_level,
            "history_windows": self.history_windows,
            "feature_count": self.feature_count,
        }
        if self.statistics:
            data["flow_count"] = self.statistics.flow_count
            data["packet_count"] = self.statistics.packet_count
            data["total_bytes"] = self.statistics.total_bytes
        return data
