from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from datetime import datetime

@dataclass
class LiveStatus:
    mode: str = "live"
    capture_active: bool = False
    interface: str = ""
    interface_index: int = 0
    host_ip: str = ""
    previous_interface: Optional[str] = None
    previous_host_ip: Optional[str] = None
    accepted_interface: str = ""
    accepted_host_ip: str = ""
    network_changed: bool = False
    rebuilding_context: bool = False
    model_name: str = "PHASE 2.3 LSTM"
    feature_set: str = "SET_R"
    window_seconds: int = 10
    history_required: int = 20
    history_collected: int = 0
    model_ready: bool = False

@dataclass
class LivePrediction:
    timestamp: datetime
    window_start: datetime
    window_end: datetime
    attack_probability: float
    model_threshold: float
    binary_prediction: int
    risk_level: str
    history_windows: int
    flow_count: int
    packet_count: int
    total_bytes: int
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "window_start": self.window_start.isoformat() if self.window_start else None,
            "window_end": self.window_end.isoformat() if self.window_end else None,
            "attack_probability": self.attack_probability,
            "model_threshold": self.model_threshold,
            "prediction": self.binary_prediction, # mapped for the frontend
            "risk": self.risk_level, # mapped for the frontend
            "history_collected": self.history_windows,
            "flow_count": self.flow_count,
            "packet_count": self.packet_count,
            "total_bytes": self.total_bytes
        }

@dataclass
class LiveTrafficSummary:
    packets_observed: int = 0
    flows_observed: int = 0
    bytes_observed: int = 0
    packets_per_sec: float = 0.0
    bytes_per_sec: float = 0.0
    active_flows: int = 0
