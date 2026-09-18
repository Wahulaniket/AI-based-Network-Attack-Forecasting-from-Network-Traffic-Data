import json
import os
from fastapi import FastAPI, HTTPException, Query, Body
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Dict, Any, Optional
from pydantic import BaseModel
from src.live.live_engine import LiveEngine

app = FastAPI(title="CyberCast Command Center API", description="Read-only & Live API for CyberCast")

origins = [
    "http://localhost:5173",
    "http://localhost:5174",
    "http://127.0.0.1:5173",
    "http://127.0.0.1:5174",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
RESULTS_DIR = os.path.join(REPO_ROOT, "results")
FORECASTS_JSON = os.path.join(RESULTS_DIR, "inference", "forecast_predictions.json")
CHAMPION_METRICS = os.path.join(RESULTS_DIR, "phase2_3", "champion_metrics.json")
FEATURE_SETS = os.path.join(RESULTS_DIR, "phase2_3", "feature_sets.json")

_forecasts = []
_metrics = {}
_feature_sets = {}

def load_data():
    global _forecasts, _metrics, _feature_sets
    if os.path.exists(FORECASTS_JSON):
        with open(FORECASTS_JSON, "r") as f:
            _forecasts = json.load(f)
    if os.path.exists(CHAMPION_METRICS):
        with open(CHAMPION_METRICS, "r") as f:
            _metrics = json.load(f)
    if os.path.exists(FEATURE_SETS):
        with open(FEATURE_SETS, "r") as f:
            _feature_sets = json.load(f)

load_data()

# Initialize Live Engine
live_engine = LiveEngine(repo_root=REPO_ROOT)

# --- Pydantic Models ---
class ForecastEvidence(BaseModel):
    feature: str
    reason: str

class FeatureImportance(BaseModel):
    feature: str
    importance: float
    direction: str

class TemporalEvidence(BaseModel):
    window_offset: int
    importance: float

class ForecastModel(BaseModel):
    timestamp: str
    window_start: str
    window_end: str
    attack_probability: float
    model_threshold: float
    binary_prediction: int
    risk_level: str
    current_stage: str
    stage_confidence: str
    stage_evidence: List[ForecastEvidence]
    forecasted_next_stage: Optional[str] = None
    forecast_confidence: str
    forecast_method: str
    probability_delta: Optional[float] = None
    risk_trend: str
    top_features: List[FeatureImportance] = []
    temporal_evidence: List[TemporalEvidence] = []

class MetricsModel(BaseModel):
    Config: str
    History: int
    Threshold: float
    Test_Metrics: Dict[str, Any]

class HealthModel(BaseModel):
    status: str
    forecasts_loaded: int
    metrics_loaded: bool

class CaptureStatusModel(BaseModel):
    capture_backend: str
    interface: str
    interface_description: Optional[str] = None
    interface_index: int
    current_ipv4: str
    capture_active: bool
    packets_captured: int
    bytes_captured: int
    flows_created: int
    packets_per_sec: float
    last_packet_time: Optional[str] = None
    error: Optional[str] = None

# --- Endpoints ---

@app.get("/api/health", response_model=HealthModel)
def health_check():
    return HealthModel(
        status="ok",
        forecasts_loaded=len(_forecasts),
        metrics_loaded=bool(_metrics)
    )

@app.get("/api/forecasts", response_model=List[ForecastModel])
def get_forecasts(limit: int = Query(100, ge=1, le=5000), offset: int = Query(0, ge=0)):
    return _forecasts[offset : offset + limit]

@app.get("/api/forecasts/latest", response_model=ForecastModel)
def get_latest_forecast():
    if not _forecasts:
        raise HTTPException(status_code=404, detail="No forecasts available")
    return _forecasts[-1]

@app.get("/api/forecasts/{timestamp}", response_model=ForecastModel)
def get_forecast_by_timestamp(timestamp: str):
    for f in _forecasts:
        if f.get("timestamp") == timestamp:
            return f
    raise HTTPException(status_code=404, detail="Forecast not found")

@app.get("/api/metrics", response_model=MetricsModel)
def get_metrics():
    if not _metrics:
        raise HTTPException(status_code=404, detail="Metrics not found")
    return _metrics

@app.get("/api/model-info")
def get_model_info():
    if not _metrics:
        raise HTTPException(status_code=404, detail="Model info not found")
    return {
        "model_name": "Phase 2.3 LSTM",
        "feature_set": _metrics.get("Config", "SET_R"),
        "features_count": len(_feature_sets.get(_metrics.get("Config", "SET_R"), [])),
        "history_windows": _metrics.get("History", 20),
        "window_size_seconds": 10,
        "context_seconds": _metrics.get("History", 20) * 10,
        "threshold": _metrics.get("Threshold")
    }

@app.get("/api/traffic-state")
def get_traffic_state():
    if not _forecasts:
        return {}
    total = len(_forecasts)
    high_risk = sum(1 for f in _forecasts if f.get("attack_probability", 0) > _metrics.get("Threshold", 0.98))
    escalating = sum(1 for f in _forecasts if f.get("risk_trend") == "escalating")
    
    return {
        "total_analyzed_windows": total,
        "high_risk_windows": high_risk,
        "escalating_windows": escalating,
        "timeline_span": {
            "start": _forecasts[0]["timestamp"] if total > 0 else None,
            "end": _forecasts[-1]["timestamp"] if total > 0 else None
        }
    }

@app.get("/api/explainability")
def get_explainability(timestamp: Optional[str] = None):
    if timestamp:
        for f in _forecasts:
            if f.get("timestamp") == timestamp:
                return {"top_features": f.get("top_features", []), "temporal_evidence": f.get("temporal_evidence", [])}
        raise HTTPException(status_code=404, detail="Forecast not found")
    else:
        if not _forecasts:
            raise HTTPException(status_code=404, detail="No forecasts available")
        f = _forecasts[-1]
        return {"top_features": f.get("top_features", []), "temporal_evidence": f.get("temporal_evidence", [])}

@app.get("/api/attack-progression")
def get_attack_progression():
    if not _forecasts:
        return []
    return [{"timestamp": f["timestamp"], "attack_probability": f["attack_probability"], "risk_level": f["risk_level"], "current_stage": f["current_stage"], "forecasted_next_stage": f.get("forecasted_next_stage"), "stage_confidence": f.get("stage_confidence")} for f in _forecasts]

# --- Live Endpoints ---

@app.get("/api/live/status")
def get_live_status():
    return live_engine.get_status()

@app.get("/api/live/capture-status", response_model=CaptureStatusModel)
def get_capture_status():
    return live_engine.get_capture_status()

@app.get("/api/live/debug")
def get_live_debug():
    return {
        "current_interface": live_engine.status.interface,
        "current_ip": live_engine.status.host_ip,
        "accepted_interface": getattr(live_engine.status, "accepted_interface", None),
        "accepted_host_ip": getattr(live_engine.status, "accepted_host_ip", None),
        "previous_interface": live_engine.status.previous_interface,
        "previous_host_ip": live_engine.status.previous_host_ip,
        "network_changed": live_engine.status.network_changed,
        "rebuilding_context": getattr(live_engine.status, "rebuilding_context", None),
        "history_collected": len(live_engine.window_context),
        "flow_buffer_count": len(live_engine.flow_buffer),
        "worker_alive": True,
        "scheduler_alive": live_engine.scheduler_thread.is_alive() if live_engine.scheduler_thread else False,
        "last_window_time": live_engine.last_window_time,
        "capture_active": live_engine.status.capture_active,
        "capture_details": live_engine.get_capture_status()
    }

@app.get("/api/live/prediction")
def get_live_prediction():
    return live_engine.get_live_prediction()

@app.get("/api/live/diagnostics")
def get_live_diagnostics():
    return live_engine.get_diagnostics()

@app.get("/api/live/latest")
def get_live_latest():
    return live_engine.get_live_prediction()

@app.post("/api/live/start")
def start_live_capture(interface: Optional[str] = Query(None)):
    live_engine.start_capture(interface_override=interface)
    return {"status": "started", "interface": live_engine.status.interface}

@app.post("/api/live/stop")
def stop_live_capture():
    live_engine.stop_capture()
    return {"status": "stopped"}

@app.post("/api/live/reset")
def reset_live_context():
    live_engine.reset_context()
    return {"status": "reset", "history_collected": 0}

@app.post("/api/live/flows")
def ingest_live_flows(flows: List[Dict[str, Any]] = Body(...)):
    live_engine.ingest_flows(flows)
    return {"status": "ingested", "count": len(flows)}
