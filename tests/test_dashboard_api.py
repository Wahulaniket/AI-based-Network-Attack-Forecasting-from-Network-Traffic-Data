import pytest
from fastapi.testclient import TestClient
import os
import sys
import json

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from src.api.main import app, _forecasts, _metrics, _feature_sets

client = TestClient(app)

def test_health_endpoint():
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "forecasts_loaded" in data

def test_metrics_endpoint():
    # If metrics exist, it should return them
    if _metrics:
        response = client.get("/api/metrics")
        assert response.status_code == 200
        data = response.json()
        assert "PR-AUC" in data.get("Test_Metrics", {})
        assert data.get("Test_Metrics", {}).get("PR-AUC") > 0.0

def test_forecasts_endpoint():
    if _forecasts:
        response = client.get("/api/forecasts?limit=10")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) <= 10
        if len(data) > 0:
            assert "attack_probability" in data[0]
            assert "current_stage" in data[0]

def test_latest_forecast_endpoint():
    if _forecasts:
        response = client.get("/api/forecasts/latest")
        assert response.status_code == 200
        data = response.json()
        assert "attack_probability" in data
        assert "timestamp" in data
        # Ensure chronological - the latest should match the last element
        assert data["timestamp"] == _forecasts[-1]["timestamp"]

def test_model_info_endpoint():
    if _metrics:
        response = client.get("/api/model-info")
        assert response.status_code == 200
        data = response.json()
        assert data["model_name"] == "Phase 2.3 LSTM"
        assert "threshold" in data

def test_traffic_state_endpoint():
    if _forecasts:
        response = client.get("/api/traffic-state")
        assert response.status_code == 200
        data = response.json()
        assert "total_analyzed_windows" in data
        assert "high_risk_windows" in data

def test_explainability_endpoint():
    if _forecasts:
        response = client.get("/api/explainability")
        assert response.status_code == 200
        data = response.json()
        assert "top_features" in data
        assert "temporal_evidence" in data

def test_no_target_leakage():
    if _forecasts:
        f = _forecasts[0]
        # Verify target fields are NOT in explainability
        if "top_features" in f:
            for feat in f["top_features"]:
                assert feat["feature"] not in ["Label", "binary_attack", "Attack"]

def test_low_risk_attack_suppression_rule():
    if _forecasts:
        for f in _forecasts:
            if f.get("attack_probability", 1.0) < f.get("model_threshold", 0.0):
                # By schema rules, it could have heuristic evidence but we suppress 
                # visual "confirmed attack" in frontend. Let's ensure the API isn't 
                # changing the JSON but accurately delivering it.
                assert "forecast_method" in f
                assert f["forecast_method"] == "heuristic"
