import pytest
from fastapi.testclient import TestClient
import sys
import os

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from src.api.main import app

client = TestClient(app)

def test_live_status():
    response = client.get("/api/live/status")
    assert response.status_code == 200
    data = response.json()
    assert "mode" in data
    assert data["mode"] == "live"
    assert "capture_active" in data
    assert "history_collected" in data

def test_live_latest_waiting():
    response = client.get("/api/live/latest")
    assert response.status_code == 200
    data = response.json()
    # It should say waiting before 20 windows
    if "status" in data:
        assert data["status"] == "Waiting for 20 window context"

def test_live_start_stop():
    # Start
    start_res = client.post("/api/live/start")
    assert start_res.status_code == 200
    assert start_res.json()["status"] == "started"
    
    # Check status is capturing
    status_res = client.get("/api/live/status")
    assert status_res.json()["capture_active"] is True
    
    # Stop
    stop_res = client.post("/api/live/stop")
    assert stop_res.status_code == 200
    assert stop_res.json()["status"] == "stopped"
    
    # Check status is not capturing
    status_res2 = client.get("/api/live/status")
    assert status_res2.json()["capture_active"] is False

def test_historical_status_intact():
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
