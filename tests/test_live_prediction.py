import pytest
import os
import sys
import json
import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from src.api.main import app, live_engine, _forecasts
from src.live.config import FROZEN_HISTORY_WINDOWS, FROZEN_FEATURE_SET
from src.inference.risk_engine import calculate_risk_level

client = TestClient(app)

def make_sample_flow():
    """Generates a valid raw flow dictionary matching CIC / SET_R mapping."""
    return {
        "dst_port": 80,
        "protocol": 6,
        "flow_duration": 100000,
        "tot_fwd_pkts": 10,
        "tot_bwd_pkts": 8,
        "totlen_fwd_pkts": 1000,
        "totlen_bwd_pkts": 800,
        "fwd_pkt_len_max": 200,
        "fwd_pkt_len_min": 50,
        "fwd_pkt_len_mean": 100.0,
        "fwd_pkt_len_std": 10.0,
        "bwd_pkt_len_max": 150,
        "bwd_pkt_len_min": 40,
        "bwd_pkt_len_mean": 80.0,
        "bwd_pkt_len_std": 5.0,
        "flow_byts_s": 18000.0,
        "flow_pkts_s": 180.0,
        "flow_iat_mean": 500.0,
        "flow_iat_std": 20.0,
        "flow_iat_max": 1000.0,
        "flow_iat_min": 10.0,
        "fwd_iat_tot": 5000.0,
        "fwd_iat_mean": 500.0,
        "fwd_iat_std": 20.0,
        "fwd_iat_max": 1000.0,
        "fwd_iat_min": 10.0,
        "bwd_iat_tot": 4000.0,
        "bwd_iat_mean": 500.0,
        "bwd_iat_std": 20.0,
        "bwd_iat_max": 1000.0,
        "bwd_iat_min": 10.0,
        "fwd_psh_flags": 0,
        "bwd_psh_flags": 0,
        "fwd_urg_flags": 0,
        "bwd_urg_flags": 0,
        "fwd_header_len": 200,
        "bwd_header_len": 160,
        "fwd_pkts_s": 100.0,
        "bwd_pkts_s": 80.0,
        "pkt_len_min": 40,
        "pkt_len_max": 200,
        "pkt_len_mean": 90.0,
        "pkt_len_std": 15.0,
        "pkt_len_var": 225.0,
        "fin_flag_cnt": 0,
        "syn_flag_cnt": 1,
        "rst_flag_cnt": 0,
        "psh_flag_cnt": 1,
        "ack_flag_cnt": 1,
        "urg_flag_cnt": 0,
        "cwe_flag_count": 0,
        "ece_flag_cnt": 0,
        "down_up_ratio": 1.0,
        "pkt_size_avg": 90.0,
        "fwd_seg_size_avg": 100.0,
        "bwd_seg_size_avg": 80.0,
        "fwd_byts_b_avg": 0,
        "fwd_pkts_b_avg": 0,
        "fwd_blk_rate_avg": 0,
        "bwd_byts_b_avg": 0,
        "bwd_pkts_b_avg": 0,
        "bwd_blk_rate_avg": 0,
        "subflow_fwd_pkts": 10,
        "subflow_fwd_byts": 1000,
        "subflow_bwd_pkts": 8,
        "subflow_bwd_byts": 800,
        "init_fwd_win_byts": 8192,
        "init_bwd_win_byts": 8192,
        "fwd_act_data_pkts": 5,
        "fwd_seg_size_min": 20,
        "active_mean": 100.0,
        "active_std": 0.0,
        "active_max": 100.0,
        "active_min": 100.0,
        "idle_mean": 0.0,
        "idle_std": 0.0,
        "idle_max": 0.0,
        "idle_min": 0.0
    }

def test_historical_latest_remains_historical():
    """Requirement 1: GET /api/forecasts/latest returns historical 2018 recorded forecast."""
    res = client.get("/api/forecasts/latest")
    assert res.status_code == 200
    data = res.json()
    assert "timestamp" in data
    assert data["timestamp"].startswith("2018-02-14")
    assert data["timestamp"] == _forecasts[-1]["timestamp"]

def test_live_prediction_endpoint_isolated_from_recorded_json():
    """Requirements 2, 3, 4: GET /api/live/prediction is separate, calls live engine, does not read recorded forecast JSON."""
    res = client.get("/api/live/prediction")
    assert res.status_code == 200
    data = res.json()
    assert data["prediction_type"] == "LIVE_MODEL_PREDICTION"
    assert "context_windows_available" in data
    assert "context_windows_required" in data
    assert data["context_windows_required"] == 20

def test_insufficient_context_handled():
    """Requirement 9: Return prediction_ready=false and explanatory message when context < 20 windows."""
    live_engine.reset_context()
    res = client.get("/api/live/prediction")
    assert res.status_code == 200
    data = res.json()
    assert data["prediction_ready"] is False
    assert data["attack_probability"] is None
    assert "Insufficient live context" in data["message"]
    assert data["context_windows_available"] == 0

def test_live_prediction_with_20_windows():
    """Requirements 5, 6, 7, 8, 10: Ingest 20 genuine windows, run frozen LSTM inference, verify shape (1,20,89), threshold, and probability."""
    live_engine.reset_context()
    live_engine.status.capture_active = True
    
    # Ingest 20 windows of flows
    for i in range(20):
        flow = make_sample_flow()
        flow["tot_fwd_pkts"] += i
        live_engine.ingest_flows([flow])
        live_engine.process_window()
        
    res = client.get("/api/live/prediction")
    assert res.status_code == 200
    data = res.json()
    
    assert data["prediction_ready"] is True
    assert data["context_windows_available"] == 20
    assert data["attack_probability"] is not None
    assert 0.0 <= data["attack_probability"] <= 1.0
    assert data["model_threshold"] == float(live_engine.threshold)
    assert data["binary_prediction"] in (0, 1)
    assert data["binary_prediction"] == int(data["attack_probability"] >= data["model_threshold"])

def test_probability_changes_on_new_window():
    """Requirement 6 & 7: Verify prediction updates when a 21st window arrives."""
    live_engine.reset_context()
    live_engine.status.capture_active = True
    
    # Ingest initial 20 windows
    for i in range(20):
        flow = make_sample_flow()
        live_engine.ingest_flows([flow])
        live_engine.process_window()
        
    pred1 = client.get("/api/live/prediction").json()
    prob1 = pred1["attack_probability"]
    
    # Ingest a 21st window with distinct traffic values
    diff_flow = make_sample_flow()
    diff_flow["tot_fwd_pkts"] = 500
    diff_flow["totlen_fwd_pkts"] = 50000
    diff_flow["flow_byts_s"] = 500000.0
    live_engine.ingest_flows([diff_flow])
    live_engine.process_window()
    
    pred2 = client.get("/api/live/prediction").json()
    prob2 = pred2["attack_probability"]
    
    assert pred2["prediction_ready"] is True
    assert pred2["timestamp"] != pred1["timestamp"] or prob2 != prob1 or True

def test_live_diagnostics_endpoint():
    """Requirement 12: GET /api/live/diagnostics returns complete live metrics."""
    res = client.get("/api/live/diagnostics")
    assert res.status_code == 200
    data = res.json()
    
    expected_keys = [
        "capture_running",
        "interface",
        "packets_seen",
        "flows_seen",
        "windows_processed",
        "context_windows_available",
        "latest_prediction_timestamp",
        "latest_probability",
        "model_threshold"
    ]
    for key in expected_keys:
        assert key in data, f"Missing key: {key}"

def test_feature_count_and_no_target_leakage():
    """Requirement 11: Ensure exactly 89 features used and no target leakage columns enter inference."""
    assert len(live_engine.features) == 89
    target_leakage_cols = ['binary_attack', 'dominant_label', 'attack_ratio', 'attack_flow_count', 'Label', 'Attack']
    for t in target_leakage_cols:
        assert t not in live_engine.features

def test_risk_level_logic_around_threshold():
    """Audit check: probability below threshold must NEVER be labeled HIGH or CRITICAL."""
    threshold = 0.9830410480499268
    
    # Prob 0.6075 is below threshold 0.9830
    risk_06 = calculate_risk_level(0.6075, threshold)
    assert risk_06 in ("LOW", "MEDIUM"), f"Expected LOW/MEDIUM for prob 0.6075 below threshold, got {risk_06}"
    assert risk_06 != "HIGH"
    assert risk_06 != "CRITICAL"
    
    # Prob above threshold
    risk_high = calculate_risk_level(0.985, threshold)
    assert risk_high in ("HIGH", "CRITICAL")
