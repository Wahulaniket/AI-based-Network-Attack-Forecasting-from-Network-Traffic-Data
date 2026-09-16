import os
import sys
import time
import pytest
import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

# Add project root to sys.path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.live.npcap_capture import NpcapCapturer
from src.live.flow_aggregator import FlowAggregator, FlowState
from src.live.live_engine import LiveEngine
from src.inference.feature_pipeline import process_features
from src.api.main import app


def test_interface_discovery():
    capturer = NpcapCapturer()
    discovery = capturer.discover_active_interface(target_iface="AUTO")
    assert "scapy_iface" in discovery
    assert "current_ipv4" in discovery
    assert discovery["current_ipv4"] != ""
    print("\n[TEST PASS] Interface discovery:", discovery["interface_name"], discovery["current_ipv4"])

def test_manual_interface_override():
    capturer = NpcapCapturer()
    discovery = capturer.discover_active_interface(target_iface="Wi-Fi")
    assert discovery["interface_name"] != ""
    print("\n[TEST PASS] Manual interface override Wi-Fi resolved to:", discovery["interface_name"])

def test_flow_aggregator_bidirectional():
    aggregator = FlowAggregator()
    
    import scapy.all as scapy
    pkt1 = scapy.IP(src="192.168.1.10", dst="192.168.1.20")/scapy.TCP(sport=12345, dport=80, flags="S", window=64240)
    pkt2 = scapy.IP(src="192.168.1.20", dst="192.168.1.10")/scapy.TCP(sport=80, dport=12345, flags="SA", window=65535)
    pkt3 = scapy.IP(src="192.168.1.10", dst="192.168.1.20")/scapy.TCP(sport=12345, dport=80, flags="A", window=64240)
    
    aggregator.process_packet(pkt1)
    aggregator.process_packet(pkt2)
    aggregator.process_packet(pkt3)
    
    flows = aggregator.flush_flows()
    assert len(flows) == 1
    f = flows[0]
    assert f["Dst Port"] == 80
    assert f["Protocol"] == 6
    assert f["Tot Fwd Pkts"] == 2
    assert f["Tot Bwd Pkts"] == 1
    assert f["SYN Flag Cnt"] == 2 # S and SA contain SYN flag
    assert f["ACK Flag Cnt"] == 2 # SA and A contain ACK flag
    assert f["Init Fwd Win Byts"] == 64240
    assert f["Init Bwd Win Byts"] == 65535
    print("\n[TEST PASS] Bidirectional flow aggregation extracted correct 5-tuple metrics.")

from src.inference.windowing import create_time_windows

def test_exact_89_feature_ordering_and_coverage():
    engine = LiveEngine(repo_root=REPO_ROOT)
    aggregator = FlowAggregator()
    
    assert len(engine.features) == 89
    
    import scapy.all as scapy
    pkt = scapy.IP(src="10.0.0.1", dst="10.0.0.2")/scapy.TCP(sport=5000, dport=80, flags="P")
    aggregator.process_packet(pkt)
    flows = aggregator.flush_flows()
    
    df_raw = pd.DataFrame(flows)
    df_mapped = engine._map_cic_to_set_r(df_raw)
    df_mapped['Timestamp'] = pd.Timestamp.now()
    df_processed = process_features(df_mapped, engine.features)
    states = create_time_windows(df_processed, engine.features)
    
    is_compat, cov_pct, status_str, missing = aggregator.validate_feature_coverage(states, engine.features)
    assert is_compat == True
    assert cov_pct == 100.0
    assert status_str == "FULL"
    assert len(missing) == 0


    print("\n[TEST PASS] 89 SET_R feature ordering and 100% coverage verified.")

def test_empty_window_behavior():
    engine = LiveEngine(repo_root=REPO_ROOT)
    engine.status.capture_active = True
    
    history_before = len(engine.window_context)
    engine.process_window() # flow_buffer is empty
    history_after = len(engine.window_context)
    
    assert history_before == history_after == 0
    assert engine.status.model_ready == False
    print("\n[TEST PASS] Empty window correctly advances wall-clock without appending fake zero-vectors.")

def test_20_window_context_progression_and_frozen_model():
    engine = LiveEngine(repo_root=REPO_ROOT)
    engine.status.capture_active = True
    
    # Ingest 20 genuine traffic windows
    for i in range(20):
        flow = {
            "dst_port": 80,
            "protocol": 6,
            "flow_duration": 100000.0,
            "tot_fwd_pkts": 5 + i,
            "tot_bwd_pkts": 5,
            "totlen_fwd_pkts": 500,
            "totlen_bwd_pkts": 500,
            "fwd_pkt_len_max": 100,
            "fwd_pkt_len_min": 50,
            "fwd_pkt_len_mean": 75,
            "fwd_pkt_len_std": 10,
            "bwd_pkt_len_max": 100,
            "bwd_pkt_len_min": 50,
            "bwd_pkt_len_mean": 75,
            "bwd_pkt_len_std": 10,
            "flow_byts_s": 10000.0,
            "flow_pkts_s": 100.0,
            "flow_iat_mean": 1000.0,
            "flow_iat_std": 10.0,
            "flow_iat_max": 2000.0,
            "flow_iat_min": 10.0,
            "fwd_iat_tot": 5000.0,
            "fwd_iat_mean": 1000.0,
            "fwd_iat_std": 10.0,
            "fwd_iat_max": 2000.0,
            "fwd_iat_min": 10.0,
            "bwd_iat_tot": 5000.0,
            "bwd_iat_mean": 1000.0,
            "bwd_iat_std": 10.0,
            "bwd_iat_max": 2000.0,
            "bwd_iat_min": 10.0,
            "fwd_psh_flags": 0,
            "bwd_psh_flags": 0,
            "fwd_urg_flags": 0,
            "bwd_urg_flags": 0,
            "fwd_header_len": 100,
            "bwd_header_len": 100,
            "fwd_pkts_s": 50.0,
            "bwd_pkts_s": 50.0,
            "pkt_len_min": 50,
            "pkt_len_max": 100,
            "pkt_len_mean": 75,
            "pkt_len_std": 10,
            "pkt_len_var": 100,
            "fin_flag_cnt": 0,
            "syn_flag_cnt": 1,
            "rst_flag_cnt": 0,
            "psh_flag_cnt": 0,
            "ack_flag_cnt": 1,
            "urg_flag_cnt": 0,
            "cwe_flag_count": 0,
            "ece_flag_cnt": 0,
            "down_up_ratio": 1.0,
            "pkt_size_avg": 75,
            "fwd_seg_size_avg": 75,
            "bwd_seg_size_avg": 75,
            "fwd_byts_b_avg": 0,
            "fwd_pkts_b_avg": 0,
            "fwd_blk_rate_avg": 0,
            "bwd_byts_b_avg": 0,
            "bwd_pkts_b_avg": 0,
            "bwd_blk_rate_avg": 0,
            "subflow_fwd_pkts": 5,
            "subflow_fwd_byts": 500,
            "subflow_bwd_pkts": 5,
            "subflow_bwd_byts": 500,
            "init_fwd_win_byts": 64240,
            "init_bwd_win_byts": 65535,
            "fwd_act_data_pkts": 2,
            "fwd_seg_size_min": 20,
            "active_mean": 0,
            "active_std": 0,
            "active_max": 0,
            "active_min": 0,
            "idle_mean": 0,
            "idle_std": 0,
            "idle_max": 0,
            "idle_min": 0
        }
        engine.ingest_flows([flow])
        engine.process_window()
        assert len(engine.window_context) == i + 1

    assert engine.status.model_ready == True
    assert engine.latest_prediction is not None
    pred = engine.latest_prediction
    assert 0.0 <= pred.attack_probability <= 1.0
    assert pred.model_threshold == 0.9830410480499268
    print(f"\n[TEST PASS] 20-window context reached. Model prediction = {pred.attack_probability:.4f}, Risk = {pred.risk_level}")

def test_api_endpoints():
    client = TestClient(app)
    
    # GET /api/live/status
    res = client.get("/api/live/status")
    assert res.status_code == 200
    data = res.json()
    assert "capture_backend" in data
    assert data["capture_backend"] == "npcap"

    # GET /api/live/capture-status
    res = client.get("/api/live/capture-status")
    assert res.status_code == 200
    cdata = res.json()
    assert cdata["capture_backend"] == "npcap"
    assert "packets_captured" in cdata
    assert "flows_created" in cdata

    # POST /api/live/reset
    res = client.post("/api/live/reset")
    assert res.status_code == 200
    assert res.json()["status"] == "reset"

    print("\n[TEST PASS] All live API endpoints verified.")
