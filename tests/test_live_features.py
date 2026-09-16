import os
import sys
import pytest
from datetime import datetime
import pandas as pd

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from src.live.live_engine import LiveEngine

def test_live_engine_feature_mapping():
    engine = LiveEngine(repo_root)
    # create a mock cicflowmeter output with all fields
    mock_flow = {
        "dst_port": 80,
        "protocol": 6,
        "flow_duration": 15000,
        "tot_fwd_pkts": 5,
        "tot_bwd_pkts": 3,
        "totlen_fwd_pkts": 500,
        "totlen_bwd_pkts": 300,
        "fwd_pkt_len_max": 100,
        "fwd_pkt_len_min": 100,
        "fwd_pkt_len_mean": 100,
        "fwd_pkt_len_std": 0,
        "bwd_pkt_len_max": 100,
        "bwd_pkt_len_min": 100,
        "bwd_pkt_len_mean": 100,
        "bwd_pkt_len_std": 0,
        "flow_byts_s": 53333.3,
        "flow_pkts_s": 533.3,
        "flow_iat_mean": 2142.8,
        "flow_iat_std": 100.0,
        "flow_iat_max": 2500,
        "flow_iat_min": 2000,
        "fwd_iat_tot": 15000,
        "fwd_iat_mean": 3750,
        "fwd_iat_std": 50,
        "fwd_iat_max": 4000,
        "fwd_iat_min": 3500,
        "bwd_iat_tot": 10000,
        "bwd_iat_mean": 5000,
        "bwd_iat_std": 50,
        "bwd_iat_max": 5200,
        "bwd_iat_min": 4800,
        "fwd_psh_flags": 0,
        "bwd_psh_flags": 0,
        "fwd_urg_flags": 0,
        "bwd_urg_flags": 0,
        "fwd_header_len": 100,
        "bwd_header_len": 60,
        "fwd_pkts_s": 333.3,
        "bwd_pkts_s": 200.0,
        "pkt_len_min": 100,
        "pkt_len_max": 100,
        "pkt_len_mean": 100,
        "pkt_len_std": 0,
        "pkt_len_var": 0,
        "fin_flag_cnt": 0,
        "syn_flag_cnt": 1,
        "rst_flag_cnt": 0,
        "psh_flag_cnt": 1,
        "ack_flag_cnt": 1,
        "urg_flag_cnt": 0,
        "cwe_flag_count": 0,
        "ece_flag_cnt": 0,
        "down_up_ratio": 0.6,
        "pkt_size_avg": 100,
        "fwd_seg_size_avg": 100,
        "bwd_seg_size_avg": 100,
        "fwd_byts_b_avg": 0,
        "fwd_pkts_b_avg": 0,
        "fwd_blk_rate_avg": 0,
        "bwd_byts_b_avg": 0,
        "bwd_pkts_b_avg": 0,
        "bwd_blk_rate_avg": 0,
        "subflow_fwd_pkts": 5,
        "subflow_fwd_byts": 500,
        "subflow_bwd_pkts": 3,
        "subflow_bwd_byts": 300,
        "init_fwd_win_byts": 65535,
        "init_bwd_win_byts": 65535,
        "fwd_act_data_pkts": 5,
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
    
    df_raw = pd.DataFrame([mock_flow])
    df_mapped = engine._map_cic_to_set_r(df_raw)
    
    # Verify title case mapping exists
    assert "Dst Port" in df_mapped.columns
    assert "Flow Duration" in df_mapped.columns
    
    # Ensure no Target variables exist
    target_cols = ['binary_attack', 'dominant_label', 'attack_ratio', 'attack_flow_count', 'Label', 'Attack']
    for t in target_cols:
        assert t not in df_mapped.columns
