# CICFlowMeter Live Compatibility Audit

## Overview
This document evaluates the compatibility of using the Python `cicflowmeter` live exporter (`cicflowmeter -i <interface>`) as the real-time feature source for CyberCast's Phase 2.3 model.

## Feature Mapping Table

The Python `cicflowmeter` package outputs exact CIC-IDS2018 features, but uses `snake_case` naming conventions. The table below maps these to the exact `Title Case` names expected by the frozen `feature_pipeline.py`.

| Frozen SET_R Feature | Required? | CICFlowMeter Output? | Exact/Equivalent Name | Transformation Required | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Dst Port** | Yes | Yes | `dst_port` | Rename | Exact |
| **Protocol** | Yes | Yes | `protocol` | Rename | Exact |
| **Flow Duration** | Yes | Yes | `flow_duration` | Rename | Exact |
| **Tot Fwd Pkts** | Yes | Yes | `tot_fwd_pkts` | Rename | Exact |
| **Tot Bwd Pkts** | Yes | Yes | `tot_bwd_pkts` | Rename | Exact |
| **TotLen Fwd Pkts** | Yes | Yes | `totlen_fwd_pkts` | Rename | Exact |
| **TotLen Bwd Pkts** | Yes | Yes | `totlen_bwd_pkts` | Rename | Exact |
| **Fwd Pkt Len Max** | Yes | Yes | `fwd_pkt_len_max` | Rename | Exact |
| **Fwd Pkt Len Min** | Yes | Yes | `fwd_pkt_len_min` | Rename | Exact |
| **Fwd Pkt Len Mean** | Yes | Yes | `fwd_pkt_len_mean` | Rename | Exact |
| **Fwd Pkt Len Std** | Yes | Yes | `fwd_pkt_len_std` | Rename | Exact |
| **Bwd Pkt Len Max** | Yes | Yes | `bwd_pkt_len_max` | Rename | Exact |
| **Bwd Pkt Len Min** | Yes | Yes | `bwd_pkt_len_min` | Rename | Exact |
| **Bwd Pkt Len Mean** | Yes | Yes | `bwd_pkt_len_mean` | Rename | Exact |
| **Bwd Pkt Len Std** | Yes | Yes | `bwd_pkt_len_std` | Rename | Exact |
| **Flow Byts/s** | Yes | Yes | `flow_byts_s` | Rename | Exact |
| **Flow Pkts/s** | Yes | Yes | `flow_pkts_s` | Rename | Exact |
| **Flow IAT Mean** | Yes | Yes | `flow_iat_mean` | Rename | Exact |
| **Flow IAT Std** | Yes | Yes | `flow_iat_std` | Rename | Exact |
| **Flow IAT Max** | Yes | Yes | `flow_iat_max` | Rename | Exact |
| **Flow IAT Min** | Yes | Yes | `flow_iat_min` | Rename | Exact |
| **Fwd IAT Tot** | Yes | Yes | `fwd_iat_tot` | Rename | Exact |
| **Fwd IAT Mean** | Yes | Yes | `fwd_iat_mean` | Rename | Exact |
| **Fwd IAT Std** | Yes | Yes | `fwd_iat_std` | Rename | Exact |
| **Fwd IAT Max** | Yes | Yes | `fwd_iat_max` | Rename | Exact |
| **Fwd IAT Min** | Yes | Yes | `fwd_iat_min` | Rename | Exact |
| **Bwd IAT Tot** | Yes | Yes | `bwd_iat_tot` | Rename | Exact |
| **Bwd IAT Mean** | Yes | Yes | `bwd_iat_mean` | Rename | Exact |
| **Bwd IAT Std** | Yes | Yes | `bwd_iat_std` | Rename | Exact |
| **Bwd IAT Max** | Yes | Yes | `bwd_iat_max` | Rename | Exact |
| **Bwd IAT Min** | Yes | Yes | `bwd_iat_min` | Rename | Exact |
| **Fwd PSH Flags** | Yes | Yes | `fwd_psh_flags` | Rename | Exact |
| **Bwd PSH Flags** | Yes | Yes | `bwd_psh_flags` | Rename | Exact |
| **Fwd URG Flags** | Yes | Yes | `fwd_urg_flags` | Rename | Exact |
| **Bwd URG Flags** | Yes | Yes | `bwd_urg_flags` | Rename | Exact |
| **Fwd Header Len** | Yes | Yes | `fwd_header_len` | Rename | Exact |
| **Bwd Header Len** | Yes | Yes | `bwd_header_len` | Rename | Exact |
| **Fwd Pkts/s** | Yes | Yes | `fwd_pkts_s` | Rename | Exact |
| **Bwd Pkts/s** | Yes | Yes | `bwd_pkts_s` | Rename | Exact |
| **Pkt Len Min** | Yes | Yes | `pkt_len_min` | Rename | Exact |
| **Pkt Len Max** | Yes | Yes | `pkt_len_max` | Rename | Exact |
| **Pkt Len Mean** | Yes | Yes | `pkt_len_mean` | Rename | Exact |
| **Pkt Len Std** | Yes | Yes | `pkt_len_std` | Rename | Exact |
| **Pkt Len Var** | Yes | Yes | `pkt_len_var` | Rename | Exact |
| **FIN Flag Cnt** | Yes | Yes | `fin_flag_cnt` | Rename | Exact |
| **SYN Flag Cnt** | Yes | Yes | `syn_flag_cnt` | Rename | Exact |
| **RST Flag Cnt** | Yes | Yes | `rst_flag_cnt` | Rename | Exact |
| **PSH Flag Cnt** | Yes | Yes | `psh_flag_cnt` | Rename | Exact |
| **ACK Flag Cnt** | Yes | Yes | `ack_flag_cnt` | Rename | Exact |
| **URG Flag Cnt** | Yes | Yes | `urg_flag_cnt` | Rename | Exact |
| **CWE Flag Count** | Yes | Yes | `cwe_flag_count` | Rename | Exact |
| **ECE Flag Cnt** | Yes | Yes | `ece_flag_cnt` | Rename | Exact |
| **Down/Up Ratio** | Yes | Yes | `down_up_ratio` | Rename | Exact |
| **Pkt Size Avg** | Yes | Yes | `pkt_size_avg` | Rename | Exact |
| **Fwd Seg Size Avg** | Yes | Yes | `fwd_seg_size_avg` | Rename | Exact |
| **Bwd Seg Size Avg** | Yes | Yes | `bwd_seg_size_avg` | Rename | Exact |
| **Fwd Byts/b Avg** | Yes | Yes | `fwd_byts_b_avg` | Rename | Exact |
| **Fwd Pkts/b Avg** | Yes | Yes | `fwd_pkts_b_avg` | Rename | Exact |
| **Fwd Blk Rate Avg** | Yes | Yes | `fwd_blk_rate_avg` | Rename | Exact |
| **Bwd Byts/b Avg** | Yes | Yes | `bwd_byts_b_avg` | Rename | Exact |
| **Bwd Pkts/b Avg** | Yes | Yes | `bwd_pkts_b_avg` | Rename | Exact |
| **Bwd Blk Rate Avg** | Yes | Yes | `bwd_blk_rate_avg` | Rename | Exact |
| **Subflow Fwd Pkts** | Yes | Yes | `subflow_fwd_pkts` | Rename | Exact |
| **Subflow Fwd Byts** | Yes | Yes | `subflow_fwd_byts` | Rename | Exact |
| **Subflow Bwd Pkts** | Yes | Yes | `subflow_bwd_pkts` | Rename | Exact |
| **Subflow Bwd Byts** | Yes | Yes | `subflow_bwd_byts` | Rename | Exact |
| **Init Fwd Win Byts** | Yes | Yes | `init_fwd_win_byts` | Rename | Exact |
| **Init Bwd Win Byts** | Yes | Yes | `init_bwd_win_byts` | Rename | Exact |
| **Fwd Act Data Pkts** | Yes | Yes | `fwd_act_data_pkts` | Rename | Exact |
| **Fwd Seg Size Min** | Yes | Yes | `fwd_seg_size_min` | Rename | Exact |
| **Active Mean** | Yes | Yes | `active_mean` | Rename | Exact |
| **Active Std** | Yes | Yes | `active_std` | Rename | Exact |
| **Active Max** | Yes | Yes | `active_max` | Rename | Exact |
| **Active Min** | Yes | Yes | `active_min` | Rename | Exact |
| **Idle Mean** | Yes | Yes | `idle_mean` | Rename | Exact |
| **Idle Std** | Yes | Yes | `idle_std` | Rename | Exact |
| **Idle Max** | Yes | Yes | `idle_max` | Rename | Exact |
| **Idle Min** | Yes | Yes | `idle_min` | Rename | Exact |
| **Fwd_Bwd_Pkt_Ratio** | Yes | No | N/A | Derived in `feature_pipeline.py` | Derived |
| **Fwd_Bwd_Byte_Ratio** | Yes | No | N/A | Derived in `feature_pipeline.py` | Derived |
| **Total_Pkts** | Yes | No | N/A | Derived in `feature_pipeline.py` | Derived |
| **Total_Bytes** | Yes | No | N/A | Derived in `feature_pipeline.py` | Derived |
| **Avg_Pkt_Size** | Yes | No | N/A | Derived in `feature_pipeline.py` | Derived |
| **Pkts_Per_Sec** | Yes | No | N/A | Derived in `feature_pipeline.py` | Derived |
| **Bytes_Per_Sec** | Yes | No | N/A | Derived in `feature_pipeline.py` | Derived |
| **SYN_FIN_Ratio** | Yes | No | N/A | Derived in `feature_pipeline.py` | Derived |
| **RST_SYN_Ratio** | Yes | No | N/A | Derived in `feature_pipeline.py` | Derived |
| **Fwd_Pkt_Proportion**| Yes | No | N/A | Derived in `feature_pipeline.py` | Derived |
| **flow_count** | Yes | No | N/A | Derived in `windowing.py` | Derived |

## Conclusion

**MODEL-COMPATIBLE**

The Python `cicflowmeter` package exactly tracks the state required for the complex Bulk and Active/Idle features that custom Scapy scripts cannot easily replicate. By mapping the snake_case keys to Title Case keys, and then applying the existing `feature_pipeline.py` and `windowing.py` logic, we can successfully yield the exact 89-feature `SET_R` representation. No feature fabrication or zero-filling is required.
