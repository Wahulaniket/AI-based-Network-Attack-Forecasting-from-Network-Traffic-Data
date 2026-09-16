# Live Feature Mapping Audit

## Overview
This document audits the 89 SET_R features expected by the frozen Phase 2.3 model. 
The objective is to determine if each feature can be faithfully calculated in real-time from live packet captures (e.g. Scapy). 
**Rule:** No feature may be filled with zeroes or approximated if it cannot be faithfully extracted.

## Feature Mapping Table

| Feature Name | CIC-IDS2018 Meaning | Live Packet Source | Live Calculation | Availability |
| :--- | :--- | :--- | :--- | :--- |
| **Dst Port** | Destination port | TCP/UDP header | Exact extraction | Exact |
| **Protocol** | IP protocol number | IP header | Exact extraction | Exact |
| **Flow Duration** | Duration of the flow (µs) | Packet timestamps | `last_pkt_time - first_pkt_time` | Exact |
| **Tot Fwd Pkts** | Total forward packets | Packet direction | Count | Exact |
| **Tot Bwd Pkts** | Total backward packets | Packet direction | Count | Exact |
| **TotLen Fwd Pkts** | Total bytes in forward direction | IP length field | Sum of lengths | Exact |
| **TotLen Bwd Pkts** | Total bytes in backward direction | IP length field | Sum of lengths | Exact |
| **Fwd Pkt Len Max** | Maximum length of forward packet | Packet length | `max(fwd_lengths)` | Exact |
| **Fwd Pkt Len Min** | Minimum length of forward packet | Packet length | `min(fwd_lengths)` | Exact |
| **Fwd Pkt Len Mean** | Mean length of forward packet | Packet length | `mean(fwd_lengths)` | Exact |
| **Fwd Pkt Len Std** | Standard deviation of forward packet length | Packet length | `std(fwd_lengths)` | Exact |
| **Bwd Pkt Len Max** | Maximum length of backward packet | Packet length | `max(bwd_lengths)` | Exact |
| **Bwd Pkt Len Min** | Minimum length of backward packet | Packet length | `min(bwd_lengths)` | Exact |
| **Bwd Pkt Len Mean** | Mean length of backward packet | Packet length | `mean(bwd_lengths)` | Exact |
| **Bwd Pkt Len Std** | Standard deviation of backward packet length | Packet length | `std(bwd_lengths)` | Exact |
| **Flow Byts/s** | Flow byte rate | `TotLen / Flow Duration` | See `feature_pipeline.py` | Exact |
| **Flow Pkts/s** | Flow packet rate | `TotPkts / Flow Duration` | See `feature_pipeline.py` | Exact |
| **Flow IAT Mean** | Mean Inter-Arrival Time | Timestamps | `mean(diff(timestamps))` | Exact |
| **Flow IAT Std** | Std dev of Inter-Arrival Time | Timestamps | `std(diff(timestamps))` | Exact |
| **Flow IAT Max** | Maximum Inter-Arrival Time | Timestamps | `max(diff(timestamps))` | Exact |
| **Flow IAT Min** | Minimum Inter-Arrival Time | Timestamps | `min(diff(timestamps))` | Exact |
| **Fwd IAT Tot** | Total forward IAT | Timestamps (Fwd) | `sum(diff(fwd_timestamps))` | Exact |
| **Fwd IAT Mean** | Mean forward IAT | Timestamps (Fwd) | `mean(diff(fwd_timestamps))` | Exact |
| **Fwd IAT Std** | Std dev of forward IAT | Timestamps (Fwd) | `std(diff(fwd_timestamps))` | Exact |
| **Fwd IAT Max** | Max forward IAT | Timestamps (Fwd) | `max(diff(fwd_timestamps))` | Exact |
| **Fwd IAT Min** | Min forward IAT | Timestamps (Fwd) | `min(diff(fwd_timestamps))` | Exact |
| **Bwd IAT Tot** | Total backward IAT | Timestamps (Bwd) | `sum(diff(bwd_timestamps))` | Exact |
| **Bwd IAT Mean** | Mean backward IAT | Timestamps (Bwd) | `mean(diff(bwd_timestamps))` | Exact |
| **Bwd IAT Std** | Std dev of backward IAT | Timestamps (Bwd) | `std(diff(bwd_timestamps))` | Exact |
| **Bwd IAT Max** | Max backward IAT | Timestamps (Bwd) | `max(diff(bwd_timestamps))` | Exact |
| **Bwd IAT Min** | Min backward IAT | Timestamps (Bwd) | `min(diff(bwd_timestamps))` | Exact |
| **Fwd PSH Flags** | Count of PSH flags (fwd) | TCP Header | Count | Exact |
| **Bwd PSH Flags** | Count of PSH flags (bwd) | TCP Header | Count | Exact |
| **Fwd URG Flags** | Count of URG flags (fwd) | TCP Header | Count | Exact |
| **Bwd URG Flags** | Count of URG flags (bwd) | TCP Header | Count | Exact |
| **Fwd Header Len** | Total bytes used for fwd headers | IP/TCP headers | Sum of header lengths | Exact |
| **Bwd Header Len** | Total bytes used for bwd headers | IP/TCP headers | Sum of header lengths | Exact |
| **Fwd Pkts/s** | Forward packets per second | Tot Fwd Pkts / Duration | `fwd_pkts / duration` | Exact |
| **Bwd Pkts/s** | Backward packets per second | Tot Bwd Pkts / Duration | `bwd_pkts / duration` | Exact |
| **Pkt Len Min** | Min length of a flow | Packet length | `min(all_lengths)` | Exact |
| **Pkt Len Max** | Max length of a flow | Packet length | `max(all_lengths)` | Exact |
| **Pkt Len Mean** | Mean length of a flow | Packet length | `mean(all_lengths)` | Exact |
| **Pkt Len Std** | Std dev of a flow length | Packet length | `std(all_lengths)` | Exact |
| **Pkt Len Var** | Variance of a flow length | Packet length | `var(all_lengths)` | Exact |
| **FIN Flag Cnt** | Count of FIN flags | TCP Header | Count | Exact |
| **SYN Flag Cnt** | Count of SYN flags | TCP Header | Count | Exact |
| **RST Flag Cnt** | Count of RST flags | TCP Header | Count | Exact |
| **PSH Flag Cnt** | Count of PSH flags | TCP Header | Count | Exact |
| **ACK Flag Cnt** | Count of ACK flags | TCP Header | Count | Exact |
| **URG Flag Cnt** | Count of URG flags | TCP Header | Count | Exact |
| **CWE Flag Count** | Count of CWE flags | TCP Header | Count | Exact |
| **ECE Flag Cnt** | Count of ECE flags | TCP Header | Count | Exact |
| **Down/Up Ratio** | Download / Upload ratio | Packet counts | `Tot Bwd Pkts / Tot Fwd Pkts` | Exact |
| **Pkt Size Avg** | Average packet size | Packet length | Same as Pkt Len Mean | Exact |
| **Fwd Seg Size Avg** | Average forward segment size | Packet length | Same as Fwd Pkt Len Mean | Exact |
| **Bwd Seg Size Avg** | Average backward segment size | Packet length | Same as Bwd Pkt Len Mean | Exact |
| **Fwd Byts/b Avg** | Avg bytes/bulk (fwd) | Subflow logic | Complex CICFlowMeter state | UNAVAILABLE |
| **Fwd Pkts/b Avg** | Avg pkts/bulk (fwd) | Subflow logic | Complex CICFlowMeter state | UNAVAILABLE |
| **Fwd Blk Rate Avg** | Avg bulk rate (fwd) | Subflow logic | Complex CICFlowMeter state | UNAVAILABLE |
| **Bwd Byts/b Avg** | Avg bytes/bulk (bwd) | Subflow logic | Complex CICFlowMeter state | UNAVAILABLE |
| **Bwd Pkts/b Avg** | Avg pkts/bulk (bwd) | Subflow logic | Complex CICFlowMeter state | UNAVAILABLE |
| **Bwd Blk Rate Avg** | Avg bulk rate (bwd) | Subflow logic | Complex CICFlowMeter state | UNAVAILABLE |
| **Subflow Fwd Pkts** | Subflow Fwd Packets | Subflow logic | Complex CICFlowMeter state | UNAVAILABLE |
| **Subflow Fwd Byts** | Subflow Fwd Bytes | Subflow logic | Complex CICFlowMeter state | UNAVAILABLE |
| **Subflow Bwd Pkts** | Subflow Bwd Packets | Subflow logic | Complex CICFlowMeter state | UNAVAILABLE |
| **Subflow Bwd Byts** | Subflow Bwd Bytes | Subflow logic | Complex CICFlowMeter state | UNAVAILABLE |
| **Init Fwd Win Byts** | Initial Fwd Window Size | TCP Header | First Fwd TCP Window | Exact |
| **Init Bwd Win Byts** | Initial Bwd Window Size | TCP Header | First Bwd TCP Window | Exact |
| **Fwd Act Data Pkts** | Fwd active data packets | Packet payload | Count (fwd pkts with payload > 0) | Exact |
| **Fwd Seg Size Min** | Minimum forward segment size | IP/TCP Headers | Min Fwd Header Length | Exact |
| **Active Mean** | Mean active time before idle | Flow timeout logic | Complex CICFlowMeter state | UNAVAILABLE |
| **Active Std** | Std dev active time | Flow timeout logic | Complex CICFlowMeter state | UNAVAILABLE |
| **Active Max** | Max active time | Flow timeout logic | Complex CICFlowMeter state | UNAVAILABLE |
| **Active Min** | Min active time | Flow timeout logic | Complex CICFlowMeter state | UNAVAILABLE |
| **Idle Mean** | Mean idle time | Flow timeout logic | Complex CICFlowMeter state | UNAVAILABLE |
| **Idle Std** | Std dev idle time | Flow timeout logic | Complex CICFlowMeter state | UNAVAILABLE |
| **Idle Max** | Max idle time | Flow timeout logic | Complex CICFlowMeter state | UNAVAILABLE |
| **Idle Min** | Min idle time | Flow timeout logic | Complex CICFlowMeter state | UNAVAILABLE |
| **Fwd_Bwd_Pkt_Ratio** | Custom Ratio | Derived | Handled in `feature_pipeline.py` | Exact |
| **Fwd_Bwd_Byte_Ratio** | Custom Ratio | Derived | Handled in `feature_pipeline.py` | Exact |
| **Total_Pkts** | Total Packets | Derived | Handled in `feature_pipeline.py` | Exact |
| **Total_Bytes** | Total Bytes | Derived | Handled in `feature_pipeline.py` | Exact |
| **Avg_Pkt_Size** | Custom Size | Derived | Handled in `feature_pipeline.py` | Exact |
| **Pkts_Per_Sec** | Rate | Derived | Handled in `feature_pipeline.py` | Exact |
| **Bytes_Per_Sec** | Rate | Derived | Handled in `feature_pipeline.py` | Exact |
| **SYN_FIN_Ratio** | Ratio | Derived | Handled in `feature_pipeline.py` | Exact |
| **RST_SYN_Ratio** | Ratio | Derived | Handled in `feature_pipeline.py` | Exact |
| **Fwd_Pkt_Proportion**| Proportion | Derived | Handled in `feature_pipeline.py` | Exact |
| **flow_count** | Window agg | Derived | Handled in `windowing.py` | Exact |

## Conclusion
A significant subset of the 89 features rely on complex CICFlowMeter "Bulk" and "Active/Idle" subflow heuristics (`Fwd Byts/b Avg`, `Active Std`, `Idle Min`, etc.). 

Since we cannot extract these safely and perform exactly the same state-tracking as Java-based CICFlowMeter without a dedicated C/Java flow exporter, **18 features are UNAVAILABLE**.

Because we are strictly prohibited from zero-filling or fabricating these values, **we must fail the Compatibility Gate**.

Live Lab Mode will capture traffic, map available features, and correctly halt at the compatibility check, displaying: `Incomplete SET_R feature representation`.
