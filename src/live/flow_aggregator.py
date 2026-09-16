import time
import math
import threading
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import pandas as pd
import scapy.all as scapy

class FlowState:
    """Tracks state and statistics for a single bidirectional 5-tuple flow."""
    def __init__(self, src_ip: str, dst_ip: str, src_port: int, dst_port: int, protocol: int, start_time: float):
        self.src_ip = src_ip
        self.dst_ip = dst_ip
        self.src_port = src_port
        self.dst_port = dst_port
        self.protocol = protocol
        
        self.start_time = start_time
        self.last_seen = start_time
        
        # Packet counts & Bytes
        self.tot_fwd_pkts = 0
        self.tot_bwd_pkts = 0
        self.totlen_fwd_pkts = 0
        self.totlen_bwd_pkts = 0
        
        # Length lists
        self.fwd_pkt_lens: List[int] = []
        self.bwd_pkt_lens: List[int] = []
        self.all_pkt_lens: List[int] = []
        
        # Timestamps
        self.fwd_timestamps: List[float] = []
        self.bwd_timestamps: List[float] = []
        self.all_timestamps: List[float] = []
        
        # Header & TCP details
        self.fwd_header_len = 0
        self.bwd_header_len = 0
        self.init_fwd_win_byts = 0
        self.init_bwd_win_byts = 0
        self.fwd_act_data_pkts = 0
        self.fwd_seg_size_min = 0
        
        # TCP Flags
        self.fin_cnt = 0
        self.syn_cnt = 0
        self.rst_cnt = 0
        self.psh_cnt = 0
        self.ack_cnt = 0
        self.urg_cnt = 0
        self.cwe_cnt = 0
        self.ece_cnt = 0
        
        self.fwd_psh_cnt = 0
        self.bwd_psh_cnt = 0
        self.fwd_urg_cnt = 0
        self.bwd_urg_cnt = 0
        
        # Active/Idle state tracking
        self.active_times: List[float] = []
        self.idle_times: List[float] = []
        self.last_active_start = start_time
        self.last_active_end = start_time

    def add_packet(self, pkt, is_fwd: bool, pkt_time: float):
        pkt_len = len(pkt)
        self.last_seen = pkt_time
        self.all_timestamps.append(pkt_time)
        self.all_pkt_lens.append(pkt_len)

        # Header Length calculation
        hdr_len = 0
        if pkt.haslayer(scapy.IP):
            ihl = getattr(pkt[scapy.IP], 'ihl', 5) or 5
            hdr_len += ihl * 4
        elif pkt.haslayer(scapy.IPv6):
            hdr_len += 40

        if pkt.haslayer(scapy.TCP):
            dataofs = getattr(pkt[scapy.TCP], 'dataofs', 5) or 5
            hdr_len += dataofs * 4

            tcp = pkt[scapy.TCP]
            flags = str(tcp.flags)
            
            if 'F' in flags: self.fin_cnt += 1
            if 'S' in flags: self.syn_cnt += 1
            if 'R' in flags: self.rst_cnt += 1
            if 'P' in flags: 
                self.psh_cnt += 1
                if is_fwd: self.fwd_psh_cnt += 1
                else: self.bwd_psh_cnt += 1
            if 'A' in flags: self.ack_cnt += 1
            if 'U' in flags: 
                self.urg_cnt += 1
                if is_fwd: self.fwd_urg_cnt += 1
                else: self.bwd_urg_cnt += 1
            if 'E' in flags: self.ece_cnt += 1
            if 'C' in flags: self.cwe_cnt += 1

            if is_fwd:
                if self.init_fwd_win_byts == 0:
                    self.init_fwd_win_byts = tcp.window
            else:
                if self.init_bwd_win_byts == 0:
                    self.init_bwd_win_byts = tcp.window

            # Payload length for active data packets
            payload_len = len(tcp.payload)
            if is_fwd and payload_len > 0:
                self.fwd_act_data_pkts += 1

        elif pkt.haslayer(scapy.UDP):
            hdr_len += 8

        if is_fwd:
            self.tot_fwd_pkts += 1
            self.totlen_fwd_pkts += pkt_len
            self.fwd_pkt_lens.append(pkt_len)
            self.fwd_timestamps.append(pkt_time)
            self.fwd_header_len += hdr_len
            if self.fwd_seg_size_min == 0 or hdr_len < self.fwd_seg_size_min:
                self.fwd_seg_size_min = hdr_len
        else:
            self.tot_bwd_pkts += 1
            self.totlen_bwd_pkts += pkt_len
            self.bwd_pkt_lens.append(pkt_len)
            self.bwd_timestamps.append(pkt_time)
            self.bwd_header_len += hdr_len

        # Idle tracking (> 1.0 second threshold)
        if len(self.all_timestamps) > 1:
            iat = pkt_time - self.all_timestamps[-2]
            if iat > 1.0:
                self.idle_times.append(iat * 1e6) # microseconds
                active_dur = (self.all_timestamps[-2] - self.last_active_start) * 1e6
                if active_dur > 0:
                    self.active_times.append(active_dur)
                self.last_active_start = pkt_time

    def to_dict(self) -> Dict[str, Any]:
        duration_sec = max(self.last_seen - self.start_time, 1e-6)
        duration_us = duration_sec * 1e6

        def calc_stats(arr):
            if not arr:
                return 0.0, 0.0, 0.0, 0.0
            a = np.array(arr, dtype=np.float64)
            return float(np.min(a)), float(np.max(a)), float(np.mean(a)), float(np.std(a))

        def calc_iats(ts_list):
            if len(ts_list) < 2:
                return 0.0, 0.0, 0.0, 0.0, 0.0
            arr = np.diff(ts_list) * 1e6 # microseconds
            return float(np.sum(arr)), float(np.mean(arr)), float(np.std(arr)), float(np.max(arr)), float(np.min(arr))

        fwd_min, fwd_max, fwd_mean, fwd_std = calc_stats(self.fwd_pkt_lens)
        bwd_min, bwd_max, bwd_mean, bwd_std = calc_stats(self.bwd_pkt_lens)
        all_min, all_max, all_mean, all_std = calc_stats(self.all_pkt_lens)
        all_var = float(np.var(self.all_pkt_lens)) if self.all_pkt_lens else 0.0

        flow_tot_iat, flow_iat_mean, flow_iat_std, flow_iat_max, flow_iat_min = calc_iats(self.all_timestamps)
        fwd_tot_iat, fwd_iat_mean, fwd_iat_std, fwd_iat_max, fwd_iat_min = calc_iats(self.fwd_timestamps)
        bwd_tot_iat, bwd_iat_mean, bwd_iat_std, bwd_iat_max, bwd_iat_min = calc_iats(self.bwd_timestamps)

        act_min, act_max, act_mean, act_std = calc_stats(self.active_times)
        idle_min, idle_max, idle_mean, idle_std = calc_stats(self.idle_times)

        total_pkts = self.tot_fwd_pkts + self.tot_bwd_pkts
        total_byts = self.totlen_fwd_pkts + self.totlen_bwd_pkts

        flow_byts_s = (total_byts / duration_sec) if duration_sec > 0 else 0.0
        flow_pkts_s = (total_pkts / duration_sec) if duration_sec > 0 else 0.0
        fwd_pkts_s = (self.tot_fwd_pkts / duration_sec) if duration_sec > 0 else 0.0
        bwd_pkts_s = (self.tot_bwd_pkts / duration_sec) if duration_sec > 0 else 0.0

        down_up_ratio = (self.tot_bwd_pkts / self.tot_fwd_pkts) if self.tot_fwd_pkts > 0 else 0.0

        return {
            "Dst Port": self.dst_port,
            "Protocol": self.protocol,
            "Flow Duration": duration_us,
            "Tot Fwd Pkts": self.tot_fwd_pkts,
            "Tot Bwd Pkts": self.tot_bwd_pkts,
            "TotLen Fwd Pkts": self.totlen_fwd_pkts,
            "TotLen Bwd Pkts": self.totlen_bwd_pkts,
            "Fwd Pkt Len Max": fwd_max,
            "Fwd Pkt Len Min": fwd_min,
            "Fwd Pkt Len Mean": fwd_mean,
            "Fwd Pkt Len Std": fwd_std,
            "Bwd Pkt Len Max": bwd_max,
            "Bwd Pkt Len Min": bwd_min,
            "Bwd Pkt Len Mean": bwd_mean,
            "Bwd Pkt Len Std": bwd_std,
            "Flow Byts/s": flow_byts_s,
            "Flow Pkts/s": flow_pkts_s,
            "Flow IAT Mean": flow_iat_mean,
            "Flow IAT Std": flow_iat_std,
            "Flow IAT Max": flow_iat_max,
            "Flow IAT Min": flow_iat_min,
            "Fwd IAT Tot": fwd_tot_iat,
            "Fwd IAT Mean": fwd_iat_mean,
            "Fwd IAT Std": fwd_iat_std,
            "Fwd IAT Max": fwd_iat_max,
            "Fwd IAT Min": fwd_iat_min,
            "Bwd IAT Tot": bwd_tot_iat,
            "Bwd IAT Mean": bwd_iat_mean,
            "Bwd IAT Std": bwd_iat_std,
            "Bwd IAT Max": bwd_iat_max,
            "Bwd IAT Min": bwd_iat_min,
            "Fwd PSH Flags": self.fwd_psh_cnt,
            "Bwd PSH Flags": self.bwd_psh_cnt,
            "Fwd URG Flags": self.fwd_urg_cnt,
            "Bwd URG Flags": self.bwd_urg_cnt,
            "Fwd Header Len": self.fwd_header_len,
            "Bwd Header Len": self.bwd_header_len,
            "Fwd Pkts/s": fwd_pkts_s,
            "Bwd Pkts/s": bwd_pkts_s,
            "Pkt Len Min": all_min,
            "Pkt Len Max": all_max,
            "Pkt Len Mean": all_mean,
            "Pkt Len Std": all_std,
            "Pkt Len Var": all_var,
            "FIN Flag Cnt": self.fin_cnt,
            "SYN Flag Cnt": self.syn_cnt,
            "RST Flag Cnt": self.rst_cnt,
            "PSH Flag Cnt": self.psh_cnt,
            "ACK Flag Cnt": self.ack_cnt,
            "URG Flag Cnt": self.urg_cnt,
            "CWE Flag Count": self.cwe_cnt,
            "ECE Flag Cnt": self.ece_cnt,
            "Down/Up Ratio": down_up_ratio,
            "Pkt Size Avg": all_mean,
            "Fwd Seg Size Avg": fwd_mean,
            "Bwd Seg Size Avg": bwd_mean,
            "Fwd Byts/b Avg": 0.0,
            "Fwd Pkts/b Avg": 0.0,
            "Fwd Blk Rate Avg": 0.0,
            "Bwd Byts/b Avg": 0.0,
            "Bwd Pkts/b Avg": 0.0,
            "Bwd Blk Rate Avg": 0.0,
            "Subflow Fwd Pkts": self.tot_fwd_pkts,
            "Subflow Fwd Byts": self.totlen_fwd_pkts,
            "Subflow Bwd Pkts": self.tot_bwd_pkts,
            "Subflow Bwd Byts": self.totlen_bwd_pkts,
            "Init Fwd Win Byts": self.init_fwd_win_byts,
            "Init Bwd Win Byts": self.init_bwd_win_byts,
            "Fwd Act Data Pkts": self.fwd_act_data_pkts,
            "Fwd Seg Size Min": self.fwd_seg_size_min,
            "Active Mean": act_mean,
            "Active Std": act_std,
            "Active Max": act_max,
            "Active Min": act_min,
            "Idle Mean": idle_mean,
            "Idle Std": idle_std,
            "Idle Max": idle_max,
            "Idle Min": idle_min
        }

class FlowAggregator:
    """Thread-safe flow aggregator mapping captured packets to 5-tuple bidirectional flows."""
    def __init__(self):
        self.lock = threading.Lock()
        self.flows: Dict[Tuple[str, str, int, int, int], FlowState] = {}
        self.total_packets_processed = 0

    def process_packet(self, pkt):
        """Extracts 5-tuple and updates active flow state."""
        if not (pkt.haslayer(scapy.IP) or pkt.haslayer(scapy.IPv6)):
            return

        pkt_time = float(getattr(pkt, 'time', time.time()))

        if pkt.haslayer(scapy.IP):
            src_ip = pkt[scapy.IP].src
            dst_ip = pkt[scapy.IP].dst
            proto = pkt[scapy.IP].proto
        else:
            src_ip = pkt[scapy.IPv6].src
            dst_ip = pkt[scapy.IPv6].dst
            proto = pkt[scapy.IPv6].nh

        src_port = 0
        dst_port = 0

        if pkt.haslayer(scapy.TCP):
            src_port = pkt[scapy.TCP].sport
            dst_port = pkt[scapy.TCP].dport
        elif pkt.haslayer(scapy.UDP):
            src_port = pkt[scapy.UDP].sport
            dst_port = pkt[scapy.UDP].dport

        # Bidirectional 5-tuple lookup
        fwd_key = (src_ip, dst_ip, src_port, dst_port, proto)
        bwd_key = (dst_ip, src_ip, dst_port, src_port, proto)

        with self.lock:
            self.total_packets_processed += 1
            if fwd_key in self.flows:
                self.flows[fwd_key].add_packet(pkt, is_fwd=True, pkt_time=pkt_time)
            elif bwd_key in self.flows:
                self.flows[bwd_key].add_packet(pkt, is_fwd=False, pkt_time=pkt_time)
            else:
                # Create new flow
                new_flow = FlowState(src_ip, dst_ip, src_port, dst_port, proto, pkt_time)
                new_flow.add_packet(pkt, is_fwd=True, pkt_time=pkt_time)
                self.flows[fwd_key] = new_flow

    def flush_flows(self) -> List[Dict[str, Any]]:
        """Flushes and returns all accumulated flows as dictionaries."""
        with self.lock:
            flushed = [flow.to_dict() for flow in self.flows.values()]
            self.flows.clear()
            return flushed

    def get_flow_count(self) -> int:
        with self.lock:
            return len(self.flows)

    def validate_feature_coverage(self, df_flows: pd.DataFrame, expected_89_features: List[str]) -> Tuple[bool, float, str, List[str]]:
        """Verifies if the aggregated flows cover all 89 SET_R features.
        
        Returns:
            (is_compatible, coverage_percent, status_str, missing_features)
        """
        if df_flows.empty:
            return False, 0.0, "FAILED", expected_89_features

        missing = [f for f in expected_89_features if f not in df_flows.columns]
        coverage_pct = round(((len(expected_89_features) - len(missing)) / len(expected_89_features)) * 100.0, 2)

        if len(missing) == 0:
            status = "FULL"
            is_compat = True
        elif coverage_pct >= 90.0:
            status = "PARTIAL"
            is_compat = True
        else:
            status = "FAILED"
            is_compat = False

        return is_compat, coverage_pct, status, missing
