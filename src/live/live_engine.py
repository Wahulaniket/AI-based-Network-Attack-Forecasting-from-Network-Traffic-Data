import os
import time
import json
import threading
import torch
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from collections import deque
from datetime import datetime, timedelta

from src.live.config import (
    PHASE2_3_RESULTS_DIR, 
    PHASE2_3_MODELS_DIR, 
    FROZEN_THRESHOLD, 
    FROZEN_HISTORY_WINDOWS,
    FROZEN_FEATURE_SET,
    WINDOW_SECONDS,
    LIVE_INTERFACE_NAME,
    LIVE_INTERFACE_INDEX,
    LIVE_HOST_IP
)
from src.live.schemas import LiveStatus, LivePrediction, LiveTrafficSummary
from src.live.npcap_capture import NpcapCapturer
from src.live.flow_aggregator import FlowAggregator
from src.inference.model_loader import load_inference_artifacts
from src.inference.feature_pipeline import process_features
from src.inference.windowing import create_time_windows, create_causal_sequences
from src.inference.risk_engine import calculate_risk_level

class LiveEngine:
    def __init__(self, repo_root: str):
        self.repo_root = repo_root
        import sys

        # Load frozen Phase 2.3 artifacts safely
        try:
            self.artifacts = load_inference_artifacts(self.repo_root)
            self.model = self.artifacts['model']
            self.scaler = self.artifacts['scaler']
            self.features = self.artifacts['features']
            self.threshold = float(self.artifacts['threshold'])
            self.device = self.artifacts['device']
        except Exception as e:
            print(f"[FATAL] Frozen Phase 2.3 artifacts could not be loaded.\nLive inference cannot start.\nError: {e}")
            sys.exit(1)

        # Flow Aggregator & Native Npcap Capturer
        self.flow_aggregator = FlowAggregator()
        self.capturer = NpcapCapturer(
            packet_callback=self.flow_aggregator.process_packet,
            on_network_changed=self._on_network_changed_callback
        )
            
        # State
        self.windows_processed = 0
        self.status = LiveStatus(
            interface=LIVE_INTERFACE_NAME,
            interface_index=LIVE_INTERFACE_INDEX,
            host_ip=LIVE_HOST_IP,
            window_seconds=WINDOW_SECONDS,
            history_required=FROZEN_HISTORY_WINDOWS,
            history_collected=0
        )
        self.traffic_summary = LiveTrafficSummary()
        self.latest_prediction: Optional[LivePrediction] = None
        self.feature_coverage_percent: float = 100.0
        self.feature_compatibility_status: str = "FULL"
        
        # Buffers
        self.flow_buffer = []  # raw incoming flows waiting to be windowed
        self.window_context = deque(maxlen=FROZEN_HISTORY_WINDOWS) # rolling 20 windows of 89 features
        self.window_stats_context = deque(maxlen=FROZEN_HISTORY_WINDOWS)
        
        # Timing state
        self.last_window_time = time.time()
        
        # Concurrency and Scheduling
        self.lock = threading.RLock()
        self.scheduler_thread = None
        self.stop_event = threading.Event()

    def start_capture(self, interface_override: Optional[str] = None):
        """Starts live packet capture via Windows Npcap and background 10s window scheduler."""
        with self.lock:
            self.status.capture_active = True
            self.last_window_time = time.time()
            
            print("[CyberCast Live] Starting native Npcap capture engine...")
            try:
                self.capturer.start(interface_override=interface_override)
                self.status.interface = self.capturer.interface_name
                self.status.interface_index = self.capturer.interface_index
                self.status.host_ip = self.capturer.current_ipv4
                self.status.accepted_interface = self.capturer.interface_name
                self.status.accepted_host_ip = self.capturer.current_ipv4
            except Exception as e:
                print(f"[CyberCast Live ERROR] Could not start Npcap capturer: {e}")

            print(f"[CyberCast Live] Interface: {self.status.interface}")
            print(f"[CyberCast Live] Interface Index: {self.status.interface_index}")
            print(f"[CyberCast Live] Host IPv4: {self.status.host_ip}")
            
            if self.scheduler_thread is None or not self.scheduler_thread.is_alive():
                self.stop_event.clear()
                self.scheduler_thread = threading.Thread(target=self._scheduler_loop, daemon=True)
                self.scheduler_thread.start()
        
    def stop_capture(self):
        """Stops Npcap capture and background scheduler cleanly."""
        with self.lock:
            self.status.capture_active = False
            self.stop_event.set()
            self.capturer.stop()
        print("[CyberCast Live] Npcap capture stopped")

    def reset_context(self):
        """Resets temporal 20-window context to 0/20."""
        with self.lock:
            self.window_context.clear()
            self.window_stats_context.clear()
            self.status.history_collected = 0
            self.status.model_ready = False
            self.status.rebuilding_context = False
            self.latest_prediction = None
            print("[CyberCast Live] Temporal context reset to 0/20")

    def _on_network_changed_callback(self):
        """Callback invoked when NpcapCapturer detects a network interface/IP change."""
        with self.lock:
            self.status.previous_host_ip = self.status.accepted_host_ip
            self.status.previous_interface = self.status.accepted_interface
            self.status.accepted_host_ip = self.capturer.current_ipv4
            self.status.accepted_interface = self.capturer.interface_name
            self.status.host_ip = self.capturer.current_ipv4
            self.status.interface = self.capturer.interface_name
            self.status.network_changed = True
            self.status.rebuilding_context = True
            self.status.model_ready = False
            
            # Reset temporal context to prevent mixing traffic across networks
            self.window_context.clear()
            self.window_stats_context.clear()
            self.status.history_collected = 0
            self.latest_prediction = None
            print(f"[CyberCast Live] NETWORK CHANGED: Temporal context reset to 0/20.")

    def _scheduler_loop(self):
        """Background thread executing process_window() every 10 seconds."""
        while not self.stop_event.is_set():
            time.sleep(1)
            with self.lock:
                if not self.status.capture_active:
                    continue
                now = time.time()
                while now - self.last_window_time >= WINDOW_SECONDS:
                    self.process_window()

    def get_status(self) -> Dict[str, Any]:
        """Returns engine status for GET /api/live/status."""
        with self.lock:
            return {
                "mode": self.status.mode,
                "capture_active": self.status.capture_active and self.capturer.capture_active,
                "capture_backend": "npcap",
                "interface": self.status.interface or self.capturer.interface_name,
                "interface_index": self.status.interface_index or self.capturer.interface_index,
                "host_ip": self.status.host_ip or self.capturer.current_ipv4,
                "previous_interface": self.status.previous_interface,
                "previous_host_ip": self.status.previous_host_ip,
                "accepted_interface": self.status.accepted_interface,
                "accepted_host_ip": self.status.accepted_host_ip,
                "network_changed": self.status.network_changed,
                "rebuilding_context": self.status.rebuilding_context,
                "model_name": self.status.model_name,
                "feature_set": self.status.feature_set,
                "window_seconds": self.status.window_seconds,
                "history_required": self.status.history_required,
                "history_collected": len(self.window_context),
                "model_ready": len(self.window_context) == self.status.history_required,
                "feature_coverage_percent": self.feature_coverage_percent,
                "feature_compatibility_status": self.feature_compatibility_status
            }

    def get_capture_status(self) -> Dict[str, Any]:
        """Returns diagnostic details for GET /api/live/capture-status."""
        diag = self.capturer.get_status_dict()
        diag["flows_created"] = self.capturer.flows_created + self.traffic_summary.flows_observed
        return diag
        
    def get_latest_prediction(self) -> Optional[Dict[str, Any]]:
        """Returns latest prediction or waiting state for GET /api/live/latest."""
        with self.lock:
            if self.latest_prediction:
                return self.latest_prediction.to_dict()
            return None

    def get_live_prediction(self) -> Dict[str, Any]:
        """Returns dedicated live prediction dict for GET /api/live/prediction."""
        with self.lock:
            pkts = self.capturer.packets_captured + self.traffic_summary.packets_observed
            flows = self.capturer.flows_created + self.traffic_summary.flows_observed
            cap_active = self.status.capture_active and self.capturer.capture_active
            cap_status = "ACTIVE" if cap_active else "INACTIVE"
            avail = len(self.window_context)
            req = FROZEN_HISTORY_WINDOWS

            if avail >= req and self.latest_prediction is not None:
                p = self.latest_prediction
                return {
                    "prediction_ready": True,
                    "prediction_type": "LIVE_MODEL_PREDICTION",
                    "timestamp": p.timestamp.isoformat() if p.timestamp else None,
                    "window_start": p.window_start.isoformat() if p.window_start else None,
                    "window_end": p.window_end.isoformat() if p.window_end else None,
                    "attack_probability": p.attack_probability,
                    "model_threshold": p.model_threshold,
                    "binary_prediction": p.binary_prediction,
                    "risk_level": p.risk_level,
                    "context_windows_available": avail,
                    "context_windows_required": req,
                    "packets_seen": pkts,
                    "flows_seen": flows,
                    "capture_status": cap_status,
                    "message": "Live model prediction active",
                    # Compatibility aliases
                    "prediction": p.binary_prediction,
                    "risk": p.risk_level,
                    "history_collected": avail,
                    "flow_count": p.flow_count,
                    "packet_count": p.packet_count,
                    "total_bytes": p.total_bytes
                }
            else:
                return {
                    "prediction_ready": False,
                    "prediction_type": "LIVE_MODEL_PREDICTION",
                    "timestamp": datetime.now().isoformat(),
                    "window_start": None,
                    "window_end": None,
                    "attack_probability": None,
                    "model_threshold": self.threshold,
                    "binary_prediction": None,
                    "risk_level": "INSUFFICIENT_CONTEXT",
                    "context_windows_available": avail,
                    "context_windows_required": req,
                    "packets_seen": pkts,
                    "flows_seen": flows,
                    "capture_status": cap_status,
                    "message": f"Insufficient live context: {avail}/{req} 10-second windows available. More live context is required.",
                    # Compatibility aliases
                    "prediction": None,
                    "risk": "INSUFFICIENT_CONTEXT",
                    "history_collected": avail
                }

    def get_diagnostics(self) -> Dict[str, Any]:
        """Returns diagnostic details for GET /api/live/diagnostics."""
        with self.lock:
            pkts = self.capturer.packets_captured + self.traffic_summary.packets_observed
            flows = self.capturer.flows_created + self.traffic_summary.flows_observed
            cap_running = self.status.capture_active and self.capturer.capture_active
            latest_ts = self.latest_prediction.timestamp.isoformat() if self.latest_prediction and self.latest_prediction.timestamp else None
            latest_prob = self.latest_prediction.attack_probability if self.latest_prediction else None

            return {
                "capture_running": cap_running,
                "interface": self.status.interface or self.capturer.interface_name,
                "packets_seen": pkts,
                "flows_seen": flows,
                "windows_processed": self.windows_processed,
                "context_windows_available": len(self.window_context),
                "latest_prediction_timestamp": latest_ts,
                "latest_probability": latest_prob,
                "model_threshold": self.threshold
            }

    def ingest_flows(self, flows: List[Dict[str, Any]]):
        """Receives external flows (e.g. from API/tests)."""
        if not self.status.capture_active:
            return
        with self.lock:
            for f in flows:
                self.traffic_summary.flows_observed += 1
                if 'tot_fwd_pkts' in f and 'tot_bwd_pkts' in f:
                    self.traffic_summary.packets_observed += (f.get('tot_fwd_pkts', 0) + f.get('tot_bwd_pkts', 0))
                if 'totlen_fwd_pkts' in f and 'totlen_bwd_pkts' in f:
                    self.traffic_summary.bytes_observed += (f.get('totlen_fwd_pkts', 0) + f.get('totlen_bwd_pkts', 0))
            self.flow_buffer.extend(flows)
        
    def process_window(self):
        """Processes 10-second traffic window. Caller must hold self.lock."""
        if not self.status.capture_active or not self.model:
            return
            
        # Enforce exact chronological window boundary
        self.last_window_time += WINDOW_SECONDS
        
        # Flush flows from Npcap FlowAggregator and flow buffer
        npcap_flows = self.flow_aggregator.flush_flows()
        buffered_flows = self.flow_buffer[:]
        self.flow_buffer.clear()
        
        current_flows = npcap_flows + buffered_flows
        self.capturer.flows_created += len(npcap_flows)
        self.traffic_summary.flows_observed += len(current_flows)

        if not current_flows:
            # Genuine zero-traffic interval.
            # Advance wall clock, retain existing valid context, do NOT append zero-vectors or increment history.
            print(f"\n[LIVE WINDOW {datetime.fromtimestamp(self.last_window_time).strftime('%H:%M:%S')}] NO GENUINE TRAFFIC WINDOW (history: {len(self.window_context)}/{FROZEN_HISTORY_WINDOWS})")
            return
            
        df_raw = pd.DataFrame(current_flows)
        df_raw = self._map_cic_to_set_r(df_raw)
        df_raw['Timestamp'] = datetime.now()

        df_processed = process_features(df_raw, self.features)

        # Validate feature coverage against exact 89 SET_R features
        is_compat, coverage_pct, status_str, missing_feats = self.flow_aggregator.validate_feature_coverage(df_processed, self.features)
        self.feature_coverage_percent = coverage_pct
        self.feature_compatibility_status = status_str

        if not is_compat:
            print(f"[CyberCast Live WARNING] Feature compatibility incomplete ({coverage_pct}%). Missing: {missing_feats[:5]}")
            # Abort window commit to avoid corrupted input to frozen model
            return

        
        # Assert no target leakage
        target_leakage_cols = ['binary_attack', 'dominant_label', 'attack_ratio', 'attack_flow_count', 'Label', 'Attack']
        for t in target_leakage_cols:
            if t in df_processed.columns:
                df_processed = df_processed.drop(columns=[t])
                
        # Aggregate to 10s window
        states = create_time_windows(df_processed, self.features, window_seconds=WINDOW_SECONDS)
        if len(states) == 0:
            return
            
        latest_state = states.iloc[-1:]
        
        # Enforce exact feature order
        live_feature_cols = [c for c in latest_state.columns if c != 'Timestamp']
        assert len(live_feature_cols) == 89, f"Feature count mismatch! Expected 89, got {len(live_feature_cols)}"
        assert live_feature_cols == self.features, "Live feature ordering does not match frozen model!"
        
        # Extract feature vector
        feature_vector = latest_state[self.features].values[0]
        flow_cnt = int(latest_state.iloc[0]['flow_count'])
        
        # Append genuine traffic window to 20-window context
        history_before = len(self.window_context)
        self.window_context.append(feature_vector)
        self.window_stats_context.append(flow_cnt)
        self.windows_processed += 1
        self.status.history_collected = len(self.window_context)
        
        print(f"\n[LIVE WINDOW {datetime.fromtimestamp(self.last_window_time).strftime('%H:%M:%S')}] flows={flow_cnt} history={history_before} -> {self.status.history_collected}/{FROZEN_HISTORY_WINDOWS}")
        
        # Infer when 20 genuine windows are collected
        if len(self.window_context) == FROZEN_HISTORY_WINDOWS:
            self.status.network_changed = False
            self.status.rebuilding_context = False
            self.status.model_ready = True
            self._run_inference(latest_state.iloc[0]['Timestamp'])
            
    def _run_inference(self, window_timestamp: pd.Timestamp):
        """Passes 20-window context (1, 20, 89) into frozen Phase 2.3 SET_R LSTM model."""
        seq_array = np.array(self.window_context) # (20, 89)
        seq_array = seq_array.reshape(1, FROZEN_HISTORY_WINDOWS, -1) # (1, 20, 89)
        
        # Scale sequence using frozen Phase 2.3 scaler
        X_flat = seq_array.reshape(-1, len(self.features))
        X_scaled_flat = self.scaler.transform(X_flat)
        X_scaled = X_scaled_flat.reshape(1, FROZEN_HISTORY_WINDOWS, len(self.features))
        
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(self.device)
        
        with torch.no_grad():
            logits = self.model(X_tensor)
            prob = float(torch.sigmoid(logits).item())
            
        is_attack = int(prob >= self.threshold)
        risk = calculate_risk_level(prob, self.threshold)
        w_flows = self.window_stats_context[-1]
        
        ts = window_timestamp.to_pydatetime() if isinstance(window_timestamp, pd.Timestamp) else datetime.now()
        
        self.latest_prediction = LivePrediction(
            timestamp=ts,
            window_start=ts - timedelta(seconds=WINDOW_SECONDS),
            window_end=ts,
            attack_probability=prob,
            model_threshold=self.threshold,
            binary_prediction=is_attack,
            risk_level=risk,
            history_windows=FROZEN_HISTORY_WINDOWS,
            flow_count=w_flows,
            packet_count=self.capturer.packets_captured,
            total_bytes=self.capturer.bytes_captured
        )
        
        print(f"[CyberCast Live FORECAST] Probability: {prob:.4f} | Threshold: {self.threshold:.4f} | Risk: {risk}")
        self._log_prediction(self.latest_prediction)

    def _log_prediction(self, pred: LivePrediction):
        log_dir = os.path.join(self.repo_root, "results", "inference", "live")
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, "live_predictions.csv")
        
        df = pd.DataFrame([pred.to_dict()])
        if not os.path.exists(log_file):
            df.to_csv(log_file, index=False)
        else:
            df.to_csv(log_file, mode='a', header=False, index=False)
            
    def _map_cic_to_set_r(self, df: pd.DataFrame) -> pd.DataFrame:
        """Map snake_case / raw flow names to Title Case expected by CyberCast."""
        mapping = {
            "dst_port": "Dst Port",
            "protocol": "Protocol",
            "flow_duration": "Flow Duration",
            "tot_fwd_pkts": "Tot Fwd Pkts",
            "tot_bwd_pkts": "Tot Bwd Pkts",
            "totlen_fwd_pkts": "TotLen Fwd Pkts",
            "totlen_bwd_pkts": "TotLen Bwd Pkts",
            "fwd_pkt_len_max": "Fwd Pkt Len Max",
            "fwd_pkt_len_min": "Fwd Pkt Len Min",
            "fwd_pkt_len_mean": "Fwd Pkt Len Mean",
            "fwd_pkt_len_std": "Fwd Pkt Len Std",
            "bwd_pkt_len_max": "Bwd Pkt Len Max",
            "bwd_pkt_len_min": "Bwd Pkt Len Min",
            "bwd_pkt_len_mean": "Bwd Pkt Len Mean",
            "bwd_pkt_len_std": "Bwd Pkt Len Std",
            "flow_byts_s": "Flow Byts/s",
            "flow_pkts_s": "Flow Pkts/s",
            "flow_iat_mean": "Flow IAT Mean",
            "flow_iat_std": "Flow IAT Std",
            "flow_iat_max": "Flow IAT Max",
            "flow_iat_min": "Flow IAT Min",
            "fwd_iat_tot": "Fwd IAT Tot",
            "fwd_iat_mean": "Fwd IAT Mean",
            "fwd_iat_std": "Fwd IAT Std",
            "fwd_iat_max": "Fwd IAT Max",
            "fwd_iat_min": "Fwd IAT Min",
            "bwd_iat_tot": "Bwd IAT Tot",
            "bwd_iat_mean": "Bwd IAT Mean",
            "bwd_iat_std": "Bwd IAT Std",
            "bwd_iat_max": "Bwd IAT Max",
            "bwd_iat_min": "Bwd IAT Min",
            "fwd_psh_flags": "Fwd PSH Flags",
            "bwd_psh_flags": "Bwd PSH Flags",
            "fwd_urg_flags": "Fwd URG Flags",
            "bwd_urg_flags": "Bwd URG Flags",
            "fwd_header_len": "Fwd Header Len",
            "bwd_header_len": "Bwd Header Len",
            "fwd_pkts_s": "Fwd Pkts/s",
            "bwd_pkts_s": "Bwd Pkts/s",
            "pkt_len_min": "Pkt Len Min",
            "pkt_len_max": "Pkt Len Max",
            "pkt_len_mean": "Pkt Len Mean",
            "pkt_len_std": "Pkt Len Std",
            "pkt_len_var": "Pkt Len Var",
            "fin_flag_cnt": "FIN Flag Cnt",
            "syn_flag_cnt": "SYN Flag Cnt",
            "rst_flag_cnt": "RST Flag Cnt",
            "psh_flag_cnt": "PSH Flag Cnt",
            "ack_flag_cnt": "ACK Flag Cnt",
            "urg_flag_cnt": "URG Flag Cnt",
            "cwe_flag_count": "CWE Flag Count",
            "ece_flag_cnt": "ECE Flag Cnt",
            "down_up_ratio": "Down/Up Ratio",
            "pkt_size_avg": "Pkt Size Avg",
            "fwd_seg_size_avg": "Fwd Seg Size Avg",
            "bwd_seg_size_avg": "Bwd Seg Size Avg",
            "fwd_byts_b_avg": "Fwd Byts/b Avg",
            "fwd_pkts_b_avg": "Fwd Pkts/b Avg",
            "fwd_blk_rate_avg": "Fwd Blk Rate Avg",
            "bwd_byts_b_avg": "Bwd Byts/b Avg",
            "bwd_pkts_b_avg": "Bwd Pkts/b Avg",
            "bwd_blk_rate_avg": "Bwd Blk Rate Avg",
            "subflow_fwd_pkts": "Subflow Fwd Pkts",
            "subflow_fwd_byts": "Subflow Fwd Byts",
            "subflow_bwd_pkts": "Subflow Bwd Pkts",
            "subflow_bwd_byts": "Subflow Bwd Byts",
            "init_fwd_win_byts": "Init Fwd Win Byts",
            "init_bwd_win_byts": "Init Bwd Win Byts",
            "fwd_act_data_pkts": "Fwd Act Data Pkts",
            "fwd_seg_size_min": "Fwd Seg Size Min",
            "active_mean": "Active Mean",
            "active_std": "Active Std",
            "active_max": "Active Max",
            "active_min": "Active Min",
            "idle_mean": "Idle Mean",
            "idle_std": "Idle Std",
            "idle_max": "Idle Max",
            "idle_min": "Idle Min"
        }
        df = df.rename(columns=mapping)
        return df
