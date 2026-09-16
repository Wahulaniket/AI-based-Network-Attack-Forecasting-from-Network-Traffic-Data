import os
import sys
import time
import json
import socket
import csv
import threading
import urllib.request
from typing import Dict, Any, List, Optional
from datetime import datetime

# Target safety bounds
TARGET_HOST = "127.0.0.1"
HTTP_PORT = 8090
API_BASE = "http://localhost:8000/api/live"

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUTPUT_DIR = os.path.join(REPO_ROOT, "results", "inference")
CSV_PATH = os.path.join(OUTPUT_DIR, "live_attack_validation.csv")
JSON_PATH = os.path.join(OUTPUT_DIR, "live_attack_validation.json")

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

class ControlledTrafficGenerator:
    """Generates non-destructive, safe local network traffic on 127.0.0.1."""
    def __init__(self):
        self.running = False
        self.server_process = None

    def start_local_http_server(self):
        """Starts background Python http.server on port 8090 if not already running."""
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            res = s.connect_ex((TARGET_HOST, HTTP_PORT))
            s.close()
            if res == 0:
                print(f"[TrafficGenerator] HTTP server already listening on {TARGET_HOST}:{HTTP_PORT}")
                return
        except Exception:
            pass

        def run_server():
            import http.server
            import socketserver
            handler = http.server.SimpleHTTPRequestHandler
            with socketserver.TCPServer((TARGET_HOST, HTTP_PORT), handler) as httpd:
                print(f"[TrafficGenerator] HTTP server started on {TARGET_HOST}:{HTTP_PORT}")
                httpd.serve_forever()

        t = threading.Thread(target=run_server, daemon=True)
        t.start()
        time.sleep(1)

    def scenario_0_baseline(self, duration_sec: int = 20):
        """Scenario 0: Moderate benign HTTP requests to local server."""
        print(f"\n--- SCENARIO 0: BENIGN BASELINE ({duration_sec}s) ---")
        end_time = time.time() + duration_sec
        req_count = 0
        while time.time() < end_time:
            try:
                urllib.request.urlopen(f"http://{TARGET_HOST}:{HTTP_PORT}/", timeout=1)
                req_count += 1
            except Exception:
                pass
            time.sleep(0.5)
        print(f"[Scenario 0] Generated {req_count} baseline HTTP requests.")

    def scenario_1_reconnaissance(self, duration_sec: int = 20):
        """Scenario 1: Low-rate TCP socket connection probes against small port range (1-100)."""
        print(f"\n--- SCENARIO 1: CONTROLLED RECONNAISSANCE ({duration_sec}s) ---")
        end_time = time.time() + duration_sec
        probe_count = 0
        port = 1
        while time.time() < end_time:
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.settimeout(0.1)
                s.connect_ex((TARGET_HOST, port))
                s.close()
                probe_count += 1
                port = (port % 100) + 1
            except Exception:
                pass
            time.sleep(0.1)
        print(f"[Scenario 1] Generated {probe_count} low-rate TCP port connect probes.")

    def scenario_2_http_enumeration(self, duration_sec: int = 20):
        """Scenario 2: Sequence of HTTP path requests (/robots.txt, /favicon.ico, /admin, /login)."""
        print(f"\n--- SCENARIO 2: CONTROLLED HTTP ENUMERATION ({duration_sec}s) ---")
        paths = ["/", "/robots.txt", "/favicon.ico", "/test", "/admin", "/login", "/config", "/api/v1"]
        end_time = time.time() + duration_sec
        idx = 0
        req_count = 0
        while time.time() < end_time:
            path = paths[idx % len(paths)]
            try:
                urllib.request.urlopen(f"http://{TARGET_HOST}:{HTTP_PORT}{path}", timeout=1)
                req_count += 1
            except Exception:
                pass
            idx += 1
            time.sleep(0.3)
        print(f"[Scenario 2] Generated {req_count} HTTP path enumeration requests.")

    def scenario_3_repeated_connections(self, duration_sec: int = 20):
        """Scenario 3: Repeated connection attempts to measure SYN/ACK/RST statistics."""
        print(f"\n--- SCENARIO 3: REPEATED CONNECTION ATTEMPTS ({duration_sec}s) ---")
        end_time = time.time() + duration_sec
        conn_count = 0
        while time.time() < end_time:
            try:
                # Alternate between open HTTP port and closed port 9999
                target_port = HTTP_PORT if (conn_count % 2 == 0) else 9999
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.settimeout(0.2)
                s.connect_ex((TARGET_HOST, target_port))
                s.close()
                conn_count += 1
            except Exception:
                pass
            time.sleep(0.2)
        print(f"[Scenario 3] Generated {conn_count} repeated connection attempts.")

    def scenario_4_mixed_session(self):
        """Scenario 4: Primary SIH Demo Multi-Stage Sequence (Baseline -> Recon -> Baseline -> Enum -> Recovery)."""
        print(f"\n==================================================")
        print(f"--- SCENARIO 4: MIXED HUMAN-LIKE SESSION (PRIMARY DEMO) ---")
        print(f"==================================================")
        print("Phase A: Normal HTTP Browsing (20s)...")
        self.scenario_0_baseline(20)
        
        print("Phase B: Controlled Reconnaissance (20s)...")
        self.scenario_1_reconnaissance(20)
        
        print("Phase C: Return to Normal Browsing (20s)...")
        self.scenario_0_baseline(20)
        
        print("Phase D: Controlled HTTP Enumeration (20s)...")
        self.scenario_2_http_enumeration(20)
        
        print("Phase E: Recovery / Return to Baseline (20s)...")
        self.scenario_0_baseline(20)
        print("[Scenario 4] Multi-stage session completed.")

class LiveTelemetryCollector:
    """Collects live pipeline telemetry and logs to CSV & JSON."""
    def __init__(self):
        self.records: List[Dict[str, Any]] = []

    def poll_api(self, scenario_name: str) -> Optional[Dict[str, Any]]:
        try:
            res_status = urllib.request.urlopen(f"{API_BASE}/status", timeout=2)
            status_data = json.loads(res_status.read().decode('utf-8'))
            
            res_cap = urllib.request.urlopen(f"{API_BASE}/capture-status", timeout=2)
            cap_data = json.loads(res_cap.read().decode('utf-8'))
            
            res_latest = urllib.request.urlopen(f"{API_BASE}/latest", timeout=2)
            latest_data = json.loads(res_latest.read().decode('utf-8'))
            
            prob = latest_data.get("attack_probability") if isinstance(latest_data, dict) and "attack_probability" in latest_data else None
            threshold = status_data.get("threshold", 0.9830410480499268)
            
            decision = "NO_PREDICTION"
            if prob is not None:
                decision = "ATTACK_THREAD_DETECTED" if prob >= threshold else "BELOW_ATTACK_THRESHOLD"

            rec = {
                "timestamp": datetime.now().isoformat(),
                "scenario": scenario_name,
                "interface": status_data.get("interface", "AUTO"),
                "host_ip": status_data.get("host_ip", "127.0.0.1"),
                "packets_captured": cap_data.get("packets_captured", 0),
                "bytes_captured": cap_data.get("bytes_captured", 0),
                "flows_created": cap_data.get("flows_created", 0),
                "history_collected": status_data.get("history_collected", 0),
                "history_required": status_data.get("history_required", 20),
                "model_ready": status_data.get("model_ready", False),
                "attack_probability": prob,
                "model_threshold": threshold,
                "model_decision": decision,
                "risk_level": latest_data.get("risk", "LOW") if isinstance(latest_data, dict) else "LOW",
                "feature_coverage_percent": status_data.get("feature_coverage_percent", 100.0),
                "feature_compatibility_status": status_data.get("feature_compatibility_status", "FULL")
            }
            self.records.append(rec)
            return rec
        except Exception as e:
            print(f"[TelemetryCollector Error] {e}")
            return None

    def save_results(self):
        """Saves collected telemetry to CSV and JSON."""
        if not self.records:
            print("[TelemetryCollector] No records to save.")
            return

        # Save JSON
        with open(JSON_PATH, "w") as f:
            json.dump(self.records, f, indent=2)
        print(f"[TelemetryCollector] Saved JSON report to {JSON_PATH}")

        # Save CSV
        fieldnames = list(self.records[0].keys())
        with open(CSV_PATH, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.records)
        print(f"[TelemetryCollector] Saved CSV report to {CSV_PATH}")

def run_controlled_adversarial_validation():
    print("==================================================")
    print("CYBERCAST LIVE SYSTEM — CONTROLLED ADVERSARIAL VALIDATION")
    print("==================================================")
    print(f"Target Safety Scope: {TARGET_HOST}:{HTTP_PORT} (Strictly Localhost)")
    print(f"API Base: {API_BASE}")
    print("==================================================")

    generator = ControlledTrafficGenerator()
    generator.start_local_http_server()
    
    collector = LiveTelemetryCollector()

    # Ensure live capture is running via API
    try:
        urllib.request.urlopen(urllib.request.Request(f"{API_BASE}/start", method="POST"), timeout=2)
        print("[AttackLab] Sent live capture start request.")
    except Exception as e:
        print(f"[AttackLab Warning] Could not reach API start endpoint: {e}")

    scenarios = [
        ("Scenario 0 - Baseline", lambda: generator.scenario_0_baseline(20)),
        ("Scenario 1 - Reconnaissance", lambda: generator.scenario_1_reconnaissance(20)),
        ("Scenario 2 - HTTP Enumeration", lambda: generator.scenario_2_http_enumeration(20)),
        ("Scenario 3 - Repeated Connections", lambda: generator.scenario_3_repeated_connections(20)),
        ("Scenario 4 - Mixed Human-Like Session", lambda: generator.scenario_4_mixed_session())
    ]

    for name, func in scenarios:
        print(f"\nExecuting {name}...")
        
        # Start background polling during scenario execution
        def poll_loop():
            t_end = time.time() + 110 # Max scenario time window
            while time.time() < t_end and not stop_poll.is_set():
                collector.poll_api(name)
                time.sleep(3)

        stop_poll = threading.Event()
        poll_thread = threading.Thread(target=poll_loop, daemon=True)
        poll_thread.start()

        # Execute scenario traffic
        func()
        stop_poll.set()

        # Final snapshot for scenario
        rec = collector.poll_api(name)
        if rec:
            print(f"-> Telemetry Snapshot: Packets={rec['packets_captured']} | Flows={rec['flows_created']} | Context={rec['history_collected']}/{rec['history_required']} | Prob={rec['attack_probability']} | Decision={rec['model_decision']}")

    collector.save_results()
    print("\n==================================================")
    print("CONTROLLED ADVERSARIAL VALIDATION HARNESS COMPLETE")
    print("==================================================")

if __name__ == "__main__":
    run_controlled_adversarial_validation()
