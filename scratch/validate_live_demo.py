import time
import requests
import urllib.request
import threading

API_BASE = "http://localhost:8000/api/live"

print("1. Testing GET /api/live/status...")
res = requests.get(f"{API_BASE}/status")
print("Status:", res.status_code, res.json())

print("\n2. Starting live capture via POST /api/live/start...")
res = requests.post(f"{API_BASE}/start")
print("Start response:", res.json())

print("\n3. Checking GET /api/live/capture-status...")
res = requests.get(f"{API_BASE}/capture-status")
print("Capture Status:", res.json())

print("\n4. Generating live HTTP traffic to trigger packet capture & flow aggregation...")
def generate_traffic():
    for i in range(25):
        try:
            urllib.request.urlopen("http://8.8.8.8", timeout=1)
        except Exception:
            pass
        try:
            urllib.request.urlopen("http://1.1.1.1", timeout=1)
        except Exception:
            pass
        time.sleep(0.5)

traffic_thread = threading.Thread(target=generate_traffic, daemon=True)
traffic_thread.start()

print("\n5. Polling Live Engine for 25 seconds to observe window processing & context progression...")
for poll in range(12):
    time.sleep(2)
    s = requests.get(f"{API_BASE}/status").json()
    c = requests.get(f"{API_BASE}/capture-status").json()
    l = requests.get(f"{API_BASE}/latest").json()
    print(f"[{poll+1}/12] Packets: {c.get('packets_captured')} | Flows: {c.get('flows_created')} | Context: {s.get('history_collected')}/{s.get('history_required')} | Ready: {s.get('model_ready')} | Latest: {l}")

print("\n[VALIDATION COMPLETE]")
