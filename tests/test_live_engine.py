import sys
import os
import time
from unittest import mock

# Mock psutil before importing LiveEngine to simulate availability
sys.modules['psutil'] = mock.Mock()

# Setup paths to import from src
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, REPO_ROOT)

from src.live.live_engine import LiveEngine
from src.live.schemas import LiveStatus

def run_tests():
    print("=== RUNNING NETWORK STATE MACHINE TESTS ===")
    
    # Initialize engine with mocked model loading to avoid heavy torch operations if needed
    with mock.patch('src.inference.model_loader.load_inference_artifacts') as mock_loader:
        mock_loader.return_value = {
            'model': mock.Mock(),
            'scaler': mock.Mock(),
            'features': ['feat_' + str(i) for i in range(89)], # 89 dummy features
            'threshold': 0.5,
            'device': 'cpu'
        }
        engine = LiveEngine(repo_root=REPO_ROOT)
        
    engine.status.capture_active = True
    
    # Force _map_cic_to_set_r and processing to bypass real logic for this state machine test
    engine._map_cic_to_set_r = lambda x: x
    
    with mock.patch('src.live.live_engine.process_features') as mock_proc:
        import pandas as pd
        # Return a mocked dataframe output of 1 window
        df_out = pd.DataFrame(columns=['Timestamp'] + ['feat_' + str(i) for i in range(89)])
        from datetime import datetime
        df_out.loc[0, 'Timestamp'] = datetime.now()
        for i in range(89):
            df_out.loc[0, 'feat_' + str(i)] = 0.0
        mock_proc.return_value = df_out
        
        # Helper to simulate process_window with a specific network
        def simulate_cycle(ip, iface, advance_time=True, with_flows=True):
            if advance_time:
                engine.last_window_time = time.time() - 11 # force wait to pass
            
            if with_flows:
                engine.flow_buffer.extend([{"fake": "flow"}])
            
            with mock.patch('socket.socket') as mock_sock:
                mock_sock_inst = mock.Mock()
                mock_sock_inst.getsockname.return_value = (ip, 12345)
                mock_sock.return_value = mock_sock_inst
                
                # Mock psutil
                import psutil
                addr = mock.Mock()
                addr.address = ip
                addr.family = 2 # socket.AF_INET
                psutil.net_if_addrs.return_value = {iface: [addr]}
                
                engine.process_window()
                
        print("\nTEST A: Initial network -> New network context reset")
        simulate_cycle("10.246.189.7", "Wi-Fi")
        # First cycle should set baseline
        assert engine.status.accepted_host_ip == "10.246.189.7"
        assert engine.status.history_collected == 1
        
        # Change network
        simulate_cycle("10.51.69.7", "Ethernet")
        assert engine.status.previous_host_ip == "10.246.189.7"
        assert engine.status.accepted_host_ip == "10.51.69.7"
        assert engine.status.network_changed == True
        assert engine.status.rebuilding_context == True
        assert engine.status.history_collected == 1 # Cleared to 0, then immediately processed 1 window
        print("PASS")

        print("\nTEST B: Next cycle remains on new network")
        simulate_cycle("10.51.69.7", "Ethernet")
        assert engine.status.accepted_host_ip == "10.51.69.7"
        assert engine.status.network_changed == True
        assert engine.status.history_collected == 2
        print("PASS")

        print("\nTEST C: Third cycle remains on new network")
        simulate_cycle("10.51.69.7", "Ethernet")
        assert engine.status.history_collected == 3
        print("PASS")

        print("\nTEST D: Reaching 20 fresh windows")
        for _ in range(17):
            simulate_cycle("10.51.69.7", "Ethernet")
            
        assert engine.status.history_collected == 20
        assert engine.status.model_ready == True
        assert engine.status.rebuilding_context == False
        assert engine.status.network_changed == False
        print("PASS")

        print("\nTEST E: Change again to third IP")
        simulate_cycle("192.168.1.5", "VPN")
        assert engine.status.previous_host_ip == "10.51.69.7"
        assert engine.status.accepted_host_ip == "192.168.1.5"
        assert engine.status.network_changed == True
        assert engine.status.rebuilding_context == True
        assert engine.status.history_collected == 1 # Cleared then +1
        assert engine.status.model_ready == False
        print("PASS")

        print("\nTEST F: Zero-traffic interval advances clock but does not fake data")
        h_before = engine.status.history_collected
        # We simulate a cycle where engine.flow_buffer is empty
        # simulate_cycle inherently pushes nothing to flow_buffer
        # In earlier tests it worked because we mocked process_features and forced create_time_windows
        # simulate_cycle with with_flows=False
        simulate_cycle("192.168.1.5", "VPN", with_flows=False)
        h_after = engine.status.history_collected
        assert h_after == h_before, "history_collected should not advance on empty traffic"
        print("PASS")

if __name__ == "__main__":
    run_tests()
