import time
import requests

print('Starting capture...')
requests.post('http://localhost:8000/api/live/start')

print('Sending flows...')
for i in range(4):
    try:
        res = requests.post('http://localhost:8000/api/live/flows', json=[
            {
                'Flow ID': '192.168.1.1-10.51.69.7-1234-80-6',
                'Source IP': '192.168.1.1',
                'Source Port': 1234,
                'Destination IP': '10.51.69.7',
                'Destination Port': 80,
                'Protocol': 6,
                'Timestamp': '16-09-2026 12:00:00',
                'Flow Duration': 100,
                'Total Fwd Packets': 1,
                'Total Backward Packets': 1,
                'Total Length of Fwd Packets': 100,
                'Total Length of Bwd Packets': 100,
                'Fwd Packet Length Max': 100,
                'Fwd Packet Length Min': 100,
                'Fwd Packet Length Mean': 100,
                'Fwd Packet Length Std': 0,
                'Bwd Packet Length Max': 100,
                'Bwd Packet Length Min': 100,
                'Bwd Packet Length Mean': 100,
                'Bwd Packet Length Std': 0,
                'Flow Bytes/s': 0,
                'Flow Packets/s': 0,
                'Flow IAT Mean': 0,
                'Flow IAT Std': 0,
                'Flow IAT Max': 0,
                'Flow IAT Min': 0,
                'Fwd IAT Total': 0,
                'Fwd IAT Mean': 0,
                'Fwd IAT Std': 0,
                'Fwd IAT Max': 0,
                'Fwd IAT Min': 0,
                'Bwd IAT Total': 0,
                'Bwd IAT Mean': 0,
                'Bwd IAT Std': 0,
                'Bwd IAT Max': 0,
                'Bwd IAT Min': 0,
                'Fwd PSH Flags': 0,
                'Bwd PSH Flags': 0,
                'Fwd URG Flags': 0,
                'Bwd URG Flags': 0,
                'Fwd Header Length': 20,
                'Bwd Header Length': 20,
                'Fwd Packets/s': 0,
                'Bwd Packets/s': 0,
                'Min Packet Length': 100,
                'Max Packet Length': 100,
                'Packet Length Mean': 100,
                'Packet Length Std': 0,
                'Packet Length Variance': 0,
                'FIN Flag Count': 0,
                'SYN Flag Count': 0,
                'RST Flag Count': 0,
                'PSH Flag Count': 0,
                'ACK Flag Count': 0,
                'URG Flag Count': 0,
                'CWE Flag Count': 0,
                'ECE Flag Count': 0,
                'Down/Up Ratio': 1,
                'Average Packet Size': 100,
                'Avg Fwd Segment Size': 100,
                'Avg Bwd Segment Size': 100,
                'Fwd Header Length.1': 20,
                'Fwd Avg Bytes/Bulk': 0,
                'Fwd Avg Packets/Bulk': 0,
                'Fwd Avg Bulk Rate': 0,
                'Bwd Avg Bytes/Bulk': 0,
                'Bwd Avg Packets/Bulk': 0,
                'Bwd Avg Bulk Rate': 0,
                'Subflow Fwd Packets': 1,
                'Subflow Fwd Bytes': 100,
                'Subflow Bwd Packets': 1,
                'Subflow Bwd Bytes': 100,
                'Init_Win_bytes_forward': 0,
                'Init_Win_bytes_backward': 0,
                'act_data_pkt_fwd': 1,
                'min_seg_size_forward': 20,
                'Active Mean': 0,
                'Active Std': 0,
                'Active Max': 0,
                'Active Min': 0,
                'Idle Mean': 0,
                'Idle Std': 0,
                'Idle Max': 0,
                'Idle Min': 0,
                'Label': 'BENIGN'
            }
        ])
        print(f'Sent batch {i}. response: {res.json()}')
    except Exception as e:
        print(e)
    time.sleep(3)

print('Checking debug endpoint...')
res = requests.get('http://localhost:8000/api/live/debug')
print(res.json())
