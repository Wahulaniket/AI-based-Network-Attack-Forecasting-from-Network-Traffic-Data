import time
import requests

API_BASE = 'http://localhost:8000/api/live'

def get_debug():
    return requests.get(f'{API_BASE}/debug').json()

def test_1():
    print('Test 1: Start Live Lab with no traffic')
    requests.post(f'{API_BASE}/start')
    time.sleep(1) # Let thread start
    d = get_debug()
    assert d['scheduler_alive'] == True, 'Scheduler not alive'
    assert d['history_collected'] == 0, 'History should be 0'
    print('Test 1 Passed.')

def test_2():
    print('Test 2: Inject one valid flow, wait >10s')
    requests.post(f'{API_BASE}/flows', json=[
        {
            'Flow ID': '1.1.1.1-2.2.2.2-80-80-6',
            'Source IP': '1.1.1.1',
            'Destination IP': '2.2.2.2',
            'Timestamp': '16-09-2026 12:00:00'
        }
    ])
    print('Waiting 12 seconds...')
    time.sleep(12)
    d = get_debug()
    assert d['history_collected'] == 1, f'History should be 1, got {d["history_collected"]}'
    print('Test 2 Passed.')

def test_3():
    print('Test 3: No traffic, wait another 10s')
    print('Waiting 12 seconds...')
    time.sleep(12)
    d = get_debug()
    assert d['history_collected'] == 1, f'History should remain 1, got {d["history_collected"]}'
    print('Test 3 Passed.')

def test_4():
    print('Test 4: Inject traffic again, wait >10s')
    requests.post(f'{API_BASE}/flows', json=[
        {
            'Flow ID': '1.1.1.1-2.2.2.2-80-80-6',
            'Source IP': '1.1.1.1',
            'Destination IP': '2.2.2.2',
            'Timestamp': '16-09-2026 12:00:00'
        }
    ])
    print('Waiting 12 seconds...')
    time.sleep(12)
    d = get_debug()
    assert d['history_collected'] == 2, f'History should be 2, got {d["history_collected"]}'
    print('Test 4 Passed.')

def test_7_8():
    print('Test 7: Start called twice')
    requests.post(f'{API_BASE}/start')
    requests.post(f'{API_BASE}/start')
    print('Test 8: Stop cleanly')
    requests.post(f'{API_BASE}/stop')
    time.sleep(1)
    d = get_debug()
    assert d['scheduler_alive'] == False, 'Scheduler should be dead'
    print('Test 7 & 8 Passed.')

if __name__ == '__main__':
    try:
        test_1()
        test_2()
        test_3()
        test_4()
        test_7_8()
        print('ALL TESTS COMPLETED SUCCESSFULLY!')
    except Exception as e:
        print(f'TEST FAILED: {e}')
