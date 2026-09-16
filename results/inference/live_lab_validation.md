# CyberCast Live Lab Validation Report

## Environment Details
- **Capture Method:** Python `cicflowmeter` live packet capture (via HTTP flow streaming).
- **Network Interface:** Wi-Fi (Index 21)
- **Lab Host IP:** 10.246.189.7
- **Kali Linux IP:** [FILL IN YOUR KALI IP]

## Architecture Confirmation
- **Flow Aggregation:** Using `cicflowmeter` to capture faithful subflow and stateful features (Active, Idle, Bulk).
- **Feature Mapping:** `src/live/live_engine.py` securely maps the 84 base `cicflowmeter` fields to the Title Case format and uses the frozen `feature_pipeline.py` for remaining derived fields.
- **Inference Integration:** `src/live/live_engine.py` batches flows into strict 10s windows, requires exactly 20 windows (200s context), and scales using the frozen Phase 2.3 artifacts before model prediction.

## Acceptance Criteria Checklist
- [x] Existing `(all)` environment used (CICFlowMeter installed into `D:\working_projects\all_env\all\`)
- [ ] Wi-Fi interface 21 captured successfully
- [ ] Real Kali → Windows packets observed
- [x] No old PCAP used
- [x] No fake traffic or forced predictions
- [x] Exact 89 SET_R features mapped
- [x] Frozen Phase 2.3 model, scaler, and threshold preserved
- [x] Historical mode remains fully functional
- [ ] Tests passed (`pytest tests/test_live_*.py`)

## CLI Validation Test Log
*Please run `D:\working_projects\all_env\all\Scripts\python.exe tests\test_live_cli.py` and paste the output below.*
```text

```

## Kali Traffic Validation Log
*Please start the API, run `cicflowmeter -i "Wi-Fi" -u http://127.0.0.1:8000/api/live/flows`, and generate controlled traffic from Kali. Document the prediction probabilities observed below.*
```text

```

## Scientific Disclaimer
> Live Kali traffic testing is an integration/generalization demonstration and is not part of the frozen Phase 2.3 benchmark.
> A high attack probability is not guaranteed because the frozen model was trained and evaluated on CIC-IDS2018-derived traffic rather than this specific live lab traffic.
