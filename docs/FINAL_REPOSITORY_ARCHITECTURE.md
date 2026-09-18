# Final Repository Architecture

```text
cyberCast/
├── archive/          # Legacy, duplicates, and temporary scripts
├── assets/           # Static assets, images for docs
├── configs/          # Configuration files
├── dashboard/        # React + FastAPI dashboard
├── data/             # PCAP, CSV, and Parquet data files
├── docs/             # Markdown documentation
├── models/           
│   ├── production/   # Frozen Phase 2.3 champion and scaler
│   ├── baselines/    # Baseline scientific models
│   ├── experiments/  # Hyperparameter and feature set experiments
│   └── legacy/       # Old Phase 1 / Phase 2.1 models
├── notebooks/        # Jupyter notebooks (Research)
├── results/          # JSON metrics, CSV reports, analysis artifacts
├── scripts/          # Standalone python scripts for execution
├── src/              # Production Source Code
│   ├── api/          # FastAPI backend
│   ├── explainability/ # SHAP/LIME logic
│   ├── forecasting/  # Pre-attack evaluation logic
│   ├── inference/    # Offline/CSV/Model loading logic
│   └── live/         # Npcap live capture logic
└── tests/            # Pytest suite
```

## Root Directory Policy
Only the following files are allowed at the repository root:
- README.md
- requirements.txt
- .gitignore
