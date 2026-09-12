# %% [markdown]
# # Notebook 01: Dataset Inspection
# 
# **Purpose:**
# Understand the CIC-IDS2018 dataset before any preprocessing. We will explore the raw CSV files, identify the schema, check data quality, and generate a metadata report.
# 
# **Input:**
# Raw CSV files in `data/raw/`
# 
# **Output:**
# `dataset_metadata.json` saved in `data/processed/`

# %%
import os
import sys
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
from pathlib import Path

# Add project root to path so we can import configs
PROJECT_ROOT = Path(os.path.abspath('')).parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from configs.config import RAW_DATA_DIR, PROCESSED_DATA_DIR

# Robustly check that directories exist
if not RAW_DATA_DIR.exists():
    raise FileNotFoundError(f"Raw data directory not found at: {RAW_DATA_DIR}")

PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

# %% [markdown]
# ### 1. Locate all CSV files under `data/raw/`

# %%
csv_files = list(RAW_DATA_DIR.glob("*.csv"))

if not csv_files:
    raise FileNotFoundError(f"No CSV files found in {RAW_DATA_DIR}")

print(f"Found {len(csv_files)} CSV files in {RAW_DATA_DIR}")

file_info = []
for f in csv_files:
    size_mb = f.stat().st_size / (1024 * 1024)
    file_info.append({"filename": f.name, "size_mb": size_mb, "path": str(f)})
    print(f"- {f.name} ({size_mb:.2f} MB)")

# We'll use the first found CSV for our inspection
primary_csv = csv_files[0]
print(f"\nUsing primary CSV for inspection: {primary_csv.name}")

# %% [markdown]
# ### 2. Load a small sample
# To avoid loading huge CSV files unnecessarily, we'll load a sample of 10,000 rows first.

# %%
SAMPLE_SIZE = 10000
print(f"Loading {SAMPLE_SIZE} rows from {primary_csv.name}...")

try:
    df_sample = pd.read_csv(primary_csv, nrows=SAMPLE_SIZE)
    print("Sample loaded successfully.")
except Exception as e:
    print(f"Error loading sample: {e}")
    raise

# %% [markdown]
# ### 3. Display Column Information

# %%
num_rows, num_cols = df_sample.shape
print(f"Number of rows in sample: {num_rows}")
print(f"Number of columns: {num_cols}")

print("\nComplete list of columns and their datatypes:")
columns_info = df_sample.dtypes.to_dict()
for col, dtype in columns_info.items():
    print(f" - {col}: {dtype}")

# %% [markdown]
# ### 4. Identify Column Types

# %%
numerical_cols = df_sample.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = df_sample.select_dtypes(exclude=[np.number]).columns.tolist()

print(f"Numerical columns ({len(numerical_cols)}): {numerical_cols[:5]}...")
print(f"Categorical columns ({len(categorical_cols)}): {categorical_cols}")

# Attempt to identify the label and timestamp columns dynamically
label_col_candidates = [c for c in df_sample.columns if 'label' in c.lower()]
timestamp_col_candidates = [c for c in df_sample.columns if 'timestamp' in c.lower()]

label_col = label_col_candidates[0] if label_col_candidates else None
timestamp_col = timestamp_col_candidates[0] if timestamp_col_candidates else None

print(f"\nIdentified Timestamp column: {timestamp_col}")
print(f"Identified Label column: {label_col}")

# %% [markdown]
# ### 5. Inspect Data Quality (on sample)

# %%
missing_values = df_sample.isnull().sum().sum()
duplicated_rows = df_sample.duplicated().sum()

# Checking for infinite values in numerical columns
# We use replace([np.inf, -np.inf], np.nan) to count infs
inf_values = np.isinf(df_sample[numerical_cols]).values.sum()

print(f"Missing values (sample): {missing_values}")
print(f"Duplicated rows (sample): {duplicated_rows}")
print(f"Infinite values (sample): {inf_values}")

if label_col:
    unique_labels = df_sample[label_col].unique().tolist()
    print(f"\nUnique Labels: {unique_labels}")
    
    print("\nLabel Value Counts:")
    print(df_sample[label_col].value_counts())

# %% [markdown]
# ### 6. Parse Timestamp and Inspect Time Information
# Let's inspect the temporal distribution.

# %%
if timestamp_col:
    # Safely parse timestamp
    df_sample[timestamp_col] = pd.to_datetime(df_sample[timestamp_col], errors='coerce', dayfirst=True)
    
    min_timestamp = df_sample[timestamp_col].min()
    max_timestamp = df_sample[timestamp_col].max()
    duration = max_timestamp - min_timestamp
    
    print(f"Minimum Timestamp: {min_timestamp}")
    print(f"Maximum Timestamp: {max_timestamp}")
    print(f"Duration covered in sample: {duration}")
    
    # Rows per time interval (e.g., per minute)
    if pd.notnull(min_timestamp) and pd.notnull(max_timestamp):
        df_sample.set_index(timestamp_col, inplace=True)
        rows_per_min = df_sample.resample('1min').size()
        print("\nRows per minute (first 5 minutes):")
        print(rows_per_min.head(5))
        df_sample.reset_index(inplace=True)

# %% [markdown]
# ### 7. Visualizations

# %%
if label_col:
    # 1. Attack label distribution
    plt.figure(figsize=(10, 6))
    df_sample[label_col].value_counts().plot(kind='bar', color=['skyblue', 'salmon', 'lightgreen'])
    plt.title('Attack Label Distribution (Sample)')
    plt.xlabel('Label')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.show()
    
    # 2. Top attack classes (excluding Benign)
    attacks_only = df_sample[df_sample[label_col].str.lower() != 'benign']
    if not attacks_only.empty:
        plt.figure(figsize=(10, 6))
        attacks_only[label_col].value_counts().plot(kind='bar', color='salmon')
        plt.title('Top Attack Classes')
        plt.xlabel('Attack Type')
        plt.ylabel('Count')
        plt.tight_layout()
        plt.show()

if timestamp_col and pd.notnull(min_timestamp):
    # 3. Traffic rows over time
    plt.figure(figsize=(12, 6))
    df_sample.set_index(timestamp_col).resample('1min').size().plot(color='purple')
    plt.title('Network Traffic Volume Over Time (per minute)')
    plt.xlabel('Time')
    plt.ylabel('Number of Flows')
    plt.tight_layout()
    plt.show()
    df_sample.reset_index(inplace=True)

# %% [markdown]
# ### 8. Generate Dataset Inspection Report
# Saving metadata to JSON.

# %%
metadata = {
    "files": file_info,
    "primary_file_inspected": primary_csv.name,
    "num_columns": int(num_cols),
    "columns": list(columns_info.keys()),
    "numerical_columns": numerical_cols,
    "categorical_columns": categorical_cols,
    "timestamp_column": timestamp_col,
    "label_column": label_col,
    "sample_metrics": {
        "sample_size": SAMPLE_SIZE,
        "missing_values": int(missing_values),
        "duplicated_rows": int(duplicated_rows),
        "infinite_values": int(inf_values)
    }
}

if label_col:
    metadata["sample_metrics"]["unique_labels"] = unique_labels

if timestamp_col and pd.notnull(min_timestamp):
    metadata["sample_metrics"]["time_coverage"] = {
        "min_timestamp": str(min_timestamp),
        "max_timestamp": str(max_timestamp),
        "duration_seconds": duration.total_seconds()
    }

metadata_path = PROCESSED_DATA_DIR / "dataset_metadata.json"
with open(metadata_path, 'w') as f:
    json.dump(metadata, f, indent=4)

print(f"Dataset metadata successfully saved to {metadata_path}")
