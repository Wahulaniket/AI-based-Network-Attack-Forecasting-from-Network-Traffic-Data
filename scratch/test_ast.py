import os
import sys
import gc
import ast
import glob
from collections import Counter
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, f1_score, recall_score, precision_recall_curve, confusion_matrix, auc
import matplotlib.pyplot as plt
import seaborn as sns

PIPELINE_FILE = '../cybercast_pipeline.py'
with open(PIPELINE_FILE, 'r', encoding='utf-8') as f:
    source = f.read()

tree = ast.parse(source)

# Extract top-level Assignments, Functions, and Classes
extracted_nodes = []
for node in tree.body:
    if isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.Import, ast.ImportFrom)):
        extracted_nodes.append(node)
    elif isinstance(node, ast.Assign):
        # Only grab simple constant assignments (all uppercase usually)
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id.isupper():
                extracted_nodes.append(node)

# Compile and execute the extracted definitions
new_tree = ast.Module(body=extracted_nodes, type_ignores=[])
compiled = compile(new_tree, filename='<ast>', mode='exec')

import glob
csv_files = sorted(glob.glob('../data/raw/*.csv'))
if len(csv_files) > 0:
    _sample_cols = pd.read_csv(csv_files[0], nrows=0).columns.str.strip().tolist()
else:
    _sample_cols = []
    
exec(compiled, globals())

print("Successfully injected pipeline constants, functions, and classes:")
print(f" - CHUNK_SIZE: {CHUNK_SIZE}")
print(f" - Model defined: {'CyberCastWorldModel' in globals()}")
print(f" - Extracted functions: {', '.join([n.name for n in extracted_nodes if isinstance(n, ast.FunctionDef)])}")
