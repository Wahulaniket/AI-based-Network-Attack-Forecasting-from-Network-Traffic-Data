import os
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (precision_recall_curve, auc, f1_score, roc_auc_score, 
                             confusion_matrix, average_precision_score, accuracy_score, precision_score, recall_score)

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

class CyberCastForecaster(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

class SequenceDataset(Dataset):
    def __init__(self, features, targets, seq_length):
        self.features = features
        self.targets = targets
        self.seq_length = seq_length
        
    def __len__(self):
        return len(self.features) - self.seq_length
        
    def __getitem__(self, idx):
        x = self.features[idx : idx + self.seq_length]
        y = self.targets[idx + self.seq_length]
        return torch.FloatTensor(x), torch.FloatTensor([y])

def evaluate(model, loader, device):
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            preds = torch.sigmoid(out).cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(y.cpu().numpy())
    all_preds = np.array(all_preds).flatten()
    all_targets = np.array(all_targets).flatten()
    return all_preds, all_targets

def run_audit():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    df = pd.read_parquet('data/processed/network_states_10s.parquet')
    df = df.sort_values('Timestamp').reset_index(drop=True)
    target_col = 'binary_attack'

    set_a_features = [
        'Flow Duration', 'Tot Fwd Pkts', 'Tot Bwd Pkts', 'TotLen Fwd Pkts', 'TotLen Bwd Pkts', 
        'Flow Byts/s', 'Flow Pkts/s', 'Fwd Pkt Len Mean', 'Bwd Pkt Len Mean', 'SYN Flag Cnt', 
        'ACK Flag Cnt', 'FIN Flag Cnt', 'RST Flag Cnt', 'PSH Flag Cnt', 'URG Flag Cnt'
    ]

    n = len(df)
    train_end = int(n * 0.8)
    val_end = int(n * 0.9)
    df_train = df.iloc[:train_end].copy()
    df_val = df.iloc[train_end:val_end].copy()
    df_test = df.iloc[val_end:].copy()

    scaler = StandardScaler()
    scaler.fit(df_train[set_a_features].fillna(0).values)
    
    val_features = scaler.transform(df_val[set_a_features].fillna(0).values)
    val_targets = df_val[target_col].values
    test_features = scaler.transform(df_test[set_a_features].fillna(0).values)
    test_targets = df_test[target_col].values

    h = 20
    val_dataset = SequenceDataset(val_features, val_targets, seq_length=h)
    test_dataset = SequenceDataset(test_features, test_targets, seq_length=h)
    
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    model = CyberCastForecaster(input_size=15, hidden_size=64, num_layers=2).to(device)
    model.load_state_dict(torch.load('models/archive/model_20_SET_A.pt', weights_only=True))
    
    val_preds, val_true = evaluate(model, val_loader, device)
    test_preds, test_true = evaluate(model, test_loader, device)
    
    precision, recall, thresholds = precision_recall_curve(val_true, val_preds)
    fscores = (2 * precision * recall) / (precision + recall + 1e-8)
    ix = np.argmax(fscores)
    best_threshold = thresholds[ix]

    def get_metrics(true, preds, thresh):
        binary_preds = (preds >= thresh).astype(int)
        tn, fp, fn, tp = confusion_matrix(true, binary_preds).ravel()
        return {
            'PR-AUC': average_precision_score(true, preds),
            'ROC-AUC': roc_auc_score(true, preds),
            'F1': f1_score(true, binary_preds),
            'Precision': precision_score(true, binary_preds, zero_division=0),
            'Recall': recall_score(true, binary_preds, zero_division=0),
            'Accuracy': accuracy_score(true, binary_preds),
            'FPR': float(fp / (fp + tn)) if (fp+tn)>0 else 0.0,
            'FNR': float(fn / (fn + tp)) if (fn+tp)>0 else 0.0,
            'TP': int(tp), 'TN': int(tn), 'FP': int(fp), 'FN': int(fn),
            'PosRate': float(np.mean(true)),
            'NumSamples': len(true)
        }

    val_metrics = get_metrics(val_true, val_preds, best_threshold)
    test_metrics = get_metrics(test_true, test_preds, best_threshold)
    
    # Num parameters
    num_params = sum(p.numel() for p in model.parameters())

    output = {
        'val_metrics': val_metrics,
        'test_metrics': test_metrics,
        'threshold': float(best_threshold),
        'num_parameters': num_params
    }
    
    with open('audit_metrics.json', 'w') as f:
        json.dump(output, f, indent=2)

if __name__ == '__main__':
    run_audit()
