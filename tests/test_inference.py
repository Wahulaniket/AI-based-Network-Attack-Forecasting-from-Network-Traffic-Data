import os
import sys
import unittest
import pandas as pd
import numpy as np

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from src.inference.model_loader import load_inference_artifacts
from src.inference.feature_pipeline import process_features
from src.inference.windowing import create_time_windows, create_causal_sequences

class TestInferencePipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.repo_root = repo_root
        cls.artifacts = load_inference_artifacts(cls.repo_root)

    def test_01_model_loading(self):
        """Test 1 - model loading"""
        self.assertIsNotNone(self.artifacts['model'])
        self.assertIsNotNone(self.artifacts['scaler'])
        print("\nTest 1 - model loading: PASS")

    def test_02_feature_dimensions(self):
        """Test 2 - feature dimensions"""
        features = self.artifacts['features']
        self.assertEqual(len(features), 89)
        print("\nTest 2 - feature dimensions: 89")

    def test_03_scaler_compatibility(self):
        """Test 3 - scaler compatibility"""
        scaler = self.artifacts['scaler']
        self.assertEqual(scaler.n_features_in_, 89)
        print("\nTest 3 - scaler compatibility: 89 input dimensions")

    def test_04_target_isolation(self):
        """Test 7 - target isolation"""
        # Create a mock dataframe with target columns
        df = pd.DataFrame({
            'Timestamp': ['14/02/2018 08:31:01', '14/02/2018 08:31:02'],
            'Label': ['Attack', 'Benign'],
            'Attack': [1, 0],
            'binary_attack': [1, 0],
            'dominant_label': ['A', 'B'],
            'attack_ratio': [1.0, 0.0],
            'attack_flow_count': [1, 0],
            'Flow Duration': [1000, 2000],
            'Tot Fwd Pkts': [5, 10],
            'Tot Bwd Pkts': [3, 8]
        })
        
        df_processed = process_features(df, self.artifacts['features'])
        
        target_cols = ['binary_attack', 'dominant_label', 'attack_ratio', 'attack_flow_count', 'Label', 'Attack']
        for col in target_cols:
            self.assertNotIn(col, df_processed.columns)
            
        print("\nTest 7 - target isolation: PASS")

    def test_05_sequence_creation_and_causality(self):
        """Test 4 & 6 - sequence creation and causal history"""
        # Create a sequence of mock data that guarantees we get 20 windows
        features = self.artifacts['features']
        
        # 30 windows of 10s each
        timestamps = pd.date_range('2018-02-14 08:00:00', periods=30, freq='10s')
        df = pd.DataFrame({'Timestamp': timestamps})
        for f in features:
            df[f] = 1.0 # mock data
            
        # mock windowing (it is already 10s intervals, so create_time_windows just keeps it)
        states = create_time_windows(df, features, window_seconds=10)
        
        # sequence creation
        X, ts_out = create_causal_sequences(states, features, seq_length=20)
        
        # We have 30 states, seq_length 20, so we should get 11 sequences (from index 19 to 29)
        self.assertEqual(len(X), 11)
        self.assertEqual(X.shape[1], 20)
        self.assertEqual(X.shape[2], 89)
        
        # Check causality: the last sequence should end at the last timestamp
        self.assertEqual(ts_out[-1], timestamps[-1])
        
        # First sequence should end at the 20th timestamp (index 19)
        self.assertEqual(ts_out[0], timestamps[19])
        
        print("\nTest 4 - sequence creation: 20 × 89")
        print("Test 6 - causal history: PASS")

if __name__ == '__main__':
    unittest.main(verbosity=2)
