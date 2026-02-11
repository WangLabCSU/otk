import unittest
import pandas as pd
import numpy as np
import os
import tempfile
from otk.data.data_processor import DataProcessor, ECDNA_Dataset

class TestDataProcessor(unittest.TestCase):
    def setUp(self):
        # Create a temporary config file
        self.config_content = """
# Model configuration
model:
  name: "ecDNA_Predictor"
  architecture:
    type: "MLP"
    layers:
      - input_dim: 58
      - hidden_dim: 128
        activation: "relu"
        dropout: 0.2
      - hidden_dim: 64
        activation: "relu"
        dropout: 0.2
      - hidden_dim: 32
        activation: "relu"
        dropout: 0.1
      - output_dim: 1
        activation: "sigmoid"
  loss_function:
    type: "BCEWithLogitsLoss"
    weight: [0.01, 0.99]
  optimizer:
    type: "Adam"
    lr: 0.001
    weight_decay: 0.0001
  metrics:
    - "auPRC"
    - "AUC"
    - "F1"
    - "Precision"
    - "Recall"

# Training configuration
training:
  batch_size: 1024
  epochs: 100
  validation_split: 0.25
  test_split: 0.25
  seed: 2026
  early_stopping:
    patience: 10
    min_delta: 0.001
  learning_rate_scheduler:
    type: "ReduceLROnPlateau"
    factor: 0.5
    patience: 5
    min_lr: 0.00001

# Data configuration
data:
  path: "test_data.csv"
  features:
    - "segVal"
    - "minor_cn"
    - "age"
  target: "y"
  sample_id: "sample"
  gene_id: "gene_id"
  missing_value_strategy:
    age: "mean"

# Prediction configuration
prediction:
  batch_size: 1024
  threshold: 0.5
  output_format: "csv"
"""
        
        # Create temporary config file
        self.config_file = tempfile.NamedTemporaryFile(mode='w', suffix='.yml', delete=False)
        self.config_file.write(self.config_content)
        self.config_file.close()
        
        # Create test data
        self.test_data = pd.DataFrame({
            'sample': ['sample1', 'sample1', 'sample2', 'sample2', 'sample3', 'sample3', 'sample4', 'sample4'],
            'gene_id': ['gene1', 'gene2', 'gene1', 'gene2', 'gene1', 'gene2', 'gene1', 'gene2'],
            'segVal': [3, 2, 4, 1, 5, 2, 3, 1],
            'minor_cn': [1, 1, 2, 0, 2, 1, 1, 0],
            'age': [55, np.nan, 60, 65, 70, 50, 65, 75],
            'y': [0, 1, 0, 1, 0, 1, 0, 1]
        })
        
        # Create temporary data file
        self.data_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
        self.test_data.to_csv(self.data_file, index=False)
        self.data_file.close()
        
        # Update config path
        import yaml
        with open(self.config_file.name, 'r') as f:
            config = yaml.safe_load(f)
        config['data']['path'] = self.data_file.name
        with open(self.config_file.name, 'w') as f:
            yaml.dump(config, f)
        
        # Initialize data processor
        self.processor = DataProcessor(self.config_file.name)
    
    def tearDown(self):
        # Clean up temporary files
        os.unlink(self.config_file.name)
        os.unlink(self.data_file.name)
    
    def test_load_data(self):
        """Test loading data"""
        df = self.processor.load_data(self.data_file.name)
        self.assertEqual(df.shape, (8, 6))
        self.assertEqual(list(df['sample'][:4]), ['sample1', 'sample1', 'sample2', 'sample2'])
    
    def test_preprocess(self):
        """Test preprocessing data"""
        df = self.processor.load_data(self.data_file.name)
        features, target, samples, genes = self.processor.preprocess(df)
        
        self.assertEqual(features.shape, (8, 3))
        self.assertEqual(target.shape, (8,))
        self.assertEqual(samples.shape, (8,))
        self.assertEqual(genes.shape, (8,))
        
        # Check if missing values are handled
        self.assertFalse(features['age'].isnull().any())
    
    def test_split_data(self):
        """Test splitting data"""
        df = self.processor.load_data(self.data_file.name)
        features, target, samples, genes = self.processor.preprocess(df)
        X_train, y_train, X_val, y_val, X_test, y_test = self.processor.split_data(features, target, samples)
        
        # Check if all splits have data
        self.assertTrue(len(X_train) > 0)
        self.assertTrue(len(X_val) > 0)
        self.assertTrue(len(X_test) > 0)
    
    def test_normalize(self):
        """Test normalizing data"""
        df = self.processor.load_data(self.data_file.name)
        features, target, samples, genes = self.processor.preprocess(df)
        X_train, y_train, X_val, y_val, X_test, y_test = self.processor.split_data(features, target, samples)
        X_train_scaled, X_val_scaled, X_test_scaled = self.processor.normalize(X_train, X_val, X_test)
        
        # Check if data is normalized (mean close to 0, std close to 1)
        self.assertAlmostEqual(np.mean(X_train_scaled), 0, delta=0.1)
        self.assertAlmostEqual(np.std(X_train_scaled), 1, delta=0.1)
    
    def test_create_dataloaders(self):
        """Test creating dataloaders"""
        df = self.processor.load_data(self.data_file.name)
        features, target, samples, genes = self.processor.preprocess(df)
        X_train, y_train, X_val, y_val, X_test, y_test = self.processor.split_data(features, target, samples)
        X_train_scaled, X_val_scaled, X_test_scaled = self.processor.normalize(X_train, X_val, X_test)
        train_loader, val_loader, test_loader = self.processor.create_dataloaders(X_train_scaled, y_train, X_val_scaled, y_val, X_test_scaled, y_test)
        
        # Check if dataloaders are created successfully
        self.assertTrue(train_loader is not None)
        self.assertTrue(val_loader is not None)
        self.assertTrue(test_loader is not None)
    
    def test_ecdna_dataset(self):
        """Test ECDNA_Dataset"""
        features = np.array([[3, 1, 55], [2, 1, 60]])
        targets = np.array([0, 1])
        dataset = ECDNA_Dataset(features, targets)
        
        self.assertEqual(len(dataset), 2)
        item = dataset[0]
        self.assertEqual(item[0].shape, (3,))
        self.assertEqual(item[1].shape, ())

if __name__ == '__main__':
    unittest.main()
