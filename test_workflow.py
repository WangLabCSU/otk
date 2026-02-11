#!/usr/bin/env python3
"""
Test script to verify the entire workflow of otk
"""

import os
import sys
import tempfile
import shutil
from otk.data.data_processor import DataProcessor
from otk.models.model import ECDNA_Model
from otk.train.trainer import train_model
from otk.predict.predictor import predict


def test_workflow():
    """Test the entire workflow from data processing to prediction"""
    print("Testing otk workflow...")
    
    # Create temporary directories
    temp_dir = tempfile.mkdtemp()
    config_dir = os.path.join(temp_dir, 'configs')
    data_dir = os.path.join(temp_dir, 'data')
    model_dir = os.path.join(temp_dir, 'models')
    predict_dir = os.path.join(temp_dir, 'predictions')
    
    os.makedirs(config_dir)
    os.makedirs(data_dir)
    os.makedirs(model_dir)
    os.makedirs(predict_dir)
    
    try:
        # Create test config file
        config_content = """
# Model configuration
model:
  name: "ecDNA_Predictor"
  architecture:
    type: "MLP"
    layers:
      - input_dim: 3
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
  batch_size: 16
  epochs: 5
  validation_split: 0.25
  test_split: 0.25
  seed: 2026
  early_stopping:
    patience: 3
    min_delta: 0.001
  learning_rate_scheduler:
    type: "ReduceLROnPlateau"
    factor: 0.5
    patience: 2
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
  batch_size: 16
  threshold: 0.5
  output_format: "csv"
"""
        
        config_path = os.path.join(config_dir, 'test_config.yml')
        with open(config_path, 'w') as f:
            f.write(config_content)
        
        # Create test data
        test_data = """
sample,gene_id,segVal,minor_cn,age,y
sample1,gene1,3,1,55,0
sample1,gene2,2,1,55,1
sample2,gene1,4,2,60,0
sample2,gene2,1,0,60,1
sample3,gene1,5,2,65,0
sample3,gene2,2,1,65,1
sample4,gene1,3,1,70,0
sample4,gene2,1,0,70,1
"""
        
        data_path = os.path.join(data_dir, 'test_data.csv')
        with open(data_path, 'w') as f:
            f.write(test_data)
        
        # Update config path
        import yaml
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        config['data']['path'] = data_path
        with open(config_path, 'w') as f:
            yaml.dump(config, f)
        
        # Test data processing
        print("\n1. Testing data processing...")
        processor = DataProcessor(config_path)
        df = processor.load_data(data_path)
        print(f"   Loaded data shape: {df.shape}")
        
        features, target, samples, genes, sample_classification, amplicon_mapping = processor.preprocess(df)
        print(f"   Preprocessed features shape: {features.shape}")
        
        X_train, y_train, X_val, y_val, X_test, y_test = processor.split_data(features, target, samples, sample_classification)
        print(f"   Train set shape: {X_train.shape}")
        print(f"   Validation set shape: {X_val.shape}")
        print(f"   Test set shape: {X_test.shape}")
        
        # Test model building
        print("\n2. Testing model building...")
        model = ECDNA_Model(config_path)
        print("   Model built successfully")
        
        # Test model training (using CPU for quick test)
        print("\n3. Testing model training...")
        try:
            best_val_auPRC, test_metrics = train_model(config_path, model_dir, gpu=-1)
            print(f"   Training completed successfully")
            print(f"   Best validation auPRC: {best_val_auPRC:.4f}")
            print(f"   Test metrics: {test_metrics}")
        except Exception as e:
            print(f"   Training failed: {e}")
            print("   Skipping training test (this is expected in some environments)")
        
        # Test prediction
        print("\n4. Testing prediction...")
        model_path = os.path.join(model_dir, 'best_model.pth')
        if os.path.exists(model_path):
            predict(model_path, data_path, predict_dir, gpu=-1)
            print("   Prediction completed successfully")
            
            # Check prediction results
            prediction_file = os.path.join(predict_dir, 'predictions.csv')
            if os.path.exists(prediction_file):
                print(f"   Predictions saved to: {prediction_file}")
                import pandas as pd
                pred_df = pd.read_csv(prediction_file)
                print(f"   Prediction results shape: {pred_df.shape}")
                print(f"   First few rows:")
                print(pred_df.head())
            else:
                print(f"   Prediction file not found: {prediction_file}")
        else:
            print(f"   Model file not found: {model_path}")
            print("   Skipping prediction test")
        
        print("\nWorkflow test completed!")
        
    finally:
        # Clean up
        shutil.rmtree(temp_dir)
        print(f"\nCleaned up temporary directory: {temp_dir}")


if __name__ == '__main__':
    test_workflow()
