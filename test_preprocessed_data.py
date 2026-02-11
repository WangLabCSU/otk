#!/usr/bin/env python3
"""
Test preprocessed data loading and splitting
"""

import sys
import os
import pandas as pd
import numpy as np

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from otk.data.data_processor import DataProcessor

def main():
    """Test preprocessed data loading and splitting"""
    config_path = os.path.join(os.path.dirname(__file__), 'configs', 'baseline_config.yml')
    
    print(f"Testing preprocessed data with config: {config_path}")
    
    # Initialize data processor
    processor = DataProcessor(config_path)
    
    # Load data (should use preprocessed sorted data)
    print("Loading data...")
    df = processor.load_data()
    print(f"Data loaded with shape: {df.shape}")
    
    # Preprocess data
    print("Preprocessing data...")
    features, target, samples, genes = processor.preprocess(df)
    print(f"Preprocessing completed")
    
    # Test split_data method (should use precomputed sample y sum)
    print("Testing split_data method...")
    X_train, y_train, X_val, y_val, X_test, y_test = processor.split_data(features, target, samples)
    print("Split completed successfully!")
    
    # Print results
    print(f"Train shape: {X_train.shape}, Positive rate: {y_train.sum()/len(y_train):.4f}")
    print(f"Validation shape: {X_val.shape}, Positive rate: {y_val.sum()/len(y_val):.4f}")
    print(f"Test shape: {X_test.shape}, Positive rate: {y_test.sum()/len(y_test):.4f}")
    print("\nTest completed successfully!")

if __name__ == "__main__":
    main()
