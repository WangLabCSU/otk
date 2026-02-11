#!/usr/bin/env python3
"""
Preprocess modeling data to speed up loading
"""

import sys
import os
import pandas as pd
import numpy as np
import pickle

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def main():
    """Preprocess modeling data"""
    # Define paths
    data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'gcap_modeling_data.csv.gz')
    output_dir = os.path.join(os.path.dirname(__file__), 'src', 'otk', 'data')
    sample_sum_path = os.path.join(output_dir, 'sample_y_sum.csv')
    sorted_data_path = os.path.join(output_dir, 'sorted_modeling_data.csv.gz')
    
    print(f"Preprocessing data from: {data_path}")
    print(f"Output directory: {output_dir}")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    print("Loading data...")
    df = pd.read_csv(data_path)
    print(f"Data loaded with shape: {df.shape}")
    
    # Calculate y sum for each sample
    print("Calculating y sum for each sample...")
    sample_y_sum = df.groupby('sample')['y'].sum().reset_index()
    sample_y_sum.columns = ['sample', 'y_sum']
    print(f"Calculated y sum for {len(sample_y_sum)} samples")
    
    # Save sample y sum
    print(f"Saving sample y sum to: {sample_sum_path}")
    sample_y_sum.to_csv(sample_sum_path, index=False)
    
    # Sort data by sample y sum
    print("Sorting data by sample y sum...")
    # Merge y sum back to original dataframe
    df_with_sum = df.merge(sample_y_sum, on='sample', how='left')
    # Sort by y_sum and then sample
    df_sorted = df_with_sum.sort_values(['y_sum', 'sample']).drop('y_sum', axis=1)
    print(f"Data sorted with shape: {df_sorted.shape}")
    
    # Save sorted data as gzipped CSV
    print(f"Saving sorted data to: {sorted_data_path}")
    df_sorted.to_csv(sorted_data_path, index=False, compression='gzip')
    
    print("Preprocessing completed successfully!")

if __name__ == "__main__":
    main()
