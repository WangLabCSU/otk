#!/usr/bin/env python3
"""
Test script to evaluate the predict functionality with various edge cases
"""

import pandas as pd
import numpy as np
import os
import sys

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from otk.predict.predictor import predict

def load_sample_data(sample_id='TCGA-21-5782-01'):
    """
    Load data for a specific sample
    
    Args:
        sample_id (str): Sample ID to load
    
    Returns:
        pd.DataFrame: Sample data
    """
    data_path = '/data/home/wsx/Projects/otk/data/gcap_modeling_data.csv.gz'
    print(f"Loading data for sample {sample_id} from {data_path}")
    
    # Load data with low memory for faster loading
    df = pd.read_csv(data_path, compression='gzip', low_memory=True)
    
    # Select data for the specific sample
    sample_df = df[df['sample'] == sample_id]
    print(f"Sample data loaded with shape: {sample_df.shape}")
    print(f"Number of genes for sample {sample_id}: {len(sample_df)}")
    
    return sample_df

def prepare_test_data(sample_df):
    """
    Prepare test data by removing freq_* features and processing type column
    
    Args:
        sample_df (pd.DataFrame): Sample data
    
    Returns:
        dict: Dictionary of test datasets
    """
    test_datasets = {}
    
    # Base test case: normal data without freq_* features
    base_df = sample_df.copy()
    
    # Remove freq_* features
    freq_columns = [col for col in base_df.columns if col.startswith('freq_')]
    base_df = base_df.drop(columns=freq_columns)
    print(f"Removed freq_* features: {freq_columns}")
    
    # Process type column - use actual cancer types
    if any(col.startswith('type_') for col in base_df.columns):
        # Convert one-hot encoded type columns to single type column
        type_columns = [col for col in base_df.columns if col.startswith('type_')]
        
        # Create a list to store types
        types = []
        for _, row in base_df.iterrows():
            found_type = None
            for col in type_columns:
                if row[col] == 1:
                    found_type = col.replace('type_', '')
                    break
            types.append(found_type if found_type else 'BRCA')
        
        # Add type column as string
        base_df['type'] = types
        
        # Drop one-hot encoded type columns
        base_df = base_df.drop(columns=type_columns)
    else:
        # If no type columns, add BRCA as default
        base_df['type'] = 'BRCA'
    
    print(f"Type column added with values: {base_df['type'].unique()}")
    
    # Test case 1: Normal data
    test_datasets['normal'] = base_df
    
    # Test case 2: Data with missing values
    missing_df = base_df.copy()
    
    # Introduce missing values in various columns
    missing_columns = ['age', 'gender', 'segVal', 'minor_cn', 'purity', 'ploidy']
    for col in missing_columns:
        if col in missing_df.columns:
            # Introduce 30% missing values
            mask = np.random.rand(len(missing_df)) < 0.3
            missing_df.loc[mask, col] = np.nan
            print(f"Introduced missing values in column: {col}")
    
    test_datasets['missing_values'] = missing_df
    
    # Test case 3: Data with inconsistent column order
    inconsistent_df = base_df.copy()
    
    # Shuffle columns randomly
    np.random.seed(42)
    shuffled_columns = np.random.permutation(inconsistent_df.columns)
    inconsistent_df = inconsistent_df[shuffled_columns]
    print(f"Shuffled columns, new order: {list(inconsistent_df.columns[:5])}...")
    
    test_datasets['inconsistent_columns'] = inconsistent_df
    
    # Test case 4: Data with both missing values and inconsistent columns
    combined_df = missing_df.copy()
    shuffled_columns = np.random.permutation(combined_df.columns)
    combined_df = combined_df[shuffled_columns]
    print(f"Created combined test case with missing values and inconsistent columns")
    
    test_datasets['combined_issues'] = combined_df
    
    return test_datasets

def save_test_data(test_datasets, output_dir):
    """
    Save test datasets to CSV files
    
    Args:
        test_datasets (dict): Dictionary of test datasets
        output_dir (str): Output directory
    """
    os.makedirs(output_dir, exist_ok=True)
    
    for test_name, test_df in test_datasets.items():
        output_path = os.path.join(output_dir, f'test_{test_name}.csv')
        test_df.to_csv(output_path, index=False)
        print(f"Test data saved to: {output_path}")

def run_prediction_tests(model_path, test_datasets, output_base_dir):
    """
    Run prediction tests on all test datasets
    
    Args:
        model_path (str): Path to trained model
        test_datasets (dict): Dictionary of test datasets
        output_base_dir (str): Base output directory
    """
    results = {}
    
    for test_name, test_df in test_datasets.items():
        print(f"\n{'='*60}")
        print(f"Testing: {test_name}")
        print(f"{'='*60}")
        
        # Save test data temporarily
        test_dir = os.path.join(output_base_dir, 'test_data')
        os.makedirs(test_dir, exist_ok=True)
        test_path = os.path.join(test_dir, f'{test_name}.csv')
        test_df.to_csv(test_path, index=False)
        
        # Define output directory for this test
        output_dir = os.path.join(output_base_dir, f'predictions_{test_name}')
        
        try:
            # Run prediction
            print(f"Running prediction for {test_name}...")
            prediction_results = predict(model_path, test_path, output_dir, gpu=0)
            
            # Analyze results
            print(f"Prediction completed for {test_name}")
            print(f"Results shape: {prediction_results.shape}")
            print(f"Prediction distribution:")
            print(prediction_results['prediction'].value_counts())
            print(f"Sample-level predictions:")
            if 'sample_level_prediction_label' in prediction_results.columns:
                print(prediction_results['sample_level_prediction_label'].value_counts())
            
            results[test_name] = prediction_results
            print(f"✓ Test {test_name} passed successfully")
            
        except Exception as e:
            print(f"✗ Test {test_name} failed with error:")
            print(f"  Error: {str(e)}")
            results[test_name] = None
    
    return results

def generate_summary(results):
    """
    Generate summary of test results
    
    Args:
        results (dict): Dictionary of test results
    """
    print(f"\n{'='*80}")
    print("TEST SUMMARY")
    print(f"{'='*80}")
    
    for test_name, result in results.items():
        if result is not None:
            print(f"\nTest: {test_name}")
            print(f"Status: PASS")
            print(f"Number of predictions: {len(result)}")
            print(f"Positive predictions: {sum(result['prediction'])}")
            print(f"Negative predictions: {len(result) - sum(result['prediction'])}")
            print(f"Average prediction probability: {result['prediction_prob'].mean():.4f}")
        else:
            print(f"\nTest: {test_name}")
            print(f"Status: FAIL")
    
    print(f"\n{'='*80}")

def main():
    """
    Main function to run prediction tests
    """
    print("Starting prediction functionality tests...")
    
    # Define model path
    model_path = os.path.join(os.path.dirname(__file__), 'output_baseline', 'best_model.pth')
    print(f"Using model: {model_path}")
    
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return
    
    # Create output directory
    output_dir = os.path.join(os.path.dirname(__file__), 'test_predict_output')
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    # Load sample data for TCGA-21-5782-01
    sample_df = load_sample_data()
    
    # Prepare test datasets
    test_datasets = prepare_test_data(sample_df)
    
    # Save test datasets
    save_test_data(test_datasets, os.path.join(output_dir, 'test_data'))
    
    # Run prediction tests
    results = run_prediction_tests(model_path, test_datasets, output_dir)
    
    # Generate summary
    generate_summary(results)
    
    print("\nPrediction functionality tests completed!")

if __name__ == "__main__":
    main()
