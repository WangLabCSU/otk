#!/usr/bin/env python
"""Test script for otk predict functionality"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import pandas as pd
import numpy as np
from otk.predict.predictor import Predictor

def create_test_data():
    """Create test data for prediction"""
    np.random.seed(42)
    n_samples = 100
    
    data = {
        'sample': [f'test_sample_{i}' for i in range(n_samples)],
        'gene_id': [f'gene_{i % 20}' for i in range(n_samples)],
        'cancer_type': np.random.choice(['BRCA', 'LUAD', 'COAD', 'GBM'], n_samples),
        'segVal': np.random.uniform(0, 10, n_samples),
        'ploidy': np.random.uniform(1.5, 4.0, n_samples),
        'cn': np.random.randint(0, 10, n_samples),
        'sv_count': np.random.randint(0, 5, n_samples),
        'amplicon_count': np.random.randint(0, 3, n_samples),
        'amplicon_type': np.random.choice(['Linear', 'BFB', 'Circular', 'HR'], n_samples),
    }
    
    df = pd.DataFrame(data)
    test_file = 'test_predict_input.csv'
    df.to_csv(test_file, index=False)
    print(f"Created test data: {test_file} with {n_samples} samples")
    return test_file

def test_predict():
    """Test prediction functionality"""
    print("=" * 60)
    print("Testing otk predict functionality")
    print("=" * 60)
    
    # Create test data
    test_file = create_test_data()
    
    # Test with different models
    models = [
        'otk_api/models/baseline_mlp/best_model.pth',
        'otk_api/models/deep_residual/best_model.pth',
        'otk_api/models/optimized_residual/best_model.pth',
        'otk_api/models/transformer/best_model.pth',
    ]
    
    results = []
    
    for model_path in models:
        if not os.path.exists(model_path):
            print(f"\nSkipping {model_path}: model not found")
            continue
        
        print(f"\n{'='*60}")
        print(f"Testing model: {model_path}")
        print(f"{'='*60}")
        
        try:
            # Initialize predictor
            predictor = Predictor(model_path, gpu=-1)
            print(f"Model loaded successfully")
            print(f"Model type: {predictor.config['model']['architecture']['type']}")
            print(f"Optimal threshold: {predictor.optimal_threshold}")
            
            # Run prediction
            output_dir = f"test_output_{os.path.basename(os.path.dirname(model_path))}"
            os.makedirs(output_dir, exist_ok=True)
            predictor.run(test_file, output_dir)
            
            # Verify output
            output_file = os.path.join(output_dir, 'predictions.csv')
            if os.path.exists(output_file):
                pred_df = pd.read_csv(output_file)
                print(f"\nPrediction output shape: {pred_df.shape}")
                print(f"Columns: {list(pred_df.columns)}")
                print(f"\nPrediction statistics:")
                print(f"  Mean probability: {pred_df['prediction_prob'].mean():.4f}")
                print(f"  Std probability: {pred_df['prediction_prob'].std():.4f}")
                print(f"  Min probability: {pred_df['prediction_prob'].min():.4f}")
                print(f"  Max probability: {pred_df['prediction_prob'].max():.4f}")
                print(f"  Predicted positive: {(pred_df['prediction'] == 1).sum()}")
                print(f"  Predicted negative: {(pred_df['prediction'] == 0).sum()}")
                
                results.append({
                    'model': os.path.basename(os.path.dirname(model_path)),
                    'status': 'SUCCESS',
                    'output_dir': output_dir,
                    'output_file': output_file,
                    'n_predictions': len(pred_df),
                    'n_positive': (pred_df['prediction'] == 1).sum()
                })
            else:
                print(f"ERROR: Output file not created: {output_file}")
                results.append({
                    'model': os.path.basename(os.path.dirname(model_path)),
                    'status': 'FAILED',
                    'error': 'Output file not created'
                })
                
        except Exception as e:
            print(f"ERROR: {str(e)}")
            import traceback
            traceback.print_exc()
            results.append({
                'model': os.path.basename(os.path.dirname(model_path)),
                'status': 'FAILED',
                'error': str(e)
            })
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    for r in results:
        status_icon = "✅" if r['status'] == 'SUCCESS' else "❌"
        print(f"{status_icon} {r['model']}: {r['status']}")
        if r['status'] == 'SUCCESS':
            print(f"   - Predictions: {r['n_predictions']}, Positive: {r['n_positive']}")
        else:
            print(f"   - Error: {r.get('error', 'Unknown')}")
    
    # Cleanup
    print("\nCleaning up test files...")
    os.remove(test_file)
    for r in results:
        if r['status'] == 'SUCCESS':
            if os.path.exists(r['output_file']):
                os.remove(r['output_file'])
                print(f"  Removed: {r['output_file']}")
            if os.path.exists(r['output_dir']):
                import shutil
                shutil.rmtree(r['output_dir'])
                print(f"  Removed dir: {r['output_dir']}")
    
    # Check if all tests passed
    all_passed = all(r['status'] == 'SUCCESS' for r in results)
    print("\n" + "=" * 60)
    if all_passed:
        print("ALL TESTS PASSED ✅")
    else:
        print("SOME TESTS FAILED ❌")
    print("=" * 60)
    
    return all_passed

if __name__ == "__main__":
    success = test_predict()
    sys.exit(0 if success else 1)
