#!/usr/bin/env python3
"""
Run both baseline and improved model training in parallel
"""

import sys
import os
import concurrent.futures
from otk.train.trainer import train_model

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def run_model(config_path, output_dir, model_name):
    """
    Run model training for a single model
    
    Args:
        config_path (str): Path to model config file
        output_dir (str): Path to output directory
        model_name (str): Name of the model
    
    Returns:
        tuple: (model_name, best_val_auPRC, test_metrics)
    """
    print(f"\n=== Starting {model_name} model training ===")
    print(f"Config: {config_path}")
    print(f"Output directory: {output_dir}")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Run training
    best_val_auPRC, test_metrics = train_model(config_path, output_dir, gpu=0)
    
    print(f"\n=== {model_name} model training completed ===")
    print(f"Best validation auPRC: {best_val_auPRC:.4f}")
    print(f"Test metrics:")
    for metric, value in test_metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    return model_name, best_val_auPRC, test_metrics

def main():
    """
    Run both baseline and improved model training in parallel
    """
    print("Starting both baseline and improved model training...")
    print("Models will be trained in parallel to save time")
    
    # Define paths for both models
    base_dir = os.path.dirname(__file__)
    
    models = [
        {
            "name": "Baseline",
            "config_path": os.path.join(base_dir, 'configs', 'baseline_config.yml'),
            "output_dir": os.path.join(base_dir, 'output_baseline')
        },
        {
            "name": "Improved",
            "config_path": os.path.join(base_dir, 'configs', 'improved_config.yml'),
            "output_dir": os.path.join(base_dir, 'output_improved')
        }
    ]
    
    # Run both models in parallel
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        # Submit both training jobs
        future_to_model = {
            executor.submit(run_model, model['config_path'], model['output_dir'], model['name']): model['name']
            for model in models
        }
        
        # Collect results
        for future in concurrent.futures.as_completed(future_to_model):
            model_name = future_to_model[future]
            try:
                result = future.result()
                results.append(result)
            except Exception as exc:
                print(f'{model_name} model generated an exception: {exc}')
    
    # Print summary of results
    print("\n" + "="*60)
    print("SUMMARY OF TRAINING RESULTS")
    print("="*60)
    
    for model_name, best_val_auPRC, test_metrics in results:
        print(f"\n{model_name} Model:")
        print(f"  Best Validation auPRC: {best_val_auPRC:.4f}")
        print(f"  Test auPRC: {test_metrics.get('auPRC', 'N/A'):.4f}")
        print(f"  Test Accuracy: {test_metrics.get('accuracy', 'N/A'):.4f}")
        print(f"  Test F1 Score: {test_metrics.get('f1', 'N/A'):.4f}")
    
    print("\n" + "="*60)
    print("Both models have been trained!")
    print("="*60)

if __name__ == "__main__":
    main()
