#!/usr/bin/env python3
"""
Hyperparameter optimization script for otk model
"""

import os
import yaml
import torch
import numpy as np
from otk.train.trainer import Trainer
from otk.data.data_processor import DataProcessor

# Define hyperparameter search space
param_grid = {
    'learning_rate': [0.001, 0.0005, 0.0001],
    'batch_size': [512, 1024, 2048],
    'dropout': [0.2, 0.3, 0.4],
    'weight_decay': [0.0001, 0.001, 0.01],
    'focal_alpha': [0.5, 1.0, 2.0],
    'focal_gamma': [1.0, 2.0, 3.0]
}

def load_base_config():
    """Load the base configuration file"""
    config_path = 'configs/model_config.yml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def save_config(config, path):
    """Save configuration to file"""
    # Ensure the directory exists
    dir_path = os.path.dirname(path)
    if dir_path:
        os.makedirs(dir_path, exist_ok=True)
    with open(path, 'w') as f:
        yaml.dump(config, f)

def run_hyperparameter_optimization():
    """Run hyperparameter optimization"""
    base_config = load_base_config()
    best_score = 0.0
    best_params = {}
    
    # Store results for all combinations
    results_list = []
    
    # Generate all combinations of hyperparameters
    import itertools
    keys, values = zip(*param_grid.items())
    param_combinations = list(itertools.product(*values))
    
    print(f"Starting hyperparameter optimization with {len(param_combinations)} combinations")
    
    # Run each combination
    for i, params in enumerate(param_combinations):
        param_dict = dict(zip(keys, params))
        print(f"\nRunning combination {i+1}/{len(param_combinations)}: {param_dict}")
        
        # Update configuration
        config = base_config.copy()
        config['model']['optimizer']['lr'] = param_dict['learning_rate']
        config['training']['batch_size'] = param_dict['batch_size']
        config['model']['loss_function']['alpha'] = param_dict['focal_alpha']
        config['model']['loss_function']['gamma'] = param_dict['focal_gamma']
        config['model']['optimizer']['weight_decay'] = param_dict['weight_decay']
        
        # Save temporary config
        temp_config_path = f'temp_config_{i}.yml'
        save_config(config, temp_config_path)
        
        try:
            # Create trainer with output directory
            output_dir = f'output_hyperopt/{i}'
            trainer = Trainer(temp_config_path, output_dir=output_dir)
            
            # Run training for fewer epochs for optimization
            config['training']['epochs'] = 5
            trainer.config = config
            
            # Start training
            best_val_auPRC, test_metrics = trainer.train()
            
            # Get validation metrics from test_metrics
            val_auprc = best_val_auPRC
            val_auc = test_metrics.get('AUC', 0.0)
            val_f1 = test_metrics.get('F1', 0.0)
            
            print(f"Validation auPRC: {val_auprc}")
            print(f"Test AUC: {val_auc}")
            print(f"Test F1: {val_f1}")
            
            # Store results
            result = param_dict.copy()
            result['validation_auPRC'] = val_auprc
            result['test_AUC'] = val_auc
            result['test_F1'] = val_f1
            results_list.append(result)
            
            # Update best score
            if val_auprc > best_score:
                best_score = val_auprc
                best_params = param_dict
                print(f"New best score: {best_score}")
                
        except Exception as e:
            print(f"Error running combination: {e}")
            # Store error result
            result = param_dict.copy()
            result['validation_auPRC'] = 0.0
            result['validation_AUC'] = 0.0
            result['validation_F1'] = 0.0
            result['error'] = str(e)
            results_list.append(result)
        finally:
            # Clean up
            if os.path.exists(temp_config_path):
                os.remove(temp_config_path)
    
    # Save results to CSV
    import pandas as pd
    results_df = pd.DataFrame(results_list)
    results_csv_path = 'hyperparameter_optimization_results.csv'
    results_df.to_csv(results_csv_path, index=False)
    print(f"\nResults saved to {results_csv_path}")
    
    # Print top 5 results
    print("\n===== Top 5 Hyperparameter Combinations =====")
    top_results = results_df.nlargest(5, 'validation_auPRC')
    print(top_results)
    
    # Print best results
    print("\n===== Best Hyperparameter Combination =====")
    print(f"Best validation auPRC: {best_score}")
    print(f"Best hyperparameters: {best_params}")
    
    # Save best configuration
    best_config = base_config.copy()
    for key, value in best_params.items():
        if key == 'learning_rate':
            best_config['model']['optimizer']['lr'] = value
        elif key == 'batch_size':
            best_config['training']['batch_size'] = value
        elif key == 'focal_alpha':
            best_config['model']['loss_function']['alpha'] = value
        elif key == 'focal_gamma':
            best_config['model']['loss_function']['gamma'] = value
        elif key == 'weight_decay':
            best_config['model']['optimizer']['weight_decay'] = value
    
    save_config(best_config, 'configs/best_model_config.yml')
    print("Best configuration saved to configs/best_model_config.yml")

if __name__ == '__main__':
    run_hyperparameter_optimization()
