#!/usr/bin/env python
import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from otk.train.trainer import Trainer

MODELS = [
    {'name': 'baseline_mlp', 'config': 'otk_api/models/baseline_mlp/config.yml', 'output': 'otk_api/models/baseline_mlp', 'gpu': 0},
    {'name': 'deep_residual', 'config': 'otk_api/models/deep_residual/config.yml', 'output': 'otk_api/models/deep_residual', 'gpu': 0},
    {'name': 'optimized_residual', 'config': 'otk_api/models/optimized_residual/config.yml', 'output': 'otk_api/models/optimized_residual', 'gpu': 0},
    {'name': 'transformer', 'config': 'otk_api/models/transformer/config.yml', 'output': 'otk_api/models/transformer', 'gpu': 0},
]

def main():
    print("="*60)
    print("Batch Training Script for otk_api/models")
    print("="*60)
    
    total_start = time.time()
    results = []
    
    for model_info in MODELS:
        name = model_info['name']
        config = model_info['config']
        output = model_info['output']
        gpu = model_info['gpu']
        
        print(f"\n{'='*60}")
        print(f"Training model: {name}")
        print(f"{'='*60}\n")
        
        start_time = time.time()
        
        try:
            trainer = Trainer(config, output, gpu)
            best_val_auPRC, test_metrics = trainer.train()
            
            elapsed = time.time() - start_time
            print(f"\n{name} completed in {elapsed:.2f} seconds")
            print(f"Best validation auPRC: {best_val_auPRC:.4f}")
            
            results.append({
                'name': name,
                'status': 'success',
                'best_val_auPRC': best_val_auPRC,
                'test_metrics': test_metrics,
                'elapsed': elapsed
            })
        except Exception as e:
            elapsed = time.time() - start_time
            print(f"\n{name} failed with error: {str(e)}")
            results.append({
                'name': name,
                'status': 'failed',
                'error': str(e),
                'elapsed': elapsed
            })
    
    total_elapsed = time.time() - total_start
    
    print("\n" + "="*60)
    print("TRAINING SUMMARY")
    print("="*60)
    
    for result in results:
        print(f"\n{result['name']}:")
        if result['status'] == 'success':
            print(f"  Status: SUCCESS")
            print(f"  Best Val auPRC: {result['best_val_auPRC']:.4f}")
            print(f"  Test auPRC: {result['test_metrics']['auPRC']:.4f}")
            print(f"  Test Precision: {result['test_metrics']['Precision']:.4f}")
            print(f"  Test Recall: {result['test_metrics']['Recall']:.4f}")
        else:
            print(f"  Status: FAILED - {result['error']}")
    
    print(f"\nTotal training time: {total_elapsed:.2f} seconds")

if __name__ == "__main__":
    main()
