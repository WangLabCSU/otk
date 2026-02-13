#!/usr/bin/env python
"""
Parallel training script for otk_api/models
Runs 4 models simultaneously using multiprocessing
"""
import sys
import os
import time
import multiprocessing as mp
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from otk.train.trainer import Trainer

MODELS = [
    {'name': 'baseline_mlp', 'config': 'otk_api/models/baseline_mlp/config.yml', 'output': 'otk_api/models/baseline_mlp', 'gpu': -1},
    {'name': 'deep_residual', 'config': 'otk_api/models/deep_residual/config.yml', 'output': 'otk_api/models/deep_residual', 'gpu': -1},
    {'name': 'optimized_residual', 'config': 'otk_api/models/optimized_residual/config.yml', 'output': 'otk_api/models/optimized_residual', 'gpu': -1},
    {'name': 'transformer', 'config': 'otk_api/models/transformer/config.yml', 'output': 'otk_api/models/transformer', 'gpu': -1},
]


def train_model(model_info):
    """Train a single model in a separate process"""
    name = model_info['name']
    config = model_info['config']
    output = model_info['output']
    gpu = model_info['gpu']
    
    print(f"[{name}] Starting training at {datetime.now().strftime('%H:%M:%S')}")
    
    start_time = time.time()
    
    try:
        trainer = Trainer(config, output, gpu)
        best_val_auPRC, test_metrics = trainer.train()
        
        elapsed = time.time() - start_time
        print(f"[{name}] Completed in {elapsed:.2f} seconds, best val auPRC: {best_val_auPRC:.4f}")
        
        return {
            'name': name,
            'status': 'success',
            'best_val_auPRC': best_val_auPRC,
            'test_metrics': test_metrics,
            'elapsed': elapsed
        }
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"[{name}] FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return {
            'name': name,
            'status': 'failed',
            'error': str(e),
            'elapsed': elapsed
        }


def main():
    print("=" * 60)
    print("Parallel Training Script for otk_api/models")
    print(f"Running {len(MODELS)} models in parallel")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    total_start = time.time()
    
    # Create process pool
    num_processes = min(len(MODELS), mp.cpu_count())
    print(f"\nUsing {num_processes} parallel processes")
    print(f"Available CPU cores: {mp.cpu_count()}")
    
    # Run training in parallel
    with mp.Pool(processes=num_processes) as pool:
        results = pool.map(train_model, MODELS)
    
    total_elapsed = time.time() - total_start
    
    # Print summary
    print("\n" + "=" * 60)
    print("TRAINING SUMMARY")
    print("=" * 60)
    
    successful = 0
    for result in results:
        print(f"\n{result['name']}:")
        if result['status'] == 'success':
            successful += 1
            print(f"  Status: SUCCESS")
            print(f"  Best Val auPRC: {result['best_val_auPRC']:.4f}")
            print(f"  Test auPRC: {result['test_metrics']['auPRC']:.4f}")
            print(f"  Test Precision: {result['test_metrics']['Precision']:.4f}")
            print(f"  Test Recall: {result['test_metrics']['Recall']:.4f}")
            print(f"  Training Time: {result['elapsed']:.2f}s")
        else:
            print(f"  Status: FAILED - {result['error']}")
    
    print(f"\nTotal wall-clock time: {total_elapsed:.2f} seconds ({total_elapsed/3600:.2f} hours)")
    print(f"Successful models: {successful}/{len(MODELS)}")
    print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
