#!/usr/bin/env python
"""
Sequential GPU training script for otk_api/models
Trains models one by one on GPU for stability
"""
import sys
import os
import time
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from otk.train.trainer import Trainer

MODELS = [
    {'name': 'baseline_mlp', 'config': 'otk_api/models/baseline_mlp/config.yml', 'output': 'otk_api/models/baseline_mlp', 'gpu': 0},
    {'name': 'deep_residual', 'config': 'otk_api/models/deep_residual/config.yml', 'output': 'otk_api/models/deep_residual', 'gpu': 0},
    {'name': 'optimized_residual', 'config': 'otk_api/models/optimized_residual/config.yml', 'output': 'otk_api/models/optimized_residual', 'gpu': 0},
    {'name': 'transformer', 'config': 'otk_api/models/transformer/config.yml', 'output': 'otk_api/models/transformer', 'gpu': 0},
]


def main():
    print("=" * 60)
    print("Sequential GPU Training Script for otk_api/models")
    print(f"Running {len(MODELS)} models on GPU")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    total_start = time.time()
    results = []
    
    for model_info in MODELS:
        name = model_info['name']
        config = model_info['config']
        output = model_info['output']
        gpu = model_info['gpu']
        
        print(f"\n{'='*60}")
        print(f"[{name}] Starting training at {datetime.now().strftime('%H:%M:%S')}")
        print(f"{'='*60}\n")
        
        start_time = time.time()
        
        try:
            trainer = Trainer(config, output, gpu)
            best_val_auPRC, test_metrics = trainer.train()
            
            elapsed = time.time() - start_time
            print(f"\n[{name}] Completed in {elapsed:.2f} seconds ({elapsed/60:.1f} min)")
            print(f"[{name}] Best val auPRC: {best_val_auPRC:.4f}, Test auPRC: {test_metrics['auPRC']:.4f}")
            
            results.append({
                'name': name,
                'status': 'success',
                'best_val_auPRC': best_val_auPRC,
                'test_metrics': test_metrics,
                'elapsed': elapsed
            })
        except Exception as e:
            elapsed = time.time() - start_time
            print(f"\n[{name}] FAILED: {str(e)}")
            import traceback
            traceback.print_exc()
            results.append({
                'name': name,
                'status': 'failed',
                'error': str(e),
                'elapsed': elapsed
            })
    
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
            tm = result['test_metrics']
            print(f"  Status: SUCCESS")
            print(f"  Best Val auPRC: {result['best_val_auPRC']:.4f}")
            print(f"  Test auPRC: {tm['auPRC']:.4f}")
            print(f"  Test AUC: {tm['AUC']:.4f}")
            print(f"  Test Precision: {tm['Precision']:.4f}")
            print(f"  Test Recall: {tm['Recall']:.4f}")
            print(f"  Test F1: {tm['F1']:.4f}")
            print(f"  Training Time: {result['elapsed']:.2f}s ({result['elapsed']/60:.1f} min)")
        else:
            print(f"  Status: FAILED - {result['error']}")
    
    print(f"\nTotal training time: {total_elapsed:.2f} seconds ({total_elapsed/3600:.2f} hours)")
    print(f"Successful models: {successful}/{len(MODELS)}")
    print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
