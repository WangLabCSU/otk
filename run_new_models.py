#!/usr/bin/env python
"""
Train DGIT Super and TabPFN models in parallel
"""
import sys
import os
import time
import multiprocessing as mp
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))


def train_dgit_super():
    """Train DGIT Super model"""
    from otk.train.trainer import Trainer
    
    print(f"[DGITSuper] Starting training at {datetime.now().strftime('%H:%M:%S')}")
    
    start_time = time.time()
    
    try:
        trainer = Trainer('configs/dgit_super.yml', 'otk_api/models/dgit_super', gpu=0)
        best_val_auPRC, test_metrics = trainer.train()
        
        elapsed = time.time() - start_time
        print(f"[DGITSuper] Completed in {elapsed/60:.1f} min, best val auPRC: {best_val_auPRC:.4f}")
        
        return {
            'name': 'DGITSuper',
            'status': 'success',
            'best_val_auPRC': best_val_auPRC,
            'test_metrics': test_metrics,
            'elapsed': elapsed
        }
    except Exception as e:
        print(f"[DGITSuper] FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return {'name': 'DGITSuper', 'status': 'failed', 'error': str(e)}


def train_tabpfn():
    """Train TabPFN model"""
    from otk.models.tabpfn_model import TabPFNTrainer
    
    print(f"[TabPFN] Starting training at {datetime.now().strftime('%H:%M:%S')}")
    
    start_time = time.time()
    
    try:
        trainer = TabPFNTrainer(
            data_path='src/otk/data/sorted_modeling_data.csv.gz',
            output_dir='otk_api/models/tabpfn',
            n_estimators=5,
            max_samples_per_estimator=5000
        )
        val_metrics, test_metrics = trainer.train_and_evaluate()
        
        elapsed = time.time() - start_time
        print(f"[TabPFN] Completed in {elapsed/60:.1f} min, test auPRC: {test_metrics['auPRC']:.4f}")
        
        return {
            'name': 'TabPFN',
            'status': 'success',
            'val_metrics': val_metrics,
            'test_metrics': test_metrics,
            'elapsed': elapsed
        }
    except Exception as e:
        print(f"[TabPFN] FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return {'name': 'TabPFN', 'status': 'failed', 'error': str(e)}


def main():
    print("=" * 60)
    print("Training DGIT Super and TabPFN models")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    total_start = time.time()
    
    # Run in parallel
    with mp.Pool(processes=2) as pool:
        results = pool.map(lambda f: f(), [train_dgit_super, train_tabpfn])
    
    total_elapsed = time.time() - total_start
    
    # Print summary
    print("\n" + "=" * 60)
    print("TRAINING SUMMARY")
    print("=" * 60)
    
    for result in results:
        print(f"\n{result['name']}:")
        if result['status'] == 'success':
            tm = result.get('test_metrics', {})
            print(f"  Status: SUCCESS")
            print(f"  Test auPRC: {tm.get('auPRC', tm.get('auPRC', 0)):.4f}")
            print(f"  Test Precision: {tm.get('Precision', 0):.4f}")
            print(f"  Test Recall: {tm.get('Recall', 0):.4f}")
            print(f"  Training Time: {result['elapsed']/60:.1f} min")
        else:
            print(f"  Status: FAILED - {result.get('error', 'Unknown')}")
    
    print(f"\nTotal time: {total_elapsed/3600:.2f} hours")
    print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
