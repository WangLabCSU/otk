#!/usr/bin/env python
"""Train DGIT model"""
import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from otk.train.trainer import Trainer


def main():
    config_path = 'configs/dgit_v1.yml'
    output_dir = 'otk_api/models/dgit_v1'
    
    print("=" * 60)
    print("Training Deep Gated Interaction Transformer (DGIT)")
    print(f"Config: {config_path}")
    print(f"Output: {output_dir}")
    print("=" * 60)
    
    start_time = time.time()
    
    try:
        trainer = Trainer(config_path, output_dir, gpu=0)
        best_val_auPRC, test_metrics = trainer.train()
        
        elapsed = time.time() - start_time
        print("\n" + "=" * 60)
        print("TRAINING COMPLETED")
        print("=" * 60)
        print(f"Best validation auPRC: {best_val_auPRC:.4f}")
        print(f"Test auPRC: {test_metrics['auPRC']:.4f}")
        print(f"Test Precision: {test_metrics['Precision']:.4f}")
        print(f"Test Recall: {test_metrics['Recall']:.4f}")
        print(f"Test F1: {test_metrics['F1']:.4f}")
        print(f"Total time: {elapsed/60:.1f} minutes")
        
    except Exception as e:
        print(f"Training failed: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
