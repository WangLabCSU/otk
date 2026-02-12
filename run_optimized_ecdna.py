#!/usr/bin/env python
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from otk.train.trainer import Trainer

if __name__ == "__main__":
    config_path = "configs/optimized_ecdna.yml"
    output_dir = "output_optimized_ecdna"
    gpu = 0
    
    print(f"Running OptimizedEcDNA model with optimized parameters")
    print(f"Config: {config_path}")
    print(f"Output directory: {output_dir}")
    print(f"Using GPU: {gpu}")
    
    trainer = Trainer(config_path, output_dir, gpu)
    trainer.train()
    
    print(f"Training completed. Results saved to {output_dir}")
