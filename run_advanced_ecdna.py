#!/usr/bin/env python3
"""
Run AdvancedEcDNA model training with optimized parameters for high auPRC and precision
"""

import os
import sys
import time
from otk.train.trainer import Trainer

if __name__ == "__main__":
    # Configuration and output settings
    config_path = "configs/advanced_ecdna.yml"
    output_dir = "output_advanced_ecdna"
    gpu = 0  # Use first GPU
    
    # Check if configuration file exists
    if not os.path.exists(config_path):
        print(f"Configuration file not found: {config_path}")
        sys.exit(1)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Running AdvancedEcDNA model with optimized parameters, config: {config_path}")
    print(f"Output directory: {output_dir}")
    
    # Initialize trainer
    trainer = Trainer(config_path, output_dir, gpu=gpu)
    
    # Start training
    start_time = time.time()
    trainer.train()
    training_time = time.time() - start_time
    
    print(f"Training completed in {training_time:.2f} seconds")
    print(f"Results saved to {output_dir}")
