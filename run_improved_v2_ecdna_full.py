#!/usr/bin/env python3
"""
Run improved V2 model with full ecDNA optimization (ecDNAOptimizedLoss)
"""

import sys
import os
from otk.train.trainer import train_model

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def main():
    """Run improved V2 model with ecDNAOptimizedLoss"""
    config_path = os.path.join(os.path.dirname(__file__), 'configs', 'improved_v2_ecdna_full.yml')
    output_dir = os.path.join(os.path.dirname(__file__), 'output_improved_v2_ecdna_full')
    
    print(f"Running improved V2 model with ecDNAOptimizedLoss")
    print(f"Config: {config_path}")
    print(f"Output directory: {output_dir}")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Run training
    best_val_auPRC, test_metrics = train_model(config_path, output_dir, gpu=0)
    
    print(f"\nTraining completed!")
    print(f"Best validation auPRC: {best_val_auPRC:.4f}")
    print(f"Test metrics:")
    for metric, value in test_metrics.items():
        print(f"  {metric}: {value:.4f}")

if __name__ == "__main__":
    main()
