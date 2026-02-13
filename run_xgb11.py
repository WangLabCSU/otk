#!/usr/bin/env python
"""Train XGB11 model - Original and Optimized versions"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from otk.models.xgb11_model import XGB11Trainer
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def main():
    # Train original XGB11
    print("="*60)
    print("Training XGB11 Original Model")
    print("="*60)
    
    trainer_original = XGB11Trainer(
        data_path='src/otk/data/sorted_modeling_data.csv.gz',
        output_dir='otk_api/models/xgb11',
        model_type='original'
    )
    
    test_metrics_orig, test_sample_metrics_orig = trainer_original.train()
    
    print("\n" + "="*60)
    print("XGB11 ORIGINAL - TRAINING COMPLETED")
    print("="*60)
    print(f"Gene-level auPRC: {test_metrics_orig['auPRC']:.4f}")
    print(f"Gene-level Precision: {test_metrics_orig['Precision']:.4f}")
    print(f"Sample-level auPRC: {test_sample_metrics_orig['auPRC']:.4f}")
    print(f"Sample-level auROC: {test_sample_metrics_orig['AUC']:.4f}")
    
    # Train optimized XGB11
    print("\n" + "="*60)
    print("Training XGB11 Optimized Model")
    print("="*60)
    
    trainer_optimized = XGB11Trainer(
        data_path='src/otk/data/sorted_modeling_data.csv.gz',
        output_dir='otk_api/models/xgb11',
        model_type='optimized'
    )
    
    test_metrics_opt, test_sample_metrics_opt = trainer_optimized.train()
    
    print("\n" + "="*60)
    print("XGB11 OPTIMIZED - TRAINING COMPLETED")
    print("="*60)
    print(f"Gene-level auPRC: {test_metrics_opt['auPRC']:.4f}")
    print(f"Gene-level Precision: {test_metrics_opt['Precision']:.4f}")
    print(f"Sample-level auPRC: {test_sample_metrics_opt['auPRC']:.4f}")
    print(f"Sample-level auROC: {test_sample_metrics_opt['AUC']:.4f}")
    
    print("\n" + "="*60)
    print("COMPARISON")
    print("="*60)
    print(f"Original  - Gene auPRC: {test_metrics_orig['auPRC']:.4f}, Sample auPRC: {test_sample_metrics_orig['auPRC']:.4f}")
    print(f"Optimized - Gene auPRC: {test_metrics_opt['auPRC']:.4f}, Sample auPRC: {test_sample_metrics_opt['auPRC']:.4f}")

if __name__ == "__main__":
    main()
