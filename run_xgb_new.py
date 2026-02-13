#!/usr/bin/env python
"""Train XGB New model with corrected feature engineering"""
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
    print("="*60)
    print("Training XGB New Model")
    print("(with corrected feature engineering)")
    print("="*60)
    
    trainer = XGB11Trainer(
        data_path='src/otk/data/sorted_modeling_data.csv.gz',
        output_dir='otk_api/models/xgb_new',
        model_type='new'
    )
    
    test_metrics, test_sample_metrics = trainer.train()
    
    print("\n" + "="*60)
    print("XGB NEW - TRAINING COMPLETED")
    print("="*60)
    print(f"Gene-level auPRC: {test_metrics['auPRC']:.4f}")
    print(f"Gene-level Precision: {test_metrics['Precision']:.4f}")
    print(f"Sample-level auPRC: {test_sample_metrics['auPRC']:.4f}")
    print(f"Sample-level auROC: {test_sample_metrics['AUC']:.4f}")
    print("="*60)

if __name__ == "__main__":
    main()
