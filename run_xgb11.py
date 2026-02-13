#!/usr/bin/env python
"""Train XGB11 model - Paper, Original, and Full versions"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from otk.models.xgb11_model import XGB11Trainer
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def train_model(model_type, output_suffix):
    """Train a specific model version"""
    print(f"\n{'='*60}")
    print(f"Training XGB11 {model_type.upper()} Model")
    print(f"{'='*60}")
    
    trainer = XGB11Trainer(
        data_path='src/otk/data/sorted_modeling_data.csv.gz',
        output_dir=f'otk_api/models/xgb11_{output_suffix}',
        model_type=model_type
    )
    
    test_metrics, test_sample_metrics = trainer.train()
    
    print(f"\n{'='*60}")
    print(f"XGB11 {model_type.upper()} - TRAINING COMPLETED")
    print(f"{'='*60}")
    print(f"Gene-level auPRC: {test_metrics['auPRC']:.4f}")
    print(f"Gene-level Precision: {test_metrics['Precision']:.4f}")
    print(f"Sample-level auPRC: {test_sample_metrics['auPRC']:.4f}")
    print(f"Sample-level auROC: {test_sample_metrics['AUC']:.4f}")
    
    return test_metrics, test_sample_metrics

def main():
    results = {}
    
    # 1. Paper version - uses paper hyperparameters with 11 features
    results['paper'] = train_model('paper', 'paper')
    
    # 2. Original version - uses actual R model hyperparameters
    results['original'] = train_model('original', 'original')
    
    # 3. Full version - uses all features with comprehensive engineering
    results['full'] = train_model('full', 'full')
    
    # Summary comparison
    print("\n" + "="*60)
    print("FINAL COMPARISON")
    print("="*60)
    print(f"{'Model':<15} {'Gene auPRC':<12} {'Precision':<12} {'Sample auPRC':<14} {'Sample auROC':<12}")
    print("-"*60)
    
    for model_name, (gene_metrics, sample_metrics) in results.items():
        print(f"{model_name:<15} {gene_metrics['auPRC']:<12.4f} {gene_metrics['Precision']:<12.4f} "
              f"{sample_metrics['auPRC']:<14.4f} {sample_metrics['AUC']:<12.4f}")
    
    print("="*60)
    
    # Identify best model
    best_gene = max(results.items(), key=lambda x: x[1][0]['auPRC'])
    best_sample = max(results.items(), key=lambda x: x[1][1]['auPRC'])
    
    print(f"\nBest Gene-level auPRC: {best_gene[0]} ({best_gene[1][0]['auPRC']:.4f})")
    print(f"Best Sample-level auPRC: {best_sample[0]} ({best_sample[1][1]['auPRC']:.4f})")

if __name__ == "__main__":
    main()
