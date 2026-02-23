#!/usr/bin/env python
"""
Update gene-level confusion matrix (TP, FP, TN, FN) for existing trained models.
This script loads trained models using UnifiedPredictor, makes predictions on train/val/test sets,
and updates training_summary.yml with the confusion matrix data.
"""
import os
import sys
import yaml
import argparse
from pathlib import Path
from typing import Dict, Any

import pandas as pd
import numpy as np

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / 'src'))

from otk.data.data_processor import DataProcessor
from otk.data.data_split import get_data_splits
from otk.predict.predictor import UnifiedPredictor


def load_and_prepare_data():
    """Load and prepare data for evaluation"""
    models_dir = Path(__file__).parent / 'models'
    
    config_path = None
    for model_dir in models_dir.iterdir():
        if model_dir.is_dir() and (model_dir / 'config.yml').exists():
            config_path = model_dir / 'config.yml'
            break
    
    if config_path is None:
        raise FileNotFoundError("No model config.yml found")
    
    data_processor = DataProcessor(str(config_path))
    df = data_processor.load_data()
    
    features, target, samples, genes = data_processor.preprocess(df)
    
    train_samples, val_samples, test_samples = get_data_splits()
    
    train_mask = samples.isin(train_samples)
    val_mask = samples.isin(val_samples)
    test_mask = samples.isin(test_samples)
    
    splits = {
        'train': {
            'df': df[train_mask].reset_index(drop=True),
            'y': target[train_mask].reset_index(drop=True),
        },
        'val': {
            'df': df[val_mask].reset_index(drop=True),
            'y': target[val_mask].reset_index(drop=True),
        },
        'test': {
            'df': df[test_mask].reset_index(drop=True),
            'y': target[test_mask].reset_index(drop=True),
        }
    }
    
    return splits


def calculate_confusion_matrix(y_true, y_pred):
    """Calculate confusion matrix components"""
    tp = ((y_pred == 1) & (y_true == 1)).sum()
    fp = ((y_pred == 1) & (y_true == 0)).sum()
    fn = ((y_pred == 0) & (y_true == 1)).sum()
    tn = ((y_pred == 0) & (y_true == 0)).sum()
    return int(tp), int(fp), int(fn), int(tn)


def update_model_gene_level_metrics(model_name: str, model_dir: Path, splits: Dict) -> bool:
    """Update gene-level metrics for a single model"""
    
    summary_path = model_dir / 'training_summary.yml'
    model_path = model_dir / 'best_model.pkl'
    
    if not summary_path.exists():
        print(f"  [SKIP] No training_summary.yml found")
        return False
    
    if not model_path.exists():
        print(f"  [SKIP] No best_model.pkl found")
        return False
    
    with open(summary_path) as f:
        summary = yaml.safe_load(f)
    
    if 'gene_level' not in summary:
        print(f"  [SKIP] No gene_level section in training_summary.yml")
        return False
    
    gene_level = summary['gene_level']
    
    all_have_cm = True
    for split_name in ['train', 'val', 'test']:
        if split_name in gene_level:
            if 'TP' not in gene_level[split_name] or 'FP' not in gene_level[split_name]:
                all_have_cm = False
                break
    
    if all_have_cm:
        print(f"  [SKIP] All splits already have confusion matrix data")
        return False
    
    try:
        predictor = UnifiedPredictor(str(model_path), gpu=-1)
        print(f"  [INFO] Model loaded successfully (type: {predictor.model_type})")
        
        updated = False
        for split_name, split_data in splits.items():
            if split_name not in gene_level:
                continue
            
            df = split_data['df']
            y = split_data['y']
            
            threshold = gene_level[split_name].get('threshold', predictor.optimal_threshold)
            
            X, feature_names = predictor.prepare_features(df)
            probs = predictor.predict_proba(X, feature_names=feature_names)
            preds = (probs >= threshold).astype(int)
            
            tp, fp, fn, tn = calculate_confusion_matrix(y.values, preds)
            
            gene_level[split_name]['TP'] = tp
            gene_level[split_name]['FP'] = fp
            gene_level[split_name]['FN'] = fn
            gene_level[split_name]['TN'] = tn
            
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            gene_level[split_name]['Specificity'] = round(specificity, 6)
            
            print(f"  [UPDATE] {split_name}: TP={tp}, FP={fp}, FN={fn}, TN={tn}, Specificity={specificity:.4f}")
            updated = True
        
        if updated:
            summary['gene_level'] = gene_level
            with open(summary_path, 'w') as f:
                yaml.dump(summary, f, default_flow_style=False, sort_keys=False)
            print(f"  [SUCCESS] Updated training_summary.yml")
        
        return True
        
    except Exception as e:
        print(f"  [ERROR] Failed to update: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description='Update gene-level confusion matrix for trained models')
    parser.add_argument('--model', type=str, default=None, help='Specific model to update (default: all models)')
    parser.add_argument('--models-dir', type=str, default=None, help='Models directory path')
    args = parser.parse_args()
    
    if args.models_dir:
        models_dir = Path(args.models_dir)
    else:
        models_dir = Path(__file__).parent / 'models'
    
    print("=" * 60)
    print("Updating gene-level confusion matrix for trained models")
    print("=" * 60)
    
    print("\n[1/2] Loading and preparing data...")
    splits = load_and_prepare_data()
    print(f"  Train: {len(splits['train']['y'])} samples")
    print(f"  Val: {len(splits['val']['y'])} samples")
    print(f"  Test: {len(splits['test']['y'])} samples")
    
    print("\n[2/2] Updating models...")
    
    if args.model:
        model_dirs = [(args.model, models_dir / args.model)]
    else:
        model_dirs = []
        for item in models_dir.iterdir():
            if item.is_dir() and (item / 'training_summary.yml').exists():
                model_dirs.append((item.name, item))
    
    success_count = 0
    for model_name, model_dir in model_dirs:
        print(f"\n[{model_name}]")
        if update_model_gene_level_metrics(model_name, model_dir, splits):
            success_count += 1
    
    print("\n" + "=" * 60)
    print(f"Done! Updated {success_count}/{len(model_dirs)} models")
    print("=" * 60)


if __name__ == '__main__':
    main()
