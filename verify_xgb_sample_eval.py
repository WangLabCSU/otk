#!/usr/bin/env python
"""Verify XGB model sample-level evaluation logic"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import pandas as pd
import numpy as np
from otk.data.data_split import load_split
from otk.models.xgb11_model import XGBNewModel
from sklearn.metrics import precision_recall_curve, auc, roc_auc_score

def evaluate_sample_level_manual(df, probs, threshold=0.5):
    """Manual calculation of sample-level metrics"""
    df = df.copy()
    df['prob'] = probs
    
    # Aggregate to sample level
    sample_agg = df.groupby('sample').agg({
        'y': 'max',
        'prob': 'max'
    }).reset_index()
    
    y_true = sample_agg['y'].values
    y_prob = sample_agg['prob'].values
    y_pred = (y_prob >= threshold).astype(int)
    
    # Calculate metrics
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    auprc = auc(recall, precision)
    auc_score = roc_auc_score(y_true, y_prob)
    
    tp = ((y_pred == 1) & (y_true == 1)).sum()
    fp = ((y_pred == 1) & (y_true == 0)).sum()
    fn = ((y_pred == 0) & (y_true == 1)).sum()
    tn = ((y_pred == 0) & (y_true == 0)).sum()
    
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    acc = (tp + tn) / len(y_true)
    
    return {
        'auPRC': auprc,
        'AUC': auc_score,
        'Accuracy': acc,
        'Precision': prec,
        'Recall': rec,
        'F1': f1,
        'total_samples': len(y_true),
        'positive_samples': int(y_true.sum()),
        'predicted_positive': int(y_pred.sum()),
        'TP': int(tp), 'FP': int(fp), 'FN': int(fn), 'TN': int(tn)
    }

def main():
    print("="*60)
    print("Verifying XGB Sample-Level Evaluation")
    print("="*60)
    
    # Load data with unified split
    train_df, val_df, test_df = load_split()
    
    print(f"\nData loaded:")
    print(f"  Train: {len(train_df)} rows, {train_df['y'].sum()} positive")
    print(f"  Val: {len(val_df)} rows, {val_df['y'].sum()} positive")
    print(f"  Test: {len(test_df)} rows, {test_df['y'].sum()} positive")
    
    # Load XGB model
    model = XGBNewModel()
    model.load('otk_api/models/xgb_new/xgb11_new_model.pkl')
    
    print(f"\nModel loaded: xgb_new")
    print(f"  Optimal threshold: {model.optimal_threshold:.4f}")
    
    # Evaluate on each split
    for split_name, split_df in [('Train', train_df), ('Val', val_df), ('Test', test_df)]:
        print(f"\n{split_name} Set:")
        print("-" * 40)
        
        # Get predictions
        probs = model.predict_proba(split_df)
        
        # Manual evaluation
        metrics = evaluate_sample_level_manual(split_df, probs, model.optimal_threshold)
        
        print(f"  Samples: {metrics['total_samples']}")
        print(f"  Positive samples: {metrics['positive_samples']}")
        print(f"  Predicted positive: {metrics['predicted_positive']}")
        print(f"  TP: {metrics['TP']}, FP: {metrics['FP']}, FN: {metrics['FN']}, TN: {metrics['TN']}")
        print(f"  auPRC: {metrics['auPRC']:.4f}")
        print(f"  auROC: {metrics['AUC']:.4f}")
        print(f"  Precision: {metrics['Precision']:.4f}")
        print(f"  Recall: {metrics['Recall']:.4f}")
        print(f"  F1: {metrics['F1']:.4f}")
        print(f"  Accuracy: {metrics['Accuracy']:.4f}")
        
        # Verify logic
        print(f"\n  Verification:")
        print(f"    TP + FN = {metrics['TP'] + metrics['FN']} (should = {metrics['positive_samples']})")
        print(f"    TP + FP = {metrics['TP'] + metrics['FP']} (should = {metrics['predicted_positive']})")
    
    print("\n" + "="*60)
    print("Verification Complete")
    print("="*60)

if __name__ == "__main__":
    main()
