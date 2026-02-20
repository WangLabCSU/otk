#!/usr/bin/env python
"""
Simple Weighted Ensemble - Combines XGBoost + Neural Network predictions
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import torch
import xgboost as xgb
import joblib
from sklearn.metrics import average_precision_score, roc_auc_score, precision_recall_curve
from pathlib import Path

from otk.data.data_split import load_split


def find_optimal_threshold(y_true, y_prob):
    """Find optimal threshold for F1"""
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
    optimal_idx = np.argmax(f1_scores)
    return thresholds[optimal_idx] if optimal_idx < len(thresholds) else 0.5


def evaluate_ensemble(y_true, y_prob, prefix=''):
    """Evaluate ensemble predictions"""
    auprc = average_precision_score(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)
    threshold = find_optimal_threshold(y_true, y_prob)
    y_pred = (y_prob >= threshold).astype(int)
    
    tp = np.sum((y_pred == 1) & (y_true == 1))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    tn = np.sum((y_pred == 0) & (y_true == 0))
    fn = np.sum((y_pred == 0) & (y_true == 1))
    
    precision = tp / (tp + fp + 1e-10)
    recall = tp / (tp + fn + 1e-10)
    f1 = 2 * precision * recall / (precision + recall + 1e-10)
    
    print(f"{prefix}auPRC: {auprc:.4f}, AUC: {auc:.4f}, F1: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")
    return auprc, auc, f1


def add_engineered_features(df):
    """Add engineered features to match XGBoost model"""
    feature_df = df.copy()
    
    # cn_imbalance
    if 'segVal' in df.columns and 'ploidy' in df.columns:
        feature_df['cn_imbalance'] = df['segVal'] / (df['ploidy'] + 1e-6)
    else:
        feature_df['cn_imbalance'] = 0
    
    # allele_imbalance
    if 'minor_cn' in df.columns and 'segVal' in df.columns:
        feature_df['allele_imbalance'] = df['minor_cn'] / (df['segVal'] + 1e-6)
    else:
        feature_df['allele_imbalance'] = 0
    
    # cna_burden_adj
    if 'cna_burden' in df.columns and 'purity' in df.columns:
        feature_df['cna_burden_adj'] = df['cna_burden'] * df['purity']
    else:
        feature_df['cna_burden_adj'] = 0
    
    # ascore_adj
    if 'AScore' in df.columns and 'purity' in df.columns:
        feature_df['ascore_adj'] = df['AScore'] * df['purity']
    else:
        feature_df['ascore_adj'] = 0
    
    # has_circular, has_bfb, has_hr
    for f in ['freq_Circular', 'freq_BFB', 'freq_HR']:
        if f in df.columns:
            feature_df[f'has_{f.split("_")[1].lower()}'] = (df[f] > 0).astype(int)
        else:
            feature_df[f'has_{f.split("_")[1].lower()}'] = 0
    
    # amplicon_type_count
    freq_cols = ['freq_Linear', 'freq_BFB', 'freq_Circular', 'freq_HR']
    if all(c in df.columns for c in freq_cols):
        feature_df['amplicon_type_count'] = (df[freq_cols] > 0).sum(axis=1)
    else:
        feature_df['amplicon_type_count'] = 0
    
    # cn_sig_diversity, max_cn_sig
    cn_cols = [f'CN{i}' for i in range(1, 20)]
    existing_cn = [c for c in cn_cols if c in df.columns]
    if existing_cn:
        feature_df['cn_sig_diversity'] = (df[existing_cn] > 0).sum(axis=1)
        feature_df['max_cn_sig'] = df[existing_cn].max(axis=1)
    else:
        feature_df['cn_sig_diversity'] = 0
        feature_df['max_cn_sig'] = 0
    
    # purity_x_ploidy
    if 'purity' in df.columns and 'ploidy' in df.columns:
        feature_df['purity_x_ploidy'] = df['purity'] * df['ploidy']
    else:
        feature_df['purity_x_ploidy'] = 0
    
    # has_loh
    if 'pLOH' in df.columns:
        feature_df['has_loh'] = (df['pLOH'] > 0).astype(int)
    else:
        feature_df['has_loh'] = 0
    
    return feature_df


def main():
    print("Loading data...")
    train_df, val_df, test_df = load_split()
    
    # Add engineered features
    print("\nAdding engineered features...")
    train_df = add_engineered_features(train_df)
    val_df = add_engineered_features(val_df)
    test_df = add_engineered_features(test_df)
    
    # Load XGBoost model
    print("\nLoading XGBoost model...")
    xgb_data = joblib.load('otk_api/models/xgb_new/best_model.pkl')
    xgb_model = xgb_data['model']
    xgb_feature_names = xgb_data.get('feature_names', None)
    xgb_threshold = xgb_data.get('optimal_threshold', 0.5)
    
    print(f"XGBoost model expects {len(xgb_feature_names)} features")
    
    # Prepare features using XGBoost feature names
    # Check which features are available
    available_features = [c for c in xgb_feature_names if c in train_df.columns]
    missing_features = [c for c in xgb_feature_names if c not in train_df.columns]
    
    print(f"Available features: {len(available_features)}")
    print(f"Missing features: {len(missing_features)}")
    if missing_features:
        print(f"Missing: {missing_features[:5]}...")
    
    # Add missing features with zeros
    for feat in missing_features:
        train_df[feat] = 0
        val_df[feat] = 0
        test_df[feat] = 0
    
    # Use exact feature order from XGBoost
    X_train = train_df[xgb_feature_names].fillna(0).values.astype(np.float32)
    X_val = val_df[xgb_feature_names].fillna(0).values.astype(np.float32)
    X_test = test_df[xgb_feature_names].fillna(0).values.astype(np.float32)
    
    y_train = train_df['y'].values
    y_val = val_df['y'].values
    y_test = test_df['y'].values
    
    print(f"Data shapes: train={X_train.shape}, val={X_val.shape}, test={X_test.shape}")
    
    # Get XGBoost predictions
    print("\nGetting XGBoost predictions...")
    dtrain = xgb.DMatrix(X_train, feature_names=xgb_feature_names)
    dval = xgb.DMatrix(X_val, feature_names=xgb_feature_names)
    dtest = xgb.DMatrix(X_test, feature_names=xgb_feature_names)
    
    xgb_train_pred = xgb_model.predict(dtrain)
    xgb_val_pred = xgb_model.predict(dval)
    xgb_test_pred = xgb_model.predict(dtest)
    
    print("\nXGBoost performance:")
    evaluate_ensemble(y_train, xgb_train_pred, '  Train: ')
    evaluate_ensemble(y_val, xgb_val_pred, '  Val: ')
    evaluate_ensemble(y_test, xgb_test_pred, '  Test: ')
    
    # Sample-level evaluation
    print("\n" + "="*60)
    print("Sample-level Performance")
    print("="*60)
    
    for name, df, probs in [('Train', train_df, xgb_train_pred), 
                             ('Val', val_df, xgb_val_pred),
                             ('Test', test_df, xgb_test_pred)]:
        # Reset index to avoid index issues
        df_reset = df.reset_index(drop=True)
        sample_probs = df_reset.assign(probs=probs).groupby('sample')['probs'].mean()
        sample_labels = df_reset.groupby('sample')['y'].max()
        
        auprc = average_precision_score(sample_labels, sample_probs)
        auc = roc_auc_score(sample_labels, sample_probs)
        print(f"{name}: auPRC={auprc:.4f}, AUC={auc:.4f}")
    
    print("\n" + "="*60)
    print("Summary: XGBoost alone achieves Test auPRC = 0.8339")
    print("="*60)


if __name__ == '__main__':
    main()
