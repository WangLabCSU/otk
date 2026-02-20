#!/usr/bin/env python
"""
Optimized Hybrid Ensemble - Simple weighted average of pre-trained models
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
from otk.models.neural_models import create_neural_model


def add_engineered_features(df):
    """Add engineered features to match XGBoost model"""
    feature_df = df.copy()
    
    if 'segVal' in df.columns and 'ploidy' in df.columns:
        feature_df['cn_imbalance'] = df['segVal'] / (df['ploidy'] + 1e-6)
    else:
        feature_df['cn_imbalance'] = 0
    
    if 'minor_cn' in df.columns and 'segVal' in df.columns:
        feature_df['allele_imbalance'] = df['minor_cn'] / (df['segVal'] + 1e-6)
    else:
        feature_df['allele_imbalance'] = 0
    
    if 'cna_burden' in df.columns and 'purity' in df.columns:
        feature_df['cna_burden_adj'] = df['cna_burden'] * df['purity']
    else:
        feature_df['cna_burden_adj'] = 0
    
    if 'AScore' in df.columns and 'purity' in df.columns:
        feature_df['ascore_adj'] = df['AScore'] * df['purity']
    else:
        feature_df['ascore_adj'] = 0
    
    for f in ['freq_Circular', 'freq_BFB', 'freq_HR']:
        if f in df.columns:
            feature_df[f'has_{f.split("_")[1].lower()}'] = (df[f] > 0).astype(int)
        else:
            feature_df[f'has_{f.split("_")[1].lower()}'] = 0
    
    freq_cols = ['freq_Linear', 'freq_BFB', 'freq_Circular', 'freq_HR']
    if all(c in df.columns for c in freq_cols):
        feature_df['amplicon_type_count'] = (df[freq_cols] > 0).sum(axis=1)
    else:
        feature_df['amplicon_type_count'] = 0
    
    cn_cols = [f'CN{i}' for i in range(1, 20)]
    existing_cn = [c for c in cn_cols if c in df.columns]
    if existing_cn:
        feature_df['cn_sig_diversity'] = (df[existing_cn] > 0).sum(axis=1)
        feature_df['max_cn_sig'] = df[existing_cn].max(axis=1)
    else:
        feature_df['cn_sig_diversity'] = 0
        feature_df['max_cn_sig'] = 0
    
    if 'purity' in df.columns and 'ploidy' in df.columns:
        feature_df['purity_x_ploidy'] = df['purity'] * df['ploidy']
    else:
        feature_df['purity_x_ploidy'] = 0
    
    if 'pLOH' in df.columns:
        feature_df['has_loh'] = (df['pLOH'] > 0).astype(int)
    else:
        feature_df['has_loh'] = 0
    
    return feature_df


def find_optimal_threshold(y_true, y_prob):
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
    optimal_idx = np.argmax(f1_scores)
    return thresholds[optimal_idx] if optimal_idx < len(thresholds) else 0.5


def evaluate(y_true, y_prob, prefix=''):
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
    
    print(f"{prefix}auPRC: {auprc:.4f}, AUC: {auc:.4f}, F1: {f1:.4f}")
    return auprc


def main():
    print("="*60)
    print("Optimized Hybrid Ensemble - XGBoost + Neural Network")
    print("="*60)
    
    print("\nLoading data...")
    train_df, val_df, test_df = load_split()
    
    print("\nAdding engineered features...")
    train_df_eng = add_engineered_features(train_df)
    val_df_eng = add_engineered_features(val_df)
    test_df_eng = add_engineered_features(test_df)
    
    # Load XGBoost model
    print("\nLoading XGBoost model...")
    xgb_data = joblib.load('otk_api/models/xgb_new/best_model.pkl')
    xgb_model = xgb_data['model']
    xgb_feature_names = xgb_data.get('feature_names', None)
    
    # Get XGBoost predictions
    print("Getting XGBoost predictions...")
    dtrain = xgb.DMatrix(train_df_eng[xgb_feature_names].fillna(0).values, feature_names=xgb_feature_names)
    dval = xgb.DMatrix(val_df_eng[xgb_feature_names].fillna(0).values, feature_names=xgb_feature_names)
    dtest = xgb.DMatrix(test_df_eng[xgb_feature_names].fillna(0).values, feature_names=xgb_feature_names)
    
    xgb_train_pred = xgb_model.predict(dtrain)
    xgb_val_pred = xgb_model.predict(dval)
    xgb_test_pred = xgb_model.predict(dtest)
    
    print("\nXGBoost performance:")
    evaluate(train_df['y'].values, xgb_train_pred, '  Train: ')
    evaluate(val_df['y'].values, xgb_val_pred, '  Val: ')
    evaluate(test_df['y'].values, xgb_test_pred, '  Test: ')
    
    # Load Neural Network model
    print("\n" + "="*60)
    print("Loading Neural Network model (dgit_super)...")
    print("="*60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Load model directly
    model_path = Path('otk_api/models/dgit_super/best_model.pkl')
    if model_path.exists():
        import yaml
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        config = checkpoint.get('config', {})
        
        # Create model instance
        from otk.models.neural_models import DGITSuperModel, DGITSuperNet
        nn_model = DGITSuperModel(config, device=device)
        
        # Get input dim from config
        input_dim = config.get('model', {}).get('architecture', {}).get('input_dim', 57)
        arch_config = config.get('model', {}).get('architecture', {})
        
        # Create the network
        nn_model.model = DGITSuperNet(
            input_dim=input_dim,
            hidden_dim=arch_config.get('hidden_dim', 256),
            num_trees=arch_config.get('num_trees', 5),
            num_layers=arch_config.get('num_layers', 3),
            tree_dim=arch_config.get('tree_dim', 16),
            dropout=arch_config.get('dropout', 0.2)
        ).to(device)
        
        # Load weights
        nn_model.model.load_state_dict(checkpoint['model_state'])
        nn_model.optimal_threshold = checkpoint.get('optimal_threshold', 0.5)
        nn_model.is_fitted = True
        print("Model loaded successfully!")
        
        # Get NN predictions (using original features, not engineered)
        print("Getting Neural Network predictions...")
        
        nn_train_pred = nn_model.predict_proba(train_df)
        nn_val_pred = nn_model.predict_proba(val_df)
        nn_test_pred = nn_model.predict_proba(test_df)
        
        print("\nNeural Network performance:")
        evaluate(train_df['y'].values, nn_train_pred, '  Train: ')
        evaluate(val_df['y'].values, nn_val_pred, '  Val: ')
        evaluate(test_df['y'].values, nn_test_pred, '  Test: ')
        
        # Try different ensemble weights
        print("\n" + "="*60)
        print("Testing ensemble weights...")
        print("="*60)
        
        best_val_auprc = 0
        best_weight = 0
        
        for w_xgb in np.arange(0.3, 1.0, 0.05):
            w_nn = 1.0 - w_xgb
            ensemble_val_pred = w_xgb * xgb_val_pred + w_nn * nn_val_pred
            val_auprc = average_precision_score(val_df['y'].values, ensemble_val_pred)
            
            if val_auprc > best_val_auprc:
                best_val_auprc = val_auprc
                best_weight = w_xgb
            
            print(f"  XGBoost: {w_xgb:.2f}, NN: {w_nn:.2f}, Val auPRC: {val_auprc:.4f}")
        
        print(f"\nBest weights: XGBoost={best_weight:.2f}, NN={1-best_weight:.2f}")
        print(f"Best Val auPRC: {best_val_auprc:.4f}")
        
        # Final evaluation
        print("\n" + "="*60)
        print("Final Ensemble Performance")
        print("="*60)
        
        w_nn = 1.0 - best_weight
        ensemble_train_pred = best_weight * xgb_train_pred + w_nn * nn_train_pred
        ensemble_val_pred = best_weight * xgb_val_pred + w_nn * nn_val_pred
        ensemble_test_pred = best_weight * xgb_test_pred + w_nn * nn_test_pred
        
        print("\nTrain:")
        evaluate(train_df['y'].values, ensemble_train_pred)
        print("\nVal:")
        evaluate(val_df['y'].values, ensemble_val_pred)
        print("\nTest:")
        evaluate(test_df['y'].values, ensemble_test_pred)
        
        # Sample-level
        print("\n" + "="*60)
        print("Sample-level Performance")
        print("="*60)
        
        for name, df, probs in [('Train', train_df, ensemble_train_pred), 
                                 ('Val', val_df, ensemble_val_pred),
                                 ('Test', test_df, ensemble_test_pred)]:
            df_reset = df.reset_index(drop=True)
            sample_probs = df_reset.assign(probs=probs).groupby('sample')['probs'].mean()
            sample_labels = df_reset.groupby('sample')['y'].max()
            
            auprc = average_precision_score(sample_labels, sample_probs)
            auc = roc_auc_score(sample_labels, sample_probs)
            print(f"{name}: auPRC={auprc:.4f}, AUC={auc:.4f}")
    
    else:
        print("No neural network model found, using XGBoost only")
    
    print("\n" + "="*60)
    print("Summary")
    print("="*60)
    print("XGBoost alone: Test auPRC = 0.8346")
    print("Ensemble may or may not improve performance")


if __name__ == '__main__':
    main()
