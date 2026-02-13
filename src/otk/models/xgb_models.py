#!/usr/bin/env python
"""
XGBoost Models for ecDNA Prediction

Unified implementation using BaseEcDNAModel interface.
"""

import numpy as np
import pandas as pd
import xgboost as xgb
from typing import Dict, Any, Optional, Union
from pathlib import Path
import pickle
import logging

from .base_model import BaseEcDNAModel

logger = logging.getLogger(__name__)


class XGB11Model(BaseEcDNAModel):
    """
    XGB11 Model - Exact reproduction of Nature Communications 2024 paper
    Uses 11 features with paper hyperparameters
    """
    
    FEATURES = [
        'total_cn', 'minor_cn', 'purity', 'ploidy', 'AScore',
        'pLOH', 'cna_burden', 'freq_Linear', 'freq_BFB', 'freq_Circular', 'freq_HR'
    ]
    
    PAPER_PARAMS = {
        'eta': 0.1, 'max_depth': 4, 'gamma': 10, 'subsample': 0.6,
        'max_delta_step': 0, 'min_child_weight': 1, 'alpha': 0, 'lambda': 1,
        'objective': 'binary:logistic', 'eval_metric': 'aucpr',
        'booster': 'gbtree', 'tree_method': 'hist', 'device': 'cuda'
    }
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.params = self.PAPER_PARAMS.copy()
        self.model = None
        
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare 11 features"""
        feature_df = pd.DataFrame()
        mapping = {
            'total_cn': 'segVal', 'minor_cn': 'minor_cn', 'purity': 'purity',
            'ploidy': 'ploidy', 'AScore': 'AScore', 'pLOH': 'pLOH',
            'cna_burden': 'cna_burden', 'freq_Linear': 'freq_Linear',
            'freq_BFB': 'freq_BFB', 'freq_Circular': 'freq_Circular', 'freq_HR': 'freq_HR'
        }
        for model_feat, data_feat in mapping.items():
            feature_df[model_feat] = df[data_feat].fillna(0) if data_feat in df.columns else 0
        return feature_df
    
    def fit(self, X_train, y_train, X_val=None, y_val=None, **kwargs):
        X_train_prepared = self.prepare_features(X_train)
        dtrain = xgb.DMatrix(X_train_prepared, label=y_train)
        evals = [(dtrain, 'train')]
        
        if X_val is not None and y_val is not None:
            X_val_prepared = self.prepare_features(X_val)
            dval = xgb.DMatrix(X_val_prepared, label=y_val)
            evals.append((dval, 'eval'))
        
        self.model = xgb.train(
            self.params, dtrain, num_boost_round=200,
            evals=evals, early_stopping_rounds=50, verbose_eval=False
        )
        
        if X_val is not None and y_val is not None:
            val_probs = self.predict_proba(X_val)
            self.optimal_threshold = self._find_optimal_threshold(y_val, val_probs)
        
        self.is_fitted = True
        return self
    
    def _find_optimal_threshold(self, y_true, y_prob):
        from sklearn.metrics import precision_recall_curve
        precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
        optimal_idx = np.argmax(f1_scores)
        return thresholds[optimal_idx] if optimal_idx < len(thresholds) else 0.5
    
    def predict_proba(self, X):
        if not self.is_fitted:
            raise ValueError("Model not fitted")
        X_prepared = self.prepare_features(X)
        return self.model.predict(xgb.DMatrix(X_prepared))
    
    def save(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump({'model': self.model, 'params': self.params, 
                        'optimal_threshold': self.optimal_threshold}, f)
    
    def load(self, path):
        with open(path, 'rb') as f:
            data = pickle.load(f)
        self.model = data['model']
        self.params = data['params']
        self.optimal_threshold = data['optimal_threshold']
        self.is_fitted = True
        return self
    
    def get_feature_importance(self):
        if not self.is_fitted:
            return None
        importance = self.model.get_score(importance_type='gain')
        return pd.DataFrame([{'feature': k, 'importance': v} 
                           for k, v in importance.items()]).sort_values('importance', ascending=False)


class XGBNewModel(BaseEcDNAModel):
    """
    XGB New - Optimized with all features and engineering
    """
    
    DEFAULT_PARAMS = {
        'eta': 0.05, 'max_depth': 6, 'gamma': 3, 'subsample': 0.8,
        'max_delta_step': 1, 'min_child_weight': 2, 'alpha': 0.1, 'lambda': 2,
        'objective': 'binary:logistic', 'eval_metric': 'aucpr',
        'booster': 'gbtree', 'tree_method': 'hist', 'device': 'cuda'
    }
    
    def __init__(self, config=None):
        super().__init__(config)
        self.params = self.DEFAULT_PARAMS.copy()
        self.model = None
    
    def prepare_features(self, df):
        """Prepare all features with engineering"""
        feature_df = pd.DataFrame()
        
        # Core features
        core = ['segVal', 'minor_cn', 'purity', 'ploidy', 'pLOH', 'AScore', 'cna_burden']
        for f in core:
            feature_df[f] = df[f].fillna(0) if f in df.columns else 0
        
        # Frequency features
        for f in ['freq_Linear', 'freq_BFB', 'freq_Circular', 'freq_HR']:
            feature_df[f] = df[f].fillna(0) if f in df.columns else 0
        
        # CN signatures
        for i in range(1, 20):
            f = f'CN{i}'
            feature_df[f] = df[f].fillna(0) if f in df.columns else 0
        
        # Clinical
        if 'age' in df.columns:
            feature_df['age'] = df['age'].fillna(df['age'].mean())
        if 'gender' in df.columns:
            feature_df['gender'] = df['gender'].fillna(0)
        
        # Cancer types
        for c in [col for col in df.columns if col.startswith('type_')]:
            feature_df[c] = df[c].fillna(0)
        
        # Engineering
        if 'segVal' in df.columns and 'ploidy' in df.columns:
            feature_df['cn_imbalance'] = df['segVal'] / (df['ploidy'] + 1e-6)
        if 'minor_cn' in df.columns and 'segVal' in df.columns:
            feature_df['allele_imbalance'] = df['minor_cn'] / (df['segVal'] + 1e-6)
        if 'cna_burden' in df.columns and 'purity' in df.columns:
            feature_df['cna_burden_adj'] = df['cna_burden'] * df['purity']
        if 'AScore' in df.columns and 'purity' in df.columns:
            feature_df['ascore_adj'] = df['AScore'] * df['purity']
        for f in ['freq_Circular', 'freq_BFB', 'freq_HR']:
            if f in df.columns:
                feature_df[f'has_{f.split("_")[1].lower()}'] = (df[f] > 0).astype(int)
        
        freq_cols = ['freq_Linear', 'freq_BFB', 'freq_Circular', 'freq_HR']
        if all(c in df.columns for c in freq_cols):
            feature_df['amplicon_type_count'] = (df[freq_cols] > 0).sum(axis=1)
        
        cn_cols = [f'CN{i}' for i in range(1, 20)]
        if any(c in df.columns for c in cn_cols):
            feature_df['cn_sig_diversity'] = (df[[c for c in cn_cols if c in df.columns]] > 0).sum(axis=1)
            feature_df['max_cn_sig'] = df[[c for c in cn_cols if c in df.columns]].max(axis=1)
        
        if 'purity' in df.columns and 'ploidy' in df.columns:
            feature_df['purity_x_ploidy'] = df['purity'] * df['ploidy']
        if 'pLOH' in df.columns:
            feature_df['has_loh'] = (df['pLOH'] > 0.1).astype(int)
        
        return feature_df
    
    def fit(self, X_train, y_train, X_val=None, y_val=None, **kwargs):
        X_train_prepared = self.prepare_features(X_train)
        dtrain = xgb.DMatrix(X_train_prepared, label=y_train)
        evals = [(dtrain, 'train')]
        
        if X_val is not None and y_val is not None:
            X_val_prepared = self.prepare_features(X_val)
            dval = xgb.DMatrix(X_val_prepared, label=y_val)
            evals.append((dval, 'eval'))
        
        self.model = xgb.train(
            self.params, dtrain, num_boost_round=10000,
            evals=evals, early_stopping_rounds=50, verbose_eval=False
        )
        
        if X_val is not None and y_val is not None:
            val_probs = self.predict_proba(X_val)
            self.optimal_threshold = self._find_optimal_threshold(y_val, val_probs)
        
        self.is_fitted = True
        return self
    
    def _find_optimal_threshold(self, y_true, y_prob):
        from sklearn.metrics import precision_recall_curve
        precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
        optimal_idx = np.argmax(f1_scores)
        return thresholds[optimal_idx] if optimal_idx < len(thresholds) else 0.5
    
    def predict_proba(self, X):
        if not self.is_fitted:
            raise ValueError("Model not fitted")
        X_prepared = self.prepare_features(X)
        return self.model.predict(xgb.DMatrix(X_prepared))
    
    def save(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump({'model': self.model, 'params': self.params,
                        'optimal_threshold': self.optimal_threshold}, f)
    
    def load(self, path):
        with open(path, 'rb') as f:
            data = pickle.load(f)
        self.model = data['model']
        self.params = data['params']
        self.optimal_threshold = data['optimal_threshold']
        self.is_fitted = True
        return self
    
    def get_feature_importance(self):
        if not self.is_fitted:
            return None
        importance = self.model.get_score(importance_type='gain')
        return pd.DataFrame([{'feature': k, 'importance': v}
                           for k, v in importance.items()]).sort_values('importance', ascending=False)
