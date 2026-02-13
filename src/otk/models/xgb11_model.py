#!/usr/bin/env python
"""
XGB11 Model - XGBoost-based ecDNA Prediction Model

Based on Nature Communications 2024 paper:
"Machine learning-based extrachromosomal DNA identification in large-scale cohorts"

Original R model features (exactly 11 features):
1. total_cn (segVal in our data)
2. minor_cn
3. purity
4. ploidy
5. AScore
6. pLOH
7. cna_burden
8. freq_Linear
9. freq_BFB
10. freq_Circular
11. freq_HR

Original hyperparameters from R model:
- eta: 0.3
- max_depth: 3
- gamma: 1
- subsample: 0.5
- max_delta_step: 1
- min_child_weight: 1
- objective: binary:logistic
- eval_metric: aucpr

Target Performance:
- Gene-level auPRC: 0.85+
- Gene-level Precision: 0.8+
- Sample-level auPRC: 0.99+
- Sample-level auROC: 0.9+
"""

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import precision_recall_curve, auc, roc_auc_score, accuracy_score
from typing import Dict, Any, Tuple, Optional
import logging
import pickle
from pathlib import Path

logger = logging.getLogger(__name__)


class XGB11Model:
    """
    XGB11 Model - Exact reproduction of Nature Communications 2024 paper
    Uses exactly 11 features with original hyperparameters
    """
    
    # Exact 11 features from the original R model
    FEATURES = [
        'total_cn',      # Maps to segVal
        'minor_cn',      # minor_cn
        'purity',        # purity
        'ploidy',        # ploidy
        'AScore',        # AScore
        'pLOH',          # pLOH
        'cna_burden',    # cna_burden
        'freq_Linear',   # freq_Linear
        'freq_BFB',      # freq_BFB
        'freq_Circular', # freq_Circular
        'freq_HR',       # freq_HR
    ]
    
    # Original hyperparameters from R model
    ORIGINAL_PARAMS = {
        'eta': 0.3,
        'max_depth': 3,
        'gamma': 1,
        'subsample': 0.5,
        'max_delta_step': 1,
        'min_child_weight': 1,
        'objective': 'binary:logistic',
        'eval_metric': 'aucpr',
        'booster': 'gbtree',
        'tree_method': 'hist',
        'device': 'cuda',
    }
    
    def __init__(self, params: Optional[Dict[str, Any]] = None, use_original_params: bool = True):
        """
        Initialize XGB11 model
        
        Args:
            params: Optional custom parameters
            use_original_params: If True, use exact params from paper; if False, use provided params
        """
        if use_original_params or params is None:
            self.params = self.ORIGINAL_PARAMS.copy()
        else:
            self.params = params
        
        self.model = None
        self.optimal_threshold = 0.5
        
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare features - exact 11 features, no additional engineering
        
        Args:
            df: Input dataframe with raw features
            
        Returns:
            DataFrame with exactly 11 features
        """
        feature_df = pd.DataFrame()
        
        # Map features from data to model features
        feature_mapping = {
            'total_cn': 'segVal',  # total_cn maps to segVal
            'minor_cn': 'minor_cn',
            'purity': 'purity',
            'ploidy': 'ploidy',
            'AScore': 'AScore',
            'pLOH': 'pLOH',
            'cna_burden': 'cna_burden',
            'freq_Linear': 'freq_Linear',
            'freq_BFB': 'freq_BFB',
            'freq_Circular': 'freq_Circular',
            'freq_HR': 'freq_HR',
        }
        
        for model_feat, data_feat in feature_mapping.items():
            if data_feat in df.columns:
                feature_df[model_feat] = df[data_feat].fillna(0)
            else:
                logger.warning(f"Feature {data_feat} not found in data, using zeros")
                feature_df[model_feat] = 0
        
        return feature_df
    
    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
        sample_weight: Optional[np.ndarray] = None,
        early_stopping_rounds: int = 50,
        verbose: bool = True
    ) -> 'XGB11Model':
        """Train the XGB11 model"""
        logger.info("Preparing features...")
        X_train_prepared = self.prepare_features(X_train)
        
        # Create DMatrix
        dtrain = xgb.DMatrix(
            X_train_prepared,
            label=y_train,
            weight=sample_weight
        )
        
        evals = [(dtrain, 'train')]
        
        if X_val is not None and y_val is not None:
            X_val_prepared = self.prepare_features(X_val)
            dval = xgb.DMatrix(X_val_prepared, label=y_val)
            evals.append((dval, 'eval'))
        
        logger.info(f"Training XGB11 model with params: {self.params}")
        
        self.model = xgb.train(
            self.params,
            dtrain,
            num_boost_round=200,
            evals=evals,
            early_stopping_rounds=early_stopping_rounds,
            verbose_eval=verbose
        )
        
        # Find optimal threshold on validation set
        if X_val is not None and y_val is not None:
            val_probs = self.predict_proba(X_val)
            self.optimal_threshold = self._find_optimal_threshold(y_val, val_probs)
            logger.info(f"Optimal threshold: {self.optimal_threshold:.4f}")
        
        return self
    
    def _find_optimal_threshold(self, y_true: np.ndarray, y_prob: np.ndarray) -> float:
        """Find optimal threshold using F1 score"""
        precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
        optimal_idx = np.argmax(f1_scores)
        
        if optimal_idx < len(thresholds):
            return thresholds[optimal_idx]
        return 0.5
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict probabilities"""
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")
        
        X_prepared = self.prepare_features(X)
        dtest = xgb.DMatrix(X_prepared)
        return self.model.predict(dtest)
    
    def predict(self, X: pd.DataFrame, threshold: Optional[float] = None) -> np.ndarray:
        """Predict labels"""
        probs = self.predict_proba(X)
        threshold = threshold if threshold is not None else self.optimal_threshold
        return (probs >= threshold).astype(int)
    
    def evaluate(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        threshold: Optional[float] = None
    ) -> Dict[str, float]:
        """Evaluate model performance - gene level"""
        probs = self.predict_proba(X)
        preds = self.predict(X, threshold)
        
        precision, recall, _ = precision_recall_curve(y, probs)
        auprc = auc(recall, precision)
        auc_score = roc_auc_score(y, probs)
        
        tp = ((preds == 1) & (y == 1)).sum()
        fp = ((preds == 1) & (y == 0)).sum()
        fn = ((preds == 0) & (y == 1)).sum()
        
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        
        return {
            'auPRC': auprc,
            'AUC': auc_score,
            'Precision': prec,
            'Recall': rec,
            'F1': f1,
            'threshold': threshold if threshold else self.optimal_threshold
        }
    
    def evaluate_sample_level(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        samples: pd.Series,
        threshold: Optional[float] = None
    ) -> Dict[str, float]:
        """Evaluate at sample level"""
        probs = self.predict_proba(X)
        
        # Aggregate to sample level
        results_df = pd.DataFrame({
            'sample': samples,
            'y': y,
            'prob': probs
        })
        
        sample_agg = results_df.groupby('sample').agg({
            'y': 'max',
            'prob': 'max'
        }).reset_index()
        
        y_true = sample_agg['y'].values
        y_prob = sample_agg['prob'].values
        y_pred = (y_prob >= (threshold if threshold else self.optimal_threshold)).astype(int)
        
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        auprc = auc(recall, precision)
        auc_score = roc_auc_score(y_true, y_prob)
        
        tp = ((y_pred == 1) & (y_true == 1)).sum()
        fp = ((y_pred == 1) & (y_true == 0)).sum()
        fn = ((y_pred == 0) & (y_true == 1)).sum()
        tn = ((y_pred == 0) & (y_true == 0)).sum()
        
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
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
            'predicted_positive': int(y_pred.sum())
        }
    
    def save(self, path: str):
        """Save model to file"""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        
        save_dict = {
            'model': self.model,
            'params': self.params,
            'optimal_threshold': self.optimal_threshold,
            'features': self.FEATURES
        }
        
        with open(path, 'wb') as f:
            pickle.dump(save_dict, f)
        
        logger.info(f"Model saved to {path}")
    
    def load(self, path: str) -> 'XGB11Model':
        """Load model from file"""
        with open(path, 'rb') as f:
            save_dict = pickle.load(f)
        
        self.model = save_dict['model']
        self.params = save_dict['params']
        self.optimal_threshold = save_dict['optimal_threshold']
        
        logger.info(f"Model loaded from {path}")
        return self
    
    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance"""
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")
        
        importance = self.model.get_score(importance_type='gain')
        
        importance_df = pd.DataFrame([
            {'feature': k, 'importance': v}
            for k, v in importance.items()
        ])
        importance_df = importance_df.sort_values('importance', ascending=False)
        
        return importance_df


class XGB11OptimizedModel(XGB11Model):
    """
    XGB11 Optimized - Enhanced version with feature engineering
    Based on XGB11 but with additional engineered features
    """
    
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare features with additional engineering
        """
        # Start with original 11 features
        feature_df = super().prepare_features(df)
        
        # Add engineered features
        # 1. CN relative to ploidy (important ratio)
        feature_df['cn_ploidy_ratio'] = feature_df['total_cn'] / (feature_df['ploidy'] + 1e-6)
        
        # 2. CNA burden normalized by purity
        feature_df['cna_burden_purity'] = feature_df['cna_burden'] * feature_df['purity']
        
        # 3. Minor CN ratio
        feature_df['minor_cn_ratio'] = feature_df['minor_cn'] / (feature_df['total_cn'] + 1e-6)
        
        # 4. Total amplification frequency (sum of all freq types)
        feature_df['total_freq'] = (
            feature_df['freq_Linear'] + 
            feature_df['freq_BFB'] + 
            feature_df['freq_Circular'] + 
            feature_df['freq_HR']
        )
        
        # 5. Circular dominance ratio
        feature_df['circular_dominance'] = feature_df['freq_Circular'] / (feature_df['total_freq'] + 1e-6)
        
        return feature_df


class XGB11Trainer:
    """Trainer for XGB11 model"""
    
    def __init__(
        self,
        data_path: str,
        output_dir: str,
        model_type: str = 'original',  # 'original' or 'optimized'
        validation_split: float = 0.12,
        test_split: float = 0.18,
        random_state: int = 42
    ):
        self.data_path = Path(data_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.model_type = model_type
        self.validation_split = validation_split
        self.test_split = test_split
        self.random_state = random_state
        
        # Choose model type
        if model_type == 'original':
            self.model = XGB11Model(use_original_params=True)
        else:
            self.model = XGB11OptimizedModel(use_original_params=True)
        
    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Load and split data"""
        import gzip
        
        logger.info(f"Loading data from {self.data_path}")
        
        with gzip.open(self.data_path, 'rt') as f:
            df = pd.read_csv(f)
        
        logger.info(f"Loaded {len(df)} rows")
        
        # Split by sample using stratified sampling
        unique_samples = df['sample'].unique()
        
        # Stratified split based on positive count per sample
        sample_pos_counts = df.groupby('sample')['y'].sum().sort_values()
        sorted_samples = sample_pos_counts.index.tolist()
        
        n_samples = len(sorted_samples)
        test_size = int(n_samples * self.test_split)
        val_size = int(n_samples * self.validation_split)
        train_size = n_samples - val_size - test_size
        
        # Equidistant sampling for balanced distribution
        train_indices = np.linspace(0, n_samples-1, train_size, dtype=int)
        remaining = list(set(range(n_samples)) - set(train_indices))
        remaining.sort()
        
        val_indices = np.linspace(0, len(remaining)-1, val_size, dtype=int)
        val_indices = [remaining[i] for i in val_indices]
        test_indices = list(set(remaining) - set(val_indices))
        
        train_samples = [sorted_samples[i] for i in train_indices]
        val_samples = [sorted_samples[i] for i in val_indices]
        test_samples = [sorted_samples[i] for i in test_indices]
        
        train_df = df[df['sample'].isin(train_samples)]
        val_df = df[df['sample'].isin(val_samples)]
        test_df = df[df['sample'].isin(test_samples)]
        
        logger.info(f"Train: {len(train_df)} rows, {train_df['y'].sum()} positive")
        logger.info(f"Val: {len(val_df)} rows, {val_df['y'].sum()} positive")
        logger.info(f"Test: {len(test_df)} rows, {test_df['y'].sum()} positive")
        
        return train_df, val_df, test_df
    
    def train(self):
        """Train XGB11 model"""
        train_df, val_df, test_df = self.load_data()
        
        # Prepare features and labels
        X_train = train_df
        y_train = train_df['y']
        X_val = val_df
        y_val = val_df['y']
        X_test = test_df
        y_test = test_df['y']
        
        # Calculate sample weights for imbalanced data
        pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
        sample_weight = np.where(y_train == 1, pos_weight, 1.0)
        
        logger.info(f"Positive weight: {pos_weight:.2f}")
        
        # Train model
        self.model.fit(
            X_train, y_train,
            X_val=X_val, y_val=y_val,
            sample_weight=sample_weight,
            early_stopping_rounds=50,
            verbose=True
        )
        
        # Evaluate gene-level
        logger.info("\nEvaluating gene-level performance...")
        train_metrics = self.model.evaluate(X_train, y_train)
        val_metrics = self.model.evaluate(X_val, y_val)
        test_metrics = self.model.evaluate(X_test, y_test)
        
        logger.info(f"Train - auPRC: {train_metrics['auPRC']:.4f}, Precision: {train_metrics['Precision']:.4f}")
        logger.info(f"Val - auPRC: {val_metrics['auPRC']:.4f}, Precision: {val_metrics['Precision']:.4f}")
        logger.info(f"Test - auPRC: {test_metrics['auPRC']:.4f}, Precision: {test_metrics['Precision']:.4f}")
        
        # Evaluate sample-level
        logger.info("\nEvaluating sample-level performance...")
        train_sample_metrics = self.model.evaluate_sample_level(
            X_train, y_train, train_df['sample']
        )
        val_sample_metrics = self.model.evaluate_sample_level(
            X_val, y_val, val_df['sample']
        )
        test_sample_metrics = self.model.evaluate_sample_level(
            X_test, y_test, test_df['sample']
        )
        
        logger.info(f"Train Sample - auPRC: {train_sample_metrics['auPRC']:.4f}, auROC: {train_sample_metrics['AUC']:.4f}")
        logger.info(f"Val Sample - auPRC: {val_sample_metrics['auPRC']:.4f}, auROC: {val_sample_metrics['AUC']:.4f}")
        logger.info(f"Test Sample - auPRC: {test_sample_metrics['auPRC']:.4f}, auROC: {test_sample_metrics['AUC']:.4f}")
        
        # Save model
        model_path = self.output_dir / f'xgb11_{self.model_type}_model.pkl'
        self.model.save(str(model_path))
        
        # Save metrics
        import yaml
        summary = {
            'model': f'XGB11_{self.model_type}',
            'gene_level': {
                'train': train_metrics,
                'val': val_metrics,
                'test': test_metrics
            },
            'sample_level': {
                'train': train_sample_metrics,
                'val': val_sample_metrics,
                'test': test_sample_metrics
            }
        }
        
        with open(self.output_dir / f'training_summary_{self.model_type}.yml', 'w') as f:
            yaml.dump(summary, f, default_flow_style=False)
        
        # Feature importance
        importance_df = self.model.get_feature_importance()
        importance_df.to_csv(self.output_dir / f'feature_importance_{self.model_type}.csv', index=False)
        logger.info(f"\nTop 10 features:\n{importance_df.head(10)}")
        
        return test_metrics, test_sample_metrics


def main():
    """Main function for training XGB11"""
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
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
