#!/usr/bin/env python
"""
XGB11 Model - XGBoost-based ecDNA Prediction Model

Based on Nature Communications 2024 paper:
"Machine learning-based extrachromosomal DNA identification in large-scale cohorts"

Features (11 predictive features):
1. segVal (segment value/total copy number)
2. purity (tumor purity)
3. ploidy (tumor ploidy)
4. cna_burden (copy number alteration burden)
5. minor_cn (minor copy number)
6. AScore (allelic score)
7. pLOH (percentage of LOH)
8. intersect_ratio (intersection ratio)
9. CN1-CN4 (copy number states 1-4)
10. freq_Linear (linear amplification frequency)
11. freq_Circular (circular amplification frequency)

Hyperparameters (from paper):
- eta: 0.1
- max_depth: 4
- min_child_weight: 1
- alpha: 0
- lambda: 1
- gamma: 10
- subsample: 0.6
- colsample_bytree: 1
- objective: logistic
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
    XGB11 Model for ecDNA prediction
    
    Implementation based on Nature Communications 2024 paper.
    Uses 11 key predictive features with optimized hyperparameters.
    """
    
    # 11 predictive features as described in the paper
    FEATURES = [
        'segVal',           # Total copy number
        'purity',           # Tumor purity
        'ploidy',           # Tumor ploidy
        'cna_burden',       # Copy number alteration burden
        'minor_cn',         # Minor copy number
        'AScore',           # Allelic score
        'pLOH',             # Percentage of LOH
        'intersect_ratio',  # Intersection ratio
        'CN1',              # Copy number state 1
        'CN2',              # Copy number state 2
        'freq_Circular',    # Circular amplification frequency
    ]
    
    # Optimal hyperparameters from the paper
    DEFAULT_PARAMS = {
        'eta': 0.1,
        'max_depth': 4,
        'min_child_weight': 1,
        'alpha': 0,
        'lambda': 1,
        'gamma': 10,
        'subsample': 0.6,
        'colsample_bytree': 1,
        'objective': 'binary:logistic',
        'eval_metric': 'aucpr',
        'tree_method': 'hist',
        'device': 'cuda',
    }
    
    def __init__(self, params: Optional[Dict[str, Any]] = None):
        """
        Initialize XGB11 model
        
        Args:
            params: Optional custom parameters. If None, uses paper's defaults.
        """
        self.params = params if params else self.DEFAULT_PARAMS.copy()
        self.model = None
        self.optimal_threshold = 0.5
        
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare features from raw data
        
        Args:
            df: Input dataframe with raw features
            
        Returns:
            DataFrame with selected and engineered features
        """
        feature_df = pd.DataFrame()
        
        for feat in self.FEATURES:
            if feat in df.columns:
                feature_df[feat] = df[feat].fillna(0)
            else:
                logger.warning(f"Feature {feat} not found in data, using zeros")
                feature_df[feat] = 0
        
        # Feature engineering based on paper insights
        # Total CN relative to ploidy
        feature_df['cn_ploidy_ratio'] = feature_df['segVal'] / (feature_df['ploidy'] + 1e-6)
        
        # CNA burden normalized by purity
        feature_df['cna_burden_norm'] = feature_df['cna_burden'] * feature_df['purity']
        
        # Minor CN ratio
        feature_df['minor_cn_ratio'] = feature_df['minor_cn'] / (feature_df['segVal'] + 1e-6)
        
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
        """
        Train the XGB11 model
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            sample_weight: Sample weights for imbalanced data
            early_stopping_rounds: Early stopping patience
            verbose: Whether to print training progress
            
        Returns:
            self
        """
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
            evals.append((dval, 'validation'))
        
        logger.info(f"Training XGB11 model with params: {self.params}")
        
        self.model = xgb.train(
            self.params,
            dtrain,
            num_boost_round=10000,
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
        """
        Evaluate model performance
        
        Returns gene-level metrics
        """
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
        """
        Evaluate at sample level
        
        A sample is predicted as circular if any gene is predicted positive.
        """
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
        
        # Map back to original features
        feature_names = self.prepare_features(pd.DataFrame(columns=self.FEATURES)).columns.tolist()
        
        importance_df = pd.DataFrame([
            {'feature': k, 'importance': v}
            for k, v in importance.items()
        ])
        importance_df = importance_df.sort_values('importance', ascending=False)
        
        return importance_df


class XGB11Trainer:
    """Trainer for XGB11 model"""
    
    def __init__(
        self,
        data_path: str,
        output_dir: str,
        validation_split: float = 0.12,
        test_split: float = 0.18,
        random_state: int = 42
    ):
        self.data_path = Path(data_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.validation_split = validation_split
        self.test_split = test_split
        self.random_state = random_state
        
        self.model = XGB11Model()
        
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
        model_path = self.output_dir / 'xgb11_model.pkl'
        self.model.save(str(model_path))
        
        # Save metrics
        import yaml
        summary = {
            'model': 'XGB11',
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
        
        with open(self.output_dir / 'training_summary.yml', 'w') as f:
            yaml.dump(summary, f, default_flow_style=False)
        
        # Feature importance
        importance_df = self.model.get_feature_importance()
        importance_df.to_csv(self.output_dir / 'feature_importance.csv', index=False)
        logger.info(f"\nTop 10 features:\n{importance_df.head(10)}")
        
        return test_metrics, test_sample_metrics


def main():
    """Main function for training XGB11"""
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    trainer = XGB11Trainer(
        data_path='src/otk/data/sorted_modeling_data.csv.gz',
        output_dir='otk_api/models/xgb11'
    )
    
    test_metrics, test_sample_metrics = trainer.train()
    
    print("\n" + "="*60)
    print("XGB11 TRAINING COMPLETED")
    print("="*60)
    print(f"Gene-level auPRC: {test_metrics['auPRC']:.4f}")
    print(f"Gene-level Precision: {test_metrics['Precision']:.4f}")
    print(f"Sample-level auPRC: {test_sample_metrics['auPRC']:.4f}")
    print(f"Sample-level auROC: {test_sample_metrics['AUC']:.4f}")


if __name__ == "__main__":
    main()
