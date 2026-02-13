#!/usr/bin/env python
"""
TabPFN Model for ecDNA Prediction

TabPFN is a foundation model for tabular data that achieves strong performance
without training. However, it has limitations:
- Max 10,000 training samples (recommended ~1000)
- Max 100 features
- Max 10 classes

For our ecDNA prediction task with 7M+ samples and 57 features:
1. Use stratified sampling to get representative training subset
2. Train multiple TabPFN models on different samples
3. Ensemble predictions for robust results
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from typing import Optional, Dict, Any, Tuple
from pathlib import Path
import logging
from sklearn.metrics import precision_recall_curve, auc, roc_auc_score

logger = logging.getLogger(__name__)


class TabPFNWrapper:
    """
    TabPFN wrapper for ecDNA prediction
    
    Handles large datasets through:
    1. Stratified sampling for training
    2. Batch prediction for inference
    3. Ensemble of multiple TabPFN models
    """
    
    def __init__(
        self,
        n_estimators: int = 5,
        max_samples_per_estimator: int = 5000,
        random_state: int = 42,
        device: str = 'auto'
    ):
        self.n_estimators = n_estimators
        self.max_samples_per_estimator = max_samples_per_estimator
        self.random_state = random_state
        self.device = device
        self.models = []
        self.is_fitted = False
        
    def _get_device(self):
        if self.device == 'auto':
            if torch.cuda.is_available():
                return 'cuda'
            return 'cpu'
        return self.device
    
    def _stratified_sample(
        self, 
        X: np.ndarray, 
        y: np.ndarray, 
        n_samples: int,
        random_state: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Stratified sampling to maintain class distribution"""
        np.random.seed(random_state)
        
        pos_idx = np.where(y == 1)[0]
        neg_idx = np.where(y == 0)[0]
        
        pos_ratio = len(pos_idx) / len(y)
        n_pos = min(int(n_samples * pos_ratio * 2), len(pos_idx))
        n_neg = min(n_samples - n_pos, len(neg_idx))
        
        selected_pos = np.random.choice(pos_idx, size=n_pos, replace=False)
        selected_neg = np.random.choice(neg_idx, size=n_neg, replace=False)
        
        selected_idx = np.concatenate([selected_pos, selected_neg])
        np.random.shuffle(selected_idx)
        
        return X[selected_idx], y[selected_idx]
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'TabPFNWrapper':
        """Fit multiple TabPFN models on stratified samples"""
        from tabpfn import TabPFNClassifier
        
        device = self._get_device()
        logger.info(f"Training {self.n_estimators} TabPFN models on device: {device}")
        
        self.models = []
        for i in range(self.n_estimators):
            X_sample, y_sample = self._stratified_sample(
                X, y, 
                self.max_samples_per_estimator,
                random_state=self.random_state + i * 100
            )
            
            logger.info(f"Model {i+1}/{self.n_estimators}: Training on {len(X_sample)} samples "
                       f"({(y_sample==1).sum()} positive, {(y_sample==0).sum()} negative)")
            
            model = TabPFNClassifier(device=device)
            model.fit(X_sample, y_sample)
            self.models.append(model)
        
        self.is_fitted = True
        logger.info(f"TabPFN ensemble trained successfully with {len(self.models)} models")
        return self
    
    def predict_proba(self, X: np.ndarray, batch_size: int = 10000) -> np.ndarray:
        """Predict probabilities with ensemble averaging"""
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        n_samples = len(X)
        all_probs = []
        
        for i in range(0, n_samples, batch_size):
            batch = X[i:i+batch_size]
            batch_probs = []
            
            for model in self.models:
                proba = model.predict_proba(batch)
                batch_probs.append(proba[:, 1])
            
            avg_prob = np.mean(batch_probs, axis=0)
            all_probs.append(avg_prob)
            
            if (i // batch_size) % 10 == 0:
                logger.info(f"Predicted {min(i+batch_size, n_samples)}/{n_samples} samples")
        
        return np.concatenate(all_probs)
    
    def predict(self, X: np.ndarray, threshold: float = 0.5, batch_size: int = 10000) -> np.ndarray:
        """Predict labels"""
        probs = self.predict_proba(X, batch_size)
        return (probs >= threshold).astype(int)
    
    def evaluate(
        self, 
        X: np.ndarray, 
        y: np.ndarray, 
        batch_size: int = 10000
    ) -> Dict[str, float]:
        """Evaluate model performance"""
        probs = self.predict_proba(X, batch_size)
        preds = (probs >= 0.5).astype(int)
        
        precision, recall, thresholds = precision_recall_curve(y, probs)
        auprc = auc(recall, precision)
        auc_score = roc_auc_score(y, probs)
        
        tp = ((preds == 1) & (y == 1)).sum()
        fp = ((preds == 1) & (y == 0)).sum()
        fn = ((preds == 0) & (y == 1)).sum()
        
        precision_score = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall_score = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision_score * recall_score / (precision_score + recall_score) \
             if (precision_score + recall_score) > 0 else 0
        
        return {
            'auPRC': auprc,
            'AUC': auc_score,
            'Precision': precision_score,
            'Recall': recall_score,
            'F1': f1
        }


class TabPFNTrainer:
    """
    Trainer for TabPFN model on ecDNA dataset
    """
    
    def __init__(
        self,
        data_path: str,
        output_dir: str,
        n_estimators: int = 5,
        max_samples_per_estimator: int = 5000,
        random_state: int = 42
    ):
        self.data_path = Path(data_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.model = TabPFNWrapper(
            n_estimators=n_estimators,
            max_samples_per_estimator=max_samples_per_estimator,
            random_state=random_state
        )
        
    def load_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Load preprocessed data"""
        import gzip
        
        logger.info(f"Loading data from {self.data_path}")
        
        with gzip.open(self.data_path, 'rt') as f:
            df = pd.read_csv(f)
        
        logger.info(f"Loaded data shape: {df.shape}")
        
        feature_cols = [col for col in df.columns if col not in ['sample', 'gene_id', 'y']]
        X = df[feature_cols].values
        y = df['y'].values
        samples = df['sample'].values
        
        unique_samples = np.unique(samples)
        np.random.seed(42)
        np.random.shuffle(unique_samples)
        
        n_samples = len(unique_samples)
        train_end = int(n_samples * 0.7)
        val_end = int(n_samples * 0.82)
        
        train_samples = set(unique_samples[:train_end])
        val_samples = set(unique_samples[train_end:val_end])
        test_samples = set(unique_samples[val_end:])
        
        train_mask = np.isin(samples, list(train_samples))
        val_mask = np.isin(samples, list(val_samples))
        test_mask = np.isin(samples, list(test_samples))
        
        X_train, y_train = X[train_mask], y[train_mask]
        X_val, y_val = X[val_mask], y[val_mask]
        X_test, y_test = X[test_mask], y[test_mask]
        
        logger.info(f"Train: {X_train.shape[0]} samples, {y_train.sum()} positive")
        logger.info(f"Val: {X_val.shape[0]} samples, {y_val.sum()} positive")
        logger.info(f"Test: {X_test.shape[0]} samples, {y_test.sum()} positive")
        
        return X_train, y_train, X_val, y_val, X_test, y_test
    
    def train_and_evaluate(self):
        """Train and evaluate TabPFN model"""
        X_train, y_train, X_val, y_val, X_test, y_test = self.load_data()
        
        logger.info("Training TabPFN ensemble...")
        self.model.fit(X_train, y_train)
        
        logger.info("Evaluating on validation set...")
        val_metrics = self.model.evaluate(X_val, y_val)
        logger.info(f"Validation metrics: {val_metrics}")
        
        logger.info("Evaluating on test set...")
        test_metrics = self.model.evaluate(X_test, y_test)
        logger.info(f"Test metrics: {test_metrics}")
        
        import yaml
        summary = {
            'model': 'TabPFN Ensemble',
            'n_estimators': self.model.n_estimators,
            'max_samples_per_estimator': self.model.max_samples_per_estimator,
            'validation_metrics': val_metrics,
            'test_metrics': test_metrics
        }
        
        summary_path = self.output_dir / 'training_summary.yml'
        with open(summary_path, 'w') as f:
            yaml.dump(summary, f, default_flow_style=False)
        
        logger.info(f"Summary saved to {summary_path}")
        
        return val_metrics, test_metrics


def main():
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    trainer = TabPFNTrainer(
        data_path='src/otk/data/sorted_modeling_data.csv.gz',
        output_dir='otk_api/models/tabpfn',
        n_estimators=5,
        max_samples_per_estimator=5000
    )
    
    trainer.train_and_evaluate()


if __name__ == "__main__":
    main()
