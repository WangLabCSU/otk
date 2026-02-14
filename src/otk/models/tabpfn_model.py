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
import numpy as np
import pandas as pd
from typing import Optional, Dict, Any, Tuple
from pathlib import Path
import pickle
import logging

from .base_model import BaseEcDNAModel

logger = logging.getLogger(__name__)

RANDOM_SEED = 2026


class TabPFNModel(BaseEcDNAModel):
    """
    TabPFN model for ecDNA prediction, inheriting from BaseEcDNAModel.
    
    Handles large datasets through:
    1. Stratified sampling for training
    2. Batch prediction for inference
    3. Ensemble of multiple TabPFN models
    """
    
    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        n_estimators: int = 5,
        max_samples_per_estimator: int = 5000
    ):
        super().__init__(config)
        self.n_estimators = n_estimators
        self.max_samples_per_estimator = max_samples_per_estimator
        self.models = []
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
    def prepare_features(self, df: pd.DataFrame) -> np.ndarray:
        """Prepare features"""
        feature_cols = [c for c in df.columns if c not in ['sample', 'gene_id', 'y']]
        return df[feature_cols].fillna(0).values.astype(np.float32)
    
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
    
    def fit(self, X_train, y_train, X_val=None, y_val=None, **kwargs):
        """Fit multiple TabPFN models on stratified samples"""
        from tabpfn import TabPFNClassifier
        
        # Set random seed
        np.random.seed(RANDOM_SEED)
        
        X_arr = self.prepare_features(X_train) if isinstance(X_train, pd.DataFrame) else X_train
        y_arr = y_train.values if isinstance(y_train, pd.Series) else y_train
        
        logger.info(f"Training {self.n_estimators} TabPFN models on device: {self.device}")
        
        self.models = []
        for i in range(self.n_estimators):
            X_sample, y_sample = self._stratified_sample(
                X_arr, y_arr, 
                self.max_samples_per_estimator,
                random_state=RANDOM_SEED + i * 100
            )
            
            logger.info(f"Model {i+1}/{self.n_estimators}: Training on {len(X_sample)} samples "
                       f"({(y_sample==1).sum()} positive, {(y_sample==0).sum()} negative)")
            
            model = TabPFNClassifier(device=self.device)
            model.fit(X_sample, y_sample)
            self.models.append(model)
        
        self.is_fitted = True
        
        # Find optimal threshold using validation set
        if X_val is not None and y_val is not None:
            val_probs = self.predict_proba(X_val)
            self.optimal_threshold = self._find_optimal_threshold(
                y_val.values if isinstance(y_val, pd.Series) else y_val, 
                val_probs
            )
        
        logger.info(f"TabPFN ensemble trained successfully with {len(self.models)} models")
        return self
    
    def _find_optimal_threshold(self, y_true, y_prob):
        from sklearn.metrics import precision_recall_curve
        precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
        optimal_idx = np.argmax(f1_scores)
        return float(thresholds[optimal_idx]) if optimal_idx < len(thresholds) else 0.5
    
    def predict_proba(self, X, batch_size: int = 10000) -> np.ndarray:
        """Predict probabilities with ensemble averaging"""
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        X_arr = self.prepare_features(X) if isinstance(X, pd.DataFrame) else X
        n_samples = len(X_arr)
        all_probs = []
        
        for i in range(0, n_samples, batch_size):
            batch = X_arr[i:i+batch_size]
            batch_probs = []
            
            for model in self.models:
                proba = model.predict_proba(batch)
                batch_probs.append(proba[:, 1])
            
            avg_prob = np.mean(batch_probs, axis=0)
            all_probs.append(avg_prob)
            
            if (i // batch_size) % 10 == 0:
                logger.info(f"Predicted {min(i+batch_size, n_samples)}/{n_samples} samples")
        
        return np.concatenate(all_probs)
    
    def save(self, path):
        """Save model to disk"""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        
        # Save model configuration and state
        save_data = {
            'n_estimators': self.n_estimators,
            'max_samples_per_estimator': self.max_samples_per_estimator,
            'optimal_threshold': float(self.optimal_threshold),
            'device': self.device,
            'models': self.models  # TabPFN models are picklable
        }
        
        with open(path, 'wb') as f:
            pickle.dump(save_data, f)
    
    def load(self, path):
        """Load model from disk"""
        with open(path, 'rb') as f:
            save_data = pickle.load(f)
        
        self.n_estimators = save_data['n_estimators']
        self.max_samples_per_estimator = save_data['max_samples_per_estimator']
        self.optimal_threshold = save_data['optimal_threshold']
        self.device = save_data['device']
        self.models = save_data['models']
        self.is_fitted = True
        
        return self


def train_tabpfn():
    """Train TabPFN model using unified interface"""
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))
    
    from otk.data.data_split import load_split
    from otk.models.base_model import ModelTrainer
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Load data
    train_df, val_df, test_df = load_split()
    
    # Create model
    model = TabPFNModel(n_estimators=5, max_samples_per_estimator=5000)
    
    # Train
    trainer = ModelTrainer(model, 'otk_api/models/tabpfn', 'tabpfn')
    results = trainer.train(train_df, val_df, test_df)
    
    return results


if __name__ == "__main__":
    train_tabpfn()
