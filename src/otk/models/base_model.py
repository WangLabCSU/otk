#!/usr/bin/env python
"""
Base Model Interface for ecDNA Prediction

All models must implement this interface for unified training and evaluation.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple, Optional, Union
import pandas as pd
import numpy as np
from pathlib import Path


class BaseEcDNAModel(ABC):
    """
    Abstract base class for all ecDNA prediction models.
    
    All models must implement:
    - fit: Train the model
    - predict_proba: Get prediction probabilities
    - predict: Get binary predictions
    - evaluate_gene_level: Evaluate at gene level
    - evaluate_sample_level: Evaluate at sample level
    - save: Save model to disk
    - load: Load model from disk
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize model
        
        Args:
            config: Model configuration dictionary
        """
        self.config = config or {}
        self.is_fitted = False
        self.optimal_threshold = 0.5
        
    @abstractmethod
    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
        **kwargs
    ) -> 'BaseEcDNAModel':
        """
        Train the model
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            **kwargs: Additional training parameters
            
        Returns:
            self
        """
        pass
    
    @abstractmethod
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict probabilities
        
        Args:
            X: Features
            
        Returns:
            Array of probabilities
        """
        pass
    
    def predict(self, X: pd.DataFrame, threshold: Optional[float] = None) -> np.ndarray:
        """
        Predict binary labels
        
        Args:
            X: Features
            threshold: Classification threshold (default: self.optimal_threshold)
            
        Returns:
            Array of binary predictions
        """
        probs = self.predict_proba(X)
        threshold = threshold if threshold is not None else self.optimal_threshold
        return (probs >= threshold).astype(int)
    
    def evaluate_gene_level(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        threshold: Optional[float] = None
    ) -> Dict[str, float]:
        """
        Evaluate model at gene level
        
        Args:
            X: Features
            y: True labels
            threshold: Classification threshold
            
        Returns:
            Dictionary of metrics
        """
        from sklearn.metrics import precision_recall_curve, auc, roc_auc_score, f1_score, precision_score, recall_score
        
        probs = self.predict_proba(X)
        preds = self.predict(X, threshold)
        
        # Calculate auPRC
        precision, recall, _ = precision_recall_curve(y, probs)
        auprc = auc(recall, precision)
        
        # Calculate other metrics
        metrics = {
            'auPRC': float(auprc),
            'AUC': float(roc_auc_score(y, probs)),
            'Precision': float(precision_score(y, preds, zero_division=0)),
            'Recall': float(recall_score(y, preds, zero_division=0)),
            'F1': float(f1_score(y, preds, zero_division=0)),
            'threshold': float(threshold if threshold is not None else self.optimal_threshold)
        }
        
        return metrics
    
    def evaluate_sample_level(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        samples: pd.Series,
        threshold: Optional[float] = None
    ) -> Dict[str, float]:
        """
        Evaluate model at sample level
        
        A sample is predicted as circular if any gene is predicted positive.
        
        Args:
            X: Features
            y: True labels (gene level)
            samples: Sample IDs for each gene
            threshold: Classification threshold
            
        Returns:
            Dictionary of metrics
        """
        from sklearn.metrics import precision_recall_curve, auc, roc_auc_score, accuracy_score
        
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
        y_pred = (y_prob >= (threshold if threshold is not None else self.optimal_threshold)).astype(int)
        
        # Calculate metrics
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        auprc = auc(recall, precision)
        
        tp = ((y_pred == 1) & (y_true == 1)).sum()
        fp = ((y_pred == 1) & (y_true == 0)).sum()
        fn = ((y_pred == 0) & (y_true == 1)).sum()
        tn = ((y_pred == 0) & (y_true == 0)).sum()
        
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        
        metrics = {
            'auPRC': float(auprc),
            'AUC': float(roc_auc_score(y_true, y_prob)),
            'Accuracy': float(accuracy_score(y_true, y_pred)),
            'Precision': float(prec),
            'Recall': float(rec),
            'F1': float(f1),
            'total_samples': int(len(y_true)),
            'positive_samples': int(y_true.sum()),
            'predicted_positive': int(y_pred.sum()),
            'TP': int(tp),
            'FP': int(fp),
            'FN': int(fn),
            'TN': int(tn)
        }
        
        return metrics
    
    @abstractmethod
    def save(self, path: Union[str, Path]) -> None:
        """
        Save model to disk
        
        Args:
            path: Path to save model
        """
        pass
    
    @abstractmethod
    def load(self, path: Union[str, Path]) -> 'BaseEcDNAModel':
        """
        Load model from disk
        
        Args:
            path: Path to load model from
            
        Returns:
            self
        """
        pass
    
    def get_feature_importance(self) -> Optional[pd.DataFrame]:
        """
        Get feature importance (if available)
        
        Returns:
            DataFrame with feature importance or None
        """
        return None


class ModelTrainer:
    """
    Unified model trainer for all ecDNA models
    """
    
    def __init__(
        self,
        model: BaseEcDNAModel,
        output_dir: Union[str, Path],
        model_name: str
    ):
        """
        Initialize trainer
        
        Args:
            model: Model instance
            output_dir: Directory to save outputs
            model_name: Name of the model
        """
        self.model = model
        self.output_dir = Path(output_dir)
        self.model_name = model_name
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def train(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
        feature_cols: Optional[list] = None
    ) -> Dict[str, Any]:
        """
        Train and evaluate model
        
        Args:
            train_df: Training data
            val_df: Validation data
            test_df: Test data
            feature_cols: List of feature columns (if None, use all except sample, gene_id, y)
            
        Returns:
            Dictionary with all metrics
        """
        import logging
        logger = logging.getLogger(__name__)
        
        # Determine feature columns
        if feature_cols is None:
            feature_cols = [c for c in train_df.columns if c not in ['sample', 'gene_id', 'y']]
        
        logger.info(f"Training {self.model_name} with {len(feature_cols)} features")
        
        # Prepare data
        X_train = train_df[feature_cols]
        y_train = train_df['y']
        X_val = val_df[feature_cols]
        y_val = val_df['y']
        X_test = test_df[feature_cols]
        y_test = test_df['y']
        
        # Train model
        logger.info("Fitting model...")
        self.model.fit(X_train, y_train, X_val, y_val)
        
        # Evaluate on all splits
        logger.info("Evaluating gene-level performance...")
        train_metrics = self.model.evaluate_gene_level(X_train, y_train)
        val_metrics = self.model.evaluate_gene_level(X_val, y_val)
        test_metrics = self.model.evaluate_gene_level(X_test, y_test)
        
        logger.info("Evaluating sample-level performance...")
        train_sample_metrics = self.model.evaluate_sample_level(X_train, y_train, train_df['sample'])
        val_sample_metrics = self.model.evaluate_sample_level(X_val, y_val, val_df['sample'])
        test_sample_metrics = self.model.evaluate_sample_level(X_test, y_test, test_df['sample'])
        
        # Compile results
        results = {
            'model_name': self.model_name,
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
        
        # Save model
        model_path = self.output_dir / f'{self.model_name}_model.pkl'
        self.model.save(model_path)
        logger.info(f"Model saved to {model_path}")
        
        # Save metrics
        import yaml
        metrics_path = self.output_dir / 'training_summary.yml'
        with open(metrics_path, 'w') as f:
            yaml.dump(results, f, default_flow_style=False)
        logger.info(f"Metrics saved to {metrics_path}")
        
        # Save feature importance if available
        importance = self.model.get_feature_importance()
        if importance is not None:
            importance_path = self.output_dir / 'feature_importance.csv'
            importance.to_csv(importance_path, index=False)
            logger.info(f"Feature importance saved to {importance_path}")
        
        # Print summary
        logger.info("\n" + "="*60)
        logger.info(f"Training completed for {self.model_name}")
        logger.info("="*60)
        logger.info(f"Gene-level Test auPRC: {test_metrics['auPRC']:.4f}")
        logger.info(f"Gene-level Test Precision: {test_metrics['Precision']:.4f}")
        logger.info(f"Sample-level Test auROC: {test_sample_metrics['AUC']:.4f}")
        logger.info(f"Sample-level Test auPRC: {test_sample_metrics['auPRC']:.4f}")
        
        return results
