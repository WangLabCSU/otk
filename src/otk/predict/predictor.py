#!/usr/bin/env python
"""
Predictor for ecDNA Models

Unified prediction interface for all model types:
- XGBoost models (xgb_new, xgb_paper)
- Neural network models (transformer, baseline_mlp, etc.)
- TabPFN model
"""

import torch
import pandas as pd
import numpy as np
import os
import yaml
import pickle
import logging
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

logger = logging.getLogger(__name__)

RANDOM_SEED = 2026


class PredictionDataset(Dataset):
    def __init__(self, features):
        self.features = torch.tensor(features, dtype=torch.float32)
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx]


class UnifiedPredictor:
    """
    Unified predictor for all ecDNA models.
    
    Supports:
    - XGBoost models (.pkl files)
    - Neural network models (.pth files)
    - TabPFN models (.pkl files)
    """
    
    CANCER_TYPES = [
        'BLCA', 'BRCA', 'CESC', 'COAD', 'DLBC', 'ESCA', 'GBM', 'HNSC',
        'KICH', 'KIRC', 'KIRP', 'LGG', 'LIHC', 'LUAD', 'LUSC', 'OV',
        'PRAD', 'READ', 'SARC', 'SKCM', 'STAD', 'THCA', 'UCEC', 'UVM'
    ]
    
    DEFAULT_VALUES = {
        'minor_cn': 0,
        'intersect_ratio': 1.0,
        'purity': 0.8,
        'ploidy': 2.0,
        'AScore': 10.0,
        'pLOH': 0.1,
        'cna_burden': 0.2,
        'age': 60,
        'gender': 0,
    }
    
    def __init__(self, model_path, gpu=-1):
        self.model_path = Path(model_path)
        self.gpu = gpu
        self.optimal_threshold = 0.5
        self.model = None
        self.model_type = None
        self.config = None
        
        # Set device
        if torch.cuda.is_available() and gpu >= 0:
            self.device = torch.device(f'cuda:{gpu}')
            logger.info(f"Using GPU: {gpu}")
        else:
            self.device = torch.device('cpu')
            logger.info("Using CPU")
        
        # Load model
        self._load_model()
        
        # Load gene frequencies
        self.gene_freqs = self._load_gene_frequencies()
    
    def _load_model(self):
        """Load model based on file extension and path"""
        model_name = self.model_path.parent.name
        model_file = self.model_path.name
        
        # Determine model type from path/name
        if 'xgb' in model_name.lower() or 'xgb' in model_file.lower():
            self.model_type = 'xgboost'
            self._load_xgboost_model()
        elif 'tabpfn' in model_name.lower():
            self.model_type = 'tabpfn'
            self._load_tabpfn_model()
        elif model_file.endswith('.pth'):
            self.model_type = 'neural'
            self._load_neural_model()
        elif model_file.endswith('.pkl'):
            # Could be XGBoost or TabPFN
            self._load_pickle_model()
        else:
            raise ValueError(f"Unknown model format: {model_file}")
        
        # Load config if available
        config_path = self.model_path.parent / 'config.yml'
        if config_path.exists():
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
        
        logger.info(f"Loaded {self.model_type} model from {self.model_path}")
    
    def _load_xgboost_model(self):
        """Load XGBoost model from pickle"""
        with open(self.model_path, 'rb') as f:
            data = pickle.load(f)
        
        if isinstance(data, dict):
            self.model = data.get('model', data.get('xgb_model'))
            self.optimal_threshold = data.get('optimal_threshold', 0.5)
            self.feature_names = data.get('feature_names', None)
        else:
            self.model = data
        
        self.optimal_threshold = float(self.optimal_threshold)
        logger.info(f"XGBoost model loaded, threshold: {self.optimal_threshold:.4f}")
        if self.feature_names:
            logger.info(f"Feature names: {len(self.feature_names)} features")
    
    def _load_tabpfn_model(self):
        """Load TabPFN model from pickle"""
        with open(self.model_path, 'rb') as f:
            data = pickle.load(f)
        
        if isinstance(data, dict):
            self.model = data.get('models', [])
            self.optimal_threshold = data.get('optimal_threshold', 0.5)
        else:
            self.model = data
        
        self.optimal_threshold = float(self.optimal_threshold)
        logger.info(f"TabPFN model loaded with {len(self.model) if isinstance(self.model, list) else 1} estimators")
    
    def _load_neural_model(self):
        """Load neural network model from checkpoint"""
        checkpoint = torch.load(self.model_path, map_location=self.device)
        
        self.optimal_threshold = checkpoint.get('optimal_threshold', 0.5)
        if self.optimal_threshold is not None:
            self.optimal_threshold = float(self.optimal_threshold)
        
        # Try to determine model architecture from config
        config = checkpoint.get('config', {})
        model_type = config.get('model', {}).get('architecture', {}).get('type', 'Baseline')
        
        # Build model
        self.model = self._build_neural_model(model_type, config)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        logger.info(f"Neural model loaded, type: {model_type}, threshold: {self.optimal_threshold:.4f}")
    
    def _load_pickle_model(self):
        """Load model from pickle file (auto-detect type)"""
        import torch
        
        # Try torch.load first (for PyTorch models saved as .pkl)
        try:
            data = torch.load(self.model_path, map_location=self.device, weights_only=False)
            
            # Check if it's a neural network model
            if isinstance(data, dict) and ('model_state' in data or 'model_state_dict' in data):
                self.model_type = 'neural'
                self.optimal_threshold = float(data.get('optimal_threshold', 0.5))
                
                # Load full config from config.yml
                config_path = self.model_path.parent / 'config.yml'
                if config_path.exists():
                    with open(config_path, 'r') as f:
                        full_config = yaml.safe_load(f)
                else:
                    full_config = data.get('config', {})
                
                # Get model type from config
                model_type = full_config.get('model', {}).get('variant', 'BaselineMLP')
                
                # Build model architecture and load state dict
                self.model = self._build_neural_model(model_type, full_config)
                state_dict = data.get('model_state', data.get('model_state_dict'))
                self.model.load_state_dict(state_dict)
                self.model.to(self.device)
                self.model.eval()
                
                logger.info(f"Loaded neural model from pickle (via torch.load), type: {model_type}")
                return
            elif isinstance(data, dict) and 'models' in data:
                # TabPFN
                self.model_type = 'tabpfn'
                self.model = data['models']
                self.optimal_threshold = float(data.get('optimal_threshold', 0.5))
                logger.info(f"Loaded TabPFN model from pickle")
                return
        except Exception as e:
            logger.debug(f"torch.load failed: {e}, trying pickle.load")
        
        # Fall back to pickle.load for XGBoost models
        with open(self.model_path, 'rb') as f:
            data = pickle.load(f)
        
        if isinstance(data, dict):
            if 'models' in data and isinstance(data['models'], list):
                # TabPFN
                self.model_type = 'tabpfn'
                self.model = data['models']
                self.optimal_threshold = float(data.get('optimal_threshold', 0.5))
            elif 'model' in data or hasattr(data.get('model', None), 'predict'):
                # XGBoost
                self.model_type = 'xgboost'
                self.model = data.get('model', data)
                self.optimal_threshold = float(data.get('optimal_threshold', 0.5))
                self.feature_names = data.get('feature_names', None)
            else:
                raise ValueError("Unknown pickle model format")
        else:
            # Assume XGBoost
            self.model_type = 'xgboost'
            self.model = data
        
        logger.info(f"Loaded {self.model_type} model from pickle")
    
    def _build_neural_model(self, model_type, config):
        """Build neural network model from config - returns the actual PyTorch model"""
        import torch.nn as nn
        from otk.models.neural_models import (
            TransformerModel, DeepResidualNet, OptimizedResidualNet, DGITSuperNet
        )
        
        # Get architecture config
        arch = config.get('model', {}).get('architecture', {})
        input_dim = arch.get('input_dim', 57)
        
        if model_type in ['Transformer', 'TransformerEcDNA']:
            hidden_dim = arch.get('hidden_dim', 128)
            num_heads = arch.get('num_heads', 4)
            num_layers = arch.get('num_layers', 3)
            dropout = arch.get('dropout', 0.3)
            return TransformerModel(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                num_layers=num_layers,
                dropout=dropout
            )
        elif model_type in ['DeepResidual', 'PrecisionFocusedEcDNA']:
            return DeepResidualNet(input_dim)
        elif model_type in ['OptimizedResidual', 'OptimizedEcDNA']:
            return OptimizedResidualNet(input_dim)
        elif model_type == 'DGITSuper':
            return DGITSuperNet(input_dim)
        elif model_type in ['Baseline', 'BaselineMLP']:
            # Simple MLP
            return nn.Sequential(
                nn.Linear(input_dim, 128),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(64, 1)
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def _load_gene_frequencies(self):
        """Load precomputed gene frequencies"""
        gene_freq_path = self.model_path.parent.parent.parent / 'src' / 'otk' / 'data' / 'gene_frequencies.csv'
        
        if gene_freq_path.exists():
            gene_freqs = pd.read_csv(gene_freq_path)
            gene_freq_dict = {}
            for _, row in gene_freqs.iterrows():
                gene_freq_dict[row['gene_id']] = {
                    'freq_Linear': row.get('freq_Linear', 0),
                    'freq_BFB': row.get('freq_BFB', 0),
                    'freq_Circular': row.get('freq_Circular', 0),
                    'freq_HR': row.get('freq_HR', 0)
                }
            logger.info(f"Loaded frequencies for {len(gene_freq_dict)} genes")
            return gene_freq_dict
        else:
            logger.warning(f"Gene frequencies file not found")
            return {}
    
    def prepare_features(self, df):
        """Prepare features for prediction - matches XGBNewModel.prepare_features"""
        feature_df = pd.DataFrame()
        
        # segVal is required
        if 'segVal' not in df.columns:
            raise ValueError("segVal is a required feature but not found in input data")
        feature_df['segVal'] = df['segVal'].fillna(0)
        
        # Core features with defaults
        for f in ['minor_cn', 'purity', 'ploidy', 'pLOH', 'AScore', 'cna_burden']:
            if f in df.columns:
                feature_df[f] = df[f].fillna(self.DEFAULT_VALUES.get(f, 0))
            else:
                feature_df[f] = self.DEFAULT_VALUES.get(f, 0)
        
        # intersect_ratio defaults to 1.0
        if 'intersect_ratio' in df.columns:
            feature_df['intersect_ratio'] = df['intersect_ratio'].fillna(1.0)
        else:
            feature_df['intersect_ratio'] = 1.0
        
        # Frequency features
        for f in ['freq_Linear', 'freq_BFB', 'freq_Circular', 'freq_HR']:
            if f in df.columns:
                feature_df[f] = df[f].fillna(0)
            else:
                feature_df[f] = 0
        
        # CN signatures
        for i in range(1, 20):
            f = f'CN{i}'
            if f in df.columns:
                feature_df[f] = df[f].fillna(0)
            else:
                feature_df[f] = 0.05
        
        # Clinical
        if 'age' in df.columns:
            feature_df['age'] = df['age'].fillna(self.DEFAULT_VALUES.get('age', 60))
        else:
            feature_df['age'] = self.DEFAULT_VALUES.get('age', 60)
        if 'gender' in df.columns:
            feature_df['gender'] = df['gender'].fillna(self.DEFAULT_VALUES.get('gender', 0))
        else:
            feature_df['gender'] = self.DEFAULT_VALUES.get('gender', 0)
        
        # Cancer types (from df columns)
        for c in [col for col in df.columns if col.startswith('type_')]:
            feature_df[c] = df[c].fillna(0)
        
        # Ensure all cancer types exist
        for cancer_type in self.CANCER_TYPES:
            col_name = f'type_{cancer_type}'
            if col_name not in feature_df.columns:
                if 'type' in df.columns:
                    feature_df[col_name] = (df['type'] == cancer_type).astype(int)
                else:
                    feature_df[col_name] = 0
        
        # Feature engineering (order matches XGBNewModel)
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
            feature_df['has_loh'] = (df['pLOH'] > 0.1).astype(int)
        else:
            feature_df['has_loh'] = 0
        
        # Get feature columns from model's saved feature_names, config, or use all
        if hasattr(self, 'feature_names') and self.feature_names:
            feature_cols = self.feature_names
        elif self.config and 'data' in self.config and 'features' in self.config['data']:
            feature_cols = self.config['data']['features']
        else:
            feature_cols = [c for c in feature_df.columns if c not in ['sample', 'gene_id', 'y', 'type']]
        
        # Ensure all features exist
        for col in feature_cols:
            if col not in feature_df.columns:
                feature_df[col] = 0
        
        return feature_df[feature_cols].fillna(0).values.astype(np.float32), feature_cols
    
    def predict_proba(self, X, feature_names=None):
        """Predict probabilities"""
        if self.model_type == 'xgboost':
            import xgboost as xgb
            dmatrix = xgb.DMatrix(X, feature_names=feature_names)
            return self.model.predict(dmatrix)
        
        elif self.model_type == 'tabpfn':
            if isinstance(self.model, list):
                # Ensemble prediction
                probs = []
                for m in self.model:
                    probs.append(m.predict_proba(X)[:, 1])
                return np.mean(probs, axis=0)
            else:
                return self.model.predict_proba(X)[:, 1]
        
        elif self.model_type == 'neural':
            self.model.eval()
            dataset = PredictionDataset(X)
            dataloader = DataLoader(dataset, batch_size=4096, shuffle=False)
            
            predictions = []
            with torch.no_grad():
                for batch in dataloader:
                    batch = batch.to(self.device)
                    outputs = self.model(batch)
                    if outputs.min() < 0 or outputs.max() > 1:
                        outputs = torch.sigmoid(outputs)
                    predictions.extend(outputs.cpu().numpy())
            
            return np.array(predictions).flatten()
        
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def predict(self, X, threshold=None):
        """Predict binary labels"""
        if threshold is None:
            threshold = self.optimal_threshold
        
        probs = self.predict_proba(X)
        return (probs >= threshold).astype(int)
    
    def run(self, input_data, output_path=None):
        """Run prediction pipeline
        
        Args:
            input_data: DataFrame or path to CSV file
            output_path: Optional path to save results
            
        Returns:
            DataFrame with predictions
        """
        # Load data if path provided
        if isinstance(input_data, (str, Path)):
            logger.info(f"Loading data from {input_data}")
            df = pd.read_csv(input_data)
        else:
            df = input_data
        
        # Store original data
        original_df = df.copy()
        
        # Prepare features
        X, feature_names = self.prepare_features(df)
        logger.info(f"Prepared features shape: {X.shape}")
        
        # Get probabilities
        probs = self.predict_proba(X, feature_names=feature_names)
        
        # Create results
        results = pd.DataFrame()
        
        if 'sample' in original_df.columns:
            results['sample'] = original_df['sample']
        if 'gene_id' in original_df.columns:
            results['gene_id'] = original_df['gene_id']
        
        results['prediction_prob'] = probs
        results['prediction'] = (probs >= self.optimal_threshold).astype(int)
        
        # Sample-level classification
        if 'sample' in results.columns:
            sample_classifications = {}
            sample_id_col = 'sample'
            
            for sample, group in results.groupby(sample_id_col):
                has_ecdna_cargo = any(group['prediction'] == 1)
                
                # Check segVal threshold
                has_segval_threshold = False
                sample_data = original_df[original_df[sample_id_col] == sample]
                if not sample_data.empty and 'segVal' in sample_data.columns and 'ploidy' in sample_data.columns:
                    ploidy = sample_data['ploidy'].dropna().iloc[0] if len(sample_data['ploidy'].dropna()) > 0 else 2.0
                    has_segval_threshold = any(sample_data['segVal'] > (ploidy + 2))
                
                if has_ecdna_cargo:
                    sample_classifications[sample] = 'circular'
                elif has_segval_threshold:
                    sample_classifications[sample] = 'noncircular'
                else:
                    sample_classifications[sample] = 'nofocal'
            
            results['sample_level_prediction_label'] = results[sample_id_col].map(sample_classifications)
            results['sample_level_prediction'] = results['sample_level_prediction_label'].map(
                {'nofocal': 0, 'noncircular': 1, 'circular': 2}
            )
        
        # Save results
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            results.to_csv(output_path, index=False)
            logger.info(f"Results saved to {output_path}")
        
        return results


def predict(model_path, input_path, output_dir, gpu=-1):
    """Run prediction using trained model"""
    predictor = UnifiedPredictor(model_path, gpu)
    
    # Load input data
    if isinstance(input_path, pd.DataFrame):
        df = input_path
    else:
        df = pd.read_csv(input_path)
    
    # Create output path
    output_path = Path(output_dir) / 'predictions.csv'
    
    # Run prediction
    results = predictor.run(df, output_path)
    
    return results


# Backward compatibility
class Predictor(UnifiedPredictor):
    """Backward compatible predictor class"""
    pass


class Prediction_Dataset(PredictionDataset):
    """Backward compatible dataset class"""
    pass
