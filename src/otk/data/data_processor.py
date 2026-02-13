import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import yaml
import os
import logging
import gzip

from .data_split import get_data_splits

logger = logging.getLogger(__name__)

class ECDNA_Dataset(Dataset):
    def __init__(self, features, targets):
        self.features = torch.tensor(features, dtype=torch.float32)
        # Convert pandas Series to numpy array if needed
        if hasattr(targets, 'values'):
            targets = targets.values
        self.targets = torch.tensor(targets, dtype=torch.float32)
        self.labels = self.targets  # Alias for compatibility
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]

class DataProcessor:
    def __init__(self, config_path):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        self.data_config = self.config['data']
        self.training_config = self.config['training']
        self.scaler = None
    
    def load_data(self, data_path=None):
        """Load data from CSV file"""
        # Use preprocessed sorted data if available
        sorted_data_path = os.path.join(os.path.dirname(__file__), 'sorted_modeling_data.csv.gz')
        
        if os.path.exists(sorted_data_path):
            logger.info(f"Loading preprocessed sorted data from {sorted_data_path}")
            df = pd.read_csv(sorted_data_path)
            logger.info(f"Preprocessed data loaded successfully with shape: {df.shape}")
            return df
        
        # Fallback to original data path if preprocessed data not available
        if data_path is None:
            data_path = self.data_config['path']
        
        # Ensure the path is absolute or relative to the project root
        if not os.path.isabs(data_path):
            data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), data_path)
        
        logger.info(f"Loading data from {data_path}")
        df = pd.read_csv(data_path)
        logger.info(f"Data loaded successfully with shape: {df.shape}")
        
        return df
    
    def preprocess(self, df):
        """Preprocess data including missing value handling"""
        # Handle missing values
        if 'age' in df.columns:
            if df['age'].isnull().sum() > 0:
                # Use 'mean' as default if missing_value_strategy is not defined
                strategy = self.data_config.get('missing_value_strategy', {}).get('age', 'mean')
                if strategy == 'mean':
                    df['age'] = df['age'].fillna(df['age'].mean())
                elif strategy == 'median':
                    df['age'] = df['age'].fillna(df['age'].median())
                elif strategy == 'mode':
                    df['age'] = df['age'].fillna(df['age'].mode()[0])
                logger.info(f"Handled missing values in 'age' column using {strategy} strategy")
        
        # Select features and target
        features = df[self.data_config['features']]
        target = df[self.data_config['target']]
        samples = df[self.data_config['sample_id']]
        genes = df[self.data_config['gene_id']]
        
        return features, target, samples, genes
    
    def split_data(self, features, target, samples):
        """Split data into train, validation, and test sets using unified split"""
        # Use unified data split (80/10/10, seed=2026)
        train_samples, val_samples, test_samples = get_data_splits()
        
        logger.info(f"Using unified split: train={len(train_samples)}, val={len(val_samples)}, test={len(test_samples)}")
        
        # Create masks for each split
        train_mask = samples.isin(train_samples)
        validation_mask = samples.isin(val_samples)
        test_mask = samples.isin(test_samples)
        
        # Split data
        X_train = features[train_mask]
        y_train = target[train_mask]
        X_val = features[validation_mask]
        y_val = target[validation_mask]
        X_test = features[test_mask]
        y_test = target[test_mask]
        
        # Print class distribution for each split
        logger.info(f"Train set shape: {X_train.shape}, Positive samples: {int(y_train.sum())}, Positive rate: {y_train.sum()/len(y_train):.4f}")
        logger.info(f"Validation set shape: {X_val.shape}, Positive samples: {int(y_val.sum())}, Positive rate: {y_val.sum()/len(y_val):.4f}")
        logger.info(f"Test set shape: {X_test.shape}, Positive samples: {int(y_test.sum())}, Positive rate: {y_test.sum()/len(y_test):.4f}")
        return X_train, y_train, X_val, y_val, X_test, y_test
    
    def normalize(self, X_train, X_val, X_test):
        """Normalize features"""
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        X_test_scaled = self.scaler.transform(X_test)
        return X_train_scaled, X_val_scaled, X_test_scaled
    
    def create_dataloaders(self, X_train, y_train, X_val, y_val, X_test, y_test):
        """Create DataLoaders for training, validation, and test sets"""
        train_dataset = ECDNA_Dataset(X_train, y_train)
        val_dataset = ECDNA_Dataset(X_val, y_val)
        test_dataset = ECDNA_Dataset(X_test, y_test)
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.training_config['batch_size'], 
            shuffle=True
        )
        val_loader = DataLoader(
            val_dataset, 
            batch_size=self.training_config['batch_size'], 
            shuffle=False
        )
        test_loader = DataLoader(
            test_dataset, 
            batch_size=self.training_config['batch_size'], 
            shuffle=False
        )
        
        logger.info(f"Created DataLoaders with batch size: {self.training_config['batch_size']}")
        logger.info(f"Train batches: {len(train_loader)}, Validation batches: {len(val_loader)}, Test batches: {len(test_loader)}")
        
        return train_loader, val_loader, test_loader
    
    def process(self, data_path=None):
        """End-to-end data processing pipeline"""
        # Load data
        df = self.load_data(data_path)
        
        # Preprocess data
        features, target, samples, genes = self.preprocess(df)
        
        # Split data
        X_train, y_train, X_val, y_val, X_test, y_test = self.split_data(features, target, samples)
        
        # Normalize data
        X_train_scaled, X_val_scaled, X_test_scaled = self.normalize(X_train, X_val, X_test)
        
        # Create dataloaders
        train_loader, val_loader, test_loader = self.create_dataloaders(
            X_train_scaled, y_train, X_val_scaled, y_val, X_test_scaled, y_test
        )
        
        return {
            'train_loader': train_loader,
            'val_loader': val_loader,
            'test_loader': test_loader,
            'scaler': self.scaler,
            'X_test': X_test_scaled,
            'y_test': y_test,
            'genes': genes
        }
