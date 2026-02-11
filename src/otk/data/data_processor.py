import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import yaml
import os

class ECDNA_Dataset(Dataset):
    def __init__(self, features, targets, amplicon_classes=None):
        self.features = torch.tensor(features, dtype=torch.float32)
        # Convert pandas Series to numpy array if needed
        if hasattr(targets, 'values'):
            targets = targets.values
        self.targets = torch.tensor(targets, dtype=torch.float32)
        self.amplicon_classes = None
        if amplicon_classes is not None:
            # Convert pandas Series to numpy array if needed
            if hasattr(amplicon_classes, 'values'):
                amplicon_classes = amplicon_classes.values
            self.amplicon_classes = torch.tensor(amplicon_classes, dtype=torch.long)
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        if self.amplicon_classes is not None:
            return self.features[idx], self.targets[idx], self.amplicon_classes[idx]
        else:
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
        if data_path is None:
            data_path = self.data_config['path']
        
        # Ensure the path is absolute or relative to the project root
        if not os.path.isabs(data_path):
            data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), data_path)
        
        print(f"Loading data from {data_path}")
        df = pd.read_csv(data_path)
        print(f"Data loaded successfully with shape: {df.shape}")
        
        # Load amplicon data if available
        amplicon_path = self.data_config.get('amplicon_path', None)
        if amplicon_path:
            if not os.path.isabs(amplicon_path):
                amplicon_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), amplicon_path)
            print(f"Loading amplicon data from {amplicon_path}")
            amplicon_df = pd.read_csv(amplicon_path)
            print(f"Amplicon data loaded successfully with shape: {amplicon_df.shape}")
            return df, amplicon_df
        
        return df
    
    def preprocess(self, df, amplicon_df=None):
        """Preprocess data including missing value handling"""
        # Handle missing values
        if 'age' in df.columns:
            if df['age'].isnull().sum() > 0:
                strategy = self.data_config['missing_value_strategy'].get('age', 'mean')
                if strategy == 'mean':
                    df['age'] = df['age'].fillna(df['age'].mean())
                elif strategy == 'median':
                    df['age'] = df['age'].fillna(df['age'].median())
                elif strategy == 'mode':
                    df['age'] = df['age'].fillna(df['age'].mode()[0])
                print(f"Handled missing values in 'age' column using {strategy} strategy")
        
        # Select features and target
        features = df[self.data_config['features']]
        target = df[self.data_config['target']]
        samples = df[self.data_config['sample_id']]
        genes = df[self.data_config['gene_id']]
        
        # Process amplicon data if available
        sample_classification = None
        amplicon_mapping = None
        if amplicon_df is not None:
            # Create amplicon classification to index mapping
            amplicon_classes = amplicon_df['amplicon_classification'].unique()
            # Add 'nofocal' to the mapping if not present
            if 'nofocal' not in amplicon_classes:
                amplicon_classes = list(amplicon_classes) + ['nofocal']
            amplicon_mapping = {cls: i for i, cls in enumerate(amplicon_classes)}
            print(f"Created amplicon classification mapping: {amplicon_mapping}")
            
            # Create sample-level classification mapping
            # For each sample, get the most common amplicon classification
            sample_classification = amplicon_df.groupby('sample_barcode')['amplicon_classification'].agg(lambda x: x.value_counts().idxmax()).to_dict()
            print(f"Created sample classification mapping for {len(sample_classification)} samples")
            
            # Add nofocal classification for samples not in amplicon_df
            all_samples = df['sample'].unique()
            missing_samples = [sample for sample in all_samples if sample not in sample_classification]
            for sample in missing_samples:
                sample_classification[sample] = 'nofocal'
            print(f"Added nofocal classification for {len(missing_samples)} missing samples")
            
            # Create gene-level amplicon classification mapping
            gene_amplicon_mapping = {}
            for _, row in amplicon_df.iterrows():
                key = (row['gene_id'], row['sample_barcode'])
                gene_amplicon_mapping[key] = row['amplicon_classification']
            print(f"Created gene amplicon mapping for {len(gene_amplicon_mapping)} gene-sample pairs")
            
            # Add amplicon classification to each gene-sample pair
            # For pairs not in gene_amplicon_mapping, set to 'nofocal'
            df['amplicon_class'] = df.apply(lambda row: gene_amplicon_mapping.get((row['gene_id'], row['sample']), 'nofocal'), axis=1)
            print(f"Added amplicon classification to dataframe")
        
        return features, target, samples, genes, sample_classification, amplicon_mapping
    
    def split_data(self, features, target, samples, sample_classification=None, amplicon_class=None):
        """Split data into train, validation, and test sets by sample"""
        # Get unique samples
        unique_samples = samples.unique()
        print(f"Number of unique samples: {len(unique_samples)}")
        
        # Set random seed for reproducibility
        np.random.seed(self.training_config['seed'])
        
        # Split samples into train, validation, and test with balanced distribution
        if sample_classification is not None:
            # Create a list of samples with their classification
            sample_list = []
            for sample in unique_samples:
                if sample in sample_classification:
                    sample_list.append((sample, sample_classification[sample]))
                else:
                    sample_list.append((sample, 'Unknown'))
            
            # Convert to DataFrame for stratified split
            sample_df = pd.DataFrame(sample_list, columns=['sample', 'classification'])
            
            # Use stratified split to ensure balanced distribution of classifications
            from sklearn.model_selection import train_test_split as stratified_split
            
            # First split into train and temp
            train_samples, temp_samples = stratified_split(
                sample_df['sample'], 
                test_size=self.training_config['validation_split'] + self.training_config['test_split'],
                stratify=sample_df['classification'],
                random_state=self.training_config['seed']
            )
            
            # Then split temp into validation and test
            temp_df = sample_df[sample_df['sample'].isin(temp_samples)]
            validation_samples, test_samples = stratified_split(
                temp_df['sample'], 
                test_size=self.training_config['test_split'] / (self.training_config['validation_split'] + self.training_config['test_split']),
                stratify=temp_df['classification'],
                random_state=self.training_config['seed']
            )
        else:
            # Use random split if no classification data available
            train_samples, temp_samples = train_test_split(
                unique_samples, 
                test_size=self.training_config['validation_split'] + self.training_config['test_split'],
                random_state=self.training_config['seed']
            )
            
            validation_samples, test_samples = train_test_split(
                temp_samples, 
                test_size=self.training_config['test_split'] / (self.training_config['validation_split'] + self.training_config['test_split']),
                random_state=self.training_config['seed']
            )
        
        print(f"Train samples: {len(train_samples)}, Validation samples: {len(validation_samples)}, Test samples: {len(test_samples)}")
        
        # Create masks for each split
        train_mask = samples.isin(train_samples)
        validation_mask = samples.isin(validation_samples)
        test_mask = samples.isin(test_samples)
        
        # Split data
        X_train = features[train_mask]
        y_train = target[train_mask]
        X_val = features[validation_mask]
        y_val = target[validation_mask]
        X_test = features[test_mask]
        y_test = target[test_mask]
        
        # Split amplicon class if provided
        if amplicon_class is not None:
            amplicon_train = amplicon_class[train_mask]
            amplicon_val = amplicon_class[validation_mask]
            amplicon_test = amplicon_class[test_mask]
            print(f"Train amplicon shape: {amplicon_train.shape}, Validation amplicon shape: {amplicon_val.shape}, Test amplicon shape: {amplicon_test.shape}")
            return X_train, y_train, X_val, y_val, X_test, y_test, amplicon_train, amplicon_val, amplicon_test
        
        # Print class distribution for each split
        print(f"Train set shape: {X_train.shape}, Positive samples: {y_train.sum()}")
        print(f"Validation set shape: {X_val.shape}, Positive samples: {y_val.sum()}")
        print(f"Test set shape: {X_test.shape}, Positive samples: {y_test.sum()}")
        return X_train, y_train, X_val, y_val, X_test, y_test
    
    def normalize(self, X_train, X_val, X_test):
        """Normalize features"""
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        X_test_scaled = self.scaler.transform(X_test)
        return X_train_scaled, X_val_scaled, X_test_scaled
    
    def create_dataloaders(self, X_train, y_train, X_val, y_val, X_test, y_test, amplicon_train=None, amplicon_val=None, amplicon_test=None):
        """Create DataLoaders for training, validation, and test sets"""
        train_dataset = ECDNA_Dataset(X_train, y_train, amplicon_train)
        val_dataset = ECDNA_Dataset(X_val, y_val, amplicon_val)
        test_dataset = ECDNA_Dataset(X_test, y_test, amplicon_test)
        
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
        
        print(f"Created DataLoaders with batch size: {self.training_config['batch_size']}")
        print(f"Train batches: {len(train_loader)}, Validation batches: {len(val_loader)}, Test batches: {len(test_loader)}")
        
        return train_loader, val_loader, test_loader
    
    def process(self, data_path=None):
        """End-to-end data processing pipeline"""
        # Load data
        load_result = self.load_data(data_path)
        if isinstance(load_result, tuple):
            df, amplicon_df = load_result
        else:
            df = load_result
            amplicon_df = None
        
        # Preprocess data
        features, target, samples, genes, sample_classification, amplicon_mapping = self.preprocess(df, amplicon_df)
        
        # Split data
        if 'amplicon_class' in df.columns:
            # Add amplicon class to split data
            amplicon_class = df['amplicon_class'].map(lambda x: amplicon_mapping.get(x, len(amplicon_mapping)) if amplicon_mapping else 0)
            X_train, y_train, X_val, y_val, X_test, y_test, amplicon_train, amplicon_val, amplicon_test = self.split_data(features, target, samples, sample_classification, amplicon_class)
        else:
            X_train, y_train, X_val, y_val, X_test, y_test = self.split_data(features, target, samples, sample_classification)
            amplicon_train, amplicon_val, amplicon_test = None, None, None
        
        # Normalize data
        X_train_scaled, X_val_scaled, X_test_scaled = self.normalize(X_train, X_val, X_test)
        
        # Create dataloaders
        train_loader, val_loader, test_loader = self.create_dataloaders(
            X_train_scaled, y_train, X_val_scaled, y_val, X_test_scaled, y_test, 
            amplicon_train, amplicon_val, amplicon_test
        )
        
        return {
            'train_loader': train_loader,
            'val_loader': val_loader,
            'test_loader': test_loader,
            'scaler': self.scaler,
            'X_test': X_test_scaled,
            'y_test': y_test,
            'genes': genes,
            'sample_classification': sample_classification,
            'amplicon_mapping': amplicon_mapping
        }
