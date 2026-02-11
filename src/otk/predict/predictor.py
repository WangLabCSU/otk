import torch
import pandas as pd
import numpy as np
import os
import yaml
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

class Prediction_Dataset(Dataset):
    def __init__(self, features):
        self.features = torch.tensor(features, dtype=torch.float32)
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx]

class Predictor:
    def __init__(self, model_path, gpu=-1):
        self.model_path = model_path
        self.gpu = gpu
        
        # Set device
        if torch.cuda.is_available() and gpu >= 0:
            self.device = torch.device(f'cuda:{gpu}')
            print(f"Using GPU: {gpu}")
        else:
            self.device = torch.device('cpu')
            print("Using CPU")
        
        # Load model
        self.model = self._load_model()
        
        # Load configuration from model checkpoint
        checkpoint = torch.load(model_path)
        self.config = checkpoint['config']
        
        self.scaler = None
    
    def _load_model(self):
        """Load the model from a saved file"""
        checkpoint = torch.load(self.model_path, map_location=self.device)
        
        # Create model architecture based on configuration
        model = self._build_model(checkpoint['config'])
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        model.eval()
        print(f"Model loaded from {self.model_path}")
        return model
    
    def _build_model(self, config):
        """Build the model based on configuration"""
        from otk.models.model import MLP
        return MLP({'model': config['model']})
    
    def load_data(self, data_path):
        """Load data for prediction"""
        print(f"Loading data from {data_path}")
        df = pd.read_csv(data_path)
        print(f"Data loaded successfully with shape: {df.shape}")
        return df
    
    def preprocess(self, df):
        """Preprocess data for prediction"""
        # Handle missing values
        if 'age' in df.columns:
            if df['age'].isnull().sum() > 0:
                strategy = self.config['data']['missing_value_strategy'].get('age', 'mean')
                if strategy == 'mean':
                    df['age'] = df['age'].fillna(df['age'].mean())
                elif strategy == 'median':
                    df['age'] = df['age'].fillna(df['age'].median())
                elif strategy == 'mode':
                    df['age'] = df['age'].fillna(df['age'].mode()[0])
                print(f"Handled missing values in 'age' column using {strategy} strategy")
        
        # Select features
        features = df[self.config['data']['features']]
        
        # Save sample and gene information if available
        sample_info = None
        gene_info = None
        if self.config['data']['sample_id'] in df.columns:
            sample_info = df[self.config['data']['sample_id']]
        if self.config['data']['gene_id'] in df.columns:
            gene_info = df[self.config['data']['gene_id']]
        
        return features, sample_info, gene_info
    
    def normalize(self, features):
        """Normalize features"""
        # For prediction, we need to use the same scaler as during training
        # However, since we don't have the training scaler, we'll fit a new one
        # This is a simplification and may not be optimal
        self.scaler = StandardScaler()
        features_scaled = self.scaler.fit_transform(features)
        return features_scaled
    
    def create_dataloader(self, features):
        """Create DataLoader for prediction"""
        dataset = Prediction_Dataset(features)
        batch_size = self.config['prediction']['batch_size']
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        print(f"Created DataLoader with batch size: {batch_size}")
        return dataloader
    
    def predict(self, dataloader):
        """Make predictions"""
        predictions = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Predicting"):
                inputs = batch.to(self.device)
                outputs = self.model(inputs)
                predictions.extend(outputs.cpu().detach().numpy())
        
        predictions = np.array(predictions)
        return predictions
    
    def postprocess(self, predictions, sample_info=None, gene_info=None):
        """Postprocess predictions"""
        # Create a DataFrame with predictions
        results = pd.DataFrame()
        
        # Add sample and gene information if available
        if sample_info is not None:
            results[self.config['data']['sample_id']] = sample_info
        if gene_info is not None:
            results[self.config['data']['gene_id']] = gene_info
        
        # Add prediction probability and binary prediction
        threshold = self.config['prediction']['threshold']
        results['prediction_prob'] = predictions.flatten()
        results['prediction'] = (predictions.flatten() > threshold).astype(int)
        
        return results
    
    def run(self, data_path, output_dir):
        """Run the prediction pipeline"""
        # Load data
        df = self.load_data(data_path)
        
        # Preprocess data
        features, sample_info, gene_info = self.preprocess(df)
        
        # Normalize features
        features_scaled = self.normalize(features)
        
        # Create dataloader
        dataloader = self.create_dataloader(features_scaled)
        
        # Make predictions
        predictions = self.predict(dataloader)
        
        # Postprocess predictions
        results = self.postprocess(predictions, sample_info, gene_info)
        
        # Save results
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, 'predictions.csv')
        results.to_csv(output_path, index=False)
        print(f"Predictions saved to {output_path}")
        
        # Calculate sample-level predictions (optional)
        if sample_info is not None:
            sample_level_results = self._calculate_sample_level_predictions(results)
            sample_output_path = os.path.join(output_dir, 'sample_level_predictions.csv')
            sample_level_results.to_csv(sample_output_path, index=False)
            print(f"Sample-level predictions saved to {sample_output_path}")
        
        return results
    
    def _calculate_sample_level_predictions(self, results):
        """Calculate sample-level predictions"""
        # Group by sample and calculate the maximum prediction probability
        sample_grouped = results.groupby(self.config['data']['sample_id']).agg({
            'prediction_prob': 'max',
            'prediction': 'max'
        }).reset_index()
        
        # Add sample-level classification
        sample_grouped['focal_amplification_type'] = sample_grouped['prediction'].apply(
            lambda x: 'circular' if x == 1 else 'noncircular'
        )
        
        return sample_grouped

def predict(model_path, input_path, output_dir, gpu=-1):
    """Run prediction using the trained model"""
    predictor = Predictor(model_path, gpu)
    results = predictor.run(input_path, output_dir)
    return results
