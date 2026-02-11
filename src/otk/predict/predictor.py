import torch
import pandas as pd
import numpy as np
import os
import yaml
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

class Prediction_Dataset(Dataset):
    def __init__(self, features, amplicon_class=None):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.amplicon_class = amplicon_class
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        if self.amplicon_class is not None:
            return self.features[idx], torch.tensor(self.amplicon_class[idx], dtype=torch.long)
        else:
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
        from otk.models.model import MLP, TransformerModel, MultiInputTransformerModel
        model_type = config['model']['architecture']['type']
        if model_type == 'MLP':
            return MLP({'model': config['model']})
        elif model_type == 'Transformer':
            return TransformerModel({'model': config['model']})
        elif model_type == 'MultiInputTransformer':
            return MultiInputTransformerModel({'model': config['model']})
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
    
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
        
        # Handle amplicon data if available
        amplicon_class = None
        if 'amplicon_class' in df.columns:
            # Create amplicon mapping if needed
            amplicon_mapping = {'nofocal': 0, 'Non-circular': 1, 'Circular': 2, 'Unknown': 3}
            amplicon_class = df['amplicon_class'].map(lambda x: amplicon_mapping.get(x, 3)).values
        
        return features, sample_info, gene_info, amplicon_class
    
    def normalize(self, features):
        """Normalize features"""
        # For prediction, we need to use the same scaler as during training
        # However, since we don't have the training scaler, we'll fit a new one
        # This is a simplification and may not be optimal
        self.scaler = StandardScaler()
        features_scaled = self.scaler.fit_transform(features)
        return features_scaled
    
    def create_dataloader(self, features, amplicon_class=None):
        """Create DataLoader for prediction"""
        dataset = Prediction_Dataset(features, amplicon_class)
        batch_size = self.config['prediction']['batch_size']
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        print(f"Created DataLoader with batch size: {batch_size}")
        return dataloader
    
    def predict(self, dataloader):
        """Make predictions"""
        gene_predictions = []
        sample_predictions = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Predicting"):
                if isinstance(batch, tuple):
                    # Multi-input case (e.g., MultiInputTransformer)
                    inputs, amplicon_class = batch
                    inputs = inputs.to(self.device)
                    amplicon_class = amplicon_class.to(self.device)
                    gene_output, sample_output = self.model(inputs, amplicon_class)
                    gene_predictions.extend(gene_output.cpu().detach().numpy())
                    sample_predictions.extend(sample_output.cpu().detach().numpy())
                else:
                    # Single-input case (e.g., MLP)
                    inputs = batch.to(self.device)
                    outputs = self.model(inputs)
                    if isinstance(outputs, tuple):
                        # Transformer models return both gene and sample predictions
                        gene_output, sample_output = outputs
                        gene_predictions.extend(gene_output.cpu().detach().numpy())
                        sample_predictions.extend(sample_output.cpu().detach().numpy())
                    else:
                        # MLP returns only gene predictions
                        gene_predictions.extend(outputs.cpu().detach().numpy())
        
        gene_predictions = np.array(gene_predictions)
        if sample_predictions:
            sample_predictions = np.array(sample_predictions)
            return gene_predictions, sample_predictions
        else:
            return gene_predictions
    
    def postprocess(self, predictions, sample_info=None, gene_info=None):
        """Postprocess predictions"""
        # Create a DataFrame with predictions
        results = pd.DataFrame()
        
        # Add sample and gene information if available
        if sample_info is not None:
            results[self.config['data']['sample_id']] = sample_info
        if gene_info is not None:
            results[self.config['data']['gene_id']] = gene_info
        
        # Handle different prediction formats
        if isinstance(predictions, tuple):
            # Case with both gene and sample predictions
            gene_predictions, sample_predictions = predictions
            
            # Add gene-level predictions
            threshold = self.config['prediction']['threshold']
            results['prediction_prob'] = gene_predictions.flatten()
            results['prediction'] = (gene_predictions.flatten() > threshold).astype(int)
            
            # Add sample-level predictions
            # 确保sample_classes列表顺序与模型的输出类别顺序一致
            sample_classes = ['nofocal', 'noncircular', 'circular']
            results['sample_level_prediction'] = np.argmax(sample_predictions, axis=1)
            results['sample_level_prediction_label'] = results['sample_level_prediction'].map(lambda x: sample_classes[x])
            
            # Add probability for each sample class
            for i, cls in enumerate(sample_classes):
                results[f'{cls}_prob'] = sample_predictions[:, i]
        else:
            # Case with only gene predictions
            threshold = self.config['prediction']['threshold']
            results['prediction_prob'] = predictions.flatten()
            results['prediction'] = (predictions.flatten() > threshold).astype(int)
        
        return results
    
    def run(self, data_path, output_dir):
        """Run the prediction pipeline"""
        # Load data
        df = self.load_data(data_path)
        
        # Preprocess data
        features, sample_info, gene_info, amplicon_class = self.preprocess(df)
        
        # Normalize features
        features_scaled = self.normalize(features)
        
        # Create dataloader
        dataloader = self.create_dataloader(features_scaled, amplicon_class)
        
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
        # Group by sample
        if 'sample_level_prediction_label' in results.columns:
            # Use the most common sample-level prediction label
            sample_grouped = results.groupby(self.config['data']['sample_id']).agg({
                'sample_level_prediction_label': lambda x: x.value_counts().idxmax(),
                'prediction_prob': 'max',
                'prediction': 'max',
                'nofocal_prob': 'mean',
                'noncircular_prob': 'mean',
                'circular_prob': 'mean'
            }).reset_index()
            
            # Rename columns for clarity
            sample_grouped.rename(columns={
                'sample_level_prediction_label': 'focal_amplification_type'
            }, inplace=True)
        else:
            # Fallback to original method
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
