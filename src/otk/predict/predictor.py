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
        
        # Load precomputed gene frequencies
        self.gene_freqs = self._load_gene_frequencies()
        
        # Define cancer type mapping for one-hot encoding
        self.cancer_types = [
            'BLCA', 'BRCA', 'CESC', 'COAD', 'DLBC', 'ESCA', 'GBM', 'HNSC',
            'KICH', 'KIRC', 'KIRP', 'LGG', 'LIHC', 'LUAD', 'LUSC', 'OV',
            'PRAD', 'READ', 'SARC', 'SKCM', 'STAD', 'THCA', 'UCEC', 'UVM'
        ]
        
        self.scaler = None
    
    def _load_gene_frequencies(self):
        """Load precomputed gene level frequencies"""
        # Determine the path to gene frequencies file
        import os
        script_dir = os.path.dirname(os.path.abspath(__file__))
        gene_freq_path = os.path.join(script_dir, '..', 'data', 'gene_frequencies.csv')
        
        if os.path.exists(gene_freq_path):
            print(f"Loading gene frequencies from {gene_freq_path}")
            gene_freqs = pd.read_csv(gene_freq_path)
            # Create a dictionary for quick lookup
            gene_freq_dict = {}
            for _, row in gene_freqs.iterrows():
                gene_freq_dict[row['gene_id']] = {
                    'freq_Linear': row['freq_Linear'],
                    'freq_BFB': row['freq_BFB'],
                    'freq_Circular': row['freq_Circular'],
                    'freq_HR': row['freq_HR']
                }
            print(f"Loaded frequencies for {len(gene_freq_dict)} genes")
            return gene_freq_dict
        else:
            print(f"Warning: Gene frequencies file not found at {gene_freq_path}")
            return {}
    
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
        
        # Add gene level frequency features
        if 'gene_id' in df.columns:
            print("Adding gene level frequency features...")
            for gene_freq_col in ['freq_Linear', 'freq_BFB', 'freq_Circular', 'freq_HR']:
                df[gene_freq_col] = df['gene_id'].apply(lambda x: self.gene_freqs.get(x, {}).get(gene_freq_col, 0) if self.gene_freqs else 0)
            print("Gene frequency features added successfully")
        else:
            print("Warning: 'gene_id' column not found, cannot add gene frequency features")
        
        # Handle cancer type conversion
        if 'type' in df.columns:
            print("Processing cancer type...")
            # Convert cancer type to one-hot encoding
            for cancer_type in self.cancer_types:
                col_name = f'type_{cancer_type}'
                df[col_name] = df['type'].apply(lambda x: 1 if str(x).strip() == cancer_type else 0)
            print("Cancer type one-hot encoding completed")
        else:
            print("Warning: 'type' column not found, cannot process cancer type")
        
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
        gene_predictions = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Predicting"):
                # Single-input case
                inputs = batch.to(self.device)
                outputs = self.model(inputs)
                
                # Get gene predictions
                gene_predictions.extend(outputs.cpu().detach().numpy())
        
        gene_predictions = np.array(gene_predictions)
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
        
        # Add gene-level predictions
        threshold = self.config['prediction']['threshold']
        results['prediction_prob'] = predictions.flatten()
        results['prediction'] = (predictions.flatten() > threshold).astype(int)
        
        # Add sample-level classification based on rules
        # 1. If segVal > ploidy + 2, sample is at least noncircular
        # 2. If there's any ecDNA cargo gene, sample is circular
        # 3. If neither, sample is nofocal
        
        # Group by sample to calculate sample-level classification
        sample_id_col = self.config['data']['sample_id']
        sample_groups = results.groupby(sample_id_col)
        sample_classifications = {}
        
        for sample, group in sample_groups:
            # Check if any gene is predicted as ecDNA cargo
            has_ecdna_cargo = any(group['prediction'] == 1)
            
            # For now, we'll assume segVal > ploidy + 2 is False by default
            # In practice, we would need to load this information from the data
            has_segval_threshold = False
            
            # Apply sample classification rules
            if has_ecdna_cargo:
                sample_classifications[sample] = 'circular'
            elif has_segval_threshold:
                sample_classifications[sample] = 'noncircular'
            else:
                sample_classifications[sample] = 'nofocal'
        
        # Add sample-level classification to results
        results['sample_level_prediction_label'] = results[sample_id_col].map(sample_classifications)
        
        # Map sample classification to numerical values
        class_mapping = {'nofocal': 0, 'noncircular': 1, 'circular': 2}
        results['sample_level_prediction'] = results['sample_level_prediction_label'].map(class_mapping)
        
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
