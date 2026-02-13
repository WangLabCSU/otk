import torch
import pandas as pd
import numpy as np
import os
import yaml
import logging
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

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
        self.optimal_threshold = None
        
        # Set device
        if torch.cuda.is_available() and gpu >= 0:
            self.device = torch.device(f'cuda:{gpu}')
            logger.info(f"Using GPU: {gpu}")
        else:
            self.device = torch.device('cpu')
            logger.info("Using CPU")
        
        # Load model and config together
        checkpoint = torch.load(model_path, map_location=self.device)
        self.config = checkpoint['config']
        self.optimal_threshold = checkpoint.get('optimal_threshold')
        if self.optimal_threshold is not None:
            logger.info(f"Using optimal threshold from model: {self.optimal_threshold:.4f}")
        
        # Build model architecture
        model = self._build_model(self.config)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        model.eval()
        self.model = model
        logger.info(f"Model loaded from {self.model_path}")
        
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
            logger.info(f"Loading gene frequencies from {gene_freq_path}")
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
            logger.info(f"Loaded frequencies for {len(gene_freq_dict)} genes")
            return gene_freq_dict
        else:
            logger.warning(f"Gene frequencies file not found at {gene_freq_path}")
            return {}
    
    def _build_model(self, config):
        """Build the model based on configuration"""
        from otk.models.model import MLP, TransformerModel, MultiInputTransformerModel, BaselineModel, ImprovedModel
        from otk.models.improved_model_v2 import ImprovedModelV2, ImprovedModelV2_Deep
        from otk.models.transformer_ecdna_model import TransformerEcDNAModel, EnhancedTransformerEcDNAModel, LightweightTransformerEcDNAModel
        from otk.models.advanced_ecdna_model import AdvancedEcDNAModel, PrecisionFocusedEcDNAModel, EnsembleEcDNAModel
        model_type = config['model']['architecture']['type']
        if model_type == 'MLP':
            return MLP({'model': config['model']})
        elif model_type == 'Transformer':
            return TransformerModel({'model': config['model']})
        elif model_type == 'MultiInputTransformer':
            return MultiInputTransformerModel({'model': config['model']})
        elif model_type == 'Baseline':
            return BaselineModel(config)
        elif model_type == 'Improved':
            return ImprovedModel(config)
        elif model_type == 'ImprovedV2':
            return ImprovedModelV2(config)
        elif model_type == 'ImprovedV2_Deep':
            return ImprovedModelV2_Deep(config)
        elif model_type == 'TransformerEcDNA':
            return TransformerEcDNAModel(config)
        elif model_type == 'EnhancedTransformerEcDNA':
            return EnhancedTransformerEcDNAModel(config)
        elif model_type == 'LightweightTransformerEcDNA':
            return LightweightTransformerEcDNAModel(config)
        elif model_type == 'AdvancedEcDNA':
            return AdvancedEcDNAModel(config)
        elif model_type == 'PrecisionFocusedEcDNA':
            return PrecisionFocusedEcDNAModel(config)
        elif model_type == 'EnsembleEcDNA':
            return EnsembleEcDNAModel(config)
        elif model_type == 'OptimizedEcDNA':
            from otk.models.optimized_ecdna_model import OptimizedEcDNA
            return OptimizedEcDNA(config)
        elif model_type == 'EnsembleOptimizedEcDNA':
            from otk.models.optimized_ecdna_model import EnsembleOptimizedEcDNA
            return EnsembleOptimizedEcDNA(config)
        elif model_type == 'DeepGatedInteractionTransformer':
            from otk.models.dgit_model import DeepGatedInteractionTransformer
            return DeepGatedInteractionTransformer(config)
        elif model_type == 'EnsembleDGIT':
            from otk.models.dgit_model import EnsembleDGIT
            return EnsembleDGIT(config)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
    
    def load_data(self, data_path):
        """Load data for prediction"""
        logger.info(f"Loading data from {data_path}")
        df = pd.read_csv(data_path)
        logger.info(f"Data loaded successfully with shape: {df.shape}")
        return df
    
    def preprocess(self, df):
        """Preprocess data for prediction"""
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
        for i in range(1, 20):
            DEFAULT_VALUES[f'CN{i}'] = 0.05
        
        required_features = self.config['data']['features']
        missing_cols = [col for col in required_features if col not in df.columns]
        
        if missing_cols:
            logger.info(f"Filling missing columns with default values: {missing_cols}")
            for col in missing_cols:
                if col in DEFAULT_VALUES:
                    df[col] = DEFAULT_VALUES[col]
                elif col.startswith('type_'):
                    df[col] = 0
                elif col.startswith('freq_'):
                    df[col] = 0
                else:
                    df[col] = 0
                    logger.warning(f"No default value for column '{col}', using 0")
        
        if 'age' in df.columns:
            if df['age'].isnull().sum() > 0:
                strategy = self.config['data']['missing_value_strategy'].get('age', 'mean')
                if strategy == 'mean':
                    df['age'] = df['age'].fillna(df['age'].mean())
                elif strategy == 'median':
                    df['age'] = df['age'].fillna(df['age'].median())
                elif strategy == 'mode':
                    df['age'] = df['age'].fillna(df['age'].mode()[0])
                logger.info(f"Handled missing values in 'age' column using {strategy} strategy")
        
        if 'gene_id' in df.columns:
            logger.info("Adding gene level frequency features...")
            for gene_freq_col in ['freq_Linear', 'freq_BFB', 'freq_Circular', 'freq_HR']:
                df[gene_freq_col] = df['gene_id'].apply(lambda x: self.gene_freqs.get(x, {}).get(gene_freq_col, 0) if self.gene_freqs else 0)
            logger.info("Gene frequency features added successfully")
        else:
            logger.warning("'gene_id' column not found, cannot add gene frequency features")
        
        if 'type' in df.columns:
            logger.info("Processing cancer type...")
            for cancer_type in self.cancer_types:
                col_name = f'type_{cancer_type}'
                df[col_name] = df['type'].apply(lambda x: 1 if str(x).strip() == cancer_type else 0)
            logger.info("Cancer type one-hot encoding completed")
        else:
            has_type_cols = any(col.startswith('type_') for col in df.columns)
            if not has_type_cols:
                logger.warning("No 'type' column and no type_* columns found, setting all type_* to 0")
                for cancer_type in self.cancer_types:
                    col_name = f'type_{cancer_type}'
                    if col_name not in df.columns:
                        df[col_name] = 0
        
        features = df[self.config['data']['features']]
        
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
        batch_size = self.config.get('prediction', {}).get('batch_size', 32)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        logger.info(f"Created DataLoader with batch size: {batch_size}")
        return dataloader
    
    def predict(self, dataloader):
        """Make predictions"""
        gene_predictions = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Predicting"):
                # Single-input case
                inputs = batch.to(self.device)
                outputs = self.model(inputs)
                
                # Apply sigmoid activation if model outputs logits
                # TransformerEcDNA models output logits, so we need to apply sigmoid
                if outputs.min() < 0 or outputs.max() > 1:
                    outputs = torch.sigmoid(outputs)
                
                # Get gene predictions
                gene_predictions.extend(outputs.cpu().detach().numpy())
        
        gene_predictions = np.array(gene_predictions)
        return gene_predictions
    
    def postprocess(self, predictions, sample_info=None, gene_info=None, data_path=None):
        """Postprocess predictions"""
        # Create a DataFrame with predictions
        results = pd.DataFrame()
        
        # Add sample and gene information if available
        if sample_info is not None:
            results[self.config['data']['sample_id']] = sample_info
        if gene_info is not None:
            results[self.config['data']['gene_id']] = gene_info
        
        # Add gene-level predictions
        threshold = self.optimal_threshold if self.optimal_threshold is not None else self.config.get('prediction', {}).get('threshold', 0.5)
        results['prediction_prob'] = predictions.flatten()
        results['prediction'] = (predictions.flatten() > threshold).astype(int)
        
        # Add sample-level classification based on rules
        # 1. If segVal > ploidy + 2, sample is at least noncircular
        # 2. If there's any ecDNA cargo gene, sample is circular
        # 3. If neither, sample is nofocal
        
        # Group by sample to calculate sample-level classification
        sample_id_col = self.config['data']['sample_id']
        
        # Get original data to access segVal and ploidy
        original_df = self.load_data(data_path)
        
        # Group by sample
        sample_groups = results.groupby(sample_id_col)
        sample_classifications = {}
        
        for sample, group in sample_groups:
            # Check if any gene is predicted as ecDNA cargo
            has_ecdna_cargo = any(group['prediction'] == 1)
            
            # Calculate has_segval_threshold from original data
            sample_data = original_df[original_df[sample_id_col] == sample]
            has_segval_threshold = False
            
            if not sample_data.empty:
                # Check if any gene in the sample has segVal > ploidy + 2
                if 'segVal' in sample_data.columns and 'ploidy' in sample_data.columns:
                    # Get unique ploidy value for the sample
                    ploidy_values = sample_data['ploidy'].dropna().unique()
                    if len(ploidy_values) > 0:
                        ploidy = ploidy_values[0]
                        # Check if any gene has segVal > ploidy + 2
                        has_segval_threshold = any(sample_data['segVal'] > (ploidy + 2))
                        logger.debug(f"Sample {sample}: ploidy = {ploidy}, has_segval_threshold = {has_segval_threshold}")
            
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
        results = self.postprocess(predictions, sample_info, gene_info, data_path)
        
        # Save results
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, 'predictions.csv')
        results.to_csv(output_path, index=False)
        logger.info(f"Predictions saved to {output_path}")
        
        # Sample-level predictions are already included in the main results file
        # No need for separate calculation as rules-based classification is already applied
        logger.info("Sample-level predictions included in main results file")
        
        return results
    


def predict(model_path, input_path, output_dir, gpu=-1):
    """Run prediction using the trained model"""
    predictor = Predictor(model_path, gpu)
    results = predictor.run(input_path, output_dir)
    return results
