import torch
import torch.nn as nn
import yaml
import os
import logging
from otk.models.baseline_model import BaselineModel, ECDNA_Baseline_Model
from otk.models.improved_model import ImprovedModel
from otk.models.improved_model_v2 import ImprovedModelV2, ImprovedModelV2_Deep
from otk.models.transformer_ecdna_model import TransformerEcDNAModel, EnhancedTransformerEcDNAModel, LightweightTransformerEcDNAModel
from otk.models.advanced_ecdna_model import AdvancedEcDNAModel, PrecisionFocusedEcDNAModel, EnsembleEcDNAModel
from otk.models.optimized_ecdna_model import OptimizedEcDNA, EnsembleOptimizedEcDNA
from otk.models.dgit_model import DeepGatedInteractionTransformer, EnsembleDGIT

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MLP(nn.Module):
    def __init__(self, config):
        super(MLP, self).__init__()
        self.config = config
        self.layers = nn.ModuleList()
        
        # Build the network layers
        architecture = config['model']['architecture']
        layers_config = architecture['layers']
        
        # Input layer
        input_dim = layers_config[0]['input_dim']
        
        # Hidden layers
        for i in range(1, len(layers_config) - 1):
            hidden_dim = layers_config[i]['hidden_dim']
            self.layers.append(nn.Linear(input_dim, hidden_dim))
            
            # Activation function
            activation = layers_config[i]['activation']
            if activation == 'relu':
                self.layers.append(nn.ReLU())
            elif activation == 'leaky_relu':
                self.layers.append(nn.LeakyReLU())
            elif activation == 'tanh':
                self.layers.append(nn.Tanh())
            elif activation == 'sigmoid':
                self.layers.append(nn.Sigmoid())
            
            # Dropout
            if 'dropout' in layers_config[i]:
                dropout = layers_config[i]['dropout']
                self.layers.append(nn.Dropout(dropout))
            
            input_dim = hidden_dim
        
        # Output layer
        output_dim = layers_config[-1]['output_dim']
        self.layers.append(nn.Linear(input_dim, output_dim))
        
        # Output activation
        output_activation = layers_config[-1]['activation']
        if output_activation == 'sigmoid':
            self.layers.append(nn.Sigmoid())
        elif output_activation == 'softmax':
            self.layers.append(nn.Softmax(dim=1))
    
    def forward(self, x):
        # Handle missing values by replacing NaNs with 0
        # This is a simple approach, but can be extended with more sophisticated methods
        x = torch.nan_to_num(x, nan=0.0)
        
        for layer in self.layers:
            x = layer(x)
        return x

class TransformerModel(nn.Module):
    def __init__(self, config):
        super(TransformerModel, self).__init__()
        self.config = config
        
        # Get input dimension
        input_dim = config['model']['architecture']['layers'][0]['input_dim']
        
        # Embedding dimension (must be divisible by nhead)
        d_model = 128  # Increased for better representation
        
        # Linear layer to map input to d_model
        self.embedding = nn.Linear(input_dim, d_model)
        
        # Layer normalization for better training stability
        self.layer_norm = nn.LayerNorm(d_model)
        
        # Transformer encoder layer with improved parameters
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=8,  # Increased for better attention
            dim_feedforward=512,  # Increased for better capacity
            dropout=0.4,  # Increased for better regularization
            activation='gelu',  # GELU for better performance
            batch_first=True  # Set batch_first=True for better performance
        )
        
        # Transformer encoder with more layers
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=3  # Increased for deeper representation
        )
        
        # Improved gene-level prediction head
        self.gene_level_head = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.GELU(),
            nn.Dropout(0.4),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Dropout(0.4),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        # Removed sample-level prediction head
        # Sample-level classification will be done based on gene-level predictions and rules
    
    def forward(self, x):
        # Handle missing values by replacing NaNs with 0
        x = torch.nan_to_num(x, nan=0.0)
        
        # Map input to d_model
        x = self.embedding(x)
        
        # Apply layer normalization
        x = self.layer_norm(x)
        
        # For transformer encoder, we need to add a sequence dimension
        # Since each sample is a single sequence, we add a sequence length of 1
        x = x.unsqueeze(1)
        
        # Pass through transformer encoder (batch_first=True)
        x = self.transformer_encoder(x)
        
        # Remove sequence dimension
        x = x.squeeze(1)
        
        # Gene-level prediction
        gene_level_output = self.gene_level_head(x)
        
        # Only return gene-level prediction
        return gene_level_output

class MultiInputTransformerModel(nn.Module):
    """Multi-input Transformer model with amplicon classification support"""
    def __init__(self, config):
        super(MultiInputTransformerModel, self).__init__()
        self.config = config
        
        # Get input dimension
        input_dim = config['model']['architecture']['layers'][0]['input_dim']
        
        # Embedding dimension (must be divisible by nhead)
        d_model = 256  # Increased for better representation
        
        # Linear layer to map gene features to d_model
        self.gene_embedding = nn.Linear(input_dim, d_model)
        
        # Amplicon classification embedding
        # Including nofocal class
        self.amplicon_embedding = nn.Embedding(4, 64)  # Increased embedding size
        
        # Layer normalization for better training stability
        self.layer_norm = nn.LayerNorm(d_model + 64)  # Combine gene and amplicon features
        
        # Transformer encoder layer with improved parameters
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model + 64,
            nhead=8,  # Increased for better attention
            dim_feedforward=1024,  # Increased for better capacity
            dropout=0.4,  # Increased for better regularization
            activation='gelu',  # GELU for better performance
            batch_first=True  # Set batch_first=True for better performance
        )
        
        # Transformer encoder with more layers
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=4  # Increased for deeper representation
        )
        
        # Improved gene-level prediction head with residual connection
        self.gene_level_head = nn.Sequential(
            nn.Linear(d_model + 64, 256),
            nn.GELU(),
            nn.BatchNorm1d(256),  # Added batch normalization
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.BatchNorm1d(128),  # Added batch normalization
            nn.Dropout(0.4),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
        # Removed sample-level prediction head
        # Sample-level classification will be done based on gene-level predictions and rules
    
    def forward(self, x, amplicon_class=None):
        # Handle missing values by replacing NaNs with 0
        x = torch.nan_to_num(x, nan=0.0)
        
        # Map gene features to d_model
        gene_embed = self.gene_embedding(x)
        
        # Add amplicon embedding if provided
        if amplicon_class is not None:
            amplicon_embed = self.amplicon_embedding(amplicon_class)
            # Combine gene and amplicon features
            x = torch.cat([gene_embed, amplicon_embed], dim=1)
        else:
            # If no amplicon class provided, use gene features only
            x = gene_embed
        
        # Apply layer normalization
        x = self.layer_norm(x)
        
        # For transformer encoder, we need to add a sequence dimension
        # Since each sample is a single sequence, we add a sequence length of 1
        x = x.unsqueeze(1)
        
        # Pass through transformer encoder (batch_first=True)
        x = self.transformer_encoder(x)
        
        # Remove sequence dimension
        x = x.squeeze(1)
        
        # Gene-level prediction
        gene_level_output = self.gene_level_head(x)
        
        # Only return gene-level prediction
        return gene_level_output

class ECDNA_Model:
    def __init__(self, config_path):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        self.model = self._build_model()
    
    def _build_model(self):
        """Build the model based on configuration"""
        model_type = self.config['model']['architecture']['type']
        if model_type == 'MLP':
            model = MLP(self.config)
        elif model_type == 'Transformer':
            model = TransformerModel(self.config)
        elif model_type == 'MultiInputTransformer':
            model = MultiInputTransformerModel(self.config)
        elif model_type == 'Baseline':
            model = BaselineModel(self.config)
        elif model_type == 'Improved':
            model = ImprovedModel(self.config)
        elif model_type == 'ImprovedV2':
            model = ImprovedModelV2(self.config)
        elif model_type == 'ImprovedV2_Deep':
            model = ImprovedModelV2_Deep(self.config)
        elif model_type == 'TransformerEcDNA':
            model = TransformerEcDNAModel(self.config)
        elif model_type == 'EnhancedTransformerEcDNA':
            model = EnhancedTransformerEcDNAModel(self.config)
        elif model_type == 'LightweightTransformerEcDNA':
            model = LightweightTransformerEcDNAModel(self.config)
        elif model_type == 'AdvancedEcDNA':
            model = AdvancedEcDNAModel(self.config)
        elif model_type == 'PrecisionFocusedEcDNA':
            model = PrecisionFocusedEcDNAModel(self.config)
        elif model_type == 'EnsembleEcDNA':
            model = EnsembleEcDNAModel(self.config)
        elif model_type == 'OptimizedEcDNA':
            model = OptimizedEcDNA(self.config)
        elif model_type == 'EnsembleOptimizedEcDNA':
            model = EnsembleOptimizedEcDNA(self.config)
        elif model_type == 'DeepGatedInteractionTransformer':
            from otk.models.dgit_model import DeepGatedInteractionTransformer
            model = DeepGatedInteractionTransformer(self.config)
        elif model_type == 'EnsembleDGIT':
            from otk.models.dgit_model import EnsembleDGIT
            model = EnsembleDGIT(self.config)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        return model
    
    def get_model(self):
        """Get the built model"""
        return self.model
    
    def save(self, path, optimal_threshold=None):
        """Save the model"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        save_dict = {
            'model_state_dict': self.model.state_dict(),
            'config': self.config
        }
        if optimal_threshold is not None:
            save_dict['optimal_threshold'] = optimal_threshold
        
        torch.save(save_dict, path)
        logger.info(f"Model saved to {path}")
    
    @classmethod
    def load(cls, path, map_location=None):
        """Load the model from a saved file"""
        checkpoint = torch.load(path, map_location=map_location)
        
        # Create a temporary config file to initialize the model
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yml', delete=False) as f:
            yaml.dump(checkpoint['config'], f)
            temp_config_path = f.name
        
        try:
            model = cls(temp_config_path)
            model.model.load_state_dict(checkpoint['model_state_dict'])
        finally:
            os.unlink(temp_config_path)
        
        logger.info(f"Model loaded from {path}")
        return model
