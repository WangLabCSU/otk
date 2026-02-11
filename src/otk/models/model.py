import torch
import torch.nn as nn
import yaml
import os
import logging

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
            dropout=0.3,  # Increased for better regularization
            activation='gelu'  # GELU for better performance
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
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        # Improved sample-level prediction head
        self.sample_level_head = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(64, 3),  # 3 classes: nofocal, noncircular, circular
            nn.Softmax(dim=1)
        )
    
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
        
        # Transformer expects sequence first
        x = x.transpose(0, 1)
        
        # Pass through transformer encoder
        x = self.transformer_encoder(x)
        
        # Back to batch first
        x = x.transpose(0, 1)
        
        # Remove sequence dimension
        x = x.squeeze(1)
        
        # Gene-level prediction
        gene_level_output = self.gene_level_head(x)
        
        # Sample-level prediction
        sample_level_output = self.sample_level_head(x)
        
        return gene_level_output, sample_level_output

class MultiInputTransformerModel(nn.Module):
    """Multi-input Transformer model with amplicon classification support"""
    def __init__(self, config):
        super(MultiInputTransformerModel, self).__init__()
        self.config = config
        
        # Get input dimension
        input_dim = config['model']['architecture']['layers'][0]['input_dim']
        
        # Embedding dimension (must be divisible by nhead)
        d_model = 128  # Increased for better representation
        
        # Linear layer to map gene features to d_model
        self.gene_embedding = nn.Linear(input_dim, d_model)
        
        # Amplicon classification embedding
        # Assuming 4 possible classes: Circular, Non-circular, etc.
        self.amplicon_embedding = nn.Embedding(4, 32)
        
        # Layer normalization for better training stability
        self.layer_norm = nn.LayerNorm(d_model + 32)  # Combine gene and amplicon features
        
        # Transformer encoder layer with improved parameters
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model + 32,
            nhead=8,  # Increased for better attention
            dim_feedforward=512,  # Increased for better capacity
            dropout=0.3,  # Increased for better regularization
            activation='gelu'  # GELU for better performance
        )
        
        # Transformer encoder with more layers
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=3  # Increased for deeper representation
        )
        
        # Improved gene-level prediction head
        self.gene_level_head = nn.Sequential(
            nn.Linear(d_model + 32, 128),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        # Improved sample-level prediction head
        self.sample_level_head = nn.Sequential(
            nn.Linear(d_model + 32, 128),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(64, 3),  # 3 classes: nofocal, noncircular, circular
            nn.Softmax(dim=1)
        )
    
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
        
        # Transformer expects sequence first
        x = x.transpose(0, 1)
        
        # Pass through transformer encoder
        x = self.transformer_encoder(x)
        
        # Back to batch first
        x = x.transpose(0, 1)
        
        # Remove sequence dimension
        x = x.squeeze(1)
        
        # Gene-level prediction
        gene_level_output = self.gene_level_head(x)
        
        # Sample-level prediction
        sample_level_output = self.sample_level_head(x)
        
        return gene_level_output, sample_level_output

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
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        return model
    
    def get_model(self):
        """Get the built model"""
        return self.model
    
    def save(self, path):
        """Save the model"""
        # Ensure the directory exists
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Save the model state dict and configuration
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': self.config
        }, path)
        print(f"Model saved to {path}")
    
    @classmethod
    def load(cls, path):
        """Load the model from a saved file"""
        checkpoint = torch.load(path)
        
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
        
        print(f"Model loaded from {path}")
        return model
