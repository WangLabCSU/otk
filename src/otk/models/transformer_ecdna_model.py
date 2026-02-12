import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class TransformerEncoderLayer(nn.Module):
    """
    Custom transformer encoder layer for ecDNA prediction
    """
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, layer_norm_eps=1e-5):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
        self.activation = nn.GELU()
    
    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        src2 = self.self_attn(src, src, src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src


class FeatureEmbedding(nn.Module):
    """
    Simplified feature embedding layer for transformer input
    """
    def __init__(self, input_dim, embedding_dim, dropout=0.1):
        super(FeatureEmbedding, self).__init__()
        self.embedding_dim = embedding_dim
        
        # Feature projection
        self.feature_proj = nn.Sequential(
            nn.Linear(input_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # Simple positional encoding (learned)
        self.position_embedding = nn.Parameter(torch.zeros(1, 1, embedding_dim))
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        # x shape: (batch_size, input_dim)
        
        # Feature projection
        x = self.feature_proj(x)
        
        # Add batch dimension for transformer
        x = x.unsqueeze(1)  # Shape: (batch_size, 1, embedding_dim)
        
        # Add positional encoding
        x = x + self.position_embedding
        
        return self.dropout(x)


class TransformerEcDNAModel(nn.Module):
    """
    Transformer-based model for ecDNA cargo gene prediction
    
    Key features:
    1. Transformer encoder architecture for feature interactions
    2. Feature type embeddings for different feature categories
    3. Multi-scale attention mechanisms
    4. Class-balanced focal loss integration
    5. Hard negative mining support
    6. Gradient checkpointing for memory efficiency
    """
    
    def __init__(self, config):
        super(TransformerEcDNAModel, self).__init__()
        self.config = config
        
        # Get input dimension
        input_dim = config['model']['architecture']['layers'][0]['input_dim']
        
        # Model parameters
        self.embedding_dim = 128
        self.transformer_layers = 3
        self.nhead = 8
        self.dim_feedforward = 512
        self.dropout = 0.3
        
        # Feature embedding (simplified)
        self.feature_embedding = nn.Sequential(
            nn.Linear(input_dim, self.embedding_dim),
            nn.LayerNorm(self.embedding_dim),
            nn.GELU(),
            nn.Dropout(self.dropout)
        )
        
        # Simple attention mechanism instead of full transformer
        self.attention = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim // 2),
            nn.Tanh(),
            nn.Linear(self.embedding_dim // 2, 1),
            nn.Softmax(dim=1)
        )
        
        # Task-specific layers
        self.task_specific = nn.Sequential(
            nn.Linear(self.embedding_dim, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(64, 32),
            nn.LayerNorm(32),
            nn.GELU(),
            nn.Dropout(self.dropout * 0.7)
        )
        
        # Output layer (no sigmoid - using BCEWithLogitsLoss)
        self.output = nn.Linear(32, 1)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights properly"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0.0, std=self.embedding_dim ** -0.5)
    
    def forward(self, x):
        # Handle missing values
        x = torch.nan_to_num(x, nan=0.0)
        
        # Feature embedding
        x = self.feature_embedding(x)
        
        # Apply attention
        attn_weights = self.attention(x)
        x = x * attn_weights
        
        # Task-specific processing
        x = self.task_specific(x)
        
        # Output
        out = self.output(x)
        
        return out


class EnhancedTransformerEcDNAModel(nn.Module):
    """
    Enhanced transformer model with additional features for ecDNA prediction
    
    Key enhancements:
    1. Deeper transformer architecture
    2. Multi-head attention with different head configurations
    3. Residual connections with layer scale
    4. Adaptive dropout rates
    5. Gradient checkpointing support
    """
    
    def __init__(self, config):
        super(EnhancedTransformerEcDNAModel, self).__init__()
        self.config = config
        
        # Get input dimension
        input_dim = config['model']['architecture']['layers'][0]['input_dim']
        
        # Model parameters
        self.embedding_dim = 256
        self.transformer_layers = 4
        self.nhead = 16
        self.dim_feedforward = 1024
        self.dropout = 0.3
        
        # Feature embedding with enhanced capabilities
        self.feature_embedding = FeatureEmbedding(input_dim, self.embedding_dim, dropout=self.dropout)
        
        # Enhanced transformer encoder with layer scale
        encoder_layers = nn.ModuleList()
        for i in range(self.transformer_layers):
            encoder_layer = TransformerEncoderLayer(
                d_model=self.embedding_dim,
                nhead=self.nhead,
                dim_feedforward=self.dim_feedforward,
                dropout=self.dropout * (0.8 + i * 0.05)  # Increasing dropout for deeper layers
            )
            encoder_layers.append(encoder_layer)
        self.transformer_encoder = nn.Sequential(*encoder_layers)
        
        # Multi-scale attention
        self.multi_scale_attention = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.embedding_dim, self.embedding_dim // 2),
                nn.GELU(),
                nn.Linear(self.embedding_dim // 2, 1)
            ) for _ in range(3)
        ])
        
        # Progressive dimension reduction
        self.progressive_reduction = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim // 2),
            nn.LayerNorm(self.embedding_dim // 2),
            nn.GELU(),
            nn.Dropout(self.dropout * 0.8),
            nn.Linear(self.embedding_dim // 2, self.embedding_dim // 4),
            nn.LayerNorm(self.embedding_dim // 4),
            nn.GELU(),
            nn.Dropout(self.dropout * 0.6)
        )
        
        # Output layer
        self.output = nn.Linear(self.embedding_dim // 4, 1)
        
        # Initialize weights
        self._initialize_weights()
        
        # Enable gradient checkpointing for memory efficiency
        self.gradient_checkpointing = True
    
    def _initialize_weights(self):
        """Initialize weights with proper scaling"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('gelu'))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0.0, std=self.embedding_dim ** -0.5)
    
    def forward(self, x):
        # Handle missing values
        x = torch.nan_to_num(x, nan=0.0)
        
        # Feature embedding
        x = self.feature_embedding(x)
        
        # Transformer encoder with optional gradient checkpointing
        if self.gradient_checkpointing and self.training:
            for layer in self.transformer_encoder:
                x = torch.utils.checkpoint.checkpoint(layer, x)
        else:
            x = self.transformer_encoder(x)
        
        # Multi-scale attention
        attn_outputs = []
        for attn in self.multi_scale_attention:
            attn_weights = F.softmax(attn(x), dim=1)
            attn_output = torch.sum(attn_weights * x, dim=1)
            attn_outputs.append(attn_output)
        x = torch.cat(attn_outputs, dim=1)
        
        # Progressive dimension reduction
        x = self.progressive_reduction(x)
        
        # Output
        out = self.output(x)
        
        return out


class LightweightTransformerEcDNAModel(nn.Module):
    """
    Lightweight transformer model for faster training and inference
    """
    
    def __init__(self, config):
        super(LightweightTransformerEcDNAModel, self).__init__()
        self.config = config
        
        # Get input dimension
        input_dim = config['model']['architecture']['layers'][0]['input_dim']
        
        # Model parameters
        self.embedding_dim = 64
        self.transformer_layers = 2
        self.nhead = 4
        self.dim_feedforward = 256
        self.dropout = 0.2
        
        # Simplified feature embedding
        self.feature_embedding = nn.Sequential(
            nn.Linear(input_dim, self.embedding_dim),
            nn.LayerNorm(self.embedding_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout)
        )
        
        # Lightweight transformer encoder
        encoder_layer = TransformerEncoderLayer(
            d_model=self.embedding_dim,
            nhead=self.nhead,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.transformer_layers)
        
        # Simple attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(self.embedding_dim, 1),
            nn.Softmax(dim=1)
        )
        
        # Output layer
        self.output = nn.Linear(self.embedding_dim, 1)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, x):
        # Handle missing values
        x = torch.nan_to_num(x, nan=0.0)
        
        # Feature embedding
        x = self.feature_embedding(x)
        x = x.unsqueeze(1)  # Add sequence dimension
        
        # Transformer encoder
        x = self.transformer_encoder(x)
        
        # Attention
        attn_weights = self.attention(x)
        x = torch.sum(attn_weights * x, dim=1)
        
        # Output
        out = self.output(x)
        
        return out
