import torch
import torch.nn as nn
import yaml

class FeatureEmbedding(nn.Module):
    """Feature embedding layer for different types of features"""
    def __init__(self, input_dim, embedding_dim):
        super(FeatureEmbedding, self).__init__()
        self.embedding = nn.Linear(input_dim, embedding_dim)
        self.layer_norm = nn.LayerNorm(embedding_dim)
        self.activation = nn.GELU()
    
    def forward(self, x):
        x = self.embedding(x)
        x = self.layer_norm(x)
        x = self.activation(x)
        return x

class ResidualBlock(nn.Module):
    """Residual block with layer normalization"""
    def __init__(self, hidden_dim, dropout=0.4):
        super(ResidualBlock, self).__init__()
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.layer_norm1 = nn.LayerNorm(hidden_dim)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.layer_norm2 = nn.LayerNorm(hidden_dim)
    
    def forward(self, x):
        residual = x
        out = self.fc1(x)
        out = self.layer_norm1(out)
        out = self.activation(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.layer_norm2(out)
        out += residual
        out = self.activation(out)
        return out

class AttentionLayer(nn.Module):
    """Attention layer to focus on important features"""
    def __init__(self, hidden_dim):
        super(AttentionLayer, self).__init__()
        self.attention = nn.Linear(hidden_dim, 1)
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x):
        # Compute attention weights
        attn_weights = self.attention(x)
        attn_weights = self.softmax(attn_weights)
        
        # Apply attention to features
        weighted_x = x * attn_weights
        return weighted_x, attn_weights

class MultiScaleFeatureFusion(nn.Module):
    """Multi-scale feature fusion module"""
    def __init__(self, hidden_dim):
        super(MultiScaleFeatureFusion, self).__init__()
        self.scale1 = nn.Linear(hidden_dim, hidden_dim)
        self.scale2 = nn.Linear(hidden_dim, hidden_dim)
        self.scale3 = nn.Linear(hidden_dim, hidden_dim)
        self.fusion = nn.Linear(hidden_dim * 3, hidden_dim)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.activation = nn.GELU()
    
    def forward(self, x):
        # Different scales of feature processing
        scale1 = self.activation(self.scale1(x))
        scale2 = self.activation(self.scale2(scale1))
        scale3 = self.activation(self.scale3(scale2))
        
        # Fusion
        fused = torch.cat([scale1, scale2, scale3], dim=1)
        fused = self.fusion(fused)
        fused = self.layer_norm(fused)
        fused = self.activation(fused)
        return fused

class ImprovedModel(nn.Module):
    """Improved model for ecDNA prediction with high auPRC"""
    def __init__(self, config):
        super(ImprovedModel, self).__init__()
        self.config = config
        
        # Get input dimension
        input_dim = config['model']['architecture']['layers'][0]['input_dim']
        
        # Embedding dimension
        embedding_dim = 256
        hidden_dim = 512
        
        # Feature embedding
        self.feature_embedding = FeatureEmbedding(input_dim, embedding_dim)
        
        # Attention layer
        self.attention = AttentionLayer(embedding_dim)
        
        # Multi-scale feature fusion
        self.feature_fusion = MultiScaleFeatureFusion(embedding_dim)
        
        # Residual blocks
        self.residual_blocks = nn.ModuleList()
        for _ in range(4):  # Deeper network with 4 residual blocks
            self.residual_blocks.append(ResidualBlock(hidden_dim))
        
        # Transition layer
        self.transition = nn.Linear(embedding_dim, hidden_dim)
        
        # Output head
        self.output_head = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.GELU(),
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(0.4),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # Handle missing values
        x = torch.nan_to_num(x, nan=0.0)
        
        # Feature embedding
        x = self.feature_embedding(x)
        
        # Apply attention
        x, attn_weights = self.attention(x)
        
        # Multi-scale feature fusion
        x = self.feature_fusion(x)
        
        # Transition to hidden dimension
        x = self.transition(x)
        
        # Residual blocks
        for block in self.residual_blocks:
            x = block(x)
        
        # Output prediction
        out = self.output_head(x)
        
        return out
