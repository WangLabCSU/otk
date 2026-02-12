import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class FeaturePyramidNetwork(nn.Module):
    """
    Feature Pyramid Network for multi-scale feature representation
    """
    def __init__(self, input_dim, levels=3):
        super(FeaturePyramidNetwork, self).__init__()
        self.levels = levels
        
        # Progressive downsampling
        self.downsample = nn.ModuleList()
        for i in range(levels):
            in_dim = input_dim // (2 ** i)
            out_dim = input_dim // (2 ** (i + 1))
            self.downsample.append(nn.Sequential(
                nn.Linear(in_dim, out_dim),
                nn.LayerNorm(out_dim),
                nn.GELU(),
                nn.Dropout(0.3)
            ))
        
        # Lateral connections
        self.lateral = nn.ModuleList()
        for i in range(levels):
            in_dim = input_dim // (2 ** (i + 1))
            out_dim = input_dim // 4
            self.lateral.append(nn.Linear(in_dim, out_dim))
    
    def forward(self, x):
        features = []
        current = x
        
        # Generate features at different scales
        for i in range(self.levels):
            current = self.downsample[i](current)
            features.append(current)
        
        # Upsample and fuse features
        fused = []
        for i in range(self.levels):
            lateral_out = self.lateral[i](features[i])
            fused.append(lateral_out)
        
        # Concatenate all fused features
        return torch.cat(fused, dim=1)

class EnhancedTransformerEncoder(nn.Module):
    """
    Enhanced transformer encoder with multi-head attention and feed-forward network
    """
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, layer_norm_eps=1e-5):
        super(EnhancedTransformerEncoder, self).__init__()
        # Multi-head attention with improved initialization
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        
        # Feed-forward network with GELU activation
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        
        # Layer normalization with eps
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
        # GELU activation for better performance
        self.activation = nn.GELU()
        
        # Initialize weights properly
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights with Xavier uniform and zeros for biases"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        # Multi-head attention with residual connection
        src2 = self.self_attn(src, src, src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        
        # Feed-forward network with residual connection
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        
        return src

class CNNTransformerHybrid(nn.Module):
    """
    Hybrid model combining CNN for local feature extraction and Transformer for global interactions
    """
    def __init__(self, input_dim, cnn_channels=64, transformer_dim=128, nhead=8, num_layers=4):
        super(CNNTransformerHybrid, self).__init__()
        
        # CNN for local feature extraction
        self.cnn = nn.Sequential(
            nn.Linear(input_dim, cnn_channels),
            nn.LayerNorm(cnn_channels),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(cnn_channels, cnn_channels * 2),
            nn.LayerNorm(cnn_channels * 2),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(cnn_channels * 2, transformer_dim),
            nn.LayerNorm(transformer_dim),
            nn.GELU(),
            nn.Dropout(0.3)
        )
        
        # Transformer encoder for global interactions
        encoder_layers = nn.ModuleList()
        for i in range(num_layers):
            encoder_layer = EnhancedTransformerEncoder(
                d_model=transformer_dim,
                nhead=nhead,
                dim_feedforward=transformer_dim * 4,
                dropout=0.3 if i < num_layers - 1 else 0.1,
                layer_norm_eps=1e-5
            )
            encoder_layers.append(encoder_layer)
        self.transformer_encoder = nn.Sequential(*encoder_layers)
    
    def forward(self, x):
        # CNN feature extraction
        x = self.cnn(x)
        
        # Add sequence dimension for transformer
        x = x.unsqueeze(1)  # Shape: (batch_size, 1, transformer_dim)
        
        # Transformer global interactions
        x = self.transformer_encoder(x)
        
        # Remove sequence dimension
        x = x.squeeze(1)
        
        return x

class AdvancedEcDNAModel(nn.Module):
    """
    Advanced model for ecDNA cargo gene prediction
    
    Key features:
    1. Hybrid CNN-Transformer architecture for both local and global feature learning
    2. Feature Pyramid Network for multi-scale feature representation
    3. Multi-head attention with improved initialization
    4. Progressive dimension reduction with layer normalization
    5. Task-specific output head with precision-focused design
    6. Comprehensive dropout and regularization strategies
    """
    
    def __init__(self, config):
        super(AdvancedEcDNAModel, self).__init__()
        self.config = config
        
        # Get input dimension
        input_dim = config['model']['architecture']['layers'][0]['input_dim']
        
        # Model parameters
        self.cnn_channels = 128
        self.transformer_dim = 256
        self.nhead = 16
        self.transformer_layers = 4
        self.dropout = 0.3
        
        # Feature extraction backbone
        self.backbone = CNNTransformerHybrid(
            input_dim=input_dim,
            cnn_channels=self.cnn_channels,
            transformer_dim=self.transformer_dim,
            nhead=self.nhead,
            num_layers=self.transformer_layers
        )
        
        # Feature pyramid network for multi-scale representation
        self.fpn = FeaturePyramidNetwork(self.transformer_dim)
        
        # Progressive dimension reduction
        self.reduction = nn.Sequential(
            nn.Linear(self.transformer_dim // 4 * 3, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(self.dropout * 0.8),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(self.dropout * 0.6),
            nn.Linear(64, 32),
            nn.LayerNorm(32),
            nn.GELU(),
            nn.Dropout(self.dropout * 0.4)
        )
        
        # Task-specific output head with precision focus
        self.output_head = nn.Sequential(
            nn.Linear(32, 16),
            nn.LayerNorm(16),
            nn.GELU(),
            nn.Dropout(self.dropout * 0.2),
            nn.Linear(16, 1)
            # No sigmoid - using BCEWithLogitsLoss
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights with proper scaling"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, x):
        # Handle missing values
        x = torch.nan_to_num(x, nan=0.0)
        
        # Feature extraction
        x = self.backbone(x)
        
        # Multi-scale feature representation
        x = self.fpn(x)
        
        # Progressive dimension reduction
        x = self.reduction(x)
        
        # Task-specific output
        out = self.output_head(x)
        
        return out

class PrecisionFocusedEcDNAModel(nn.Module):
    """
    Specialized model focused on high precision while maintaining good recall
    """
    
    def __init__(self, config):
        super(PrecisionFocusedEcDNAModel, self).__init__()
        self.config = config
        
        # Get input dimension
        input_dim = config['model']['architecture']['layers'][0]['input_dim']
        
        # Model parameters
        self.hidden_dim = 512
        self.num_layers = 5
        self.dropout = 0.3
        
        # Create a deep network with residual connections
        layers = []
        in_dim = input_dim
        
        for i in range(self.num_layers):
            # Progressive dimension reduction
            out_dim = self.hidden_dim // (2 ** i) if i < 3 else 64
            
            layer = nn.Sequential(
                nn.Linear(in_dim, out_dim),
                nn.LayerNorm(out_dim),
                nn.GELU(),
                nn.Dropout(self.dropout * (0.8 - i * 0.1))
            )
            layers.append(layer)
            in_dim = out_dim
        
        self.network = nn.ModuleList(layers)
        
        # Residual connections
        self.residuals = nn.ModuleList()
        for i in range(self.num_layers - 1):
            in_dim_res = self.hidden_dim // (2 ** i) if i < 3 else 64
            out_dim_res = self.hidden_dim // (2 ** (i + 1)) if i + 1 < 3 else 64
            if in_dim_res != out_dim_res:
                self.residuals.append(nn.Linear(in_dim_res, out_dim_res))
            else:
                self.residuals.append(nn.Identity())
        
        # Precision-focused output head
        self.output_head = nn.Sequential(
            nn.Linear(64, 32),
            nn.LayerNorm(32),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1)
            # No sigmoid - using BCEWithLogitsLoss
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights with proper scaling"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, x):
        # Handle missing values
        x = torch.nan_to_num(x, nan=0.0)
        
        # Forward pass with residual connections
        for i, layer in enumerate(self.network):
            if i == 0:
                out = layer(x)
            else:
                residual = self.residuals[i-1](out)
                out = layer(out) + residual
        
        # Precision-focused output
        out = self.output_head(out)
        
        return out

class EnsembleEcDNAModel(nn.Module):
    """
    Ensemble model combining multiple architectures for improved performance
    """
    
    def __init__(self, config):
        super(EnsembleEcDNAModel, self).__init__()
        self.config = config
        
        # Create multiple base models
        self.models = nn.ModuleList([
            AdvancedEcDNAModel(config),
            PrecisionFocusedEcDNAModel(config)
        ])
        
        # Ensemble layer
        self.ensemble = nn.Sequential(
            nn.Linear(len(self.models), 1)
            # No sigmoid - using BCEWithLogitsLoss
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights for ensemble layer"""
        if hasattr(self, 'ensemble'):
            for m in self.ensemble.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
    
    def forward(self, x):
        # Get predictions from all base models
        predictions = []
        for model in self.models:
            pred = model(x)
            predictions.append(pred)
        
        # Concatenate predictions
        predictions = torch.cat(predictions, dim=1)
        
        # Ensemble prediction
        out = self.ensemble(predictions)
        
        return out
