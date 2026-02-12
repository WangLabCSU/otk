import torch
import torch.nn as nn
import torch.nn.functional as F

class ImprovedModelV2(nn.Module):
    """
    Improved model for ecDNA prediction with proper architecture design
    
    Key improvements:
    1. Proper dimension handling throughout the network
    2. Residual connections for better gradient flow
    3. Attention mechanism for feature importance
    4. Batch normalization for training stability
    5. Proper regularization with dropout
    """
    
    def __init__(self, config):
        super(ImprovedModelV2, self).__init__()
        self.config = config
        
        # Get input dimension
        input_dim = config['model']['architecture']['layers'][0]['input_dim']
        
        # Network dimensions
        self.embedding_dim = 128
        self.hidden_dim = 256
        
        # Feature embedding layer
        self.feature_embedding = nn.Sequential(
            nn.Linear(input_dim, self.embedding_dim),
            nn.BatchNorm1d(self.embedding_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # Attention mechanism for feature importance
        self.attention = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim // 2),
            nn.Tanh(),
            nn.Linear(self.embedding_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # First hidden layer with residual connection preparation
        self.hidden1 = nn.Sequential(
            nn.Linear(self.embedding_dim, self.hidden_dim),
            nn.BatchNorm1d(self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.4)
        )
        
        # Residual blocks
        self.residual_blocks = nn.ModuleList([
            self._make_residual_block(self.hidden_dim) for _ in range(3)
        ])
        
        # Second hidden layer
        self.hidden2 = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.BatchNorm1d(self.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # Output layer
        self.output = nn.Sequential(
            nn.Linear(self.hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _make_residual_block(self, dim):
        """Create a residual block"""
        return nn.ModuleDict({
            'fc1': nn.Linear(dim, dim),
            'bn1': nn.BatchNorm1d(dim),
            'fc2': nn.Linear(dim, dim),
            'bn2': nn.BatchNorm1d(dim),
            'dropout': nn.Dropout(0.3)
        })
    
    def _residual_forward(self, x, block):
        """Forward pass through a residual block"""
        residual = x
        out = block['fc1'](x)
        out = block['bn1'](out)
        out = F.relu(out)
        out = block['dropout'](out)
        out = block['fc2'](out)
        out = block['bn2'](out)
        out += residual  # Residual connection
        out = F.relu(out)
        return out
    
    def _initialize_weights(self):
        """Initialize weights using Xavier initialization"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, x):
        # Handle missing values
        x = torch.nan_to_num(x, nan=0.0)
        
        # Feature embedding
        x = self.feature_embedding(x)
        
        # Apply attention
        attn_weights = self.attention(x)
        x = x * attn_weights
        
        # First hidden layer
        x = self.hidden1(x)
        
        # Residual blocks
        for block in self.residual_blocks:
            x = self._residual_forward(x, block)
        
        # Second hidden layer
        x = self.hidden2(x)
        
        # Output
        out = self.output(x)
        
        return out


class ImprovedModelV2_Deep(nn.Module):
    """
    Deeper version of ImprovedModelV2 for better performance
    """
    
    def __init__(self, config):
        super(ImprovedModelV2_Deep, self).__init__()
        self.config = config
        
        # Get input dimension
        input_dim = config['model']['architecture']['layers'][0]['input_dim']
        
        # Network dimensions
        self.embedding_dim = 256
        self.hidden_dim = 512
        
        # Feature embedding layer
        self.feature_embedding = nn.Sequential(
            nn.Linear(input_dim, self.embedding_dim),
            nn.BatchNorm1d(self.embedding_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim // 4),
            nn.Tanh(),
            nn.Linear(self.embedding_dim // 4, self.embedding_dim),
            nn.Sigmoid()
        )
        
        # Hidden layers with increasing complexity
        self.hidden1 = nn.Sequential(
            nn.Linear(self.embedding_dim, self.hidden_dim),
            nn.BatchNorm1d(self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.4)
        )
        
        # Multiple residual blocks
        self.residual_blocks = nn.ModuleList([
            self._make_residual_block(self.hidden_dim) for _ in range(5)
        ])
        
        # Progressive dimension reduction
        self.hidden2 = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.BatchNorm1d(self.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.35)
        )
        
        self.hidden3 = nn.Sequential(
            nn.Linear(self.hidden_dim // 2, self.hidden_dim // 4),
            nn.BatchNorm1d(self.hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # Output layer
        self.output = nn.Sequential(
            nn.Linear(self.hidden_dim // 4, 1),
            nn.Sigmoid()
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _make_residual_block(self, dim):
        """Create a residual block"""
        return nn.ModuleDict({
            'fc1': nn.Linear(dim, dim),
            'bn1': nn.BatchNorm1d(dim),
            'fc2': nn.Linear(dim, dim),
            'bn2': nn.BatchNorm1d(dim),
            'dropout': nn.Dropout(0.3)
        })
    
    def _residual_forward(self, x, block):
        """Forward pass through a residual block"""
        residual = x
        out = block['fc1'](x)
        out = block['bn1'](out)
        out = F.relu(out)
        out = block['dropout'](out)
        out = block['fc2'](out)
        out = block['bn2'](out)
        out += residual
        out = F.relu(out)
        return out
    
    def _initialize_weights(self):
        """Initialize weights using Xavier initialization"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, x):
        # Handle missing values
        x = torch.nan_to_num(x, nan=0.0)
        
        # Feature embedding
        x = self.feature_embedding(x)
        
        # Apply attention
        attn_weights = self.attention(x)
        x = x * attn_weights
        
        # First hidden layer
        x = self.hidden1(x)
        
        # Residual blocks
        for block in self.residual_blocks:
            x = self._residual_forward(x, block)
        
        # Progressive dimension reduction
        x = self.hidden2(x)
        x = self.hidden3(x)
        
        # Output
        out = self.output(x)
        
        return out
