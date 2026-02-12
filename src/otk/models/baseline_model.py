import torch
import torch.nn as nn
import yaml

class BaselineModel(nn.Module):
    """Baseline model for ecdna prediction"""
    def __init__(self, config):
        super(BaselineModel, self).__init__()
        
        # Get input dimension from config
        input_dim = config['model']['architecture']['layers'][0]['input_dim']
        
        # Simple MLP architecture
        self.model = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # Handle missing values by replacing NaNs with 0
        x = torch.nan_to_num(x, nan=0.0)
        
        # Forward pass
        output = self.model(x)
        
        return output

class ECDNA_Baseline_Model:
    """ECDNA baseline model wrapper"""
    def __init__(self, config_path):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        self.model = BaselineModel(self.config)
    
    def get_model(self):
        return self.model
    
    def save(self, path):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': self.config
        }, path)
    
    @classmethod
    def load(cls, path):
        checkpoint = torch.load(path)
        config = checkpoint['config']
        model = cls(config)
        model.model.load_state_dict(checkpoint['model_state_dict'])
        return model