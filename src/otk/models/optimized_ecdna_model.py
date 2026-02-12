import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any
import numpy as np


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
        inputs = torch.sigmoid(inputs)
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()
        dice = (2. * intersection + self.smooth) / (inputs.sum() + targets.sum() + self.smooth)
        
        return 1 - dice


class CombinedLoss(nn.Module):
    def __init__(self, focal_weight=0.4, dice_weight=0.3, bce_weight=0.3, 
                 alpha=0.75, gamma=2.0, pos_weight=100.0):
        super(CombinedLoss, self).__init__()
        self.focal_weight = focal_weight
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        
        self.focal_loss = FocalLoss(alpha=alpha, gamma=gamma)
        self.dice_loss = DiceLoss()
        self.pos_weight = pos_weight

    def forward(self, inputs, targets):
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        focal = self.focal_loss(inputs, targets)
        dice = self.dice_loss(inputs, targets)
        
        pos_weight = torch.tensor([self.pos_weight], device=inputs.device)
        bce = F.binary_cross_entropy_with_logits(inputs, targets, pos_weight=pos_weight)
        
        loss = self.focal_weight * focal + self.dice_weight * dice + self.bce_weight * bce
        return loss


class ResidualBlock(nn.Module):
    def __init__(self, in_features, out_features, dropout_rate=0.3):
        super(ResidualBlock, self).__init__()
        self.fc1 = nn.Linear(in_features, out_features)
        self.bn1 = nn.BatchNorm1d(out_features)
        self.fc2 = nn.Linear(out_features, out_features)
        self.bn2 = nn.BatchNorm1d(out_features)
        self.dropout = nn.Dropout(dropout_rate)
        
        self.shortcut = nn.Sequential()
        if in_features != out_features:
            self.shortcut = nn.Sequential(
                nn.Linear(in_features, out_features),
                nn.BatchNorm1d(out_features)
            )

    def forward(self, x):
        residual = self.shortcut(x)
        
        out = self.fc1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.dropout(out)
        
        out = self.fc2(out)
        out = self.bn2(out)
        out = self.dropout(out)
        
        out += residual
        out = F.relu(out)
        return out


class OptimizedEcDNA(nn.Module):
    def __init__(self, input_dim: int = 57, hidden_dims: list = [128, 64, 32], 
                 dropout_rate: float = 0.4, use_residual: bool = True):
        super(OptimizedEcDNA, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.dropout_rate = dropout_rate
        self.use_residual = use_residual
        
        self.input_bn = nn.BatchNorm1d(input_dim)
        self.input_dropout = nn.Dropout(dropout_rate * 0.5)
        
        layers = []
        prev_dim = input_dim
        
        for i, hidden_dim in enumerate(hidden_dims):
            if use_residual and i < len(hidden_dims) - 1:
                layers.append(ResidualBlock(prev_dim, hidden_dim, dropout_rate))
            else:
                layers.append(nn.Linear(prev_dim, hidden_dim))
                layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout_rate))
            
            prev_dim = hidden_dim
        
        self.features = nn.Sequential(*layers)
        
        self.classifier = nn.Sequential(
            nn.Linear(prev_dim, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(16, 1)
        )
        
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_bn(x)
        x = self.input_dropout(x)
        
        x = self.features(x)
        
        x = self.classifier(x)
        
        return x

    def get_loss_function(self, **kwargs):
        return CombinedLoss(
            focal_weight=kwargs.get('focal_weight', 0.4),
            dice_weight=kwargs.get('dice_weight', 0.3),
            bce_weight=kwargs.get('bce_weight', 0.3),
            alpha=kwargs.get('alpha', 0.75),
            gamma=kwargs.get('gamma', 2.0),
            pos_weight=kwargs.get('pos_weight', 100.0)
        )


class EnsembleOptimizedEcDNA(nn.Module):
    def __init__(self, input_dim: int = 57, num_models: int = 3, 
                 hidden_dims: list = [128, 64, 32], dropout_rate: float = 0.4):
        super(EnsembleOptimizedEcDNA, self).__init__()
        
        self.num_models = num_models
        self.models = nn.ModuleList([
            OptimizedEcDNA(input_dim, hidden_dims, dropout_rate, use_residual=True)
            for _ in range(num_models)
        ])
        
        self.meta_classifier = nn.Sequential(
            nn.Linear(num_models, 8),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(8, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = []
        for model in self.models:
            out = model(x)
            outputs.append(out)
        
        outputs = torch.cat(outputs, dim=1)
        
        final_output = self.meta_classifier(outputs)
        
        return final_output

    def get_loss_function(self, **kwargs):
        return CombinedLoss(
            focal_weight=kwargs.get('focal_weight', 0.4),
            dice_weight=kwargs.get('dice_weight', 0.3),
            bce_weight=kwargs.get('bce_weight', 0.3),
            alpha=kwargs.get('alpha', 0.75),
            gamma=kwargs.get('gamma', 2.0),
            pos_weight=kwargs.get('pos_weight', 100.0)
        )
