#!/usr/bin/env python
"""
Deep Gated Interaction Transformer (DGIT) - Simplified Version

Architecture innovations:
1. Feature Gating: 学习的特征重要性权重
2. Gated Residual Blocks: 门控残差连接
3. Multi-head Self-Attention: 捕捉复杂特征模式
4. Pairwise Interaction: 特征两两交互

Target: auPRC >= 0.85, Precision >= 0.8
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any


class FeatureGate(nn.Module):
    """学习的特征重要性门控"""
    
    def __init__(self, input_dim: int = 57, hidden_dim: int = 32):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_values = self.gate(x)
        return x * gate_values


class GatedResidualBlock(nn.Module):
    """带门控的残差块"""
    
    def __init__(self, dim: int, dropout: float = 0.3):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim * 2)
        self.fc2 = nn.Linear(dim * 2, dim)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
        self.gate = nn.Linear(dim, dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        
        out = self.norm1(x)
        out = self.fc1(out)
        out = F.gelu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        
        gate = torch.sigmoid(self.gate(residual))
        out = gate * out + (1 - gate) * residual
        
        out = self.norm2(out)
        return out


class DeepGatedInteractionTransformer(nn.Module):
    """
    Deep Gated Interaction Transformer (DGIT)
    """
    
    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        input_dim: int = 57,
        hidden_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 4,
        dropout: float = 0.3
    ):
        super().__init__()
        
        if config is not None:
            arch = config.get('model', {}).get('architecture', {})
            input_dim = arch.get('input_dim', 57)
            hidden_dim = arch.get('hidden_dim', 128)
            num_heads = arch.get('num_heads', 4)
            num_layers = arch.get('num_layers', 4)
            dropout = arch.get('dropout_rate', 0.3)
        
        self.config = config
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        self.feature_gate = FeatureGate(input_dim, hidden_dim // 4)
        
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout * 0.5)
        )
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=False
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.residual_blocks = nn.ModuleList([
            GatedResidualBlock(hidden_dim, dropout)
            for _ in range(2)
        ])
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gated = self.feature_gate(x)
        
        h = self.input_projection(gated)
        
        h = h.unsqueeze(1)
        h = self.transformer(h)
        h = h.squeeze(1)
        
        for block in self.residual_blocks:
            h = block(h)
        
        logits = self.classifier(h)
        
        return logits.squeeze(-1)
    
    def get_loss_function(self, **kwargs):
        return TripleMarginRankingLoss(
            margin=kwargs.get('margin', 0.5),
            ranking_weight=kwargs.get('ranking_weight', 0.7),
            pos_weight=kwargs.get('pos_weight', 100.0)
        )


class TripleMarginRankingLoss(nn.Module):
    """三元组排序损失"""
    
    def __init__(
        self,
        margin: float = 0.5,
        ranking_weight: float = 0.7,
        pos_weight: float = 100.0
    ):
        super().__init__()
        self.margin = margin
        self.ranking_weight = ranking_weight
        self.pos_weight = pos_weight
        
    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        if targets.dim() > 1:
            targets = targets.squeeze(-1)
        
        pos_mask = targets == 1
        neg_mask = targets == 0
        
        bce_loss = F.binary_cross_entropy_with_logits(
            logits, targets,
            pos_weight=torch.tensor([self.pos_weight], device=logits.device)
        )
        
        if pos_mask.sum() > 10 and neg_mask.sum() > 10:
            pos_scores = logits[pos_mask]
            neg_scores = logits[neg_mask]
            
            pos_scores = pos_scores.unsqueeze(0)
            neg_scores = neg_scores.unsqueeze(1)
            
            ranking_matrix = self.margin - (pos_scores - neg_scores)
            ranking_loss = F.relu(ranking_matrix).mean()
            
            total_loss = (
                self.ranking_weight * ranking_loss +
                (1 - self.ranking_weight) * bce_loss
            )
            return total_loss
        
        return bce_loss


class EnsembleDGIT(nn.Module):
    """DGIT集成模型"""
    
    def __init__(self, config=None, num_models: int = 3, **kwargs):
        super().__init__()
        self.models = nn.ModuleList([
            DeepGatedInteractionTransformer(config, **kwargs)
            for _ in range(num_models)
        ])
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = [model(x) for model in self.models]
        return torch.stack(outputs).mean(dim=0)
