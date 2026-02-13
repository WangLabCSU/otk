#!/usr/bin/env python
"""
DGIT Super V2 - Enhanced Deep Gated Interaction Transformer

Key Improvements:
1. Sample-Level Awareness: Model learns to predict both gene-level and sample-level ecDNA
2. Hierarchical Attention: Gene-level features -> Sample-level aggregation -> Prediction
3. Multi-Task Learning: Joint optimization for gene and sample level predictions
4. Advanced Data Augmentation: Handle severe class imbalance better
5. Curriculum Learning: Start with easier samples, gradually increase difficulty

Target Performance:
- Gene-level auPRC: 0.85+
- Gene-level Precision: 0.8+
- Sample-level auPRC: 0.99+
- Sample-level auROC: 0.9+
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any, Tuple
import numpy as np


class SampleAwareAttention(nn.Module):
    """
    Sample-aware attention mechanism
    Aggregates gene-level features to sample-level representation
    """
    
    def __init__(self, hidden_dim: int, num_heads: int = 4):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        
        # Gene-level feature projection
        self.gene_proj = nn.Linear(hidden_dim, hidden_dim)
        
        # Sample-level query
        self.sample_query = nn.Parameter(torch.randn(1, 1, hidden_dim))
        
        # Multi-head attention
        self.attention = nn.MultiheadAttention(
            hidden_dim, num_heads, 
            batch_first=True,
            dropout=0.1
        )
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.2)
        )
        
    def forward(self, gene_features: torch.Tensor, sample_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            gene_features: [batch_size, num_genes, hidden_dim]
            sample_mask: [batch_size, num_genes] - indicates which genes belong to which sample
        
        Returns:
            sample_features: [batch_size, hidden_dim]
        """
        batch_size = gene_features.size(0)
        
        # Project gene features
        gene_proj = self.gene_proj(gene_features)
        
        # Expand sample query for batch
        sample_query = self.sample_query.expand(batch_size, -1, -1)
        
        # Apply attention (sample query attends to genes)
        attn_output, _ = self.attention(
            sample_query,  # [batch, 1, hidden]
            gene_proj,      # [batch, num_genes, hidden]
            gene_proj,      # [batch, num_genes, hidden]
            key_padding_mask=~sample_mask.bool() if sample_mask is not None else None
        )
        
        # Project output
        sample_features = self.output_proj(attn_output.squeeze(1))
        
        return sample_features


class HierarchicalTransformer(nn.Module):
    """
    Hierarchical Transformer with two-level attention:
    1. Gene-level self-attention
    2. Sample-level cross-attention
    """
    
    def __init__(
        self,
        hidden_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 3,
        dropout: float = 0.3
    ):
        super().__init__()
        
        # Gene-level transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=False
        )
        self.gene_transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Sample-aware attention
        self.sample_attention = SampleAwareAttention(hidden_dim, num_heads)
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
    def forward(
        self, 
        gene_features: torch.Tensor, 
        sample_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            gene_features: [batch_size, num_genes, hidden_dim]
            sample_mask: [batch_size, num_genes]
        
        Returns:
            gene_output: [batch_size, num_genes, hidden_dim]
            sample_output: [batch_size, hidden_dim]
        """
        # Gene-level self-attention
        gene_encoded = self.gene_transformer(gene_features)
        
        # Sample-level aggregation
        sample_features = self.sample_attention(gene_encoded, sample_mask)
        
        # Expand sample features to match gene dimension for fusion
        sample_expanded = sample_features.unsqueeze(1).expand(-1, gene_encoded.size(1), -1)
        
        # Fuse gene and sample features
        fused = torch.cat([gene_encoded, sample_expanded], dim=-1)
        gene_output = self.fusion(fused)
        
        return gene_output, sample_features


class DGITSuperV2(nn.Module):
    """
    DGIT Super V2 - Enhanced with sample-level awareness
    """
    
    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        input_dim: int = 57,
        hidden_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 3,
        dropout: float = 0.3,
        use_sample_aware: bool = True
    ):
        super().__init__()
        
        if config is not None:
            arch = config.get('model', {}).get('architecture', {})
            input_dim = arch.get('input_dim', 57)
            hidden_dim = arch.get('hidden_dim', 128)
            num_heads = arch.get('num_heads', 4)
            num_layers = arch.get('num_layers', 3)
            dropout = arch.get('dropout_rate', 0.3)
            use_sample_aware = arch.get('use_sample_aware', True)
        
        self.config = config
        self.use_sample_aware = use_sample_aware
        
        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout * 0.5)
        )
        
        # Hierarchical transformer
        if use_sample_aware:
            self.hierarchical_transformer = HierarchicalTransformer(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                num_layers=num_layers,
                dropout=dropout
            )
        else:
            # Standard transformer
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
        
        # Gene-level classifier
        self.gene_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # Sample-level classifier
        self.sample_classifier = nn.Sequential(
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
    
    def forward(
        self, 
        x: torch.Tensor,
        sample_ids: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: [batch_size, num_genes, input_dim] or [batch_size, input_dim]
            sample_ids: [batch_size, num_genes] - sample identifier for each gene
        
        Returns:
            Dictionary with gene_logits and sample_logits
        """
        # Handle single gene input
        if x.dim() == 2:
            x = x.unsqueeze(1)
        
        batch_size, num_genes, _ = x.shape
        
        # Input projection
        h = self.input_proj(x)
        
        if self.use_sample_aware and sample_ids is not None:
            # Create sample mask
            unique_samples = torch.unique(sample_ids)
            sample_mask = (sample_ids.unsqueeze(1) == unique_samples.unsqueeze(0)).float()
            
            # Hierarchical processing
            gene_features, sample_features = self.hierarchical_transformer(h, sample_mask)
        else:
            # Standard transformer
            gene_features = self.transformer(h)
            sample_features = gene_features.mean(dim=1)
        
        # Gene-level prediction
        gene_logits = self.gene_classifier(gene_features).squeeze(-1)
        
        # Sample-level prediction
        sample_logits = self.sample_classifier(sample_features).squeeze(-1)
        
        return {
            'gene_logits': gene_logits,
            'sample_logits': sample_logits,
            'gene_features': gene_features,
            'sample_features': sample_features
        }


class HierarchicalLoss(nn.Module):
    """
    Multi-task loss for hierarchical prediction
    Combines gene-level and sample-level losses
    """
    
    def __init__(
        self,
        gene_weight: float = 0.6,
        sample_weight: float = 0.4,
        pos_weight: float = 100.0,
        focal_gamma: float = 2.0
    ):
        super().__init__()
        self.gene_weight = gene_weight
        self.sample_weight = sample_weight
        self.pos_weight = pos_weight
        self.focal_gamma = focal_gamma
        
    def focal_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Focal loss for handling class imbalance"""
        bce_loss = F.binary_cross_entropy_with_logits(
            logits, targets, reduction='none'
        )
        probs = torch.sigmoid(logits)
        pt = torch.where(targets == 1, probs, 1 - probs)
        focal_weight = (1 - pt) ** self.focal_gamma
        
        # Apply positive weight
        pos_weight = torch.where(targets == 1, self.pos_weight, 1.0)
        
        return (focal_weight * bce_loss * pos_weight).mean()
    
    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        gene_targets: torch.Tensor,
        sample_targets: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Args:
            outputs: Dictionary with gene_logits and sample_logits
            gene_targets: [batch_size, num_genes]
            sample_targets: [batch_size]
        
        Returns:
            total_loss, loss_dict
        """
        gene_logits = outputs['gene_logits']
        sample_logits = outputs['sample_logits']
        
        # Gene-level loss
        gene_loss = self.focal_loss(gene_logits, gene_targets)
        
        # Sample-level loss
        sample_loss = self.focal_loss(sample_logits, sample_targets)
        
        # Combined loss
        total_loss = self.gene_weight * gene_loss + self.sample_weight * sample_loss
        
        loss_dict = {
            'total': total_loss.item(),
            'gene': gene_loss.item(),
            'sample': sample_loss.item()
        }
        
        return total_loss, loss_dict


class CurriculumSampler:
    """
    Curriculum learning sampler
    Starts with easier samples (more positive genes), gradually includes harder ones
    """
    
    def __init__(self, sample_pos_counts: Dict[str, int], initial_ratio: float = 0.5):
        self.sample_pos_counts = sample_pos_counts
        self.initial_ratio = initial_ratio
        self.epoch = 0
        
        # Sort samples by positive count (descending)
        self.sorted_samples = sorted(
            sample_pos_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
    def set_epoch(self, epoch: int, total_epochs: int):
        """Update sampling ratio based on current epoch"""
        self.epoch = epoch
        # Gradually increase ratio from initial_ratio to 1.0
        progress = epoch / total_epochs
        self.current_ratio = self.initial_ratio + (1.0 - self.initial_ratio) * progress
        
    def get_samples(self) -> list:
        """Get samples for current epoch"""
        n_samples = int(len(self.sorted_samples) * self.current_ratio)
        return [s[0] for s in self.sorted_samples[:n_samples]]


# For compatibility with existing code
def create_dgit_super_v2_model(config: Dict[str, Any]):
    """Factory function to create DGIT Super V2 model"""
    return DGITSuperV2(config)
