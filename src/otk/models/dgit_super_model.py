#!/usr/bin/env python
"""
DGIT Super - Super Powerful Deep Gated Interaction Transformer

Target: auPRC >= 0.85, Precision >= 0.8

Key Innovations:
1. Multi-Scale Feature Extraction: 不同尺度捕捉特征模式
2. Deep Gated Residual Networks: 深层网络保持梯度流动
3. Contrastive Learning Head: 对比学习增强正负样本区分
4. Adaptive Feature Reweighting: 自适应特征重加权
5. Multi-Task Learning: 分类 + 排序 + 密度估计
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any, Tuple
import math


class MultiScaleFeatureExtractor(nn.Module):
    """多尺度特征提取器"""
    
    def __init__(self, input_dim: int = 57, hidden_dim: int = 128):
        super().__init__()
        
        self.scale_1 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim // 4),
            nn.LayerNorm(hidden_dim // 4),
            nn.GELU()
        )
        
        self.scale_2 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU()
        )
        
        self.scale_3 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )
        
        self.fusion = nn.Linear(hidden_dim // 4 + hidden_dim // 2 + hidden_dim, hidden_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s1 = self.scale_1(x)
        s2 = self.scale_2(x)
        s3 = self.scale_3(x)
        
        multi_scale = torch.cat([s1, s2, s3], dim=-1)
        return self.fusion(multi_scale)


class AdaptiveFeatureGate(nn.Module):
    """自适应特征门控"""
    
    def __init__(self, input_dim: int = 57, num_gates: int = 4):
        super().__init__()
        self.num_gates = num_gates
        
        self.gate_networks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, input_dim // 2),
                nn.ReLU(),
                nn.Linear(input_dim // 2, input_dim),
                nn.Sigmoid()
            ) for _ in range(num_gates)
        ])
        
        self.gate_selector = nn.Sequential(
            nn.Linear(input_dim, num_gates),
            nn.Softmax(dim=-1)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_weights = self.gate_selector(x)
        
        gated_features = []
        for i, gate_net in enumerate(self.gate_networks):
            gate = gate_net(x)
            gated_features.append(x * gate)
        
        stacked = torch.stack(gated_features, dim=-1)
        weighted = (stacked * gate_weights.unsqueeze(1)).sum(dim=-1)
        
        return weighted


class DeepGatedResidualBlock(nn.Module):
    """深度门控残差块"""
    
    def __init__(self, dim: int, expansion: int = 4, dropout: float = 0.3):
        super().__init__()
        
        self.norm1 = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, dim * expansion)
        self.act1 = nn.GELU()
        self.dropout1 = nn.Dropout(dropout)
        
        self.norm2 = nn.LayerNorm(dim * expansion)
        self.fc2 = nn.Linear(dim * expansion, dim)
        self.dropout2 = nn.Dropout(dropout)
        
        self.gate = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.Sigmoid()
        )
        
        self.norm3 = nn.LayerNorm(dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        
        out = self.norm1(x)
        out = self.fc1(out)
        out = self.act1(out)
        out = self.dropout1(out)
        
        out = self.norm2(out)
        out = self.fc2(out)
        out = self.dropout2(out)
        
        gate_input = torch.cat([residual, out], dim=-1)
        gate = self.gate(gate_input)
        
        out = gate * out + (1 - gate) * residual
        out = self.norm3(out)
        
        return out


class ContrastiveHead(nn.Module):
    """对比学习头 - 增强正负样本区分"""
    
    def __init__(self, hidden_dim: int = 128, proj_dim: int = 64):
        super().__init__()
        
        self.projector = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, proj_dim),
            nn.LayerNorm(proj_dim)
        )
        
        self.classifier = nn.Linear(proj_dim, 1)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        proj = self.projector(x)
        proj = F.normalize(proj, p=2, dim=-1)
        logits = self.classifier(proj)
        return logits.squeeze(-1), proj


class DensityEstimationHead(nn.Module):
    """密度估计头 - 辅助学习数据分布"""
    
    def __init__(self, hidden_dim: int = 128):
        super().__init__()
        
        self.mu_head = nn.Linear(hidden_dim, 1)
        self.logvar_head = nn.Linear(hidden_dim, 1)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mu = self.mu_head(x).squeeze(-1)
        logvar = self.logvar_head(x).squeeze(-1)
        logvar = torch.clamp(logvar, min=-10, max=10)
        return mu, logvar


class DGITSuper(nn.Module):
    """
    DGIT Super - Super Powerful Deep Gated Interaction Transformer
    
    Architecture:
    1. Adaptive Feature Gate: 自适应特征选择
    2. Multi-Scale Feature Extractor: 多尺度特征
    3. Transformer Encoder: 自注意力机制
    4. Deep Gated Residual Blocks: 深层残差网络
    5. Contrastive Head: 对比学习
    6. Density Head: 密度估计
    """
    
    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        input_dim: int = 57,
        hidden_dim: int = 256,
        num_heads: int = 8,
        num_transformer_layers: int = 4,
        num_residual_blocks: int = 6,
        dropout: float = 0.3,
        use_contrastive: bool = True,
        use_density: bool = True
    ):
        super().__init__()
        
        if config is not None:
            arch = config.get('model', {}).get('architecture', {})
            input_dim = arch.get('input_dim', 57)
            hidden_dim = arch.get('hidden_dim', 256)
            num_heads = arch.get('num_heads', 8)
            num_transformer_layers = arch.get('num_transformer_layers', 4)
            num_residual_blocks = arch.get('num_residual_blocks', 6)
            dropout = arch.get('dropout_rate', 0.3)
            use_contrastive = arch.get('use_contrastive', True)
            use_density = arch.get('use_density', True)
        
        self.config = config
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.use_contrastive = use_contrastive
        self.use_density = use_density
        
        self.feature_gate = AdaptiveFeatureGate(input_dim, num_gates=4)
        
        self.multi_scale_extractor = MultiScaleFeatureExtractor(input_dim, hidden_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=False
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_transformer_layers)
        
        self.residual_blocks = nn.ModuleList([
            DeepGatedResidualBlock(hidden_dim, expansion=4, dropout=dropout)
            for _ in range(num_residual_blocks)
        ])
        
        if use_contrastive:
            self.contrastive_head = ContrastiveHead(hidden_dim, proj_dim=64)
        else:
            self.classifier = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.LayerNorm(hidden_dim // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, 1)
            )
        
        if use_density:
            self.density_head = DensityEstimationHead(hidden_dim)
        
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
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        gated = self.feature_gate(x)
        
        h = self.multi_scale_extractor(gated)
        
        h = h.unsqueeze(1)
        h = self.transformer(h)
        h = h.squeeze(1)
        
        for block in self.residual_blocks:
            h = block(h)
        
        outputs = {}
        
        if self.use_contrastive:
            logits, proj = self.contrastive_head(h)
            outputs['logits'] = logits
            outputs['projection'] = proj
        else:
            outputs['logits'] = self.classifier(h).squeeze(-1)
        
        if self.use_density:
            mu, logvar = self.density_head(h)
            outputs['density_mu'] = mu
            outputs['density_logvar'] = logvar
        
        return outputs
    
    def get_logits(self, x: torch.Tensor) -> torch.Tensor:
        outputs = self.forward(x)
        return outputs['logits']


class DGITSuperLoss(nn.Module):
    """
    DGIT Super Loss - 多任务损失函数
    
    组合:
    1. Focal Loss: 关注难分类样本
    2. Ranking Loss: 直接优化排序
    3. Contrastive Loss: 增强正负样本区分
    4. Density Loss: 辅助学习分布
    """
    
    def __init__(
        self,
        focal_gamma: float = 2.0,
        ranking_margin: float = 0.5,
        contrastive_temp: float = 0.1,
        pos_weight: float = 100.0,
        focal_weight: float = 0.4,
        ranking_weight: float = 0.4,
        contrastive_weight: float = 0.15,
        density_weight: float = 0.05
    ):
        super().__init__()
        self.focal_gamma = focal_gamma
        self.ranking_margin = ranking_margin
        self.contrastive_temp = contrastive_temp
        self.pos_weight = pos_weight
        self.focal_weight = focal_weight
        self.ranking_weight = ranking_weight
        self.contrastive_weight = contrastive_weight
        self.density_weight = density_weight
        
    def focal_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.binary_cross_entropy_with_logits(
            logits, targets, reduction='none'
        )
        probs = torch.sigmoid(logits)
        pt = torch.where(targets == 1, probs, 1 - probs)
        focal_weight = (1 - pt) ** self.focal_gamma
        
        pos_weight = torch.where(targets == 1, self.pos_weight, 1.0)
        
        return (focal_weight * ce_loss * pos_weight).mean()
    
    def ranking_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        pos_mask = targets == 1
        neg_mask = targets == 0
        
        if pos_mask.sum() < 10 or neg_mask.sum() < 10:
            return torch.tensor(0.0, device=logits.device)
        
        pos_scores = logits[pos_mask]
        neg_scores = logits[neg_mask]
        
        pos_scores = pos_scores.unsqueeze(0)
        neg_scores = neg_scores.unsqueeze(1)
        
        ranking_matrix = self.ranking_margin - (pos_scores - neg_scores)
        return F.relu(ranking_matrix).mean()
    
    def contrastive_loss(self, proj: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        pos_mask = targets == 1
        neg_mask = targets == 0
        
        if pos_mask.sum() < 10 or neg_mask.sum() < 10:
            return torch.tensor(0.0, device=proj.device)
        
        pos_proj = proj[pos_mask]
        neg_proj = proj[neg_mask]
        
        pos_proj = F.normalize(pos_proj, p=2, dim=-1)
        neg_proj = F.normalize(neg_proj, p=2, dim=-1)
        
        sim_matrix = torch.mm(pos_proj, neg_proj.t()) / self.contrastive_temp
        
        labels = torch.zeros(pos_proj.size(0), dtype=torch.long, device=proj.device)
        
        return F.cross_entropy(sim_matrix, labels)
    
    def density_loss(
        self, 
        mu: torch.Tensor, 
        logvar: torch.Tensor, 
        targets: torch.Tensor
    ) -> torch.Tensor:
        pos_mask = targets == 1
        neg_mask = targets == 0
        
        if pos_mask.sum() < 10 or neg_mask.sum() < 10:
            return torch.tensor(0.0, device=mu.device)
        
        pos_mu = mu[pos_mask].mean()
        neg_mu = mu[neg_mask].mean()
        
        separation_loss = F.relu(0.5 - (pos_mu - neg_mu).abs())
        
        kl_loss = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).mean()
        
        return separation_loss + 0.1 * kl_loss
    
    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        if targets.dim() > 1:
            targets = targets.squeeze(-1)
        
        logits = outputs['logits']
        
        loss_dict = {}
        
        focal = self.focal_loss(logits, targets)
        loss_dict['focal'] = focal.item()
        
        ranking = self.ranking_loss(logits, targets)
        loss_dict['ranking'] = ranking.item()
        
        total = self.focal_weight * focal + self.ranking_weight * ranking
        
        if 'projection' in outputs:
            contrastive = self.contrastive_loss(outputs['projection'], targets)
            loss_dict['contrastive'] = contrastive.item()
            total = total + self.contrastive_weight * contrastive
        
        if 'density_mu' in outputs:
            density = self.density_loss(
                outputs['density_mu'], 
                outputs['density_logvar'], 
                targets
            )
            loss_dict['density'] = density.item()
            total = total + self.density_weight * density
        
        loss_dict['total'] = total.item()
        
        return total, loss_dict


class EnsembleDGITSuper(nn.Module):
    """DGIT Super 集成模型"""
    
    def __init__(self, config=None, num_models: int = 3, **kwargs):
        super().__init__()
        self.models = nn.ModuleList([
            DGITSuper(config, **kwargs)
            for _ in range(num_models)
        ])
        
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        outputs_list = [model(x) for model in self.models]
        
        logits = torch.stack([o['logits'] for o in outputs_list]).mean(dim=0)
        
        outputs = {'logits': logits}
        
        if 'projection' in outputs_list[0]:
            proj = torch.stack([o['projection'] for o in outputs_list]).mean(dim=0)
            outputs['projection'] = proj
        
        if 'density_mu' in outputs_list[0]:
            mu = torch.stack([o['density_mu'] for o in outputs_list]).mean(dim=0)
            logvar = torch.stack([o['density_logvar'] for o in outputs_list]).mean(dim=0)
            outputs['density_mu'] = mu
            outputs['density_logvar'] = logvar
        
        return outputs
    
    def get_logits(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)['logits']
