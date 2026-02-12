"""
Specialized loss functions and training strategies for ecDNA cargo gene prediction

Key challenges:
1. Extreme class imbalance (0.35% positive)
2. High precision (primary) and recall requirement
3. Precision requirement (avoid false positives in vast negative space)
4. Data distribution overlap (copy number variations are subtle)
5. auPRC optimization (primary metric)

Strategies:
1. Asymmetric loss with strong recall bias
2. Hard negative mining
3. Cost-sensitive learning
4. Label smoothing for overlapping distributions
5. Metric-aware training (directly optimize auPRC proxy)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class RecallBiasedTverskyLoss(nn.Module):
    """
    Tversky Loss with strong bias toward recall for ecDNA detection
    
    alpha: weight for false negatives (higher = more focus on recall)
    beta: weight for false positives (lower = less penalty on FP)
    
    For ecDNA: alpha > beta to prioritize finding all positives
    """
    def __init__(self, alpha=0.9, beta=0.1, smooth=1.0, gamma=1.5):
        super(RecallBiasedTverskyLoss, self).__init__()
        self.alpha = alpha  # High weight on FN (missed ecDNA)
        self.beta = beta    # Low weight on FP (false alarms)
        self.smooth = smooth
        self.gamma = gamma  # Focal-like focusing
        
    def forward(self, inputs, targets):
        probs = torch.sigmoid(inputs)
        
        # Flatten
        probs = probs.view(-1)
        targets = targets.view(-1)
        
        # True positives, false positives, false negatives
        tp = (probs * targets).sum()
        fp = (probs * (1 - targets)).sum()
        fn = ((1 - probs) * targets).sum()
        
        # Tversky index with recall bias
        tversky = (tp + self.smooth) / (tp + self.alpha * fn + self.beta * fp + self.smooth)
        
        # Apply focal-like focusing
        loss = (1 - tversky) ** self.gamma
        
        return loss


class HardNegativeMiningLoss(nn.Module):
    """
    Hard Negative Mining Loss for ecDNA prediction
    
    Focuses on hard negative examples (high copy number but not ecDNA)
    and hard positive examples (ecDNA with subtle signals)
    """
    def __init__(self, pos_weight=285.0, hard_neg_ratio=0.3, hard_pos_ratio=1.0, 
                 segval_threshold=4.0, margin=0.3):
        super(HardNegativeMiningLoss, self).__init__()
        self.pos_weight = pos_weight
        self.hard_neg_ratio = hard_neg_ratio  # Ratio of hard negatives to mine
        self.hard_pos_ratio = hard_pos_ratio  # Ratio of hard positives to mine
        self.segval_threshold = segval_threshold  # Threshold for hard negatives
        self.margin = margin  # Margin for hard examples
        self.bce = nn.BCEWithLogitsLoss(reduction='none')
        
    def forward(self, inputs, targets, features=None):
        """
        Args:
            inputs: model logits
            targets: ground truth labels
            features: input features (for identifying hard negatives by segVal)
        """
        # Ensure inputs and targets have the same shape
        if inputs.dim() > 1:
            inputs = inputs.view(-1)
        if targets.dim() > 1:
            targets = targets.view(-1)
        
        # Standard BCE loss
        bce_loss = F.binary_cross_entropy_with_logits(
            inputs, targets, reduction='none'
        )
        
        # Calculate probabilities
        probs = torch.sigmoid(inputs)
        
        # Identify hard examples
        with torch.no_grad():
            # Hard positives: predicted as negative but actually positive
            hard_pos_mask = (targets == 1) & (probs < 0.5)
            
            # Hard negatives: high confidence negative predictions that might be false
            # or samples with high segVal but not ecDNA
            hard_neg_mask = (targets == 0) & (probs > 0.3)  # High confidence false positives
            
            # If features available, also consider high segVal negatives as hard
            if features is not None and 'segVal' in features:
                segval = features['segVal']
                high_segval_neg = (targets == 0) & (segval > self.segval_threshold)
                hard_neg_mask = hard_neg_mask | high_segval_neg
        
        # Apply higher weights to hard examples
        weights = torch.ones_like(targets)
        weights[hard_pos_mask] *= 3.0  # Triple weight for hard positives
        weights[hard_neg_mask] *= 2.0  # Double weight for hard negatives
        
        # Apply pos_weight to positive samples
        weights[targets == 1] *= self.pos_weight
        
        # Weighted loss
        weighted_loss = (bce_loss * weights).mean()
        
        return weighted_loss


class auPRCProxyLoss(nn.Module):
    """
    Loss function that directly optimizes a proxy for auPRC
    
    Uses a differentiable approximation of Precision-Recall curve
    """
    def __init__(self, num_thresholds=20, temperature=0.1, pos_weight=285.0):
        super(auPRCProxyLoss, self).__init__()
        self.num_thresholds = num_thresholds
        self.temperature = temperature  # Temperature for soft thresholding
        self.pos_weight = pos_weight
        
        # Create thresholds from 0 to 1
        self.register_buffer('thresholds', torch.linspace(0, 1, num_thresholds))
        
    def forward(self, inputs, targets):
        # Ensure inputs and targets have the same shape
        if inputs.dim() > 1:
            inputs = inputs.view(-1)
        if targets.dim() > 1:
            targets = targets.view(-1)
        
        probs = torch.sigmoid(inputs)
        
        # Calculate precision at each threshold (soft version)
        precisions = []
        recalls = []
        
        for thresh in self.thresholds:
            # Soft predictions (differentiable)
            soft_pred = torch.sigmoid((probs - thresh) / self.temperature)
            
            # Soft TP, FP, FN
            tp = (soft_pred * targets).sum()
            fp = (soft_pred * (1 - targets)).sum()
            fn = ((1 - soft_pred) * targets).sum()
            
            # Soft precision and recall
            precision = tp / (tp + fp + 1e-8)
            recall = tp / (tp + fn + 1e-8)
            
            precisions.append(precision)
            recalls.append(recall)
        
        # Stack and calculate auPRC using trapezoidal rule
        precisions = torch.stack(precisions)
        recalls = torch.stack(recalls)
        
        # Sort by recall
        recalls_sorted, indices = torch.sort(recalls)
        precisions_sorted = precisions[indices]
        
        # Calculate auPRC (negative because we minimize)
        au_prc = torch.trapz(precisions_sorted, recalls_sorted)
        
        # Combine with BCE for stability
        bce_loss = F.binary_cross_entropy_with_logits(
            inputs, targets,
            pos_weight=torch.tensor([self.pos_weight], device=inputs.device)
        )
        
        # Combined loss: maximize auPRC (minimize negative) + BCE
        loss = -au_prc + 0.5 * bce_loss
        
        return loss


class CostSensitiveRecallLoss(nn.Module):
    """
    Cost-sensitive loss with asymmetric costs for ecDNA detection
    
    Assigns much higher cost to false negatives (missed ecDNA)
    than to false positives
    """
    def __init__(self, fn_cost=100.0, fp_cost=1.0, pos_weight=285.0):
        super(CostSensitiveRecallLoss, self).__init__()
        self.fn_cost = fn_cost   # Cost of missing an ecDNA (very high)
        self.fp_cost = fp_cost   # Cost of false alarm (lower)
        self.pos_weight = pos_weight
        
    def forward(self, inputs, targets):
        probs = torch.sigmoid(inputs).view(-1)
        targets = targets.view(-1)
        
        # Calculate costs
        # FN: predicted 0 but actually 1 -> cost = fn_cost
        # FP: predicted 1 but actually 0 -> cost = fp_cost
        # TP, TN: no cost
        
        fn_mask = (targets == 1) & (probs < 0.5)
        fp_mask = (targets == 0) & (probs > 0.5)
        
        # Weighted BCE with cost-sensitive weights
        weights = torch.ones_like(targets)
        weights[targets == 1] = self.pos_weight * self.fn_cost  # High weight on positives
        weights[fp_mask] = self.fp_cost  # Additional penalty for FP
        
        loss = F.binary_cross_entropy_with_logits(
            inputs, targets,
            weight=weights,
            reduction='mean'
        )
        
        return loss


class LabelSmoothingBCELoss(nn.Module):
    """
    BCE Loss with label smoothing for overlapping distributions
    
    Helps when copy number variations create overlapping distributions
    between positive and negative classes
    """
    def __init__(self, pos_weight=285.0, smoothing=0.1):
        super(LabelSmoothingBCELoss, self).__init__()
        self.pos_weight = pos_weight
        self.smoothing = smoothing
        
    def forward(self, inputs, targets):
        # Ensure inputs and targets have the same shape
        if inputs.dim() > 1:
            inputs = inputs.view(-1)
        if targets.dim() > 1:
            targets = targets.view(-1)
        
        # Apply label smoothing
        # Positive labels become (1 - smoothing)
        # Negative labels become smoothing
        targets_smooth = targets * (1 - self.smoothing) + (1 - targets) * self.smoothing
        
        loss = F.binary_cross_entropy_with_logits(
            inputs, targets_smooth,
            pos_weight=torch.tensor([self.pos_weight], device=inputs.device)
        )
        
        return loss


class ecDNAOptimizedLoss(nn.Module):
    """
    Optimized loss function specifically for ecDNA cargo gene prediction
    
    Combines multiple strategies:
    1. Recall-biased Tversky loss (high recall priority)
    2. Hard negative mining (focus on difficult examples)
    3. auPRC proxy optimization (direct metric optimization)
    4. Label smoothing (handle distribution overlap)
    """
    def __init__(self, 
                 tversky_weight=0.3,
                 hard_mining_weight=0.3,
                 auprc_weight=0.2,
                 bce_weight=0.2,
                 pos_weight=285.0,
                 tversky_alpha=0.9,
                 tversky_beta=0.1):
        super(ecDNAOptimizedLoss, self).__init__()
        
        self.tversky_weight = tversky_weight
        self.hard_mining_weight = hard_mining_weight
        self.auprc_weight = auprc_weight
        self.bce_weight = bce_weight
        
        # Initialize component losses
        self.tversky_loss = RecallBiasedTverskyLoss(
            alpha=tversky_alpha, 
            beta=tversky_beta
        )
        self.hard_mining_loss = HardNegativeMiningLoss(pos_weight=pos_weight)
        self.auprc_proxy_loss = auPRCProxyLoss(pos_weight=pos_weight)
        self.pos_weight = pos_weight
        
    def forward(self, inputs, targets, features=None):
        # Ensure inputs and targets have the same shape
        if inputs.dim() > 1:
            inputs = inputs.view(-1)
        if targets.dim() > 1:
            targets = targets.view(-1)
        
        # Component losses
        tversky = self.tversky_loss(inputs, targets)
        hard_mining = self.hard_mining_loss(inputs, targets, features)
        auprc = self.auprc_proxy_loss(inputs, targets)
        bce = F.binary_cross_entropy_with_logits(
            inputs, targets,
            pos_weight=torch.tensor([self.pos_weight], device=inputs.device)
        )
        
        # Combined weighted loss
        total_loss = (self.tversky_weight * tversky +
                     self.hard_mining_weight * hard_mining +
                     self.auprc_weight * auprc +
                     self.bce_weight * bce)
        
        return total_loss


class BalancedPrecisionRecallLoss(nn.Module):
    """
    Balanced loss function for ecDNA prediction that maintains high precision
    while improving recall and auPRC
    
    Key features:
    1. Balanced Tversky Loss (alpha ≈ beta)
    2. Precision-focused hard negative mining
    3. auPRC proxy optimization
    4. Adaptive weight adjustment
    """
    def __init__(self, 
                 balanced_tversky_weight=0.4,
                 precision_mining_weight=0.3,
                 auprc_weight=0.2,
                 bce_weight=0.1,
                 pos_weight=285.0,
                 tversky_alpha=0.6,
                 tversky_beta=0.4,
                 precision_threshold=0.8):
        super(BalancedPrecisionRecallLoss, self).__init__()
        
        self.balanced_tversky_weight = balanced_tversky_weight
        self.precision_mining_weight = precision_mining_weight
        self.auprc_weight = auprc_weight
        self.bce_weight = bce_weight
        self.precision_threshold = precision_threshold
        
        # Initialize component losses with balanced parameters
        self.balanced_tversky_loss = RecallBiasedTverskyLoss(
            alpha=tversky_alpha,  # More balanced alpha and beta
            beta=tversky_beta     # Higher beta to penalize FP more
        )
        
        # Hard negative mining with focus on precision
        self.precision_mining_loss = HardNegativeMiningLoss(
            pos_weight=pos_weight,
            hard_neg_ratio=0.4,  # Mine more hard negatives
            hard_pos_ratio=0.8,  # Focus on important hard positives
            segval_threshold=5.0  # Higher threshold for hard negatives
        )
        
        self.auprc_proxy_loss = auPRCProxyLoss(pos_weight=pos_weight)
        self.pos_weight = pos_weight
        
    def forward(self, inputs, targets, features=None):
        # Ensure inputs and targets have the same shape
        if inputs.dim() > 1:
            inputs = inputs.view(-1)
        if targets.dim() > 1:
            targets = targets.view(-1)
        
        # Component losses
        balanced_tversky = self.balanced_tversky_loss(inputs, targets)
        precision_mining = self.precision_mining_loss(inputs, targets, features)
        auprc = self.auprc_proxy_loss(inputs, targets)
        bce = F.binary_cross_entropy_with_logits(
            inputs, targets,
            pos_weight=torch.tensor([self.pos_weight], device=inputs.device)
        )
        
        # Calculate current precision to adjust weights
        probs = torch.sigmoid(inputs)
        predictions = (probs > 0.5).float()
        tp = (predictions * targets).sum()
        fp = (predictions * (1 - targets)).sum()
        current_precision = tp / (tp + fp + 1e-8)
        
        # Adaptive weight adjustment
        # If precision drops below threshold, increase precision-focused weights
        if current_precision < self.precision_threshold:
            precision_boost = max(0.0, (self.precision_threshold - current_precision) * 2)
            adjusted_precision_mining_weight = min(0.5, self.precision_mining_weight + precision_boost)
            adjusted_balanced_tversky_weight = max(0.2, self.balanced_tversky_weight - precision_boost * 0.5)
        else:
            adjusted_precision_mining_weight = self.precision_mining_weight
            adjusted_balanced_tversky_weight = self.balanced_tversky_weight
        
        # Combined weighted loss with adaptive weights
        total_loss = (
            adjusted_balanced_tversky_weight * balanced_tversky +
            adjusted_precision_mining_weight * precision_mining +
            self.auprc_weight * auprc +
            self.bce_weight * bce
        )
        
        return total_loss


def get_ecdna_loss_function(config, pos_ratio=None):
    """
    Factory function to get ecDNA-optimized loss function
    
    Args:
        config: Configuration dictionary
        pos_ratio: Positive sample ratio
    
    Returns:
        Loss function instance optimized for ecDNA prediction
    """
    loss_type = config.get('model', {}).get('loss_function', {}).get('type', 'ecDNAOptimizedLoss')
    
    # Calculate pos_weight
    if pos_ratio is not None and pos_ratio > 0:
        pos_weight = (1 - pos_ratio) / pos_ratio
    else:
        pos_weight = 285.0  # Default for ~0.35% positive rate
    
    if loss_type == 'RecallBiasedTverskyLoss':
        alpha = config.get('model', {}).get('loss_function', {}).get('alpha', 0.9)
        beta = config.get('model', {}).get('loss_function', {}).get('beta', 0.1)
        return RecallBiasedTverskyLoss(alpha=alpha, beta=beta)
    
    elif loss_type == 'HardNegativeMiningLoss':
        hard_neg_ratio = config.get('model', {}).get('loss_function', {}).get('hard_neg_ratio', 0.3)
        return HardNegativeMiningLoss(pos_weight=pos_weight, hard_neg_ratio=hard_neg_ratio)
    
    elif loss_type == 'auPRCProxyLoss':
        return auPRCProxyLoss(pos_weight=pos_weight)
    
    elif loss_type == 'CostSensitiveRecallLoss':
        fn_cost = config.get('model', {}).get('loss_function', {}).get('fn_cost', 100.0)
        fp_cost = config.get('model', {}).get('loss_function', {}).get('fp_cost', 1.0)
        return CostSensitiveRecallLoss(fn_cost=fn_cost, fp_cost=fp_cost, pos_weight=pos_weight)
    
    elif loss_type == 'LabelSmoothingBCELoss':
        smoothing = config.get('model', {}).get('loss_function', {}).get('smoothing', 0.1)
        return LabelSmoothingBCELoss(pos_weight=pos_weight, smoothing=smoothing)
    
    elif loss_type == 'ecDNAOptimizedLoss':
        return ecDNAOptimizedLoss(pos_weight=pos_weight)
    
    elif loss_type == 'BalancedPrecisionRecallLoss':
        tversky_alpha = config.get('model', {}).get('loss_function', {}).get('tversky_alpha', 0.6)
        tversky_beta = config.get('model', {}).get('loss_function', {}).get('tversky_beta', 0.4)
        precision_threshold = config.get('model', {}).get('loss_function', {}).get('precision_threshold', 0.8)
        return BalancedPrecisionRecallLoss(
            pos_weight=pos_weight,
            tversky_alpha=tversky_alpha,
            tversky_beta=tversky_beta,
            precision_threshold=precision_threshold
        )
    
    else:
        raise ValueError(f"Unknown ecDNA loss type: {loss_type}")
