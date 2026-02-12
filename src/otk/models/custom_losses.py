"""
Custom loss functions for highly imbalanced ecDNA prediction task
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class WeightedFocalLoss(nn.Module):
    """
    Weighted Focal Loss for highly imbalanced classification
    
    Combines class weighting with focal loss to handle extreme imbalance
    """
    def __init__(self, pos_weight=100.0, gamma=2.0, alpha=0.25, reduction='mean'):
        super(WeightedFocalLoss, self).__init__()
        self.pos_weight = pos_weight
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        # Apply pos_weight to BCE
        bce_loss = F.binary_cross_entropy_with_logits(
            inputs, targets, 
            pos_weight=torch.tensor([self.pos_weight], device=inputs.device),
            reduction='none'
        )
        
        # Calculate focal weighting
        probs = torch.sigmoid(inputs)
        pt = torch.where(targets == 1, probs, 1 - probs)
        focal_weight = (1 - pt) ** self.gamma
        
        # Apply alpha weighting
        alpha_t = torch.where(targets == 1, self.alpha, 1 - self.alpha)
        
        # Combined loss
        loss = alpha_t * focal_weight * bce_loss
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class DiceLoss(nn.Module):
    """
    Dice Loss for imbalanced segmentation/classification
    
    Good for handling extreme class imbalance as it focuses on overlap
    """
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        
    def forward(self, inputs, targets):
        # Apply sigmoid to get probabilities
        probs = torch.sigmoid(inputs)
        
        # Flatten
        probs = probs.view(-1)
        targets = targets.view(-1)
        
        # Calculate intersection and union
        intersection = (probs * targets).sum()
        dice = (2. * intersection + self.smooth) / (probs.sum() + targets.sum() + self.smooth)
        
        return 1 - dice


class TverskyLoss(nn.Module):
    """
    Tversky Loss - generalization of Dice loss with better control over false positives/negatives
    
    alpha: weight for false negatives
    beta: weight for false positives
    
    For high recall (detect all positives), use alpha > beta
    """
    def __init__(self, alpha=0.7, beta=0.3, smooth=1.0):
        super(TverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth
        
    def forward(self, inputs, targets):
        probs = torch.sigmoid(inputs)
        
        # Flatten
        probs = probs.view(-1)
        targets = targets.view(-1)
        
        # True positives, false positives, false negatives
        tp = (probs * targets).sum()
        fp = (probs * (1 - targets)).sum()
        fn = ((1 - probs) * targets).sum()
        
        tversky = (tp + self.smooth) / (tp + self.alpha * fn + self.beta * fp + self.smooth)
        
        return 1 - tversky


class ComboLoss(nn.Module):
    """
    Combined loss for ecDNA prediction
    
    Combines multiple loss functions to handle:
    1. Class imbalance (Weighted BCE)
    2. Hard example mining (Focal)
    3. Overlap optimization (Dice)
    """
    def __init__(self, 
                 bce_weight=0.5, 
                 focal_weight=0.3, 
                 dice_weight=0.2,
                 pos_weight=100.0,
                 focal_gamma=2.0,
                 focal_alpha=0.25):
        super(ComboLoss, self).__init__()
        
        self.bce_weight = bce_weight
        self.focal_weight = focal_weight
        self.dice_weight = dice_weight
        self.pos_weight = pos_weight
        self.focal_gamma = focal_gamma
        self.focal_alpha = focal_alpha
        
    def forward(self, inputs, targets):
        # Weighted BCE
        bce_loss = F.binary_cross_entropy_with_logits(
            inputs, targets,
            pos_weight=torch.tensor([self.pos_weight], device=inputs.device),
            reduction='mean'
        )
        
        # Focal component
        probs = torch.sigmoid(inputs)
        pt = torch.where(targets == 1, probs, 1 - probs)
        focal_weight = (1 - pt) ** self.focal_gamma
        alpha_t = torch.where(targets == 1, self.focal_alpha, 1 - self.focal_alpha)
        focal_loss = (alpha_t * focal_weight * F.binary_cross_entropy_with_logits(
            inputs, targets, reduction='none'
        )).mean()
        
        # Dice component
        probs_flat = probs.view(-1)
        targets_flat = targets.view(-1)
        intersection = (probs_flat * targets_flat).sum()
        dice = (2. * intersection + 1.0) / (probs_flat.sum() + targets_flat.sum() + 1.0)
        dice_loss = 1 - dice
        
        # Combined loss
        total_loss = (self.bce_weight * bce_loss + 
                     self.focal_weight * focal_loss + 
                     self.dice_weight * dice_loss)
        
        return total_loss


class AsymmetricLoss(nn.Module):
    """
    Asymmetric Loss for extremely imbalanced datasets
    
    Asymmetric focusing: different gamma for positive and negative samples
    This helps when negative samples are overwhelming
    """
    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8, disable_torch_grad_focal_loss=False):
        super(AsymmetricLoss, self).__init__()
        
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.eps = eps
        
    def forward(self, inputs, targets):
        # Calculate probabilities
        probs = torch.sigmoid(inputs)
        
        # Asymmetric clipping for negative samples
        if self.clip is not None and self.clip > 0:
            probs_neg = (probs + self.clip).clamp(max=1)
        else:
            probs_neg = probs
        
        # Calculate loss
        loss = targets * torch.log(probs + self.eps)
        loss_neg = (1 - targets) * torch.log(1 - probs_neg + self.eps)
        
        loss = loss + loss_neg
        
        # Asymmetric focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(False)
            
            probs_temp = probs.detach() if self.disable_torch_grad_focal_loss else probs
            
            # Different gamma for positive and negative
            focal_weight = torch.where(
                targets == 1,
                (1 - probs_temp) ** self.gamma_pos,
                probs_temp ** self.gamma_neg
            )
            
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(True)
                loss = loss * focal_weight.detach()
            else:
                loss = loss * focal_weight
        
        return -loss.mean()


class LDAMLoss(nn.Module):
    """
    Label-Distribution-Aware Margin Loss
    
    Specifically designed for imbalanced classification
    Adds larger margins for rare classes
    """
    def __init__(self, cls_num_list, max_m=0.5, s=30):
        super(LDAMLoss, self).__init__()
        
        # Calculate margins based on class frequency
        # For binary classification, we use pos_ratio to calculate margin
        if isinstance(cls_num_list, (list, tuple)) and len(cls_num_list) == 2:
            pos_ratio = cls_num_list[1] / sum(cls_num_list)
        else:
            pos_ratio = 0.01  # Default assumption
            
        self.m_list = torch.tensor([0, max_m * ((1 / pos_ratio) ** 0.5)])
        self.s = s
        
    def forward(self, inputs, targets):
        # Add margin to logits
        # For positive class (target=1), subtract margin
        # For negative class (target=0), add margin
        margins = self.m_list.to(inputs.device)[targets.long()]
        inputs_m = inputs - margins.unsqueeze(1)
        
        # Scale logits
        inputs_m = inputs_m * self.s
        
        # Standard BCE with modified logits
        return F.binary_cross_entropy_with_logits(inputs_m, targets)


def get_loss_function(config, pos_ratio=None):
    """
    Factory function to get the appropriate loss function based on config
    
    Args:
        config: Configuration dictionary
        pos_ratio: Positive sample ratio (for calculating pos_weight)
    
    Returns:
        Loss function instance
    """
    loss_type = config.get('model', {}).get('loss_function', {}).get('type', 'BCEWithLogitsLoss')
    
    # Calculate pos_weight if not provided and pos_ratio is available
    if pos_ratio is not None and pos_ratio > 0:
        calculated_pos_weight = (1 - pos_ratio) / pos_ratio
    else:
        calculated_pos_weight = 100.0  # Default for highly imbalanced data
    
    if loss_type == 'BCEWithLogitsLoss':
        pos_weight = config.get('model', {}).get('loss_function', {}).get('pos_weight', calculated_pos_weight)
        return nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]))
    
    elif loss_type == 'WeightedFocalLoss':
        pos_weight = config.get('model', {}).get('loss_function', {}).get('pos_weight', calculated_pos_weight)
        gamma = config.get('model', {}).get('loss_function', {}).get('gamma', 2.0)
        alpha = config.get('model', {}).get('loss_function', {}).get('alpha', 0.25)
        return WeightedFocalLoss(pos_weight=pos_weight, gamma=gamma, alpha=alpha)
    
    elif loss_type == 'DiceLoss':
        return DiceLoss()
    
    elif loss_type == 'TverskyLoss':
        alpha = config.get('model', {}).get('loss_function', {}).get('alpha', 0.7)
        beta = config.get('model', {}).get('loss_function', {}).get('beta', 0.3)
        return TverskyLoss(alpha=alpha, beta=beta)
    
    elif loss_type == 'ComboLoss':
        bce_weight = config.get('model', {}).get('loss_function', {}).get('bce_weight', 0.5)
        focal_weight = config.get('model', {}).get('loss_function', {}).get('focal_weight', 0.3)
        dice_weight = config.get('model', {}).get('loss_function', {}).get('dice_weight', 0.2)
        pos_weight = config.get('model', {}).get('loss_function', {}).get('pos_weight', calculated_pos_weight)
        return ComboLoss(
            bce_weight=bce_weight,
            focal_weight=focal_weight,
            dice_weight=dice_weight,
            pos_weight=pos_weight
        )
    
    elif loss_type == 'AsymmetricLoss':
        gamma_neg = config.get('model', {}).get('loss_function', {}).get('gamma_neg', 4)
        gamma_pos = config.get('model', {}).get('loss_function', {}).get('gamma_pos', 1)
        return AsymmetricLoss(gamma_neg=gamma_neg, gamma_pos=gamma_pos)
    
    elif loss_type == 'LDAMLoss':
        cls_num_list = config.get('model', {}).get('loss_function', {}).get('cls_num_list', None)
        if cls_num_list is None and pos_ratio is not None:
            # Estimate from pos_ratio
            total = 1000
            pos = int(total * pos_ratio)
            neg = total - pos
            cls_num_list = [neg, pos]
        return LDAMLoss(cls_num_list=cls_num_list)
    
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")
