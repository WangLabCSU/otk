import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import average_precision_score, roc_auc_score, f1_score, precision_score, recall_score
import numpy as np
import os
import time
from tqdm import tqdm
import yaml
import logging
import datetime

from otk.data.data_processor import DataProcessor
from otk.models.model import ECDNA_Model

# Set up logging
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)

# Create a unique log file name based on timestamp
log_timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
log_file = os.path.join(log_dir, f'training_{log_timestamp}.log')

# Create logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Clear existing handlers to prevent duplicate logs
logger.handlers.clear()

# Create formatters
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Create console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

# Create file handler
file_handler = logging.FileHandler(log_file)
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

# Ensure logs are flushed immediately
for handler in logger.handlers:
    handler.flush = lambda: None

# Prevent propagation to root logger to avoid duplicate logs
logger.propagate = False

class FocalLoss(nn.Module):
    """Focal Loss for imbalanced classification"""
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.bce = nn.BCEWithLogitsLoss(reduction='none')
    
    def forward(self, input, target):
        bce_loss = self.bce(input, target)
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class CombinedLoss(nn.Module):
    """Combined loss function for imbalanced classification"""
    def __init__(self, bce_weight=0.5, focal_weight=0.5, alpha=1, gamma=2):
        super(CombinedLoss, self).__init__()
        self.bce_weight = bce_weight
        self.focal_weight = focal_weight
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.focal_loss = FocalLoss(alpha=alpha, gamma=gamma)
    
    def forward(self, input, target):
        bce_loss = self.bce_loss(input, target)
        focal_loss = self.focal_loss(input, target)
        return self.bce_weight * bce_loss + self.focal_weight * focal_loss

class Trainer:
    def __init__(self, config_path, output_dir, gpu=0):
        self.config_path = config_path
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        self.output_dir = output_dir
        self.gpu = gpu
        
        # Set random seed for reproducibility
        seed = self.config.get('training', {}).get('seed', 2026)
        self._set_seed(seed)
        
        # Set device
        if torch.cuda.is_available() and gpu >= 0:
            self.device = torch.device(f'cuda:{gpu}')
            logger.info(f"Using GPU: {gpu}")
        else:
            self.device = torch.device('cpu')
            logger.info("Using CPU")
        
        # Initialize data processor
        self.data_processor = DataProcessor(config_path)
        
        # Process data and record start time
        self.data_processing_start_time = time.time()
        logger.info("Starting data processing...")
        self.data_dict = self.data_processor.process()
        
        # Initialize model
        self.ecdna_model = ECDNA_Model(config_path)
        self.model = self.ecdna_model.get_model().to(self.device)
        
        # Initialize loss function
        self.loss_fn = self._get_loss_function()
        
        # Initialize optimizer
        self.optimizer = self._get_optimizer()
        
        # Initialize learning rate scheduler
        self.scheduler = self._get_scheduler()
        
        # Initialize metrics
        self.metrics = []

    def _set_seed(self, seed):
        """Set random seed for reproducibility"""
        import random
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        logger.info(f"Random seed set to {seed} for reproducibility")

    def _get_loss_function(self):
        """Get loss function based on configuration"""
        from otk.models.custom_losses import (
            WeightedFocalLoss, DiceLoss, TverskyLoss, ComboLoss, 
            AsymmetricLoss, LDAMLoss
        )
        
        loss_type = self.config['model']['loss_function']['type']
        
        # Calculate pos_weight from training data if available
        pos_weight = 100.0  # Default for highly imbalanced data
        if hasattr(self, 'data_dict') and 'train_loader' in self.data_dict:
            # Estimate pos_weight from training data
            train_dataset = self.data_dict['train_loader'].dataset
            if hasattr(train_dataset, 'labels'):
                pos_ratio = train_dataset.labels.mean()
                if pos_ratio > 0:
                    pos_weight = (1 - pos_ratio) / pos_ratio
        
        if loss_type == 'BCEWithLogitsLoss':
            weights = self.config['model']['loss_function'].get('weight', None)
            if weights:
                weights = torch.tensor(weights, dtype=torch.float32).to(self.device)
                return nn.BCEWithLogitsLoss(pos_weight=weights[1])
            else:
                # Only use pos_weight if explicitly specified in config
                config_pos_weight = self.config['model']['loss_function'].get('pos_weight', None)
                if config_pos_weight is not None:
                    return nn.BCEWithLogitsLoss(pos_weight=torch.tensor([config_pos_weight], device=self.device))
                else:
                    # No pos_weight - use standard BCE loss (matches original behavior)
                    return nn.BCEWithLogitsLoss()
        elif loss_type == 'CrossEntropyLoss':
            return nn.CrossEntropyLoss()
        elif loss_type == 'FocalLoss':
            alpha = self.config['model']['loss_function'].get('alpha', 1)
            gamma = self.config['model']['loss_function'].get('gamma', 2)
            return FocalLoss(alpha=alpha, gamma=gamma)
        elif loss_type == 'CombinedLoss':
            from otk.models.optimized_ecdna_model import CombinedLoss as OptimizedCombinedLoss
            focal_weight = self.config['model']['loss_function'].get('focal_weight', 0.4)
            dice_weight = self.config['model']['loss_function'].get('dice_weight', 0.3)
            bce_weight = self.config['model']['loss_function'].get('bce_weight', 0.3)
            alpha = self.config['model']['loss_function'].get('alpha', 0.75)
            gamma = self.config['model']['loss_function'].get('gamma', 2.0)
            config_pos_weight = self.config['model']['loss_function'].get('pos_weight', pos_weight)
            return OptimizedCombinedLoss(
                focal_weight=focal_weight,
                dice_weight=dice_weight,
                bce_weight=bce_weight,
                alpha=alpha,
                gamma=gamma,
                pos_weight=config_pos_weight
            )
        elif loss_type == 'WeightedFocalLoss':
            config_pos_weight = self.config['model']['loss_function'].get('pos_weight', pos_weight)
            gamma = self.config['model']['loss_function'].get('gamma', 2.0)
            alpha = self.config['model']['loss_function'].get('alpha', 0.25)
            return WeightedFocalLoss(pos_weight=config_pos_weight, gamma=gamma, alpha=alpha)
        elif loss_type == 'DiceLoss':
            return DiceLoss()
        elif loss_type == 'TverskyLoss':
            alpha = self.config['model']['loss_function'].get('alpha', 0.7)
            beta = self.config['model']['loss_function'].get('beta', 0.3)
            return TverskyLoss(alpha=alpha, beta=beta)
        elif loss_type == 'ComboLoss':
            bce_weight = self.config['model']['loss_function'].get('bce_weight', 0.5)
            focal_weight = self.config['model']['loss_function'].get('focal_weight', 0.3)
            dice_weight = self.config['model']['loss_function'].get('dice_weight', 0.2)
            config_pos_weight = self.config['model']['loss_function'].get('pos_weight', pos_weight)
            return ComboLoss(
                bce_weight=bce_weight,
                focal_weight=focal_weight,
                dice_weight=dice_weight,
                pos_weight=config_pos_weight
            )
        elif loss_type == 'AsymmetricLoss':
            gamma_neg = self.config['model']['loss_function'].get('gamma_neg', 4)
            gamma_pos = self.config['model']['loss_function'].get('gamma_pos', 1)
            return AsymmetricLoss(gamma_neg=gamma_neg, gamma_pos=gamma_pos)
        elif loss_type == 'LDAMLoss':
            return LDAMLoss()
        elif loss_type == 'TripleMarginRankingLoss':
            from otk.models.dgit_model import TripleMarginRankingLoss
            margin = self.config['model']['loss_function'].get('margin', 0.5)
            ranking_weight = self.config['model']['loss_function'].get('ranking_weight', 0.7)
            config_pos_weight = self.config['model']['loss_function'].get('pos_weight', pos_weight)
            return TripleMarginRankingLoss(
                margin=margin,
                ranking_weight=ranking_weight,
                pos_weight=config_pos_weight
            )
        elif loss_type in ['RecallBiasedTverskyLoss', 'HardNegativeMiningLoss', 'auPRCProxyLoss', 
                          'CostSensitiveRecallLoss', 'LabelSmoothingBCELoss', 'ecDNAOptimizedLoss', 
                          'BalancedPrecisionRecallLoss', 'auPRCOptimizedLoss']:
            # Import ecDNA specialized losses
            from otk.models.ecdna_losses import get_ecdna_loss_function
            return get_ecdna_loss_function(self.config, pos_ratio=None)
        else:
            raise ValueError(f"Unsupported loss function: {loss_type}")
    
    def _get_optimizer(self):
        """Get optimizer based on configuration"""
        optimizer_type = self.config['model']['optimizer']['type']
        lr = self.config['model']['optimizer']['lr']
        weight_decay = self.config['model']['optimizer'].get('weight_decay', 0)
        
        if optimizer_type == 'Adam':
            return optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_type == 'SGD':
            momentum = self.config['model']['optimizer'].get('momentum', 0.9)
            return optim.SGD(self.model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
        elif optimizer_type == 'AdamW':
            return optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_type}")
    
    def _get_scheduler(self):
        """Get learning rate scheduler based on configuration"""
        scheduler_type = self.config['training']['learning_rate_scheduler']['type']
        if scheduler_type == 'ReduceLROnPlateau':
            factor = self.config['training']['learning_rate_scheduler']['factor']
            patience = self.config['training']['learning_rate_scheduler']['patience']
            min_lr = self.config['training']['learning_rate_scheduler']['min_lr']
            return ReduceLROnPlateau(self.optimizer, mode='max', factor=factor, patience=patience, min_lr=min_lr)
        elif scheduler_type == 'CosineAnnealingLR':
            T_max = self.config['training']['epochs']
            eta_min = self.config['training']['learning_rate_scheduler'].get('min_lr', 0.00001)
            return torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=T_max, eta_min=eta_min)
        elif scheduler_type == 'CosineAnnealingWarmRestarts':
            T_0 = self.config['training']['learning_rate_scheduler'].get('T_0', 10)
            T_mult = self.config['training']['learning_rate_scheduler'].get('T_mult', 2)
            eta_min = self.config['training']['learning_rate_scheduler'].get('eta_min', 0.000001)
            return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.optimizer, T_0=T_0, T_mult=T_mult, eta_min=eta_min
            )
        else:
            return None
    
    def train_epoch(self, train_loader, epoch):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        all_preds = []
        all_labels = []
        
        start_time = time.time()
        
        for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Training Epoch {epoch}")):
            inputs, labels = batch
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            
            # Forward pass
            outputs = self.model(inputs)
            
            # Zero the parameter gradients
            self.optimizer.zero_grad()
            
            # Calculate loss
            gene_outputs = outputs
            loss = self.loss_fn(gene_outputs, labels.unsqueeze(1))
            
            # Backward pass and optimize
            loss.backward()
            
            # Gradient clipping if enabled
            if self.config['training'].get('gradient_clipping', {}).get('enabled', False):
                max_norm = self.config['training']['gradient_clipping'].get('max_norm', 1.0)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm)
            
            self.optimizer.step()
            
            # Accumulate loss
            total_loss += loss.item() * inputs.size(0)
            
            # Collect predictions and labels
            all_preds.extend(gene_outputs.cpu().detach().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            # Log batch progress every 50 batches for better real-time feedback
            if (batch_idx + 1) % 50 == 0:
                batch_loss = loss.item()
                current_lr = self.optimizer.param_groups[0]['lr']
                logger.info(f"Epoch {epoch}, Batch {batch_idx + 1}/{len(train_loader)}, Loss: {batch_loss:.4f}, LR: {current_lr:.6f}")
        
        # Calculate metrics
        epoch_loss = total_loss / len(train_loader.dataset)
        epoch_metrics = self._calculate_metrics(np.array(all_preds), np.array(all_labels))
        
        # Log epoch results
        epoch_time = time.time() - start_time
        current_lr = self.optimizer.param_groups[0]['lr']
        logger.info(f"Epoch {epoch} completed in {epoch_time:.2f} seconds")
        logger.info(f"Train Loss: {epoch_loss:.4f}, LR: {current_lr:.6f}")
        for metric, value in epoch_metrics.items():
            logger.info(f"Train {metric}: {value:.4f}")
        
        return epoch_loss, epoch_metrics
    
    def validate(self, val_loader):
        """Validate the model"""
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []
        
        start_time = time.time()
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(val_loader, desc="Validation")):
                inputs, labels = batch
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                
                # Forward pass
                outputs = self.model(inputs)
                
                # Calculate loss
                gene_outputs = outputs
                loss = self.loss_fn(gene_outputs, labels.unsqueeze(1))
                
                # Accumulate loss
                total_loss += loss.item() * inputs.size(0)
                
                # Collect predictions and labels
                all_preds.extend(gene_outputs.cpu().detach().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Calculate metrics
        val_loss = total_loss / len(val_loader.dataset)
        val_metrics = self._calculate_metrics(np.array(all_preds), np.array(all_labels))
        
        # Log validation results
        val_time = time.time() - start_time
        logger.info(f"Validation completed in {val_time:.2f} seconds")
        logger.info(f"Val Loss: {val_loss:.4f}")
        for metric, value in val_metrics.items():
            logger.info(f"Val {metric}: {value:.4f}")
        
        return val_loss, val_metrics
    
    def _calculate_metrics(self, preds, labels):
        """Calculate metrics"""
        import numpy as np
        metrics = {}
        
        # Ensure preds is 1D array
        if preds.ndim > 1:
            preds = preds.ravel()
        
        # Check if predictions are logits (not in [0, 1]) and apply sigmoid
        if preds.min() < 0 or preds.max() > 1:
            logger.info("Applying sigmoid to logits for metric calculation")
            # Apply sigmoid using numpy
            preds = 1 / (1 + np.exp(-preds))
        
        # Find optimal threshold based on F1 score
        thresholds = np.linspace(0, 1, 100)
        f1_scores = []
        for threshold in thresholds:
            binary_preds = (preds > threshold).astype(int)
            f1 = f1_score(labels, binary_preds, zero_division=0)
            f1_scores.append(f1)
        
        # Get optimal threshold
        optimal_threshold = thresholds[np.argmax(f1_scores)]
        # Convert to Python float to avoid YAML serialization issues
        metrics['optimal_threshold'] = float(optimal_threshold)
        
        # Binarize predictions with optimal threshold
        binary_preds = (preds > optimal_threshold).astype(int)
        
        # Debug: Print prediction statistics
        if len(preds) > 0:
            logger.info(f"Prediction stats: min={preds.min():.4f}, max={preds.max():.4f}, mean={preds.mean():.4f}, std={preds.std():.4f}")
            logger.info(f"Binary prediction stats: sum={int(binary_preds.sum())}, count={len(binary_preds)}")
            logger.info(f"Label stats: sum={int(labels.sum())}, count={len(labels)}")
            logger.info(f"Optimal threshold: {optimal_threshold:.4f}")
        
        # Calculate auPRC
        try:
            metrics['auPRC'] = average_precision_score(labels, preds)
        except Exception as e:
            logger.error(f"Error calculating auPRC: {e}")
            metrics['auPRC'] = 0.0
        
        # Calculate AUC
        try:
            metrics['AUC'] = roc_auc_score(labels, preds)
        except Exception as e:
            logger.error(f"Error calculating AUC: {e}")
            metrics['AUC'] = 0.0
        
        # Calculate F1 score
        try:
            metrics['F1'] = f1_score(labels, binary_preds, zero_division=0)
        except Exception as e:
            logger.error(f"Error calculating F1: {e}")
            metrics['F1'] = 0.0
        
        # Calculate precision
        try:
            metrics['Precision'] = precision_score(labels, binary_preds, zero_division=0)
        except Exception as e:
            logger.error(f"Error calculating Precision: {e}")
            metrics['Precision'] = 0.0
        
        # Calculate recall
        try:
            metrics['Recall'] = recall_score(labels, binary_preds, zero_division=0)
        except Exception as e:
            logger.error(f"Error calculating Recall: {e}")
            metrics['Recall'] = 0.0
        
        return metrics
    
    def train(self):
        """Train the model"""
        # Get data from the already processed data_dict
        train_loader = self.data_dict['train_loader']
        val_loader = self.data_dict['val_loader']
        test_loader = self.data_dict['test_loader']
        data_processing_time = time.time() - self.data_processing_start_time
        logger.info(f"Data processing completed in {data_processing_time:.2f} seconds")
        
        # Initialize training variables
        best_val_auPRC = 0
        patience = self.config['training']['early_stopping']['patience']
        min_delta = self.config['training']['early_stopping']['min_delta']
        epochs_no_improve = 0
        best_model_path = os.path.join(self.output_dir, 'best_model.pth')
        
        # Initialize training history tracking
        training_history = {
            'train_loss': [],
            'val_loss': [],
            'train_metrics': [],
            'val_metrics': [],
            'learning_rates': [],
            'epochs': []
        }
        best_val_metrics = None
        best_train_metrics = None
        best_epoch = 0
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        logger.info(f"Output directory created: {self.output_dir}")
        
        # Save configuration
        config_save_path = os.path.join(self.output_dir, 'config.yml')
        with open(config_save_path, 'w') as f:
            yaml.dump(self.config, f)
        logger.info(f"Configuration saved to {config_save_path}")
        
        # Training loop
        total_training_start = time.time()
        logger.info(f"Starting training for {self.config['training']['epochs']} epochs")
        for epoch in range(1, self.config['training']['epochs'] + 1):
            logger.info(f"\n=== Epoch {epoch}/{self.config['training']['epochs']} ===")
            
            # Train for one epoch
            train_loss, train_metrics = self.train_epoch(train_loader, epoch)
            
            # Validate
            val_loss, val_metrics = self.validate(val_loader)
            
            # Record training history
            training_history['epochs'].append(epoch)
            training_history['train_loss'].append(float(train_loss))
            training_history['val_loss'].append(float(val_loss))
            training_history['train_metrics'].append({k: float(v) for k, v in train_metrics.items()})
            training_history['val_metrics'].append({k: float(v) for k, v in val_metrics.items()})
            training_history['learning_rates'].append(float(self.optimizer.param_groups[0]['lr']))
            
            # Update learning rate scheduler
            if self.scheduler:
                # 根据调度器类型决定是否传递参数
                scheduler_type = self.config['training']['learning_rate_scheduler']['type']
                if scheduler_type == 'ReduceLROnPlateau':
                    self.scheduler.step(val_metrics['auPRC'])
                else:
                    # 对于其他调度器（如CosineAnnealingLR），不需要传递参数
                    self.scheduler.step()
                current_lr = self.optimizer.param_groups[0]['lr']
                logger.info(f"Learning rate updated to: {current_lr:.6f}")
            
            # Early stopping
            if val_metrics['auPRC'] > best_val_auPRC + min_delta:
                best_val_auPRC = val_metrics['auPRC']
                best_val_metrics = val_metrics.copy()
                best_train_metrics = train_metrics.copy()
                best_epoch = epoch
                epochs_no_improve = 0
                optimal_threshold = best_val_metrics.get('optimal_threshold')
                self.ecdna_model.save(best_model_path, optimal_threshold)
                threshold_str = f"{optimal_threshold:.4f}" if optimal_threshold is not None else "N/A"
                logger.info(f"New best model saved with auPRC: {best_val_auPRC:.4f}, threshold: {threshold_str}")
            else:
                epochs_no_improve += 1
                logger.info(f"No improvement in validation auPRC for {epochs_no_improve} epochs")
                if epochs_no_improve >= patience:
                    logger.info(f"Early stopping triggered after {epoch} epochs")
                    break
        
        total_training_time = time.time() - total_training_start
        logger.info(f"Total training time: {total_training_time:.2f} seconds")
        
        # Load best model
        logger.info(f"Loading best model from {best_model_path}")
        self.ecdna_model = ECDNA_Model.load(best_model_path)
        self.model = self.ecdna_model.get_model().to(self.device)
        logger.info("Best model loaded successfully")
        
        # Evaluate on test set
        logger.info("Evaluating on test set...")
        test_start_time = time.time()
        test_loss, test_metrics = self.validate(test_loader)
        test_time = time.time() - test_start_time
        logger.info(f"Test evaluation completed in {test_time:.2f} seconds")
        logger.info(f"Test Loss: {test_loss:.4f}")
        logger.info("Test Metrics:")
        for metric, value in test_metrics.items():
            logger.info(f"{metric}: {value:.4f}")
        
        # Get dataset statistics
        train_dataset = train_loader.dataset
        val_dataset = val_loader.dataset
        test_dataset = test_loader.dataset
        
        train_labels = train_dataset.labels if hasattr(train_dataset, 'labels') else []
        val_labels = val_dataset.labels if hasattr(val_dataset, 'labels') else []
        test_labels = test_dataset.labels if hasattr(test_dataset, 'labels') else []
        
        dataset_stats = {
            'train_samples': len(train_dataset),
            'val_samples': len(val_dataset),
            'test_samples': len(test_dataset),
            'train_positive': int(sum(train_labels)) if len(train_labels) > 0 else 0,
            'val_positive': int(sum(val_labels)) if len(val_labels) > 0 else 0,
            'test_positive': int(sum(test_labels)) if len(test_labels) > 0 else 0,
            'train_positive_rate': float(sum(train_labels) / len(train_labels)) if len(train_labels) > 0 else 0.0,
            'val_positive_rate': float(sum(val_labels) / len(val_labels)) if len(val_labels) > 0 else 0.0,
            'test_positive_rate': float(sum(test_labels) / len(test_labels)) if len(test_labels) > 0 else 0.0
        }
        
        # Calculate overfitting analysis
        overfitting_analysis = {}
        if best_train_metrics and best_val_metrics:
            overfitting_analysis = {
                'train_val_auPRC_gap': float(best_train_metrics.get('auPRC', 0) - best_val_metrics.get('auPRC', 0)),
                'train_val_precision_gap': float(best_train_metrics.get('Precision', 0) - best_val_metrics.get('Precision', 0)),
                'train_val_recall_gap': float(best_train_metrics.get('Recall', 0) - best_val_metrics.get('Recall', 0)),
                'overfitting_severity': 'high' if (best_train_metrics.get('auPRC', 0) - best_val_metrics.get('auPRC', 0)) > 0.15 else 'medium' if (best_train_metrics.get('auPRC', 0) - best_val_metrics.get('auPRC', 0)) > 0.05 else 'low'
            }
        
        # Save test metrics
        test_metrics_path = os.path.join(self.output_dir, 'test_metrics.yml')
        with open(test_metrics_path, 'w') as f:
            yaml.dump({k: float(v) for k, v in test_metrics.items()}, f)
        logger.info(f"Test metrics saved to {test_metrics_path}")
        
        # Save training history
        history_path = os.path.join(self.output_dir, 'training_history.yml')
        with open(history_path, 'w') as f:
            yaml.dump(training_history, f)
        logger.info(f"Training history saved to {history_path}")
        
        # Save comprehensive training summary
        training_summary = {
            'model_info': {
                'model_type': self.config['model']['architecture']['type'],
                'input_dim': self.config['model']['architecture'].get('input_dim', 'N/A'),
                'hidden_dims': self.config['model']['architecture'].get('hidden_dims', 'N/A'),
                'dropout_rate': self.config['model']['architecture'].get('dropout_rate', 'N/A'),
                'loss_function': self.config['model']['loss_function']['type'],
                'optimizer': self.config['model']['optimizer']['type'],
                'learning_rate': self.config['model']['optimizer']['lr'],
                'weight_decay': self.config['model']['optimizer'].get('weight_decay', 0),
                'batch_size': self.config['training']['batch_size']
            },
            'dataset_statistics': dataset_stats,
            'training_progress': {
                'epochs_trained': epoch,
                'best_epoch': best_epoch,
                'early_stopped': epochs_no_improve >= patience,
                'total_training_time_seconds': float(total_training_time),
                'data_processing_time_seconds': float(data_processing_time),
                'test_evaluation_time_seconds': float(test_time)
            },
            'performance': {
                'training_set': {k: float(v) for k, v in best_train_metrics.items()} if best_train_metrics else {},
                'validation_set': {k: float(v) for k, v in best_val_metrics.items()} if best_val_metrics else {},
                'test_set': {k: float(v) for k, v in test_metrics.items()}
            },
            'overfitting_analysis': overfitting_analysis,
            'best_val_auPRC': float(best_val_auPRC)
        }
        summary_path = os.path.join(self.output_dir, 'training_summary.yml')
        with open(summary_path, 'w') as f:
            yaml.dump(training_summary, f)
        logger.info(f"Training summary saved to {summary_path}")
        
        # Save log file path to summary
        log_info = {
            'log_file': log_file
        }
        log_info_path = os.path.join(self.output_dir, 'log_info.yml')
        with open(log_info_path, 'w') as f:
            yaml.dump(log_info, f)
        logger.info(f"Log information saved to {log_info_path}")
        
        return best_val_auPRC, test_metrics

def train_model(config_path, output_dir, gpu=0):
    """Train the model"""
    trainer = Trainer(config_path, output_dir, gpu)
    best_val_auPRC, test_metrics = trainer.train()
    logger.info(f"Training completed. Best validation auPRC: {best_val_auPRC:.4f}")
    return best_val_auPRC, test_metrics
