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
        
        # Set device
        if torch.cuda.is_available() and gpu >= 0:
            self.device = torch.device(f'cuda:{gpu}')
            print(f"Using GPU: {gpu}")
        else:
            self.device = torch.device('cpu')
            print("Using CPU")
        
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

    def _get_loss_function(self):
        """Get loss function based on configuration"""
        loss_type = self.config['model']['loss_function']['type']
        if loss_type == 'BCEWithLogitsLoss':
            weights = self.config['model']['loss_function'].get('weight', None)
            if weights:
                weights = torch.tensor(weights, dtype=torch.float32).to(self.device)
                return nn.BCEWithLogitsLoss(pos_weight=weights[1])
            else:
                return nn.BCEWithLogitsLoss()
        elif loss_type == 'CrossEntropyLoss':
            return nn.CrossEntropyLoss()
        elif loss_type == 'FocalLoss':
            alpha = self.config['model']['loss_function'].get('alpha', 1)
            gamma = self.config['model']['loss_function'].get('gamma', 2)
            return FocalLoss(alpha=alpha, gamma=gamma)
        elif loss_type == 'CombinedLoss':
            bce_weight = self.config['model']['loss_function'].get('bce_weight', 0.5)
            focal_weight = self.config['model']['loss_function'].get('focal_weight', 0.5)
            alpha = self.config['model']['loss_function'].get('alpha', 1)
            gamma = self.config['model']['loss_function'].get('gamma', 2)
            return CombinedLoss(bce_weight=bce_weight, focal_weight=focal_weight, alpha=alpha, gamma=gamma)
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
        else:
            return None
    
    def train_epoch(self, train_loader, epoch):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        all_preds = []
        all_labels = []
        all_sample_preds = []
        
        start_time = time.time()
        
        for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Training Epoch {epoch}")):
            # Handle different batch formats
            if len(batch) == 3:
                inputs, labels, amplicon_class = batch
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                amplicon_class = amplicon_class.to(self.device)
                # Forward pass with amplicon class
                outputs = self.model(inputs, amplicon_class)
            else:
                inputs, labels = batch
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                # Forward pass without amplicon class
                outputs = self.model(inputs)
            
            # Zero the parameter gradients
            self.optimizer.zero_grad()
            
            # Handle different model outputs
            if isinstance(outputs, tuple):
                # Transformer model with multiple outputs
                gene_outputs, sample_outputs = outputs
                loss = self.loss_fn(gene_outputs, labels.unsqueeze(1))
                # For sample-level prediction, we would need sample-level labels
                # This is a placeholder for now
            else:
                # MLP model with single output
                gene_outputs = outputs
                loss = self.loss_fn(gene_outputs, labels.unsqueeze(1))
            
            # Backward pass and optimize
            loss.backward()
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
        all_sample_preds = []
        
        start_time = time.time()
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(val_loader, desc="Validation")):
                # Handle different batch formats
                if len(batch) == 3:
                    inputs, labels, amplicon_class = batch
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)
                    amplicon_class = amplicon_class.to(self.device)
                    # Forward pass with amplicon class
                    outputs = self.model(inputs, amplicon_class)
                else:
                    inputs, labels = batch
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)
                    # Forward pass without amplicon class
                    outputs = self.model(inputs)
                
                # Handle different model outputs
                if isinstance(outputs, tuple):
                    # Transformer model with multiple outputs
                    gene_outputs, sample_outputs = outputs
                    loss = self.loss_fn(gene_outputs, labels.unsqueeze(1))
                else:
                    # MLP model with single output
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
        metrics = {}
        
        # Binarize predictions for classification metrics
        binary_preds = (preds > 0.5).astype(int)
        
        # Calculate auPRC
        try:
            metrics['auPRC'] = average_precision_score(labels, preds)
        except:
            metrics['auPRC'] = 0.0
        
        # Calculate AUC
        try:
            metrics['AUC'] = roc_auc_score(labels, preds)
        except:
            metrics['AUC'] = 0.0
        
        # Calculate F1 score
        metrics['F1'] = f1_score(labels, binary_preds, zero_division=0)
        
        # Calculate precision
        metrics['Precision'] = precision_score(labels, binary_preds, zero_division=0)
        
        # Calculate recall
        metrics['Recall'] = recall_score(labels, binary_preds, zero_division=0)
        
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
            
            # Update learning rate scheduler
            if self.scheduler:
                self.scheduler.step(val_metrics['auPRC'])
                current_lr = self.optimizer.param_groups[0]['lr']
                logger.info(f"Learning rate updated to: {current_lr:.6f}")
            
            # Early stopping
            if val_metrics['auPRC'] > best_val_auPRC + min_delta:
                best_val_auPRC = val_metrics['auPRC']
                epochs_no_improve = 0
                # Save best model
                self.ecdna_model.save(best_model_path)
                logger.info(f"New best model saved with auPRC: {best_val_auPRC:.4f} at {best_model_path}")
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
        
        # Save test metrics
        test_metrics_path = os.path.join(self.output_dir, 'test_metrics.yml')
        with open(test_metrics_path, 'w') as f:
            yaml.dump(test_metrics, f)
        logger.info(f"Test metrics saved to {test_metrics_path}")
        
        # Save training summary
        training_summary = {
            'best_val_auPRC': best_val_auPRC,
            'test_metrics': test_metrics,
            'epochs_trained': epoch,
            'early_stopped': epochs_no_improve >= patience,
            'total_training_time': total_training_time,
            'data_processing_time': data_processing_time,
            'test_evaluation_time': test_time
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
    print(f"Training completed. Best validation auPRC: {best_val_auPRC:.4f}")
    return best_val_auPRC, test_metrics
