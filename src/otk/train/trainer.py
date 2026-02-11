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

from otk.data.data_processor import DataProcessor
from otk.models.model import ECDNA_Model

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
    
    def train_epoch(self, train_loader):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        all_preds = []
        all_labels = []
        
        for batch in tqdm(train_loader, desc="Training"):
            inputs, labels = batch
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            
            # Zero the parameter gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(inputs)
            loss = self.loss_fn(outputs, labels.unsqueeze(1))
            
            # Backward pass and optimize
            loss.backward()
            self.optimizer.step()
            
            # Accumulate loss
            total_loss += loss.item() * inputs.size(0)
            
            # Collect predictions and labels
            all_preds.extend(outputs.cpu().detach().numpy())
            all_labels.extend(labels.cpu().numpy())
        
        # Calculate metrics
        epoch_loss = total_loss / len(train_loader.dataset)
        epoch_metrics = self._calculate_metrics(np.array(all_preds), np.array(all_labels))
        
        return epoch_loss, epoch_metrics
    
    def validate(self, val_loader):
        """Validate the model"""
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                inputs, labels = batch
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                
                # Forward pass
                outputs = self.model(inputs)
                loss = self.loss_fn(outputs, labels.unsqueeze(1))
                
                # Accumulate loss
                total_loss += loss.item() * inputs.size(0)
                
                # Collect predictions and labels
                all_preds.extend(outputs.cpu().detach().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Calculate metrics
        val_loss = total_loss / len(val_loader.dataset)
        val_metrics = self._calculate_metrics(np.array(all_preds), np.array(all_labels))
        
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
        # Process data
        data_dict = self.data_processor.process()
        train_loader = data_dict['train_loader']
        val_loader = data_dict['val_loader']
        test_loader = data_dict['test_loader']
        
        # Initialize training variables
        best_val_auPRC = 0
        patience = self.config['training']['early_stopping']['patience']
        min_delta = self.config['training']['early_stopping']['min_delta']
        epochs_no_improve = 0
        best_model_path = os.path.join(self.output_dir, 'best_model.pth')
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Training loop
        for epoch in range(self.config['training']['epochs']):
            print(f"\nEpoch {epoch+1}/{self.config['training']['epochs']}")
            
            # Train for one epoch
            train_loss, train_metrics = self.train_epoch(train_loader)
            
            # Validate
            val_loss, val_metrics = self.validate(val_loader)
            
            # Print metrics
            print(f"Train Loss: {train_loss:.4f}, Train auPRC: {train_metrics['auPRC']:.4f}")
            print(f"Val Loss: {val_loss:.4f}, Val auPRC: {val_metrics['auPRC']:.4f}")
            
            # Update learning rate scheduler
            if self.scheduler:
                self.scheduler.step(val_metrics['auPRC'])
            
            # Early stopping
            if val_metrics['auPRC'] > best_val_auPRC + min_delta:
                best_val_auPRC = val_metrics['auPRC']
                epochs_no_improve = 0
                # Save best model
                self.ecdna_model.save(best_model_path)
                print(f"New best model saved with auPRC: {best_val_auPRC:.4f}")
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    print(f"Early stopping triggered after {epoch+1} epochs")
                    break
        
        # Load best model
        self.ecdna_model = ECDNA_Model.load(best_model_path)
        self.model = self.ecdna_model.get_model().to(self.device)
        
        # Evaluate on test set
        print("\nEvaluating on test set...")
        test_loss, test_metrics = self.validate(test_loader)
        print(f"Test Loss: {test_loss:.4f}")
        print("Test Metrics:")
        for metric, value in test_metrics.items():
            print(f"{metric}: {value:.4f}")
        
        # Save test metrics
        with open(os.path.join(self.output_dir, 'test_metrics.yml'), 'w') as f:
            yaml.dump(test_metrics, f)
        
        return best_val_auPRC, test_metrics

def train_model(config_path, output_dir, gpu=0):
    """Train the model"""
    trainer = Trainer(config_path, output_dir, gpu)
    best_val_auPRC, test_metrics = trainer.train()
    print(f"Training completed. Best validation auPRC: {best_val_auPRC:.4f}")
    return best_val_auPRC, test_metrics
