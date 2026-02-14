#!/usr/bin/env python
"""
Neural Network Models for ecDNA Prediction

Unified implementation using BaseEcDNAModel interface.
All models use seed=2026 for reproducibility.
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional
from pathlib import Path
import logging

from .base_model import BaseEcDNAModel

logger = logging.getLogger(__name__)

RANDOM_SEED = 2026


def set_random_seed(seed: int = RANDOM_SEED):
    """Set random seed for reproducibility"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def train_neural_model(
    model, train_loader, X_val, y_val, device,
    n_epochs=100, lr=0.001, weight_decay=0.01, pos_weight=10.0,
    patience=15, log_interval=5, model_name="Model"
):
    """Generic training function with detailed logging"""
    from sklearn.metrics import average_precision_score, roc_auc_score
    import time
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]).to(device))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10)
    
    best_val_auprc = 0
    patience_counter = 0
    
    logger.info(f"Starting {model_name} training: {n_epochs} epochs, {len(train_loader)} batches/epoch")
    start_time = time.time()
    
    for epoch in range(n_epochs):
        model.train()
        epoch_loss = 0
        n_batches = 0
        
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs.squeeze(-1), batch_y.squeeze(-1))
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1
        
        avg_loss = epoch_loss / n_batches
        scheduler.step(avg_loss)
        
        # Validation
        if X_val is not None and y_val is not None and (epoch + 1) % log_interval == 0:
            model.eval()
            with torch.no_grad():
                val_probs = []
                for batch_x, _ in train_loader:
                    batch_x = batch_x.to(device)
                    outputs = torch.sigmoid(model(batch_x))
                    val_probs.extend(outputs.cpu().numpy().flatten())
            
            # Use a subset for validation to speed up
            val_size = min(100000, len(y_val))
            val_idx = np.random.choice(len(y_val), val_size, replace=False)
            val_probs_subset = np.array(val_probs)[:val_size] if len(val_probs) >= val_size else np.array(val_probs)
            y_val_subset = y_val[val_idx] if hasattr(y_val, '__getitem__') else y_val[:val_size]
            
            logger.info(f"Epoch {epoch+1}/{n_epochs} - Loss: {avg_loss:.4f}")
            
            # Early stopping check
            if patience > 0:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break
        elif (epoch + 1) % 10 == 0:
            logger.info(f"Epoch {epoch+1}/{n_epochs} - Loss: {avg_loss:.4f}")
    
    training_time = time.time() - start_time
    logger.info(f"Training completed in {training_time:.1f}s")
    
    return model


class BaselineMLPModel(BaseEcDNAModel):
    """Baseline MLP model"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None, device: str = 'auto'):
        super().__init__(config)
        self.config = config or {}
        self.model = None
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
    def prepare_features(self, df: pd.DataFrame) -> np.ndarray:
        """Prepare features"""
        feature_cols = [c for c in df.columns if c not in ['sample', 'gene_id', 'y']]
        return df[feature_cols].fillna(0).values.astype(np.float32)
    
    def fit(self, X_train, y_train, X_val=None, y_val=None, **kwargs):
        from torch.utils.data import DataLoader, TensorDataset
        from sklearn.metrics import average_precision_score, roc_auc_score, precision_score, recall_score, f1_score
        import time
        
        set_random_seed(RANDOM_SEED)
        
        X_train_arr = self.prepare_features(X_train)
        y_train_arr = y_train.values.astype(np.float32)
        
        input_dim = X_train_arr.shape[1]
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1)
        ).to(self.device)
        
        train_dataset = TensorDataset(
            torch.tensor(X_train_arr),
            torch.tensor(y_train_arr).unsqueeze(1)
        )
        train_loader = DataLoader(train_dataset, batch_size=4096, shuffle=True)
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([10.0]).to(self.device))
        
        best_val_auprc = 0
        best_model_state = None
        best_epoch = 0
        patience_counter = 0
        max_patience = 10
        n_epochs = 100
        
        logger.info(f"Starting BaselineMLP training: {n_epochs} epochs, {len(train_loader)} batches/epoch")
        start_time = time.time()
        
        for epoch in range(n_epochs):
            self.model.train()
            epoch_loss = 0
            n_batches = 0
            
            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                n_batches += 1
            
            train_loss = epoch_loss / n_batches
            
            # Compute train metrics
            self.is_fitted = True
            train_probs = self.predict_proba(X_train)
            train_auprc = average_precision_score(y_train.values, train_probs)
            train_auc = roc_auc_score(y_train.values, train_probs)
            
            # Compute val metrics
            val_loss = None
            val_auprc = 0
            val_auc = 0
            
            if X_val is not None and y_val is not None:
                val_probs = self.predict_proba(X_val)
                val_auprc = average_precision_score(y_val.values, val_probs)
                val_auc = roc_auc_score(y_val.values, val_probs)
            
            # Log every epoch
            logger.info(f"Epoch {epoch+1}/{n_epochs}")
            logger.info(f"  Train - Loss: {train_loss:.4f}, auPRC: {train_auprc:.4f}, AUC: {train_auc:.4f}")
            if X_val is not None and y_val is not None:
                logger.info(f"  Val   - auPRC: {val_auprc:.4f}, AUC: {val_auc:.4f}")
            
            # Save best model and early stopping
            if val_auprc > best_val_auprc:
                best_val_auprc = val_auprc
                best_model_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                best_epoch = epoch + 1
                logger.info(f"  New best model! Val auPRC: {best_val_auprc:.4f}")
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= max_patience:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break
        
        # Load best model state
        if best_model_state is not None:
            self.model.load_state_dict({k: v.to(self.device) for k, v in best_model_state.items()})
            logger.info(f"Loaded best model from epoch {best_epoch} with Val auPRC: {best_val_auprc:.4f}")
        
        training_time = time.time() - start_time
        logger.info(f"Training completed in {training_time:.1f}s")
        
        if X_val is not None and y_val is not None:
            val_probs = self.predict_proba(X_val)
            self.optimal_threshold = self._find_optimal_threshold(y_val.values, val_probs)
            logger.info(f"Optimal threshold: {self.optimal_threshold:.4f}")
        
        self.is_fitted = True
        return self
    
    def _find_optimal_threshold(self, y_true, y_prob):
        from sklearn.metrics import precision_recall_curve
        precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
        optimal_idx = np.argmax(f1_scores)
        return thresholds[optimal_idx] if optimal_idx < len(thresholds) else 0.5
    
    def predict_proba(self, X):
        if not self.is_fitted:
            raise ValueError("Model not fitted")
        
        from torch.utils.data import DataLoader, TensorDataset
        
        X_arr = self.prepare_features(X)
        dataset = TensorDataset(torch.tensor(X_arr))
        loader = DataLoader(dataset, batch_size=4096, shuffle=False)
        
        self.model.eval()
        probs = []
        with torch.no_grad():
            for batch in loader:
                batch_x = batch[0].to(self.device)
                outputs = torch.sigmoid(self.model(batch_x))
                probs.extend(outputs.cpu().numpy().flatten())
        
        return np.array(probs)
    
    def save(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            'model_state': self.model.state_dict(),
            'config': self.config,
            'optimal_threshold': self.optimal_threshold
        }, path)
    
    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        # Model will be initialized on first fit
        self.optimal_threshold = checkpoint['optimal_threshold']
        self.is_fitted = True
        return self


class TransformerEcDNAModel(BaseEcDNAModel):
    """Transformer-based ecDNA prediction model"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None, device: str = 'auto'):
        super().__init__(config)
        self.config = config or {
            'input_dim': 57,
            'hidden_dim': 128,
            'num_heads': 4,
            'num_layers': 3,
            'dropout': 0.3
        }
        self.model = None
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
    
    def prepare_features(self, df: pd.DataFrame) -> np.ndarray:
        """Prepare features"""
        feature_cols = [c for c in df.columns if c not in ['sample', 'gene_id', 'y']]
        return df[feature_cols].fillna(0).values.astype(np.float32)
    
    def fit(self, X_train, y_train, X_val=None, y_val=None, **kwargs):
        from torch.utils.data import DataLoader, TensorDataset
        from sklearn.metrics import average_precision_score, roc_auc_score, precision_score, recall_score, f1_score
        import time
        
        set_random_seed(RANDOM_SEED)
        
        X_train_arr = self.prepare_features(X_train)
        y_train_arr = y_train.values.astype(np.float32)
        
        # Create transformer model
        input_dim = X_train_arr.shape[1]
        self.model = TransformerModel(
            input_dim=input_dim,
            hidden_dim=self.config['hidden_dim'],
            num_heads=self.config['num_heads'],
            num_layers=self.config['num_layers'],
            dropout=self.config['dropout']
        ).to(self.device)
        
        # Initialize weights for stability
        def init_weights(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        self.model.apply(init_weights)
        
        # Training
        train_dataset = TensorDataset(
            torch.tensor(X_train_arr),
            torch.tensor(y_train_arr).unsqueeze(1)
        )
        train_loader = DataLoader(train_dataset, batch_size=4096, shuffle=True)
        
        # Use lower learning rate for stability
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=0.0001, weight_decay=0.01)
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([10.0]).to(self.device))
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10)
        
        max_grad_norm = 1.0
        best_val_auprc = 0
        patience_counter = 0
        max_patience = 15
        n_epochs = 150
        
        logger.info(f"Starting Transformer training: {n_epochs} epochs, {len(train_loader)} batches/epoch")
        logger.info(f"Model config: hidden_dim={self.config['hidden_dim']}, num_heads={self.config['num_heads']}, num_layers={self.config['num_layers']}")
        logger.info(f"Learning rate: 0.0001, Gradient clipping: {max_grad_norm}")
        start_time = time.time()
        
        for epoch in range(n_epochs):
            self.model.train()
            epoch_loss = 0
            n_batches = 0
            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(batch_x)
                loss = criterion(outputs.squeeze(-1), batch_y.squeeze(-1))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
                optimizer.step()
                epoch_loss += loss.item()
                n_batches += 1
            
            train_loss = epoch_loss / n_batches
            scheduler.step(train_loss)
            
            # Check for NaN loss
            if np.isnan(train_loss) or np.isinf(train_loss):
                logger.warning(f"NaN/Inf loss detected at epoch {epoch+1}, stopping training")
                break
            
            # Compute train metrics
            self.is_fitted = True
            train_probs = self.predict_proba(X_train)
            if np.any(np.isnan(train_probs)):
                logger.warning(f"NaN train predictions at epoch {epoch+1}")
                train_probs = np.nan_to_num(train_probs, nan=0.5)
            train_preds = (train_probs >= 0.5).astype(int)
            train_auprc = average_precision_score(y_train.values, train_probs)
            train_auc = roc_auc_score(y_train.values, train_probs)
            train_precision = precision_score(y_train.values, train_preds, zero_division=0)
            train_recall = recall_score(y_train.values, train_preds, zero_division=0)
            train_f1 = f1_score(y_train.values, train_preds, zero_division=0)
            
            # Compute val metrics
            val_loss = None
            val_auprc = 0
            val_auc = 0
            val_precision = 0
            val_recall = 0
            val_f1 = 0
            
            if X_val is not None and y_val is not None:
                self.is_fitted = True
                val_probs = self.predict_proba(X_val)
                if np.any(np.isnan(val_probs)):
                    logger.warning(f"NaN val predictions at epoch {epoch+1}")
                    val_probs = np.nan_to_num(val_probs, nan=0.5)
                val_preds = (val_probs >= 0.5).astype(int)
                
                # Compute val loss
                val_dataset = TensorDataset(
                    torch.tensor(self.prepare_features(X_val)),
                    torch.tensor(y_val.values.astype(np.float32)).unsqueeze(1)
                )
                val_loader_tmp = DataLoader(val_dataset, batch_size=4096, shuffle=False)
                self.model.eval()
                val_loss_sum = 0
                val_batches = 0
                with torch.no_grad():
                    for batch_x, batch_y in val_loader_tmp:
                        batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                        outputs = self.model(batch_x)
                        loss = criterion(outputs.squeeze(-1), batch_y.squeeze(-1))
                        val_loss_sum += loss.item()
                        val_batches += 1
                val_loss = val_loss_sum / val_batches
                
                val_auprc = average_precision_score(y_val.values, val_probs)
                val_auc = roc_auc_score(y_val.values, val_probs)
                val_precision = precision_score(y_val.values, val_preds, zero_division=0)
                val_recall = recall_score(y_val.values, val_preds, zero_division=0)
                val_f1 = f1_score(y_val.values, val_preds, zero_division=0)
            
            # Log every epoch
            logger.info(f"Epoch {epoch+1}/{n_epochs}")
            logger.info(f"  Train - Loss: {train_loss:.4f}, auPRC: {train_auprc:.4f}, AUC: {train_auc:.4f}, P: {train_precision:.4f}, R: {train_recall:.4f}, F1: {train_f1:.4f}")
            if val_loss is not None:
                logger.info(f"  Val   - Loss: {val_loss:.4f}, auPRC: {val_auprc:.4f}, AUC: {val_auc:.4f}, P: {val_precision:.4f}, R: {val_recall:.4f}, F1: {val_f1:.4f}")
            
            # Save best model and early stopping
            if val_auprc > best_val_auprc:
                best_val_auprc = val_auprc
                patience_counter = 0
                # Save best model state
                self.best_model_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                self.best_epoch = epoch + 1
                self.best_val_auprc = best_val_auprc
                logger.info(f"  New best model! Val auPRC: {best_val_auprc:.4f}")
            else:
                patience_counter += 1
                if patience_counter >= max_patience:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break
        
        # Load best model state
        if hasattr(self, 'best_model_state') and self.best_model_state is not None:
            self.model.load_state_dict({k: v.to(self.device) for k, v in self.best_model_state.items()})
            logger.info(f"Loaded best model from epoch {self.best_epoch} with Val auPRC: {self.best_val_auprc:.4f}")
        
        training_time = time.time() - start_time
        logger.info(f"Training completed in {training_time:.1f}s")
        
        if X_val is not None and y_val is not None:
            self.is_fitted = True
            val_probs = self.predict_proba(X_val)
            if not np.any(np.isnan(val_probs)):
                self.optimal_threshold = self._find_optimal_threshold(y_val.values, val_probs)
                logger.info(f"Optimal threshold: {self.optimal_threshold:.4f}")
        
        self.is_fitted = True
        return self
    
    def _find_optimal_threshold(self, y_true, y_prob):
        from sklearn.metrics import precision_recall_curve
        precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
        optimal_idx = np.argmax(f1_scores)
        return thresholds[optimal_idx] if optimal_idx < len(thresholds) else 0.5
    
    def predict_proba(self, X):
        if not self.is_fitted:
            raise ValueError("Model not fitted")
        
        from torch.utils.data import DataLoader, TensorDataset
        
        X_arr = self.prepare_features(X)
        dataset = TensorDataset(torch.tensor(X_arr))
        loader = DataLoader(dataset, batch_size=4096, shuffle=False)
        
        self.model.eval()
        probs = []
        with torch.no_grad():
            for batch in loader:
                batch_x = batch[0].to(self.device)
                outputs = torch.sigmoid(self.model(batch_x))
                probs.extend(outputs.cpu().numpy().flatten())
        
        return np.array(probs)
    
    def save(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            'model_state': self.model.state_dict(),
            'config': self.config,
            'optimal_threshold': self.optimal_threshold
        }, path)
    
    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.config = checkpoint['config']
        self.optimal_threshold = checkpoint['optimal_threshold']
        self.is_fitted = True
        return self


class TransformerModel(nn.Module):
    """Transformer architecture for ecDNA prediction with improved stability"""
    
    def __init__(self, input_dim: int, hidden_dim: int, num_heads: int, num_layers: int, dropout: float):
        super().__init__()
        
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout)
        )
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
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
                nn.init.xavier_uniform_(module.weight, gain=0.5)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(self, x):
        x = torch.clamp(x, min=-100, max=100)
        x = torch.nan_to_num(x, nan=0.0, posinf=100.0, neginf=-100.0)
        
        x = x.unsqueeze(1)
        h = self.input_proj(x)
        h = self.transformer(h)
        h = h.squeeze(1)
        output = self.classifier(h)
        return output.squeeze(-1)


class DeepResidualModel(BaseEcDNAModel):
    """Deep Residual Network model"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None, device: str = 'auto'):
        super().__init__(config)
        self.config = config or {}
        self.model = None
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
    
    def prepare_features(self, df: pd.DataFrame) -> np.ndarray:
        feature_cols = [c for c in df.columns if c not in ['sample', 'gene_id', 'y']]
        return df[feature_cols].fillna(0).values.astype(np.float32)
    
    def fit(self, X_train, y_train, X_val=None, y_val=None, **kwargs):
        from torch.utils.data import DataLoader, TensorDataset
        from sklearn.metrics import average_precision_score, roc_auc_score
        import time
        
        set_random_seed(RANDOM_SEED)
        
        X_train_arr = self.prepare_features(X_train)
        y_train_arr = y_train.values.astype(np.float32)
        
        input_dim = X_train_arr.shape[1]
        self.model = DeepResidualNet(input_dim).to(self.device)
        
        train_dataset = TensorDataset(
            torch.tensor(X_train_arr),
            torch.tensor(y_train_arr).unsqueeze(1)
        )
        train_loader = DataLoader(train_dataset, batch_size=4096, shuffle=True)
        
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=0.001, weight_decay=0.01)
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([10.0]).to(self.device))
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10)
        
        best_val_auprc = 0
        best_model_state = None
        best_epoch = 0
        patience_counter = 0
        max_patience = 15
        n_epochs = 150
        
        logger.info(f"Starting DeepResidual training: {n_epochs} epochs, {len(train_loader)} batches/epoch")
        start_time = time.time()
        
        for epoch in range(n_epochs):
            self.model.train()
            epoch_loss = 0
            n_batches = 0
            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(batch_x)
                loss = criterion(outputs.squeeze(-1), batch_y.squeeze(-1))
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                n_batches += 1
            
            train_loss = epoch_loss / n_batches
            scheduler.step(train_loss)
            
            # Compute train metrics
            self.is_fitted = True
            train_probs = self.predict_proba(X_train)
            train_auprc = average_precision_score(y_train.values, train_probs)
            train_auc = roc_auc_score(y_train.values, train_probs)
            
            # Compute val metrics
            val_auprc = 0
            val_auc = 0
            if X_val is not None and y_val is not None:
                val_probs = self.predict_proba(X_val)
                val_auprc = average_precision_score(y_val.values, val_probs)
                val_auc = roc_auc_score(y_val.values, val_probs)
            
            # Log every epoch
            logger.info(f"Epoch {epoch+1}/{n_epochs}")
            logger.info(f"  Train - Loss: {train_loss:.4f}, auPRC: {train_auprc:.4f}, AUC: {train_auc:.4f}")
            if X_val is not None and y_val is not None:
                logger.info(f"  Val   - auPRC: {val_auprc:.4f}, AUC: {val_auc:.4f}")
            
            # Save best model and early stopping
            if val_auprc > best_val_auprc:
                best_val_auprc = val_auprc
                best_model_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                best_epoch = epoch + 1
                logger.info(f"  New best model! Val auPRC: {best_val_auprc:.4f}")
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= max_patience:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break
        
        # Load best model state
        if best_model_state is not None:
            self.model.load_state_dict({k: v.to(self.device) for k, v in best_model_state.items()})
            logger.info(f"Loaded best model from epoch {best_epoch} with Val auPRC: {best_val_auprc:.4f}")
        
        training_time = time.time() - start_time
        logger.info(f"Training completed in {training_time:.1f}s")
        
        self.is_fitted = True
        if X_val is not None and y_val is not None:
            val_probs = self.predict_proba(X_val)
            self.optimal_threshold = self._find_optimal_threshold(y_val.values, val_probs)
            logger.info(f"Optimal threshold: {self.optimal_threshold:.4f}")
        return self
    
    def _find_optimal_threshold(self, y_true, y_prob):
        from sklearn.metrics import precision_recall_curve
        precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
        optimal_idx = np.argmax(f1_scores)
        return thresholds[optimal_idx] if optimal_idx < len(thresholds) else 0.5
    
    def predict_proba(self, X):
        if not self.is_fitted:
            raise ValueError("Model not fitted")
        from torch.utils.data import DataLoader, TensorDataset
        X_arr = self.prepare_features(X)
        dataset = TensorDataset(torch.tensor(X_arr))
        loader = DataLoader(dataset, batch_size=4096, shuffle=False)
        self.model.eval()
        probs = []
        with torch.no_grad():
            for batch in loader:
                batch_x = batch[0].to(self.device)
                outputs = torch.sigmoid(self.model(batch_x))
                probs.extend(outputs.cpu().numpy().flatten())
        return np.array(probs)
    
    def save(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({'model_state': self.model.state_dict(), 'config': self.config,
                    'optimal_threshold': self.optimal_threshold}, path)
    
    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.optimal_threshold = checkpoint['optimal_threshold']
        self.is_fitted = True
        return self


class OptimizedResidualModel(BaseEcDNAModel):
    """Optimized Residual Network model"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None, device: str = 'auto'):
        super().__init__(config)
        self.config = config or {}
        self.model = None
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
    
    def prepare_features(self, df: pd.DataFrame) -> np.ndarray:
        feature_cols = [c for c in df.columns if c not in ['sample', 'gene_id', 'y']]
        return df[feature_cols].fillna(0).values.astype(np.float32)
    
    def fit(self, X_train, y_train, X_val=None, y_val=None, **kwargs):
        from torch.utils.data import DataLoader, TensorDataset
        from sklearn.metrics import average_precision_score, roc_auc_score
        import time
        
        set_random_seed(RANDOM_SEED)
        
        X_train_arr = self.prepare_features(X_train)
        y_train_arr = y_train.values.astype(np.float32)
        
        input_dim = X_train_arr.shape[1]
        self.model = OptimizedResidualNet(input_dim).to(self.device)
        
        train_dataset = TensorDataset(
            torch.tensor(X_train_arr),
            torch.tensor(y_train_arr).unsqueeze(1)
        )
        train_loader = DataLoader(train_dataset, batch_size=4096, shuffle=True)
        
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=0.001, weight_decay=0.01)
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([10.0]).to(self.device))
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10)
        
        best_val_auprc = 0
        best_model_state = None
        best_epoch = 0
        patience_counter = 0
        max_patience = 15
        n_epochs = 150
        
        logger.info(f"Starting OptimizedResidual training: {n_epochs} epochs, {len(train_loader)} batches/epoch")
        start_time = time.time()
        
        for epoch in range(n_epochs):
            self.model.train()
            epoch_loss = 0
            n_batches = 0
            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(batch_x)
                loss = criterion(outputs.squeeze(-1), batch_y.squeeze(-1))
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                n_batches += 1
            
            train_loss = epoch_loss / n_batches
            scheduler.step(train_loss)
            
            # Compute train metrics
            self.is_fitted = True
            train_probs = self.predict_proba(X_train)
            train_auprc = average_precision_score(y_train.values, train_probs)
            train_auc = roc_auc_score(y_train.values, train_probs)
            
            # Compute val metrics
            val_auprc = 0
            val_auc = 0
            if X_val is not None and y_val is not None:
                val_probs = self.predict_proba(X_val)
                val_auprc = average_precision_score(y_val.values, val_probs)
                val_auc = roc_auc_score(y_val.values, val_probs)
            
            # Log every epoch
            logger.info(f"Epoch {epoch+1}/{n_epochs}")
            logger.info(f"  Train - Loss: {train_loss:.4f}, auPRC: {train_auprc:.4f}, AUC: {train_auc:.4f}")
            if X_val is not None and y_val is not None:
                logger.info(f"  Val   - auPRC: {val_auprc:.4f}, AUC: {val_auc:.4f}")
            
            # Save best model and early stopping
            if val_auprc > best_val_auprc:
                best_val_auprc = val_auprc
                best_model_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                best_epoch = epoch + 1
                logger.info(f"  New best model! Val auPRC: {best_val_auprc:.4f}")
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= max_patience:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break
        
        # Load best model state
        if best_model_state is not None:
            self.model.load_state_dict({k: v.to(self.device) for k, v in best_model_state.items()})
            logger.info(f"Loaded best model from epoch {best_epoch} with Val auPRC: {best_val_auprc:.4f}")
        
        training_time = time.time() - start_time
        logger.info(f"Training completed in {training_time:.1f}s")
        
        self.is_fitted = True
        if X_val is not None and y_val is not None:
            val_probs = self.predict_proba(X_val)
            self.optimal_threshold = self._find_optimal_threshold(y_val.values, val_probs)
            logger.info(f"Optimal threshold: {self.optimal_threshold:.4f}")
        return self
    
    def _find_optimal_threshold(self, y_true, y_prob):
        from sklearn.metrics import precision_recall_curve
        precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
        optimal_idx = np.argmax(f1_scores)
        return thresholds[optimal_idx] if optimal_idx < len(thresholds) else 0.5
    
    def predict_proba(self, X):
        if not self.is_fitted:
            raise ValueError("Model not fitted")
        from torch.utils.data import DataLoader, TensorDataset
        X_arr = self.prepare_features(X)
        dataset = TensorDataset(torch.tensor(X_arr))
        loader = DataLoader(dataset, batch_size=4096, shuffle=False)
        self.model.eval()
        probs = []
        with torch.no_grad():
            for batch in loader:
                batch_x = batch[0].to(self.device)
                outputs = torch.sigmoid(self.model(batch_x))
                probs.extend(outputs.cpu().numpy().flatten())
        return np.array(probs)
    
    def save(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({'model_state': self.model.state_dict(), 'config': self.config,
                    'optimal_threshold': self.optimal_threshold}, path)
    
    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.optimal_threshold = checkpoint['optimal_threshold']
        self.is_fitted = True
        return self


class DGITSuperModel(BaseEcDNAModel):
    """DGIT Super model"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None, device: str = 'auto'):
        super().__init__(config)
        self.config = config or {}
        self.model = None
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
    
    def prepare_features(self, df: pd.DataFrame) -> np.ndarray:
        feature_cols = [c for c in df.columns if c not in ['sample', 'gene_id', 'y']]
        return df[feature_cols].fillna(0).values.astype(np.float32)
    
    def fit(self, X_train, y_train, X_val=None, y_val=None, **kwargs):
        from torch.utils.data import DataLoader, TensorDataset
        from sklearn.metrics import average_precision_score, roc_auc_score
        import time
        
        set_random_seed(RANDOM_SEED)
        
        X_train_arr = self.prepare_features(X_train)
        y_train_arr = y_train.values.astype(np.float32)
        
        input_dim = X_train_arr.shape[1]
        self.model = DGITSuperNet(input_dim).to(self.device)
        
        train_dataset = TensorDataset(
            torch.tensor(X_train_arr),
            torch.tensor(y_train_arr).unsqueeze(1)
        )
        train_loader = DataLoader(train_dataset, batch_size=4096, shuffle=True)
        
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=0.001, weight_decay=0.01)
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([10.0]).to(self.device))
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10)
        
        best_val_auprc = 0
        best_model_state = None
        best_epoch = 0
        patience_counter = 0
        max_patience = 15
        n_epochs = 150
        
        logger.info(f"Starting DGITSuper training: {n_epochs} epochs, {len(train_loader)} batches/epoch")
        start_time = time.time()
        
        for epoch in range(n_epochs):
            self.model.train()
            epoch_loss = 0
            n_batches = 0
            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(batch_x)
                loss = criterion(outputs.squeeze(-1), batch_y.squeeze(-1))
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                n_batches += 1
            
            train_loss = epoch_loss / n_batches
            scheduler.step(train_loss)
            
            # Compute train metrics
            self.is_fitted = True
            train_probs = self.predict_proba(X_train)
            train_auprc = average_precision_score(y_train.values, train_probs)
            train_auc = roc_auc_score(y_train.values, train_probs)
            
            # Compute val metrics
            val_auprc = 0
            val_auc = 0
            if X_val is not None and y_val is not None:
                val_probs = self.predict_proba(X_val)
                val_auprc = average_precision_score(y_val.values, val_probs)
                val_auc = roc_auc_score(y_val.values, val_probs)
            
            # Log every epoch
            logger.info(f"Epoch {epoch+1}/{n_epochs}")
            logger.info(f"  Train - Loss: {train_loss:.4f}, auPRC: {train_auprc:.4f}, AUC: {train_auc:.4f}")
            if X_val is not None and y_val is not None:
                logger.info(f"  Val   - auPRC: {val_auprc:.4f}, AUC: {val_auc:.4f}")
            
            # Save best model and early stopping
            if val_auprc > best_val_auprc:
                best_val_auprc = val_auprc
                best_model_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                best_epoch = epoch + 1
                logger.info(f"  New best model! Val auPRC: {best_val_auprc:.4f}")
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= max_patience:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break
        
        # Load best model state
        if best_model_state is not None:
            self.model.load_state_dict({k: v.to(self.device) for k, v in best_model_state.items()})
            logger.info(f"Loaded best model from epoch {best_epoch} with Val auPRC: {best_val_auprc:.4f}")
        
        training_time = time.time() - start_time
        logger.info(f"Training completed in {training_time:.1f}s")
        
        self.is_fitted = True
        if X_val is not None and y_val is not None:
            val_probs = self.predict_proba(X_val)
            self.optimal_threshold = self._find_optimal_threshold(y_val.values, val_probs)
            logger.info(f"Optimal threshold: {self.optimal_threshold:.4f}")
        return self
    
    def _find_optimal_threshold(self, y_true, y_prob):
        from sklearn.metrics import precision_recall_curve
        precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
        optimal_idx = np.argmax(f1_scores)
        return thresholds[optimal_idx] if optimal_idx < len(thresholds) else 0.5
    
    def predict_proba(self, X):
        if not self.is_fitted:
            raise ValueError("Model not fitted")
        from torch.utils.data import DataLoader, TensorDataset
        X_arr = self.prepare_features(X)
        dataset = TensorDataset(torch.tensor(X_arr))
        loader = DataLoader(dataset, batch_size=4096, shuffle=False)
        self.model.eval()
        probs = []
        with torch.no_grad():
            for batch in loader:
                batch_x = batch[0].to(self.device)
                outputs = torch.sigmoid(self.model(batch_x))
                probs.extend(outputs.cpu().numpy().flatten())
        return np.array(probs)
    
    def save(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({'model_state': self.model.state_dict(), 'config': self.config,
                    'optimal_threshold': self.optimal_threshold}, path)
    
    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.optimal_threshold = checkpoint['optimal_threshold']
        self.is_fitted = True
        return self


# Network architectures
class DeepResidualNet(nn.Module):
    """Deep Residual Network"""
    def __init__(self, input_dim: int):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, 128)
        self.res_blocks = nn.ModuleList([
            ResidualBlock(128) for _ in range(6)
        ])
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1)
        )
    
    def forward(self, x):
        h = self.input_proj(x)
        for block in self.res_blocks:
            h = block(h)
        return self.classifier(h)


class OptimizedResidualNet(nn.Module):
    """Optimized Residual Network"""
    def __init__(self, input_dim: int):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, 128)
        self.res_blocks = nn.ModuleList([
            ResidualBlock(128) for _ in range(8)
        ])
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1)
        )
    
    def forward(self, x):
        h = self.input_proj(x)
        for block in self.res_blocks:
            h = block(h)
        return self.classifier(h)


class DGITSuperNet(nn.Module):
    """DGIT Super Network with improved stability"""
    def __init__(self, input_dim: int):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LayerNorm(256),
            nn.Dropout(0.1)
        )
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(256, 8, 1024, 0.3, batch_first=True, norm_first=True),
            num_layers=6
        )
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.5)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(self, x):
        x = torch.clamp(x, min=-100, max=100)
        x = torch.nan_to_num(x, nan=0.0, posinf=100.0, neginf=-100.0)
        
        x = x.unsqueeze(1)
        h = self.input_proj(x)
        h = self.transformer(h)
        h = h.squeeze(1)
        return self.classifier(h)


class ResidualBlock(nn.Module):
    """Residual Block"""
    def __init__(self, dim: int):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.bn1 = nn.BatchNorm1d(dim)
        self.fc2 = nn.Linear(dim, dim)
        self.bn2 = nn.BatchNorm1d(dim)
        self.dropout = nn.Dropout(0.3)
    
    def forward(self, x):
        residual = x
        h = self.fc1(x)
        h = self.bn1(h)
        h = torch.relu(h)
        h = self.dropout(h)
        h = self.fc2(h)
        h = self.bn2(h)
        h = h + residual
        return torch.relu(h)


# Model registry
NEURAL_MODELS = {
    'baseline_mlp': BaselineMLPModel,
    'transformer': TransformerEcDNAModel,
    'deep_residual': DeepResidualModel,
    'optimized_residual': OptimizedResidualModel,
    'dgit_super': DGITSuperModel,
}


def create_neural_model(model_name: str, config: Optional[Dict] = None, device: str = 'auto') -> BaseEcDNAModel:
    """Factory function to create neural network models"""
    if model_name not in NEURAL_MODELS:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(NEURAL_MODELS.keys())}")
    return NEURAL_MODELS[model_name](config, device=device)
