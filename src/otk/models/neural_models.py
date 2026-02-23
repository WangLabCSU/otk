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
        
        logger.info(f"Epoch {epoch+1}/{n_epochs} - Loss: {avg_loss:.4f}")
        
        if X_val is not None and y_val is not None and (epoch + 1) % log_interval == 0:
            model.eval()
            with torch.no_grad():
                val_probs = []
                for batch_x, _ in train_loader:
                    batch_x = batch_x.to(device)
                    outputs = torch.sigmoid(model(batch_x))
                    val_probs.extend(outputs.cpu().numpy().flatten())
            
            val_size = min(100000, len(y_val))
            val_idx = np.random.choice(len(y_val), val_size, replace=False)
            val_probs_subset = np.array(val_probs)[:val_size] if len(val_probs) >= val_size else np.array(val_probs)
            y_val_subset = y_val[val_idx] if hasattr(y_val, '__getitem__') else y_val[:val_size]
            
            if patience > 0:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break
    
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
    """DGIT Super Model V3 - Anti-Overfitting Edition
    
    Key improvements:
    1. ComboLoss: BCE + Focal + Dice for better generalization
    2. Mixup data augmentation
    3. Stochastic depth in residual blocks
    4. Label smoothing
    5. Strong regularization (dropout, weight decay)
    6. Early stopping with validation
    """
    
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
    
    def _listnet_loss(self, logits, targets, temperature=1.0):
        """ListNet Loss - directly optimizes ranking
        
        ListNet uses softmax to convert scores to probability distributions,
        then minimizes KL divergence between predicted and target distributions.
        This is a differentiable approximation to auPRC optimization.
        """
        logits_flat = logits.view(-1)
        targets_flat = targets.view(-1).float()
        
        # Only compute on samples with both positive and negative
        if targets_flat.sum() == 0 or targets_flat.sum() == len(targets_flat):
            return torch.tensor(0.0, device=logits.device)
        
        # Softmax over scores (temperature scaling)
        pred_probs = nn.functional.softmax(logits_flat / temperature, dim=0)
        target_probs = nn.functional.softmax(targets_flat / temperature, dim=0)
        
        # KL divergence
        loss = -torch.sum(target_probs * torch.log(pred_probs + 1e-10))
        
        return loss
    
    def fit(self, X_train, y_train, X_val=None, y_val=None, teacher_probs=None, train_df=None, **kwargs):
        """Fit model with Transformer + Multi-Task Learning
        
        Key improvements:
        1. Transformer Encoder - learns global feature interactions
        2. Multi-task heads - Gene-level + Sample-level prediction
        3. Focal Loss - handles class imbalance
        4. Early stopping with validation
        """
        from torch.utils.data import DataLoader, TensorDataset
        from sklearn.metrics import average_precision_score, roc_auc_score
        import time
        
        set_random_seed(RANDOM_SEED)
        
        X_train_arr = self.prepare_features(X_train) if isinstance(X_train, pd.DataFrame) else X_train
        y_train_arr = y_train.values.astype(np.float32) if isinstance(y_train, pd.Series) else y_train.astype(np.float32)
        
        # Prepare sample-level labels from original DataFrame
        if train_df is not None and 'sample' in train_df.columns:
            train_samples = train_df['sample'].values
            sample_to_label = train_df.groupby('sample')['y'].max().to_dict()
            sample_labels = np.array([sample_to_label.get(s, 0) for s in train_samples], dtype=np.float32)
        else:
            # Fallback: use gene labels as sample labels
            sample_labels = y_train_arr.copy()
        
        input_dim = X_train_arr.shape[1]
        
        arch_config = self.config.get('model', {}).get('architecture', {})
        hidden_dim = arch_config.get('hidden_dim', 192)
        num_heads = arch_config.get('num_heads', 8)
        num_layers = arch_config.get('num_layers', 3)
        dropout = arch_config.get('dropout', 0.2)
        drop_path_rate = arch_config.get('drop_path_rate', 0.1)
        
        loss_config = self.config.get('model', {}).get('loss_function', {})
        pos_weight = loss_config.get('pos_weight', 10.0)
        focal_gamma = loss_config.get('focal_gamma', 2.0)
        sample_weight = loss_config.get('sample_weight', 0.3)
        label_smoothing = loss_config.get('label_smoothing', 0.05)
        mixup_alpha = loss_config.get('mixup_alpha', 0.2)
        use_mixup = loss_config.get('use_mixup', True)
        
        self.model = DGITSuperNet(
            input_dim, 
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout,
            drop_path_rate=drop_path_rate
        ).to(self.device)
        
        train_dataset = TensorDataset(
            torch.tensor(X_train_arr),
            torch.tensor(y_train_arr).unsqueeze(1),
            torch.tensor(sample_labels).unsqueeze(1)
        )
        train_loader = DataLoader(train_dataset, batch_size=4096, shuffle=True)
        
        optimizer = torch.optim.AdamW(
            self.model.parameters(), 
            lr=0.001,
            weight_decay=0.01
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
        
        best_val_auprc = 0
        best_model_state = None
        best_epoch = 0
        patience_counter = 0
        max_patience = 20
        n_epochs = 150
        
        logger.info(f"Starting DGITSuper V14 (FT-Transformer + Multi-Task) training: {n_epochs} epochs")
        logger.info(f"Architecture: hidden_dim={hidden_dim}, num_heads={num_heads}, num_layers={num_layers}, dropout={dropout}, drop_path_rate={drop_path_rate}")
        logger.info(f"Multi-task: sample_weight={sample_weight}, label_smoothing={label_smoothing}, mixup={use_mixup}")
        start_time = time.time()
        
        for epoch in range(n_epochs):
            self.model.train()
            epoch_loss = 0
            n_batches = 0
            
            for batch_x, batch_y, batch_sample_y in train_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                batch_sample_y = batch_sample_y.to(self.device)
                
                # Mixup augmentation
                if use_mixup and np.random.random() < 0.5:
                    lam = np.random.beta(mixup_alpha, mixup_alpha)
                    indices = torch.randperm(batch_x.size(0), device=self.device)
                    batch_x = lam * batch_x + (1 - lam) * batch_x[indices]
                    batch_y = lam * batch_y + (1 - lam) * batch_y[indices]
                    batch_sample_y = lam * batch_sample_y + (1 - lam) * batch_sample_y[indices]
                
                # Label smoothing
                if label_smoothing > 0:
                    batch_y = batch_y * (1 - label_smoothing) + 0.5 * label_smoothing
                    batch_sample_y = batch_sample_y * (1 - label_smoothing) + 0.5 * label_smoothing
                
                optimizer.zero_grad()
                
                # Multi-task forward
                gene_out, sample_out = self.model(batch_x, return_sample_pred=True)
                
                # Gene-level Focal Loss
                logits_flat = gene_out.view(-1)
                targets_flat = batch_y.view(-1).float()
                probs = torch.sigmoid(logits_flat)
                pt = targets_flat * probs + (1.0 - targets_flat) * (1.0 - probs)
                focal_weight = (1.0 - pt) ** focal_gamma
                bce = nn.functional.binary_cross_entropy_with_logits(
                    logits_flat, targets_flat,
                    pos_weight=torch.tensor([pos_weight], device=self.device),
                    reduction='none'
                )
                gene_loss = (focal_weight * bce).mean()
                
                # Sample-level BCE Loss
                sample_loss = nn.functional.binary_cross_entropy_with_logits(
                    sample_out.view(-1), batch_sample_y.view(-1).float(),
                    pos_weight=torch.tensor([pos_weight], device=self.device)
                )
                
                # Combined loss
                loss = gene_loss + sample_weight * sample_loss
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                epoch_loss += loss.item()
                n_batches += 1
            
            train_loss = epoch_loss / n_batches
            scheduler.step()
            
            self.is_fitted = True
            train_probs = self.predict_proba(X_train)
            train_auprc = average_precision_score(y_train.values, train_probs)
            train_auc = roc_auc_score(y_train.values, train_probs)
            
            val_auprc = 0
            val_auc = 0
            if X_val is not None and y_val is not None:
                val_probs = self.predict_proba(X_val)
                val_auprc = average_precision_score(y_val.values, val_probs)
                val_auc = roc_auc_score(y_val.values, val_probs)
            
            if (epoch + 1) % 1 == 0:
                gap = train_auprc - val_auprc
                logger.info(f"Epoch {epoch+1}/{n_epochs}")
                logger.info(f"  Train - Loss: {train_loss:.4f}, auPRC: {train_auprc:.4f}, AUC: {train_auc:.4f}")
                if X_val is not None and y_val is not None:
                    logger.info(f"  Val   - auPRC: {val_auprc:.4f}, AUC: {val_auc:.4f}, Gap: {gap:.4f}")
            
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


class MultiHeadSelfAttention(nn.Module):
    """Multi-Head Self Attention for feature interactions"""
    def __init__(self, dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        B, D = x.shape
        x = x.unsqueeze(1)  # (B, 1, D)
        
        qkv = self.qkv(x).reshape(B, 1, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)
        
        out = (attn @ v).reshape(B, self.num_heads, self.head_dim)
        out = out.reshape(B, -1)
        return self.proj(out)


class DropPath(nn.Module):
    """Stochastic Depth - randomly drop entire residual branches"""
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob
        
    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.size(0),) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor


class FeatureTokenizer(nn.Module):
    """Feature Tokenizer for Tabular Data
    
    Converts continuous features to tokens for transformer processing.
    Each feature is projected to hidden_dim, creating a sequence of tokens.
    """
    def __init__(self, num_features: int, hidden_dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(num_features, hidden_dim) * 0.02)
        self.bias = nn.Parameter(torch.zeros(num_features, hidden_dim))
        
    def forward(self, x):
        # x: [batch, num_features]
        # Output: [batch, num_features, hidden_dim]
        return x.unsqueeze(-1) * self.weight.unsqueeze(0) + self.bias.unsqueeze(0)


class TransformerBlock(nn.Module):
    """Transformer Block with Pre-LN and Stochastic Depth"""
    def __init__(self, hidden_dim: int, num_heads: int, ffn_dim: int, 
                 dropout: float, drop_path_rate: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, hidden_dim),
            nn.Dropout(dropout)
        )
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0 else nn.Identity()
        
    def forward(self, x):
        # Pre-LN Attention
        residual = x
        x = self.norm1(x)
        x, _ = self.attn(x, x, x, need_weights=False)
        x = self.drop_path(x) + residual
        
        # Pre-LN FFN
        residual = x
        x = self.norm2(x)
        x = self.ffn(x)
        x = self.drop_path(x) + residual
        
        return x


class DGITSuperNet(nn.Module):
    """DGIT Super Network V14 - FT-Transformer + Multi-Task Learning
    
    Key innovations:
    1. Feature Tokenizer - converts each feature to a token
    2. CLS Token - aggregates global information for prediction
    3. Multi-Head Attention - learns feature interactions
    4. Stochastic Depth - prevents overfitting
    5. Multi-task heads - Gene-level + Sample-level prediction
    
    Reference: "Revisiting Deep Learning Models for Tabular Data" (2024)
    """
    def __init__(self, input_dim: int, hidden_dim: int = 192, num_heads: int = 8,
                 num_layers: int = 3, dropout: float = 0.2, drop_path_rate: float = 0.1):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Feature Tokenizer - converts each feature to a token
        self.tokenizer = FeatureTokenizer(input_dim, hidden_dim)
        
        # CLS Token - for global aggregation
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)
        
        # Positional embedding (learnable)
        self.pos_embedding = nn.Parameter(torch.randn(1, input_dim + 1, hidden_dim) * 0.02)
        
        # Transformer blocks with stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, num_layers)]
        self.blocks = nn.ModuleList([
            TransformerBlock(hidden_dim, num_heads, hidden_dim * 4, dropout, dpr[i])
            for i in range(num_layers)
        ])
        
        # Final LayerNorm
        self.norm = nn.LayerNorm(hidden_dim)
        
        # Gene-level prediction head
        self.gene_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # Sample-level prediction head
        self.sample_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, x, return_sample_pred=False):
        x = torch.clamp(x, min=-100, max=100)
        x = torch.nan_to_num(x, nan=0.0, posinf=100.0, neginf=-100.0)
        
        batch_size = x.size(0)
        
        # Feature Tokenization: [batch, num_features] -> [batch, num_features, hidden_dim]
        tokens = self.tokenizer(x)
        
        # Add CLS token: [batch, num_features+1, hidden_dim]
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        tokens = torch.cat([cls_tokens, tokens], dim=1)
        
        # Add positional embedding
        tokens = tokens + self.pos_embedding
        
        # Transformer blocks
        for block in self.blocks:
            tokens = block(tokens)
        
        # Final norm
        tokens = self.norm(tokens)
        
        # Extract CLS token representation for prediction
        cls_output = tokens[:, 0, :]  # [batch, hidden_dim]
        
        # Gene-level prediction
        gene_out = self.gene_head(cls_output)
        
        if return_sample_pred:
            sample_out = self.sample_head(cls_output)
            return gene_out, sample_out
        
        return gene_out


class ObliviousDecisionTreeLayer(nn.Module):
    """Oblivious Decision Tree Layer
    
    Implements a layer of oblivious decision trees where all nodes at the same depth
    use the same splitting feature. This mimics tree-based models' decision boundaries.
    """
    def __init__(self, input_dim: int, num_trees: int, tree_dim: int):
        super().__init__()
        
        self.num_trees = num_trees
        self.tree_dim = tree_dim
        
        # Feature selection for each tree (which features to split on)
        self.feature_weights = nn.Parameter(torch.randn(num_trees, input_dim) * 0.1)
        
        # Split thresholds for each tree
        self.split_thresholds = nn.Parameter(torch.zeros(num_trees, tree_dim))
        
        # Leaf responses
        self.leaf_responses = nn.Parameter(torch.randn(num_trees, tree_dim) * 0.1)
        
    def forward(self, x):
        # Compute feature responses
        # x: [batch, input_dim]
        # feature_weights: [num_trees, input_dim]
        feature_responses = torch.einsum('bi,ti->bt', x, self.feature_weights)
        
        # Apply sigmoid to get soft splits
        splits = torch.sigmoid(feature_responses.unsqueeze(-1) - self.split_thresholds.unsqueeze(0))
        # splits: [batch, num_trees, tree_dim]
        
        # Weighted leaf responses
        responses = splits * self.leaf_responses.unsqueeze(0)
        # responses: [batch, num_trees, tree_dim]
        
        # Flatten to [batch, num_trees * tree_dim]
        return responses.view(responses.size(0), -1)


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


class EnsembleSuperNet(nn.Module):
    """Ensemble Super Network - Combines multiple architectures for higher auPRC
    
    Architecture:
    - 3 diverse sub-networks with different inductive biases
    - Meta-learner to combine predictions
    - Stacking ensemble for optimal fusion
    """
    def __init__(self, input_dim: int, hidden_dim: int = 256, dropout: float = 0.2):
        super().__init__()
        
        # Sub-network 1: Deep Residual (high precision)
        self.resnet = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            *[ResidualBlock(hidden_dim) for _ in range(4)]
        )
        
        # Sub-network 2: Attention-based (balanced)
        self.attn_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        self.attention = MultiHeadSelfAttention(hidden_dim, num_heads=8, dropout=dropout)
        
        # Sub-network 3: Wide network (high recall)
        self.wide_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )
        
        # Individual classifiers
        self.cls1 = nn.Linear(hidden_dim, 1)
        self.cls2 = nn.Linear(hidden_dim, 1)
        self.cls3 = nn.Linear(hidden_dim, 1)
        
        # Meta-learner: combines predictions + features
        self.meta_learner = nn.Sequential(
            nn.Linear(hidden_dim * 3 + 3, 128),  # 3 features + 3 predictions
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
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
        
        # Sub-network 1: Residual
        h1 = self.resnet(x)
        out1 = self.cls1(h1)
        
        # Sub-network 2: Attention
        h2 = self.attn_net(x)
        h2 = h2 + self.attention(h2)
        out2 = self.cls2(h2)
        
        # Sub-network 3: Wide
        h3 = self.wide_net(x)
        out3 = self.cls3(h3)
        
        # Meta-learner
        meta_features = torch.cat([h1, h2, h3, out1, out2, out3], dim=-1)
        return self.meta_learner(meta_features)


class EnsembleSuperModel(BaseEcDNAModel):
    """Ensemble Super Model - Combines multiple architectures
    
    Target: auPRC >= 0.85
    """
    
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
        
        arch_config = self.config.get('model', {}).get('architecture', {})
        hidden_dim = arch_config.get('hidden_dim', 256)
        dropout = arch_config.get('dropout', 0.2)
        
        self.model = EnsembleSuperNet(input_dim, hidden_dim=hidden_dim, dropout=dropout).to(self.device)
        
        train_dataset = TensorDataset(
            torch.tensor(X_train_arr),
            torch.tensor(y_train_arr).unsqueeze(1)
        )
        train_loader = DataLoader(train_dataset, batch_size=4096, shuffle=True)
        
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=0.001, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=2)
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([10.0]).to(self.device))
        
        best_val_auprc = 0
        best_model_state = None
        best_epoch = 0
        patience_counter = 0
        max_patience = 25
        n_epochs = 200
        
        logger.info(f"Starting EnsembleSuper training: {n_epochs} epochs, {len(train_loader)} batches/epoch")
        logger.info(f"Architecture: hidden_dim={hidden_dim}, dropout={dropout}")
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
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                epoch_loss += loss.item()
                n_batches += 1
            
            train_loss = epoch_loss / n_batches
            scheduler.step()
            
            self.is_fitted = True
            train_probs = self.predict_proba(X_train)
            train_auprc = average_precision_score(y_train.values, train_probs)
            train_auc = roc_auc_score(y_train.values, train_probs)
            
            val_auprc = 0
            val_auc = 0
            if X_val is not None and y_val is not None:
                val_probs = self.predict_proba(X_val)
                val_auprc = average_precision_score(y_val.values, val_probs)
                val_auc = roc_auc_score(y_val.values, val_probs)
            
            if (epoch + 1) % 5 == 0:
                logger.info(f"Epoch {epoch+1}/{n_epochs}")
                logger.info(f"  Train - Loss: {train_loss:.4f}, auPRC: {train_auprc:.4f}, AUC: {train_auc:.4f}")
                if X_val is not None and y_val is not None:
                    logger.info(f"  Val   - auPRC: {val_auprc:.4f}, AUC: {val_auc:.4f}")
            
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
        self.model.eval()
        X_arr = self.prepare_features(X) if isinstance(X, pd.DataFrame) else X
        with torch.no_grad():
            x = torch.tensor(X_arr, dtype=torch.float32).to(self.device)
            logits = self.model(x)
            probs = torch.sigmoid(logits).cpu().numpy().flatten()
        return probs
    
    def predict(self, X, threshold=None):
        probs = self.predict_proba(X)
        threshold = threshold or self.optimal_threshold or 0.5
        return (probs >= threshold).astype(int)
    
    def save(self, path: str):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': self.config,
            'optimal_threshold': getattr(self, 'optimal_threshold', 0.5)
        }, path)
    
    def load(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimal_threshold = checkpoint.get('optimal_threshold', 0.5)
        self.is_fitted = True
        return self


class HybridEnsembleNet(nn.Module):
    """Hybrid Ensemble Network - Combines XGBoost predictions with Neural Network
    
    Architecture:
    1. XGBoost predictions as additional features
    2. Neural Network learns residuals and interactions
    3. Meta-learner combines all signals
    """
    def __init__(self, input_dim: int, hidden_dim: int = 256, dropout: float = 0.2):
        super().__init__()
        
        # Feature embedding
        self.feature_embed = nn.Sequential(
            nn.Linear(input_dim + 1, hidden_dim),  # +1 for XGBoost prediction
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # Residual learning branch
        self.residual_branch = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2)
        )
        
        # Interaction learning branch
        self.interaction_branch = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2)
        )
        
        # Final prediction head
        self.head = nn.Sequential(
            nn.Linear(hidden_dim // 2, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
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
    
    def forward(self, x, xgb_pred):
        x = torch.clamp(x, min=-100, max=100)
        x = torch.nan_to_num(x, nan=0.0, posinf=100.0, neginf=-100.0)
        
        # Concatenate features with XGBoost prediction
        combined = torch.cat([x, xgb_pred.unsqueeze(-1)], dim=-1)
        
        # Embed
        h = self.feature_embed(combined)
        
        # Two branches
        residual = self.residual_branch(h)
        interaction = self.interaction_branch(h)
        
        # Combine
        features = residual + interaction
        
        return self.head(features)


class HybridEnsembleModel(BaseEcDNAModel):
    """Hybrid Ensemble Model - XGBoost + Neural Network
    
    Combines the strengths of both approaches:
    - XGBoost: Handles irregular decision boundaries
    - Neural Network: Learns residuals and feature interactions
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None, device: str = 'auto'):
        super().__init__(config)
        self.config = config or {}
        self.model = None
        self.xgb_model = None
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
        import xgboost as xgb
        import time
        
        set_random_seed(RANDOM_SEED)
        
        X_train_arr = self.prepare_features(X_train)
        y_train_arr = y_train.values.astype(np.float32)
        
        input_dim = X_train_arr.shape[1]
        
        # Train XGBoost first
        logger.info("Training XGBoost model...")
        xgb_params = {
            'objective': 'binary:logistic',
            'eval_metric': 'aucpr',
            'max_depth': 8,
            'learning_rate': 0.1,
            'n_estimators': 200,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'scale_pos_weight': 10,
            'random_state': 2026,
            'n_jobs': -1
        }
        self.xgb_model = xgb.XGBClassifier(**xgb_params)
        self.xgb_model.fit(X_train_arr, y_train_arr)
        
        # Get XGBoost predictions
        xgb_train_pred = self.xgb_model.predict_proba(X_train_arr)[:, 1]
        
        arch_config = self.config.get('model', {}).get('architecture', {})
        hidden_dim = arch_config.get('hidden_dim', 256)
        dropout = arch_config.get('dropout', 0.2)
        
        self.model = HybridEnsembleNet(input_dim, hidden_dim=hidden_dim, dropout=dropout).to(self.device)
        
        train_dataset = TensorDataset(
            torch.tensor(X_train_arr),
            torch.tensor(xgb_train_pred),
            torch.tensor(y_train_arr).unsqueeze(1)
        )
        train_loader = DataLoader(train_dataset, batch_size=4096, shuffle=True)
        
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=0.001, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=2)
        
        best_val_auprc = 0
        best_model_state = None
        best_epoch = 0
        patience_counter = 0
        max_patience = 20
        n_epochs = 100
        
        logger.info(f"Starting Hybrid Ensemble (XGBoost + NN) training: {n_epochs} epochs")
        start_time = time.time()
        
        for epoch in range(n_epochs):
            self.model.train()
            epoch_loss = 0
            n_batches = 0
            
            for batch_x, batch_xgb, batch_y in train_loader:
                batch_x = batch_x.to(self.device)
                batch_xgb = batch_xgb.to(self.device)
                batch_y = batch_y.to(self.device)
                
                optimizer.zero_grad()
                
                outputs = self.model(batch_x, batch_xgb)
                
                loss = nn.functional.binary_cross_entropy_with_logits(
                    outputs.squeeze(-1), batch_y.squeeze(-1).float(),
                    pos_weight=torch.tensor([10.0], device=self.device)
                )
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                epoch_loss += loss.item()
                n_batches += 1
            
            train_loss = epoch_loss / n_batches
            scheduler.step()
            
            self.is_fitted = True
            train_probs = self.predict_proba(X_train)
            train_auprc = average_precision_score(y_train.values, train_probs)
            
            val_auprc = 0
            if X_val is not None and y_val is not None:
                val_probs = self.predict_proba(X_val)
                val_auprc = average_precision_score(y_val.values, val_probs)
            
            if (epoch + 1) % 1 == 0:
                logger.info(f"Epoch {epoch+1}/{n_epochs} - Loss: {train_loss:.4f}, Train auPRC: {train_auprc:.4f}, Val auPRC: {val_auprc:.4f}")
            
            if val_auprc > best_val_auprc:
                best_val_auprc = val_auprc
                best_model_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                best_epoch = epoch + 1
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= max_patience:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break
        
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
        self.model.eval()
        X_arr = self.prepare_features(X) if isinstance(X, pd.DataFrame) else X
        
        # Get XGBoost predictions
        xgb_pred = self.xgb_model.predict_proba(X_arr)[:, 1]
        
        with torch.no_grad():
            x = torch.tensor(X_arr, dtype=torch.float32).to(self.device)
            xgb = torch.tensor(xgb_pred, dtype=torch.float32).to(self.device)
            logits = self.model(x, xgb)
            probs = torch.sigmoid(logits).cpu().numpy().flatten()
        return probs
    
    def predict(self, X, threshold=None):
        probs = self.predict_proba(X)
        threshold = threshold or self.optimal_threshold or 0.5
        return (probs >= threshold).astype(int)
    
    def save(self, path: str):
        import joblib
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': self.config,
            'optimal_threshold': getattr(self, 'optimal_threshold', 0.5)
        }, path)
        joblib.dump(self.xgb_model, path.replace('.pkl', '_xgb.pkl'))
    
    def load(self, path: str):
        import joblib
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimal_threshold = checkpoint.get('optimal_threshold', 0.5)
        self.xgb_model = joblib.load(path.replace('.pkl', '_xgb.pkl'))
        self.is_fitted = True
        return self


# Model registry
NEURAL_MODELS = {
    'baseline_mlp': BaselineMLPModel,
    'transformer': TransformerEcDNAModel,
    'deep_residual': DeepResidualModel,
    'optimized_residual': OptimizedResidualModel,
    'dgit_super': DGITSuperModel,
    'ensemble_super': EnsembleSuperModel,
    'hybrid_ensemble': HybridEnsembleModel,
}


def create_neural_model(model_name: str, config: Optional[Dict] = None, device: str = 'auto') -> BaseEcDNAModel:
    """Factory function to create neural network models
    
    Automatically loads config from otk_api/models/{model_name}/config.yml if not provided
    """
    if model_name not in NEURAL_MODELS:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(NEURAL_MODELS.keys())}")
    
    if config is None:
        import yaml
        from pathlib import Path
        config_path = Path(__file__).parent.parent.parent.parent / 'otk_api' / 'models' / model_name / 'config.yml'
        if config_path.exists():
            with open(config_path) as f:
                config = yaml.safe_load(f)
            logger.info(f"Loaded config from {config_path}")
    
    return NEURAL_MODELS[model_name](config, device=device)
