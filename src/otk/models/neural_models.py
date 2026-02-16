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
    
    def fit(self, X_train, y_train, X_val=None, y_val=None, teacher_probs=None, **kwargs):
        """Fit model with strong regularization
        
        Key improvements:
        1. L1 regularization (feature sparsity)
        2. Feature dropout
        3. ListNet loss for ranking
        4. Early stopping with validation
        """
        from torch.utils.data import DataLoader, TensorDataset
        from sklearn.metrics import average_precision_score, roc_auc_score
        import time
        
        set_random_seed(RANDOM_SEED)
        
        X_train_arr = self.prepare_features(X_train)
        y_train_arr = y_train.values.astype(np.float32)
        
        input_dim = X_train_arr.shape[1]
        
        arch_config = self.config.get('model', {}).get('architecture', {})
        hidden_dim = arch_config.get('hidden_dim', 128)
        dropout = arch_config.get('dropout', 0.3)
        
        loss_config = self.config.get('model', {}).get('loss_function', {})
        pos_weight = loss_config.get('pos_weight', 10.0)
        l1_lambda = loss_config.get('l1_lambda', 0.001)
        feature_dropout = loss_config.get('feature_dropout', 0.1)
        
        self.model = DGITSuperNet(input_dim, hidden_dim=hidden_dim, dropout=dropout).to(self.device)
        
        train_dataset = TensorDataset(
            torch.tensor(X_train_arr),
            torch.tensor(y_train_arr).unsqueeze(1)
        )
        train_loader = DataLoader(train_dataset, batch_size=2048, shuffle=True)
        
        optimizer = torch.optim.AdamW(
            self.model.parameters(), 
            lr=0.001,
            weight_decay=0.1  # Strong L2 regularization
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
        
        best_val_auprc = 0
        best_model_state = None
        best_epoch = 0
        patience_counter = 0
        max_patience = 15
        n_epochs = 100
        
        logger.info(f"Starting DGITSuper V6 (Simplified & Regularized) training: {n_epochs} epochs")
        logger.info(f"Architecture: hidden_dim={hidden_dim}, dropout={dropout}")
        logger.info(f"Regularization: L1={l1_lambda}, L2=0.1, feature_dropout={feature_dropout}")
        start_time = time.time()
        
        for epoch in range(n_epochs):
            self.model.train()
            epoch_loss = 0
            n_batches = 0
            
            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                
                # Feature dropout (randomly zero out features)
                if feature_dropout > 0:
                    mask = torch.rand(batch_x.shape[1], device=self.device) > feature_dropout
                    batch_x = batch_x * mask.float()
                
                optimizer.zero_grad()
                
                outputs = self.model(batch_x)
                
                # Combined loss: BCE + ListNet
                bce_loss = nn.functional.binary_cross_entropy_with_logits(
                    outputs.squeeze(-1), batch_y.squeeze(-1),
                    pos_weight=torch.tensor([pos_weight], device=self.device)
                )
                
                listnet_loss = self._listnet_loss(outputs, batch_y)
                
                loss = 0.7 * bce_loss + 0.3 * listnet_loss
                
                # L1 regularization
                l1_reg = torch.tensor(0.0, device=self.device)
                for param in self.model.parameters():
                    l1_reg += torch.abs(param).sum()
                loss += l1_lambda * l1_reg
                
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


class DGITSuperNet(nn.Module):
    """DGIT Super Network V6 - Simplified & Regularized
    
    Key design principles:
    1. Simple architecture - avoid overfitting
    2. Feature binning - mimic tree models
    3. Strong regularization - L1+L2+Feature dropout
    4. Direct ranking optimization
    """
    def __init__(self, input_dim: int, hidden_dim: int = 128, num_bins: int = 16, dropout: float = 0.3):
        super().__init__()
        
        self.num_bins = num_bins
        
        # Feature embedding with binning (mimics tree splits)
        self.feature_bins = nn.ModuleList([
            nn.Sequential(
                nn.Linear(1, 8),
                nn.ReLU(),
                nn.Linear(8, 1)
            ) for _ in range(min(input_dim, 20))  # Only for top 20 features
        ])
        
        # Main network - simple and effective
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # L1 regularization will be applied in training
        self._init_weights()
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x):
        x = torch.clamp(x, min=-100, max=100)
        x = torch.nan_to_num(x, nan=0.0, posinf=100.0, neginf=-100.0)
        
        return self.net(x)


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


# Model registry
NEURAL_MODELS = {
    'baseline_mlp': BaselineMLPModel,
    'transformer': TransformerEcDNAModel,
    'deep_residual': DeepResidualModel,
    'optimized_residual': OptimizedResidualModel,
    'dgit_super': DGITSuperModel,
    'ensemble_super': EnsembleSuperModel,
}


def create_neural_model(model_name: str, config: Optional[Dict] = None, device: str = 'auto') -> BaseEcDNAModel:
    """Factory function to create neural network models"""
    if model_name not in NEURAL_MODELS:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(NEURAL_MODELS.keys())}")
    return NEURAL_MODELS[model_name](config, device=device)
