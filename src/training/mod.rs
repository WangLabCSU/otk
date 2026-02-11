pub mod trainer;
pub mod scheduler;
pub mod callbacks;

use serde::{Deserialize, Serialize};

/// Training configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingConfig {
    /// Number of training epochs
    pub epochs: usize,
    /// Batch size
    pub batch_size: usize,
    /// Learning rate
    pub learning_rate: f64,
    /// Learning rate scheduler type
    pub lr_scheduler: String,
    /// Weight decay (L2 regularization)
    pub weight_decay: f64,
    /// Dropout rate
    pub dropout: f64,
    /// Early stopping patience
    pub early_stopping_patience: usize,
    /// Gradient clipping threshold (0 = disabled)
    pub gradient_clip: f64,
    /// Number of workers for data loading
    pub num_workers: usize,
    /// Device to use (cpu, cuda, wgpu)
    pub device: String,
    /// Random seed
    pub seed: u64,
    /// Validation frequency (epochs)
    pub val_frequency: usize,
    /// Checkpoint frequency (epochs)
    pub checkpoint_frequency: usize,
    /// Whether to use mixed precision training
    pub mixed_precision: bool,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            epochs: 100,
            batch_size: 256,
            learning_rate: 0.001,
            lr_scheduler: "cosine".to_string(),
            weight_decay: 0.0001,
            dropout: 0.3,
            early_stopping_patience: 15,
            gradient_clip: 1.0,
            num_workers: 4,
            device: "cpu".to_string(),
            seed: 2026,
            val_frequency: 1,
            checkpoint_frequency: 5,
            mixed_precision: false,
        }
    }
}

impl TrainingConfig {
    /// Create configuration for quick testing
    pub fn quick_test() -> Self {
        Self {
            epochs: 5,
            batch_size: 64,
            learning_rate: 0.01,
            ..Default::default()
        }
    }
    
    /// Create configuration for production training
    pub fn production() -> Self {
        Self {
            epochs: 200,
            batch_size: 512,
            learning_rate: 0.001,
            lr_scheduler: "cosine_with_restarts".to_string(),
            weight_decay: 0.0001,
            dropout: 0.4,
            early_stopping_patience: 20,
            ..Default::default()
        }
    }
}

/// Training state
#[derive(Debug, Clone)]
pub struct TrainingState {
    /// Current epoch
    pub epoch: usize,
    /// Current global step
    pub global_step: usize,
    /// Best validation metric
    pub best_metric: f64,
    /// Number of epochs without improvement
    pub epochs_without_improvement: usize,
    /// Training loss history
    pub train_loss_history: Vec<f64>,
    /// Validation metric history
    pub val_metric_history: Vec<f64>,
    /// Learning rate history
    pub lr_history: Vec<f64>,
}

impl TrainingState {
    /// Create new training state
    pub fn new() -> Self {
        Self {
            epoch: 0,
            global_step: 0,
            best_metric: 0.0,
            epochs_without_improvement: 0,
            train_loss_history: Vec::new(),
            val_metric_history: Vec::new(),
            lr_history: Vec::new(),
        }
    }
    
    /// Update after epoch
    pub fn update_epoch(&mut self, train_loss: f64, val_metric: f64, lr: f64) {
        self.epoch += 1;
        self.train_loss_history.push(train_loss);
        self.val_metric_history.push(val_metric);
        self.lr_history.push(lr);
        
        if val_metric > self.best_metric {
            self.best_metric = val_metric;
            self.epochs_without_improvement = 0;
        } else {
            self.epochs_without_improvement += 1;
        }
    }
    
    /// Check if should stop early
    pub fn should_stop_early(&self, patience: usize) -> bool {
        patience > 0 && self.epochs_without_improvement >= patience
    }
}

impl Default for TrainingState {
    fn default() -> Self {
        Self::new()
    }
}

/// Training result
#[derive(Debug, Clone)]
pub struct TrainingResult {
    /// Final training state
    pub state: TrainingState,
    /// Path to best model checkpoint
    pub best_checkpoint: Option<std::path::PathBuf>,
    /// Final validation metrics
    pub final_metrics: crate::model::checkpoint::Metrics,
    /// Training duration in seconds
    pub duration_secs: f64,
}