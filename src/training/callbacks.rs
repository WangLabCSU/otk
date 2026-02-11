use crate::model::checkpoint::Metrics;
use crate::training::TrainingState;
use std::time::Duration;

/// Training callback trait
///
/// Callbacks allow custom actions to be performed at various points during training.
pub trait TrainingCallback: Send + Sync {
    /// Called at the start of training
    fn on_train_begin(&mut self) {}
    
    /// Called at the end of training
    fn on_train_end(&mut self, state: &TrainingState) {}
    
    /// Called at the start of each epoch
    fn on_epoch_begin(&mut self, epoch: usize) {}
    
    /// Called at the end of each epoch
    fn on_epoch_end(&mut self, epoch: usize, state: &TrainingState, metrics: &Metrics) {}
    
    /// Called at the start of each batch
    fn on_batch_begin(&mut self, batch: usize) {}
    
    /// Called at the end of each batch
    fn on_batch_end(&mut self, batch: usize, loss: f64) {}
    
    /// Called when validation starts
    fn on_validation_begin(&mut self) {}
    
    /// Called when validation ends
    fn on_validation_end(&mut self, metrics: &Metrics) {}
}

/// Callback manager that handles multiple callbacks
pub struct CallbackManager {
    callbacks: Vec<Box<dyn TrainingCallback>>,
}

impl CallbackManager {
    /// Create new callback manager
    pub fn new() -> Self {
        Self {
            callbacks: Vec::new(),
        }
    }
    
    /// Add a callback
    pub fn add_callback<C: TrainingCallback + 'static>(&mut self, callback: C) {
        self.callbacks.push(Box::new(callback));
    }
    
    /// Call on_train_begin for all callbacks
    pub fn on_train_begin(&mut self) {
        for callback in &mut self.callbacks {
            callback.on_train_begin();
        }
    }
    
    /// Call on_train_end for all callbacks
    pub fn on_train_end(&mut self, state: &TrainingState) {
        for callback in &mut self.callbacks {
            callback.on_train_end(state);
        }
    }
    
    /// Call on_epoch_begin for all callbacks
    pub fn on_epoch_begin(&mut self, epoch: usize) {
        for callback in &mut self.callbacks {
            callback.on_epoch_begin(epoch);
        }
    }
    
    /// Call on_epoch_end for all callbacks
    pub fn on_epoch_end(&mut self, epoch: usize, state: &TrainingState, metrics: &Metrics) {
        for callback in &mut self.callbacks {
            callback.on_epoch_end(epoch, state, metrics);
        }
    }
    
    /// Call on_batch_begin for all callbacks
    pub fn on_batch_begin(&mut self, batch: usize) {
        for callback in &mut self.callbacks {
            callback.on_batch_begin(batch);
        }
    }
    
    /// Call on_batch_end for all callbacks
    pub fn on_batch_end(&mut self, batch: usize, loss: f64) {
        for callback in &mut self.callbacks {
            callback.on_batch_end(batch, loss);
        }
    }
    
    /// Call on_validation_begin for all callbacks
    pub fn on_validation_begin(&mut self) {
        for callback in &mut self.callbacks {
            callback.on_validation_begin();
        }
    }
    
    /// Call on_validation_end for all callbacks
    pub fn on_validation_end(&mut self, metrics: &Metrics) {
        for callback in &mut self.callbacks {
            callback.on_validation_end(metrics);
        }
    }
}

impl Default for CallbackManager {
    fn default() -> Self {
        Self::new()
    }
}

/// Early stopping callback
pub struct EarlyStoppingCallback {
    patience: usize,
    min_delta: f64,
    best_metric: f64,
    counter: usize,
    should_stop: bool,
    monitor: String,
    mode: String,
}

impl EarlyStoppingCallback {
    /// Create new early stopping callback
    pub fn new(patience: usize, monitor: &str, mode: &str) -> Self {
        let best_metric = if mode == "max" {
            f64::NEG_INFINITY
        } else {
            f64::INFINITY
        };
        
        Self {
            patience,
            min_delta: 1e-4,
            best_metric,
            counter: 0,
            should_stop: false,
            monitor: monitor.to_string(),
            mode: mode.to_string(),
        }
    }
    
    /// Check if training should stop
    pub fn should_stop(&self) -> bool {
        self.should_stop
    }
}

impl TrainingCallback for EarlyStoppingCallback {
    fn on_epoch_end(&mut self, _epoch: usize, _state: &TrainingState, metrics: &Metrics) {
        let current_metric = match self.monitor.as_str() {
            "loss" => metrics.loss,
            "accuracy" => metrics.accuracy,
            "precision" => metrics.precision,
            "recall" => metrics.recall,
            "f1" => metrics.f1,
            "auprc" => metrics.auprc,
            _ => metrics.loss,
        };
        
        let improved = if self.mode == "max" {
            current_metric > self.best_metric + self.min_delta
        } else {
            current_metric < self.best_metric - self.min_delta
        };
        
        if improved {
            self.best_metric = current_metric;
            self.counter = 0;
        } else {
            self.counter += 1;
            if self.counter >= self.patience {
                self.should_stop = true;
            }
        }
    }
}

/// Model checkpoint callback
pub struct ModelCheckpointCallback {
    checkpoint_dir: std::path::PathBuf,
    save_best_only: bool,
    monitor: String,
    mode: String,
    best_metric: f64,
}

impl ModelCheckpointCallback {
    /// Create new model checkpoint callback
    pub fn new<P: AsRef<std::path::Path>>(checkpoint_dir: P, monitor: &str, mode: &str) -> Self {
        let best_metric = if mode == "max" {
            f64::NEG_INFINITY
        } else {
            f64::INFINITY
        };
        
        Self {
            checkpoint_dir: checkpoint_dir.as_ref().to_path_buf(),
            save_best_only: false,
            monitor: monitor.to_string(),
            mode: mode.to_string(),
            best_metric,
        }
    }
    
    /// Set save best only
    pub fn save_best_only(mut self, save_best_only: bool) -> Self {
        self.save_best_only = save_best_only;
        self
    }
}

impl TrainingCallback for ModelCheckpointCallback {
    fn on_epoch_end(&mut self, epoch: usize, _state: &TrainingState, metrics: &Metrics) {
        let current_metric = match self.monitor.as_str() {
            "loss" => metrics.loss,
            "accuracy" => metrics.accuracy,
            "precision" => metrics.precision,
            "recall" => metrics.recall,
            "f1" => metrics.f1,
            "auprc" => metrics.auprc,
            _ => metrics.loss,
        };
        
        let improved = if self.mode == "max" {
            current_metric > self.best_metric
        } else {
            current_metric < self.best_metric
        };
        
        if improved {
            self.best_metric = current_metric;
        }
        
        if !self.save_best_only || improved {
            // Save checkpoint
            // Note: Actual saving is handled by CheckpointManager
            tracing::debug!(
                "Checkpoint callback triggered for epoch {} with metric {:.4}",
                epoch, current_metric
            );
        }
    }
}

/// Learning rate scheduler callback
pub struct LearningRateSchedulerCallback {
    scheduler: crate::training::scheduler::LearningRateScheduler,
}

impl LearningRateSchedulerCallback {
    /// Create new LR scheduler callback
    pub fn new(scheduler: crate::training::scheduler::LearningRateScheduler) -> Self {
        Self { scheduler }
    }
}

impl TrainingCallback for LearningRateSchedulerCallback {
    fn on_epoch_end(&mut self, epoch: usize, _state: &TrainingState, _metrics: &Metrics) {
        let lr = self.scheduler.get_lr(epoch);
        tracing::debug!("Learning rate for epoch {}: {}", epoch + 1, lr);
    }
}

/// Progress logging callback
pub struct ProgressLoggerCallback {
    log_frequency: usize,
}

impl ProgressLoggerCallback {
    /// Create new progress logger
    pub fn new(log_frequency: usize) -> Self {
        Self { log_frequency }
    }
}

impl TrainingCallback for ProgressLoggerCallback {
    fn on_epoch_end(&mut self, epoch: usize, state: &TrainingState, metrics: &Metrics) {
        if (epoch + 1) % self.log_frequency == 0 {
            tracing::info!(
                "Epoch {} - Loss: {:.4}, AUPRC: {:.4}, F1: {:.4}, Acc: {:.4}",
                epoch + 1,
                metrics.loss,
                metrics.auprc,
                metrics.f1,
                metrics.accuracy
            );
        }
    }
}

/// Metrics logger callback that saves metrics to file
pub struct MetricsLoggerCallback {
    log_file: std::path::PathBuf,
    metrics_history: Vec<(usize, Metrics)>,
}

impl MetricsLoggerCallback {
    /// Create new metrics logger
    pub fn new<P: AsRef<std::path::Path>>(log_file: P) -> Self {
        Self {
            log_file: log_file.as_ref().to_path_buf(),
            metrics_history: Vec::new(),
        }
    }
    
    /// Save metrics to file
    fn save_metrics(&self) -> anyhow::Result<()> {
        let json = serde_json::to_string_pretty(&self.metrics_history)?;
        std::fs::write(&self.log_file, json)?;
        Ok(())
    }
}

impl TrainingCallback for MetricsLoggerCallback {
    fn on_epoch_end(&mut self, epoch: usize, _state: &TrainingState, metrics: &Metrics) {
        self.metrics_history.push((epoch, metrics.clone()));
        
        if let Err(e) = self.save_metrics() {
            tracing::warn!("Failed to save metrics: {}", e);
        }
    }
    
    fn on_train_end(&mut self, _state: &TrainingState) {
        if let Err(e) = self.save_metrics() {
            tracing::warn!("Failed to save final metrics: {}", e);
        }
    }
}

/// Timer callback that tracks training time
pub struct TimerCallback {
    start_time: Option<std::time::Instant>,
    epoch_times: Vec<Duration>,
}

impl TimerCallback {
    /// Create new timer callback
    pub fn new() -> Self {
        Self {
            start_time: None,
            epoch_times: Vec::new(),
        }
    }
    
    /// Get average epoch time
    pub fn average_epoch_time(&self) -> Option<Duration> {
        if self.epoch_times.is_empty() {
            None
        } else {
            let total: Duration = self.epoch_times.iter().sum();
            Some(total / self.epoch_times.len() as u32)
        }
    }
    
    /// Get total training time
    pub fn total_time(&self) -> Option<Duration> {
        self.start_time.map(|t| t.elapsed())
    }
}

impl Default for TimerCallback {
    fn default() -> Self {
        Self::new()
    }
}

impl TrainingCallback for TimerCallback {
    fn on_train_begin(&mut self) {
        self.start_time = Some(std::time::Instant::now());
        tracing::info!("Training started");
    }
    
    fn on_epoch_begin(&mut self, _epoch: usize) {
        // Start timing epoch
    }
    
    fn on_epoch_end(&mut self, epoch: usize, _state: &TrainingState, _metrics: &Metrics) {
        // In a real implementation, we'd track per-epoch timing
        tracing::debug!("Epoch {} completed", epoch + 1);
    }
    
    fn on_train_end(&mut self, state: &TrainingState) {
        if let Some(start) = self.start_time {
            let duration = start.elapsed();
            tracing::info!(
                "Training completed in {:.2?} ({} epochs)",
                duration,
                state.epoch
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_early_stopping() {
        let mut callback = EarlyStoppingCallback::new(2, "loss", "min");
        
        let mut metrics = Metrics::default();
        metrics.loss = 1.0;
        
        // First epoch - should not stop
        callback.on_epoch_end(0, &TrainingState::new(), &metrics);
        assert!(!callback.should_stop());
        
        // Second epoch - worse loss
        metrics.loss = 1.1;
        callback.on_epoch_end(1, &TrainingState::new(), &metrics);
        assert!(!callback.should_stop());
        
        // Third epoch - still worse
        metrics.loss = 1.2;
        callback.on_epoch_end(2, &TrainingState::new(), &metrics);
        assert!(callback.should_stop());
    }
    
    #[test]
    fn test_callback_manager() {
        let mut manager = CallbackManager::new();
        manager.add_callback(ProgressLoggerCallback::new(1));
        manager.add_callback(TimerCallback::new());
        
        manager.on_train_begin();
        manager.on_epoch_begin(0);
        
        let mut metrics = Metrics::default();
        metrics.loss = 0.5;
        manager.on_epoch_end(0, &TrainingState::new(), &metrics);
        
        manager.on_train_end(&TrainingState::new());
    }
}