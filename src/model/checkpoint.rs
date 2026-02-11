use burn::prelude::*;
use burn::record::{CompactRecorder, Recorder};
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::{Path, PathBuf};
use anyhow::{Context, Result};
use tracing::{debug, info, warn};

use crate::model::architecture::EcDnaModel;
use crate::model::ModelConfig;

/// Checkpoint metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckpointMetadata {
    /// Model version
    pub version: String,
    /// Training epoch
    pub epoch: usize,
    /// Training step
    pub step: usize,
    /// Validation metrics
    pub val_metrics: Metrics,
    /// Model configuration
    pub model_config: ModelConfig,
    /// Training configuration
    pub training_config: TrainingConfigMetadata,
    /// Timestamp
    pub timestamp: String,
}

/// Training configuration metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingConfigMetadata {
    /// Learning rate
    pub learning_rate: f64,
    /// Batch size
    pub batch_size: usize,
    /// Number of epochs
    pub epochs: usize,
    /// Optimizer type
    pub optimizer: String,
}

/// Training metrics
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct Metrics {
    /// Loss value
    pub loss: f64,
    /// Accuracy
    pub accuracy: f64,
    /// Precision
    pub precision: f64,
    /// Recall
    pub recall: f64,
    /// F1 score
    pub f1: f64,
    /// Area under precision-recall curve
    pub auprc: f64,
}

/// Checkpoint manager for saving and loading model checkpoints
pub struct CheckpointManager {
    /// Directory to save checkpoints
    checkpoint_dir: PathBuf,
    /// Maximum number of checkpoints to keep
    max_checkpoints: usize,
    /// Best model metric
    best_metric: Option<f64>,
}

impl CheckpointManager {
    /// Create new checkpoint manager
    pub fn new<P: AsRef<Path>>(checkpoint_dir: P) -> Result<Self> {
        let checkpoint_dir = checkpoint_dir.as_ref().to_path_buf();
        fs::create_dir_all(&checkpoint_dir)
            .context("Failed to create checkpoint directory")?;
        
        Ok(Self {
            checkpoint_dir,
            max_checkpoints: 5,
            best_metric: None,
        })
    }
    
    /// Set maximum number of checkpoints to keep
    pub fn with_max_checkpoints(mut self, max: usize) -> Self {
        self.max_checkpoints = max;
        self
    }
    
    /// Save model checkpoint
    pub fn save_checkpoint<B: Backend>(
        &mut self,
        model: &EcDnaModel<B>,
        metadata: &CheckpointMetadata,
    ) -> Result<PathBuf> {
        let checkpoint_name = format!("checkpoint_epoch_{}.mpk", metadata.epoch);
        let checkpoint_path = self.checkpoint_dir.join(&checkpoint_name);
        
        // Save model weights
        let record = model.clone().into_record();
        CompactRecorder::new()
            .record(record, checkpoint_path.clone())
            .context("Failed to save model checkpoint")?;
        
        // Save metadata
        let metadata_path = checkpoint_path.with_extension("json");
        let metadata_json = serde_json::to_string_pretty(metadata)
            .context("Failed to serialize metadata")?;
        fs::write(&metadata_path, metadata_json)
            .context("Failed to write metadata file")?;
        
        info!("Saved checkpoint: {:?}", checkpoint_path);
        
        // Clean up old checkpoints
        self.cleanup_old_checkpoints()?;
        
        Ok(checkpoint_path)
    }
    
    /// Save best model based on metric
    pub fn save_best<B: Backend>(
        &mut self,
        model: &EcDnaModel<B>,
        metadata: &CheckpointMetadata,
        metric: f64,
        metric_name: &str,
    ) -> Result<Option<PathBuf>> {
        let is_best = match self.best_metric {
            None => true,
            Some(best) => metric > best, // Higher is better (e.g., AUPRC)
        };
        
        if is_best {
            info!(
                "New best {}: {:.4} (previous: {:?})",
                metric_name, metric, self.best_metric
            );
            self.best_metric = Some(metric);
            
            let best_path = self.checkpoint_dir.join("best_model.mpk");
            let record = model.clone().into_record();
            CompactRecorder::new()
                .record(record, best_path.clone())
                .context("Failed to save best model")?;
            
            // Save metadata
            let metadata_path = best_path.with_extension("json");
            let metadata_json = serde_json::to_string_pretty(metadata)
                .context("Failed to serialize metadata")?;
            fs::write(&metadata_path, metadata_json)
                .context("Failed to write metadata file")?;
            
            info!("Saved best model: {:?}", best_path);
            Ok(Some(best_path))
        } else {
            debug!(
                "Model did not improve. Current {}: {:.4}, Best: {:?}",
                metric_name, metric, self.best_metric
            );
            Ok(None)
        }
    }
    
    /// Load model from checkpoint
    pub fn load_checkpoint<B: Backend>(
        &self,
        checkpoint_path: &Path,
        device: &B::Device,
    ) -> Result<(EcDnaModel<B>, CheckpointMetadata)> {
        info!("Loading checkpoint from {:?}", checkpoint_path);
        
        // Load model
        let record = CompactRecorder::new()
            .load(checkpoint_path.to_path_buf(), device)
            .context("Failed to load model checkpoint")?;
        
        let model_config = ModelConfig::ecdna_default();
        let model = crate::model::architecture::init_model::<B>(&model_config, device)
            .load_record(record);
        
        // Load metadata
        let metadata_path = checkpoint_path.with_extension("json");
        let metadata_json = fs::read_to_string(&metadata_path)
            .context("Failed to read metadata file")?;
        let metadata: CheckpointMetadata = serde_json::from_str(&metadata_json)
            .context("Failed to parse metadata")?;
        
        info!("Loaded checkpoint from epoch {}", metadata.epoch);
        Ok((model, metadata))
    }
    
    /// Load best model
    pub fn load_best<B: Backend>(
        &self,
        device: &B::Device,
    ) -> Result<(EcDnaModel<B>, CheckpointMetadata)> {
        let best_path = self.checkpoint_dir.join("best_model.mpk");
        if !best_path.exists() {
            anyhow::bail!("No best model found at {:?}", best_path);
        }
        self.load_checkpoint(&best_path, device)
    }
    
    /// Load latest checkpoint
    pub fn load_latest<B: Backend>(
        &self,
        device: &B::Device,
    ) -> Result<(EcDnaModel<B>, CheckpointMetadata)> {
        let checkpoints = self.list_checkpoints()?;
        if checkpoints.is_empty() {
            anyhow::bail!("No checkpoints found in {:?}", self.checkpoint_dir);
        }
        
        // Sort by epoch (descending) and take the latest
        let latest = checkpoints
            .into_iter()
            .max_by_key(|(epoch, _)| *epoch)
            .map(|(_, path)| path)
            .unwrap();
        
        self.load_checkpoint(&latest, device)
    }
    
    /// List all checkpoints
    pub fn list_checkpoints(&self) -> Result<Vec<(usize, PathBuf)>> {
        let mut checkpoints = Vec::new();
        
        for entry in fs::read_dir(&self.checkpoint_dir)
            .context("Failed to read checkpoint directory")? {
            let entry = entry.context("Failed to read directory entry")?;
            let path = entry.path();
            
            if path.extension().map(|e| e == "mpk").unwrap_or(false) {
                // Extract epoch from filename
                let filename = path.file_stem()
                    .and_then(|s| s.to_str())
                    .unwrap_or("");
                
                if let Some(epoch_str) = filename.strip_prefix("checkpoint_epoch_") {
                    if let Ok(epoch) = epoch_str.parse::<usize>() {
                        checkpoints.push((epoch, path));
                    }
                }
            }
        }
        
        Ok(checkpoints)
    }
    
    /// Clean up old checkpoints, keeping only the most recent ones
    fn cleanup_old_checkpoints(&self) -> Result<()> {
        let mut checkpoints = self.list_checkpoints()?;
        
        if checkpoints.len() > self.max_checkpoints {
            // Sort by epoch (ascending)
            checkpoints.sort_by_key(|(epoch, _)| *epoch);
            
            // Remove oldest checkpoints
            let to_remove = checkpoints.len() - self.max_checkpoints;
            for (_, path) in checkpoints.into_iter().take(to_remove) {
                debug!("Removing old checkpoint: {:?}", path);
                fs::remove_file(&path).ok();
                fs::remove_file(path.with_extension("json")).ok();
            }
        }
        
        Ok(())
    }
    
    /// Get checkpoint directory path
    pub fn checkpoint_dir(&self) -> &Path {
        &self.checkpoint_dir
    }
}

/// Simple model saver for inference
pub struct ModelSaver;

impl ModelSaver {
    /// Save model for inference
    pub fn save_for_inference<B: Backend, P: AsRef<Path>>(
        model: &EcDnaModel<B>,
        path: P,
        metadata: &CheckpointMetadata,
    ) -> Result<()> {
        let path = path.as_ref();
        
        // Create parent directory if needed
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        
        // Save model
        let record = model.clone().into_record();
        CompactRecorder::new()
            .record(record, path.to_path_buf())
            .context("Failed to save model")?;
        
        // Save metadata
        let metadata_path = path.with_extension("json");
        let metadata_json = serde_json::to_string_pretty(metadata)
            .context("Failed to serialize metadata")?;
        fs::write(&metadata_path, metadata_json)
            .context("Failed to write metadata file")?;
        
        info!("Saved model for inference: {:?}", path);
        Ok(())
    }
    
    /// Load model for inference
    pub fn load_for_inference<B: Backend, P: AsRef<Path>>(
        path: P,
        device: &B::Device,
    ) -> Result<(EcDnaModel<B>, CheckpointMetadata)> {
        let path = path.as_ref();
        info!("Loading model from {:?}", path);
        
        // Load model
        let record = CompactRecorder::new()
            .load(path.to_path_buf(), device)
            .context("Failed to load model")?;
        
        let model_config = ModelConfig::ecdna_default();
        let model = crate::model::architecture::init_model::<B>(&model_config, device)
            .load_record(record);
        
        // Load metadata
        let metadata_path = path.with_extension("json");
        let metadata_json = fs::read_to_string(&metadata_path)
            .context("Failed to read metadata file")?;
        let metadata: CheckpointMetadata = serde_json::from_str(&metadata_json)
            .context("Failed to parse metadata")?;
        
        info!("Loaded model from epoch {}", metadata.epoch);
        Ok((model, metadata))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;
    use tempfile::TempDir;
    
    type TestBackend = NdArray<f32>;
    
    fn create_test_metadata() -> CheckpointMetadata {
        CheckpointMetadata {
            version: "0.1.0".to_string(),
            epoch: 10,
            step: 1000,
            val_metrics: Metrics {
                loss: 0.1,
                accuracy: 0.95,
                precision: 0.9,
                recall: 0.85,
                f1: 0.875,
                auprc: 0.92,
            },
            model_config: ModelConfig::ecdna_default(),
            training_config: TrainingConfigMetadata {
                learning_rate: 0.001,
                batch_size: 64,
                epochs: 100,
                optimizer: "Adam".to_string(),
            },
            timestamp: "2024-01-01T00:00:00Z".to_string(),
        }
    }
    
    #[test]
    fn test_checkpoint_save_load() {
        let temp_dir = TempDir::new().unwrap();
        let mut manager = CheckpointManager::new(temp_dir.path()).unwrap();
        
        let device = <TestBackend as Backend>::Device::default();
        let model = crate::model::architecture::init_model::<TestBackend>(
            &ModelConfig::ecdna_default(),
            &device,
        );
        
        let metadata = create_test_metadata();
        let checkpoint_path = manager.save_checkpoint(&model, &metadata).unwrap();
        
        assert!(checkpoint_path.exists());
        
        let (loaded_model, loaded_metadata) = manager
            .load_checkpoint(&checkpoint_path, &device)
            .unwrap();
        
        assert_eq!(loaded_metadata.epoch, metadata.epoch);
    }
}
