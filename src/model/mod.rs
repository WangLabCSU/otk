pub mod architecture;
pub mod checkpoint;
pub mod loss;
pub mod missing_value_layer;

use burn::prelude::*;

/// Model configuration
#[derive(Config, Debug)]
pub struct ModelConfig {
    /// Number of input features
    #[config(default = "85")]
    pub input_size: usize,
    
    /// Number of hidden units in first layer
    #[config(default = "256")]
    pub hidden_size_1: usize,
    
    /// Number of hidden units in second layer
    #[config(default = "128")]
    pub hidden_size_2: usize,
    
    /// Number of hidden units in third layer
    #[config(default = "64")]
    pub hidden_size_3: usize,
    
    /// Dropout rate
    #[config(default = "0.3")]
    pub dropout: f64,
    
    /// Use batch normalization
    #[config(default = "true")]
    pub use_batch_norm: bool,
    
    /// Use missing value handling layer
    #[config(default = "true")]
    pub use_missing_value_layer: bool,
    
    /// Missing value layer hidden size
    #[config(default = "32")]
    pub missing_value_hidden_size: usize,
}

impl ModelConfig {
    /// Create model configuration for ecDNA prediction
    pub fn ecdna_default() -> Self {
        Self::new()
            .with_input_size(85)
            .with_hidden_size_1(256)
            .with_hidden_size_2(128)
            .with_hidden_size_3(64)
            .with_dropout(0.3)
            .with_use_batch_norm(true)
            .with_use_missing_value_layer(true)
            .with_missing_value_hidden_size(32)
    }
    
    /// Create a smaller model for faster training
    pub fn small() -> Self {
        Self::new()
            .with_input_size(85)
            .with_hidden_size_1(128)
            .with_hidden_size_2(64)
            .with_hidden_size_3(32)
            .with_dropout(0.2)
            .with_use_batch_norm(true)
            .with_use_missing_value_layer(true)
            .with_missing_value_hidden_size(16)
    }
    
    /// Create a larger model for better performance
    pub fn large() -> Self {
        Self::new()
            .with_input_size(85)
            .with_hidden_size_1(512)
            .with_hidden_size_2(256)
            .with_hidden_size_3(128)
            .with_dropout(0.4)
            .with_use_batch_norm(true)
            .with_use_missing_value_layer(true)
            .with_missing_value_hidden_size(64)
    }
}