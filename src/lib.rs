//! # OTK: ecDNA Prediction Tool
//!
//! OTK is a deep learning-based tool for predicting extrachromosomal DNA (ecDNA)
//! cargo genes from genomic data.
//!
//! ## Features
//!
//! - Gene-level ecDNA cargo prediction
//! - Sample-level focal amplification classification
//! - Missing value handling with learned imputation
//! - Support for various input formats (CSV, TSV, gzipped)
//!
//! ## Example Usage
//!
//! ```rust,no_run
//! use otk::data::{DataLoader, convert_to_features, Preprocessor, SplitConfig};
//! use otk::model::{ModelConfig, architecture::init_model};
//! use otk::training::{TrainingConfig, trainer::Trainer};
//!
//! // Load data
//! let loader = DataLoader::new();
//! let records = loader.load("data.csv").unwrap();
//!
//! // Preprocess
//! let mut vectors = convert_to_features(&records);
//! let mut preprocessor = Preprocessor::new();
//! preprocessor.fit_transform(&mut vectors).unwrap();
//!
//! // Split dataset
//! let dataset = otk::data::preprocessing::split_by_sample(
//!     vectors,
//!     &SplitConfig::default()
//! );
//!
//! // Train model
//! let model_config = ModelConfig::ecdna_default();
//! let training_config = TrainingConfig::default();
//! let device = burn::backend::ndarray::NdArrayDevice::default();
//! let mut trainer = Trainer::new(training_config, model_config, device);
//! let result = trainer.train(&dataset).unwrap();
//! ```

pub mod cli;
pub mod data;
pub mod model;
pub mod predict;
pub mod training;
pub mod utils;

use burn_ndarray::NdArray;

/// Default backend type
pub type DefaultBackend = NdArray<f32>;

/// Re-export commonly used types
pub use data::loader::DataLoader;
pub use data::{FeatureVector, GeneRecord, Dataset};
pub use model::{ModelConfig, architecture::EcDnaModel};
pub use predict::{GenePrediction, BatchPredictionResult};
pub use training::{TrainingConfig, TrainingResult};

/// Library version
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Library name
pub const NAME: &str = env!("CARGO_PKG_NAME");

/// Get library information
pub fn info() -> String {
    format!(
        "{} v{} - ecDNA prediction tool using deep learning",
        NAME, VERSION
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_version() {
        assert!(!VERSION.is_empty());
    }
    
    #[test]
    fn test_info() {
        let info_str = info();
        assert!(info_str.contains("otk"));
        assert!(info_str.contains(VERSION));
    }
}
