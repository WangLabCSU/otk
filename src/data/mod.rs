pub mod loader;
pub mod preprocessing;
pub mod feature_engineering;

use serde::{Deserialize, Serialize};

/// Number of cancer type features (one-hot encoded)
pub const NUM_CANCER_TYPES: usize = 24;

/// Number of prior frequency features
pub const NUM_FREQ_FEATURES: usize = 4;

/// Number of copy number signature features (CN1-CN19)
pub const NUM_CN_SIGNATURES: usize = 19;

/// Number of base features (excluding cancer types and frequencies)
pub const NUM_BASE_FEATURES: usize = 25;

/// Total number of input features
pub const TOTAL_FEATURES: usize = NUM_BASE_FEATURES + NUM_FREQ_FEATURES + NUM_CANCER_TYPES;

/// Cancer type names for one-hot encoding
pub const CANCER_TYPES: &[&str] = &[
    "BLCA", "BRCA", "CESC", "COAD", "DLBC",
    "ESCA", "GBM", "HNSC", "KICH", "KIRC",
    "KIRP", "LGG", "LIHC", "LUAD", "LUSC",
    "OV", "PRAD", "READ", "SARC", "SKCM",
    "STAD", "THCA", "UCEC", "UVM",
];

/// Gene-level ecDNA prediction record
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeneRecord {
    /// Sample ID
    pub sample: String,
    /// Gene ID
    pub gene_id: String,
    
    // Base features
    /// Total copy number
    pub seg_val: Option<f32>,
    /// Minor copy number
    pub minor_cn: Option<f32>,
    /// Intersection ratio
    pub intersect_ratio: Option<f32>,
    /// Tumor purity
    pub purity: Option<f32>,
    /// Tumor ploidy
    pub ploidy: Option<f32>,
    /// Aneuploidy score
    pub a_score: Option<f32>,
    /// Proportion of LOH
    pub p_loh: Option<f32>,
    /// CNA burden
    pub cna_burden: Option<f32>,
    
    // Copy number signatures (CN1-CN19)
    pub cn_signatures: [Option<f32>; NUM_CN_SIGNATURES],
    
    // Clinical features
    /// Patient age
    pub age: Option<f32>,
    /// Patient gender (0: female, 1: male)
    pub gender: Option<f32>,
    
    // Cancer type (will be converted to one-hot)
    pub cancer_type: Option<String>,
    
    // Prior frequencies
    /// Linear amplification frequency
    pub freq_linear: Option<f32>,
    /// BFB frequency
    pub freq_bfb: Option<f32>,
    /// Circular amplification frequency
    pub freq_circular: Option<f32>,
    /// HR frequency
    pub freq_hr: Option<f32>,
    
    // Target variable
    /// Whether gene is ecDNA cargo (1) or not (0)
    pub y: Option<u8>,
}

impl GeneRecord {
    /// Create a new empty gene record
    pub fn new(sample: String, gene_id: String) -> Self {
        Self {
            sample,
            gene_id,
            seg_val: None,
            minor_cn: None,
            intersect_ratio: None,
            purity: None,
            ploidy: None,
            a_score: None,
            p_loh: None,
            cna_burden: None,
            cn_signatures: [None; NUM_CN_SIGNATURES],
            age: None,
            gender: None,
            cancer_type: None,
            freq_linear: None,
            freq_bfb: None,
            freq_circular: None,
            freq_hr: None,
            y: None,
        }
    }
    
    /// Check if record has valid target
    pub fn has_target(&self) -> bool {
        self.y.is_some()
    }
    
    /// Get cancer type index for one-hot encoding
    pub fn cancer_type_index(&self) -> Option<usize> {
        self.cancer_type.as_ref().and_then(|ct| {
            CANCER_TYPES.iter().position(|&t| t == ct.as_str())
        })
    }
}

/// Processed feature vector for model input
#[derive(Debug, Clone)]
pub struct FeatureVector {
    /// Sample ID
    pub sample: String,
    /// Gene ID
    pub gene_id: String,
    /// Feature values (85 dimensions)
    pub features: Vec<f32>,
    /// Target label (if available)
    pub target: Option<u8>,
    /// Mask indicating which features are missing
    pub mask: Vec<bool>,
}

impl FeatureVector {
    /// Create new feature vector with NaN values
    pub fn new(sample: String, gene_id: String) -> Self {
        Self {
            sample,
            gene_id,
            features: vec![f32::NAN; TOTAL_FEATURES],
            target: None,
            mask: vec![false; TOTAL_FEATURES],
        }
    }
    
    /// Set feature value at index
    pub fn set_feature(&mut self, index: usize, value: f32) {
        if index < TOTAL_FEATURES {
            self.features[index] = value;
            self.mask[index] = true;
        }
    }
    
    /// Get feature value at index
    pub fn get_feature(&self, index: usize) -> Option<f32> {
        if index < TOTAL_FEATURES && self.mask[index] {
            Some(self.features[index])
        } else {
            None
        }
    }
    
    /// Check if feature is present
    pub fn has_feature(&self, index: usize) -> bool {
        index < TOTAL_FEATURES && self.mask[index]
    }
}

/// Dataset split configuration
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct SplitConfig {
    /// Training set ratio
    pub train_ratio: f32,
    /// Validation set ratio
    pub val_ratio: f32,
    /// Test set ratio
    pub test_ratio: f32,
    /// Random seed
    pub seed: u64,
}

impl Default for SplitConfig {
    fn default() -> Self {
        Self {
            train_ratio: 0.7,
            val_ratio: 0.1,
            test_ratio: 0.2,
            seed: 2026,
        }
    }
}

/// Dataset container
#[derive(Debug, Clone)]
pub struct Dataset {
    /// Training samples
    pub train: Vec<FeatureVector>,
    /// Validation samples
    pub val: Vec<FeatureVector>,
    /// Test samples
    pub test: Vec<FeatureVector>,
}

impl Dataset {
    /// Create empty dataset
    pub fn new() -> Self {
        Self {
            train: Vec::new(),
            val: Vec::new(),
            test: Vec::new(),
        }
    }
    
    /// Get total number of samples
    pub fn total_samples(&self) -> usize {
        self.train.len() + self.val.len() + self.test.len()
    }
    
    /// Get positive samples count
    pub fn positive_count(&self) -> usize {
        let count = |data: &[FeatureVector]| {
            data.iter().filter(|v| v.target == Some(1)).count()
        };
        count(&self.train) + count(&self.val) + count(&self.test)
    }
}