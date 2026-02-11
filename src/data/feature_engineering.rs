//! Feature engineering utilities for ecDNA prediction

use crate::data::{GeneRecord, FeatureVector, CANCER_TYPES, NUM_BASE_FEATURES, NUM_FREQ_FEATURES, NUM_CANCER_TYPES, TOTAL_FEATURES};

/// Feature engineering configuration
#[derive(Debug, Clone)]
pub struct FeatureEngineeringConfig {
    /// Whether to use log transformation for copy number features
    pub log_transform_cn: bool,
    /// Whether to normalize features
    pub normalize: bool,
}

impl Default for FeatureEngineeringConfig {
    fn default() -> Self {
        Self {
            log_transform_cn: false,
            normalize: true,
        }
    }
}

/// Feature engineer for creating additional features
pub struct FeatureEngineer {
    config: FeatureEngineeringConfig,
}

impl FeatureEngineer {
    /// Create new feature engineer
    pub fn new() -> Self {
        Self {
            config: FeatureEngineeringConfig::default(),
        }
    }
    
    /// Create with custom config
    pub fn with_config(config: FeatureEngineeringConfig) -> Self {
        Self { config }
    }
    
    /// Apply feature engineering to records
    pub fn transform(&self, records: &[GeneRecord]) -> Vec<FeatureVector> {
        records.iter().map(|record| {
            let mut vector = FeatureVector::new(record.sample.clone(), record.gene_id.clone());
            vector.target = record.y;
            
            // Base features (indices 0-24)
            let base_features = [
                (0, record.seg_val),
                (1, record.minor_cn),
                (2, record.intersect_ratio),
                (3, record.purity),
                (4, record.ploidy),
                (5, record.a_score),
                (6, record.p_loh),
                (7, record.cna_burden),
            ];
            
            for (idx, value) in base_features {
                if let Some(v) = value {
                    let v = if self.config.log_transform_cn && idx < 2 && v > 0.0 {
                        v.ln()
                    } else {
                        v
                    };
                    vector.set_feature(idx, v);
                }
            }
            
            // CN signatures (indices 8-26)
            for (i, &sig) in record.cn_signatures.iter().enumerate() {
                if let Some(v) = sig {
                    vector.set_feature(8 + i, v);
                }
            }
            
            // Clinical features (indices 27-28)
            if let Some(v) = record.age {
                vector.set_feature(27, v);
            }
            if let Some(v) = record.gender {
                vector.set_feature(28, v);
            }
            
            // Cancer type one-hot encoding (indices 29-52)
            if let Some(idx) = record.cancer_type_index() {
                vector.set_feature(29 + idx, 1.0);
            }
            
            // Prior frequencies (indices 53-56)
            let freq_features = [
                (53, record.freq_linear),
                (54, record.freq_bfb),
                (55, record.freq_circular),
                (56, record.freq_hr),
            ];
            
            for (idx, value) in freq_features {
                if let Some(v) = value {
                    vector.set_feature(idx, v);
                }
            }
            
            vector
        }).collect()
    }
}

impl Default for FeatureEngineer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_feature_engineer() {
        let engineer = FeatureEngineer::new();
        
        let record = GeneRecord::new("S1".to_string(), "G1".to_string());
        let vectors = engineer.transform(&[record]);
        
        assert_eq!(vectors.len(), 1);
        assert_eq!(vectors[0].features.len(), TOTAL_FEATURES);
    }
}