use crate::data::{FeatureVector, GeneRecord, Dataset, SplitConfig, CANCER_TYPES, NUM_BASE_FEATURES, NUM_FREQ_FEATURES, NUM_CANCER_TYPES, TOTAL_FEATURES};
use anyhow::{Context, Result};
use rand::seq::SliceRandom;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use std::collections::{HashMap, HashSet};
use tracing::{debug, info, warn};

/// Missing value handling strategy
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum MissingValueStrategy {
    /// Use mean imputation
    Mean,
    /// Use median imputation
    Median,
    /// Use zero imputation
    Zero,
    /// Use indicator variable for missing values
    Indicator,
    /// Learned imputation (model-based)
    Learned,
}

impl Default for MissingValueStrategy {
    fn default() -> Self {
        MissingValueStrategy::Mean
    }
}

/// Feature statistics for imputation
#[derive(Debug, Clone)]
pub struct FeatureStats {
    /// Mean value for each feature
    pub means: Vec<f32>,
    /// Standard deviation for each feature
    pub stds: Vec<f32>,
    /// Median value for each feature
    pub medians: Vec<f32>,
    /// Minimum value for each feature
    pub mins: Vec<f32>,
    /// Maximum value for each feature
    pub maxs: Vec<f32>,
}

impl FeatureStats {
    /// Compute statistics from feature vectors
    pub fn compute(vectors: &[FeatureVector]) -> Self {
        let n_features = TOTAL_FEATURES;
        let mut means = vec![0.0f32; n_features];
        let mut stds = vec![0.0f32; n_features];
        let mut medians = vec![0.0f32; n_features];
        let mut mins = vec![f32::INFINITY; n_features];
        let mut maxs = vec![f32::NEG_INFINITY; n_features];
        
        // Compute means and min/max
        let mut counts = vec![0usize; n_features];
        for vector in vectors {
            for (i, (&value, &mask)) in vector.features.iter().zip(&vector.mask).enumerate() {
                if mask && !value.is_nan() {
                    means[i] += value;
                    counts[i] += 1;
                    mins[i] = mins[i].min(value);
                    maxs[i] = maxs[i].max(value);
                }
            }
        }
        
        for i in 0..n_features {
            if counts[i] > 0 {
                means[i] /= counts[i] as f32;
            } else {
                means[i] = 0.0;
                mins[i] = 0.0;
                maxs[i] = 0.0;
            }
        }
        
        // Compute standard deviations
        let mut variances = vec![0.0f32; n_features];
        for vector in vectors {
            for (i, (&value, &mask)) in vector.features.iter().zip(&vector.mask).enumerate() {
                if mask && !value.is_nan() {
                    let diff = value - means[i];
                    variances[i] += diff * diff;
                }
            }
        }
        
        for i in 0..n_features {
            if counts[i] > 1 {
                stds[i] = (variances[i] / (counts[i] - 1) as f32).sqrt();
            } else {
                stds[i] = 1.0;
            }
            if stds[i] < 1e-8 {
                stds[i] = 1.0;
            }
        }
        
        // Compute medians
        for i in 0..n_features {
            let mut values: Vec<f32> = vectors
                .iter()
                .filter_map(|v| {
                    if v.mask[i] && !v.features[i].is_nan() {
                        Some(v.features[i])
                    } else {
                        None
                    }
                })
                .collect();
            
            if !values.is_empty() {
                values.sort_by(|a, b| a.partial_cmp(b).unwrap());
                let mid = values.len() / 2;
                medians[i] = if values.len() % 2 == 0 {
                    (values[mid - 1] + values[mid]) / 2.0
                } else {
                    values[mid]
                };
            } else {
                medians[i] = 0.0;
            }
        }
        
        Self {
            means,
            stds,
            medians,
            mins,
            maxs,
        }
    }
    
    /// Standardize features using z-score normalization
    pub fn standardize(&self, features: &mut [f32]) {
        for (i, value) in features.iter_mut().enumerate() {
            if i < self.means.len() && !value.is_nan() {
                *value = (*value - self.means[i]) / self.stds[i];
            }
        }
    }
    
    /// Impute missing values
    pub fn impute(&self, features: &mut [f32], mask: &[bool], strategy: MissingValueStrategy) {
        for (i, (value, &is_present)) in features.iter_mut().zip(mask).enumerate() {
            if !is_present || value.is_nan() {
                *value = match strategy {
                    MissingValueStrategy::Mean => self.means[i],
                    MissingValueStrategy::Median => self.medians[i],
                    MissingValueStrategy::Zero => 0.0,
                    MissingValueStrategy::Indicator => -1.0,
                    MissingValueStrategy::Learned => self.means[i], // Fallback to mean
                };
            }
        }
    }
}

/// Data preprocessor
pub struct Preprocessor {
    /// Missing value handling strategy
    pub missing_strategy: MissingValueStrategy,
    /// Whether to standardize features
    pub standardize: bool,
    /// Feature statistics (computed during fit)
    pub stats: Option<FeatureStats>,
}

impl Preprocessor {
    /// Create new preprocessor
    pub fn new() -> Self {
        Self {
            missing_strategy: MissingValueStrategy::default(),
            standardize: true,
            stats: None,
        }
    }
    
    /// Set missing value strategy
    pub fn with_missing_strategy(mut self, strategy: MissingValueStrategy) -> Self {
        self.missing_strategy = strategy;
        self
    }
    
    /// Set standardization flag
    pub fn with_standardization(mut self, standardize: bool) -> Self {
        self.standardize = standardize;
        self
    }
    
    /// Fit preprocessor on training data
    pub fn fit(&mut self, vectors: &[FeatureVector]) -> Result<()> {
        info!("Fitting preprocessor on {} samples", vectors.len());
        self.stats = Some(FeatureStats::compute(vectors));
        debug!("Feature statistics computed");
        Ok(())
    }
    
    /// Transform feature vectors
    pub fn transform(&self, vectors: &mut [FeatureVector]) -> Result<()> {
        let stats = self.stats.as_ref()
            .context("Preprocessor must be fitted before transform")?;
        
        for vector in vectors.iter_mut() {
            // Impute missing values
            stats.impute(&mut vector.features, &vector.mask, self.missing_strategy);
            
            // Standardize if enabled
            if self.standardize {
                stats.standardize(&mut vector.features);
            }
        }
        
        Ok(())
    }
    
    /// Fit and transform in one step
    pub fn fit_transform(&mut self, vectors: &mut [FeatureVector]) -> Result<()> {
        self.fit(vectors)?;
        self.transform(vectors)
    }
}

impl Default for Preprocessor {
    fn default() -> Self {
        Self::new()
    }
}

/// Convert GeneRecord to FeatureVector
pub fn convert_to_features(records: &[GeneRecord]) -> Vec<FeatureVector> {
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

/// Split dataset by samples to avoid data leakage
pub fn split_by_sample(
    vectors: Vec<FeatureVector>,
    config: &SplitConfig,
) -> Dataset {
    info!("Splitting dataset with seed {}", config.seed);
    
    // Group vectors by sample
    let mut sample_groups: HashMap<String, Vec<FeatureVector>> = HashMap::new();
    for vector in vectors {
        sample_groups
            .entry(vector.sample.clone())
            .or_default()
            .push(vector);
    }
    
    let mut samples: Vec<String> = sample_groups.keys().cloned().collect();
    
    // Shuffle samples
    let mut rng = ChaCha8Rng::seed_from_u64(config.seed);
    samples.shuffle(&mut rng);
    
    // Calculate split indices
    let n_samples = samples.len();
    let n_train = (n_samples as f32 * config.train_ratio) as usize;
    let n_val = (n_samples as f32 * config.val_ratio) as usize;
    
    let train_samples: HashSet<String> = samples[..n_train].iter().cloned().collect();
    let val_samples: HashSet<String> = samples[n_train..n_train + n_val].iter().cloned().collect();
    let test_samples: HashSet<String> = samples[n_train + n_val..].iter().cloned().collect();
    
    // Split vectors
    let mut dataset = Dataset::new();
    
    for (sample, vectors) in sample_groups {
        if train_samples.contains(&sample) {
            dataset.train.extend(vectors);
        } else if val_samples.contains(&sample) {
            dataset.val.extend(vectors);
        } else {
            dataset.test.extend(vectors);
        }
    }
    
    info!(
        "Dataset split: train={}, val={}, test={}",
        dataset.train.len(),
        dataset.val.len(),
        dataset.test.len()
    );
    
    // Log class distribution
    let log_distribution = |name: &str, data: &[FeatureVector]| {
        let pos = data.iter().filter(|v| v.target == Some(1)).count();
        let neg = data.iter().filter(|v| v.target == Some(0)).count();
        let unk = data.len() - pos - neg;
        info!(
            "{} distribution: positive={}, negative={}, unknown={}",
            name, pos, neg, unk
        );
    };
    
    log_distribution("Train", &dataset.train);
    log_distribution("Val", &dataset.val);
    log_distribution("Test", &dataset.test);
    
    dataset
}

/// Balance dataset using undersampling or oversampling
pub fn balance_dataset(vectors: &mut Vec<FeatureVector>, seed: u64) {
    let pos_count = vectors.iter().filter(|v| v.target == Some(1)).count();
    let neg_count = vectors.iter().filter(|v| v.target == Some(0)).count();
    
    info!("Balancing dataset: positive={}, negative={}", pos_count, neg_count);
    
    if pos_count == 0 || neg_count == 0 {
        warn!("Cannot balance dataset: one class is empty");
        return;
    }
    
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    
    // Separate positive and negative samples
    let mut positives: Vec<_> = vectors.iter().filter(|v| v.target == Some(1)).cloned().collect();
    let mut negatives: Vec<_> = vectors.iter().filter(|v| v.target == Some(0)).cloned().collect();
    
    // Balance by undersampling majority class
    let target_count = pos_count.max(neg_count);
    
    if pos_count < target_count {
        // Oversample positive class
        while positives.len() < target_count {
            positives.extend(positives.clone());
        }
        positives.truncate(target_count);
    }
    
    if neg_count < target_count {
        // Oversample negative class
        while negatives.len() < target_count {
            negatives.extend(negatives.clone());
        }
        negatives.truncate(target_count);
    }
    
    // Shuffle and combine
    positives.shuffle(&mut rng);
    negatives.shuffle(&mut rng);
    
    vectors.clear();
    vectors.extend(positives);
    vectors.extend(negatives);
    
    info!("Balanced dataset size: {}", vectors.len());
}

#[cfg(test)]
mod tests {
    use super::*;
    
    fn create_test_vectors() -> Vec<FeatureVector> {
        vec![
            FeatureVector {
                sample: "S1".to_string(),
                gene_id: "G1".to_string(),
                features: vec![1.0, 2.0, f32::NAN],
                target: Some(1),
                mask: vec![true, true, false],
            },
            FeatureVector {
                sample: "S1".to_string(),
                gene_id: "G2".to_string(),
                features: vec![3.0, 4.0, 5.0],
                target: Some(0),
                mask: vec![true, true, true],
            },
        ]
    }
    
    #[test]
    fn test_feature_stats_compute() {
        let vectors = create_test_vectors();
        let stats = FeatureStats::compute(&vectors);
        
        assert_eq!(stats.means[0], 2.0); // (1.0 + 3.0) / 2
        assert_eq!(stats.means[1], 3.0); // (2.0 + 4.0) / 2
    }
    
    #[test]
    fn test_preprocessor_fit_transform() {
        let mut vectors = create_test_vectors();
        let mut preprocessor = Preprocessor::new();
        
        preprocessor.fit_transform(&mut vectors).unwrap();
        
        // After transformation, NaN should be imputed
        assert!(!vectors[0].features[2].is_nan());
    }
}