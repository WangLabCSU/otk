pub mod predictor;

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Prediction result for a single gene
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenePrediction {
    /// Sample ID
    pub sample: String,
    /// Gene ID
    pub gene_id: String,
    /// Probability of being ecDNA cargo gene
    pub probability: f64,
    /// Binary prediction (0 or 1)
    pub prediction: u8,
    /// Confidence score (distance from decision boundary)
    pub confidence: f64,
}

impl GenePrediction {
    /// Create new gene prediction
    pub fn new(sample: String, gene_id: String, probability: f64) -> Self {
        let prediction = if probability >= 0.5 { 1 } else { 0 };
        let confidence = (probability - 0.5).abs() * 2.0; // Scale to [0, 1]
        
        Self {
            sample,
            gene_id,
            probability,
            prediction,
            confidence,
        }
    }
    
    /// Check if prediction is positive
    pub fn is_positive(&self) -> bool {
        self.prediction == 1
    }
    
    /// Get prediction as string
    pub fn prediction_label(&self) -> &'static str {
        if self.is_positive() {
            "ecDNA_cargo"
        } else {
            "non_ecDNA"
        }
    }
}

/// Sample-level prediction result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SamplePrediction {
    /// Sample ID
    pub sample: String,
    /// Number of genes predicted as ecDNA cargo
    pub num_ecdna_genes: usize,
    /// Total number of genes
    pub total_genes: usize,
    /// Proportion of ecDNA cargo genes
    pub ecdna_proportion: f64,
    /// Focal amplification type prediction
    pub focal_type: FocalAmplificationType,
    /// Confidence score for sample-level prediction
    pub confidence: f64,
}

/// Focal amplification type
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum FocalAmplificationType {
    /// No focal amplification
    NoFocal,
    /// Non-circular focal amplification
    NonCircular,
    /// Circular focal amplification (ecDNA)
    Circular,
}

impl FocalAmplificationType {
    /// Get type as string
    pub fn as_str(&self) -> &'static str {
        match self {
            FocalAmplificationType::NoFocal => "nofocal",
            FocalAmplificationType::NonCircular => "noncircular",
            FocalAmplificationType::Circular => "circular",
        }
    }
    
    /// Parse from string
    pub fn from_str(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "nofocal" | "no_focal" | "none" => Some(FocalAmplificationType::NoFocal),
            "noncircular" | "non_circular" | "linear" | "bfb" | "hr" => Some(FocalAmplificationType::NonCircular),
            "circular" | "ecdna" => Some(FocalAmplificationType::Circular),
            _ => None,
        }
    }
}

impl std::fmt::Display for FocalAmplificationType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

/// Batch prediction results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchPredictionResult {
    /// Gene-level predictions
    pub gene_predictions: Vec<GenePrediction>,
    /// Sample-level predictions
    pub sample_predictions: Vec<SamplePrediction>,
    /// Summary statistics
    pub summary: PredictionSummary,
}

impl BatchPredictionResult {
    /// Create new batch prediction result
    pub fn new(gene_predictions: Vec<GenePrediction>) -> Self {
        let sample_predictions = Self::aggregate_by_sample(&gene_predictions);
        let summary = PredictionSummary::from_predictions(&gene_predictions, &sample_predictions);
        
        Self {
            gene_predictions,
            sample_predictions,
            summary,
        }
    }
    
    /// Aggregate gene predictions by sample
    fn aggregate_by_sample(gene_predictions: &[GenePrediction]) -> Vec<SamplePrediction> {
        let mut sample_map: HashMap<String, Vec<&GenePrediction>> = HashMap::new();
        
        for pred in gene_predictions {
            sample_map.entry(pred.sample.clone())
                .or_default()
                .push(pred);
        }
        
        sample_map.into_iter()
            .map(|(sample, preds)| {
                let num_ecdna = preds.iter().filter(|p| p.is_positive()).count();
                let total = preds.len();
                let proportion = if total > 0 {
                    num_ecdna as f64 / total as f64
                } else {
                    0.0
                };
                
                // Determine focal type based on proportion and count
                let focal_type = if num_ecdna == 0 {
                    FocalAmplificationType::NoFocal
                } else if proportion < 0.1 {
                    FocalAmplificationType::NonCircular
                } else {
                    FocalAmplificationType::Circular
                };
                
                // Calculate confidence based on prediction consistency
                let confidence = if total > 0 {
                    let avg_conf = preds.iter().map(|p| p.confidence).sum::<f64>() / total as f64;
                    avg_conf
                } else {
                    0.0
                };
                
                SamplePrediction {
                    sample,
                    num_ecdna_genes: num_ecdna,
                    total_genes: total,
                    ecdna_proportion: proportion,
                    focal_type,
                    confidence,
                }
            })
            .collect()
    }
    
    /// Get predictions for a specific sample
    pub fn get_sample_predictions(&self, sample: &str) -> Vec<&GenePrediction> {
        self.gene_predictions
            .iter()
            .filter(|p| p.sample == sample)
            .collect()
    }
    
    /// Get positive predictions only
    pub fn get_positive_predictions(&self) -> Vec<&GenePrediction> {
        self.gene_predictions
            .iter()
            .filter(|p| p.is_positive())
            .collect()
    }
    
    /// Export to CSV format
    pub fn to_csv(&self) -> String {
        let mut csv = String::from("sample,gene_id,probability,prediction,confidence\n");
        
        for pred in &self.gene_predictions {
            csv.push_str(&format!(
                "{},{},{:.6},{},{}\n",
                pred.sample,
                pred.gene_id,
                pred.probability,
                pred.prediction,
                pred.confidence
            ));
        }
        
        csv
    }
    
    /// Export sample predictions to CSV
    pub fn sample_predictions_to_csv(&self) -> String {
        let mut csv = String::from("sample,num_ecdna_genes,total_genes,ecdna_proportion,focal_type,confidence\n");
        
        for pred in &self.sample_predictions {
            csv.push_str(&format!(
                "{},{},{},{:.6},{},{}\n",
                pred.sample,
                pred.num_ecdna_genes,
                pred.total_genes,
                pred.ecdna_proportion,
                pred.focal_type.as_str(),
                pred.confidence
            ));
        }
        
        csv
    }
}

/// Prediction summary statistics
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct PredictionSummary {
    /// Total number of predictions
    pub total_predictions: usize,
    /// Number of positive predictions
    pub positive_predictions: usize,
    /// Number of negative predictions
    pub negative_predictions: usize,
    /// Proportion of positive predictions
    pub positive_rate: f64,
    /// Average probability
    pub avg_probability: f64,
    /// Average confidence
    pub avg_confidence: f64,
    /// Number of samples
    pub num_samples: usize,
    /// Number of samples with ecDNA
    pub samples_with_ecdna: usize,
}

impl PredictionSummary {
    /// Create summary from predictions
    pub fn from_predictions(
        gene_predictions: &[GenePrediction],
        sample_predictions: &[SamplePrediction],
    ) -> Self {
        let total = gene_predictions.len();
        let positive = gene_predictions.iter().filter(|p| p.is_positive()).count();
        let negative = total - positive;
        
        let avg_prob = if total > 0 {
            gene_predictions.iter().map(|p| p.probability).sum::<f64>() / total as f64
        } else {
            0.0
        };
        
        let avg_conf = if total > 0 {
            gene_predictions.iter().map(|p| p.confidence).sum::<f64>() / total as f64
        } else {
            0.0
        };
        
        let samples_with_ecdna = sample_predictions
            .iter()
            .filter(|s| s.num_ecdna_genes > 0)
            .count();
        
        Self {
            total_predictions: total,
            positive_predictions: positive,
            negative_predictions: negative,
            positive_rate: if total > 0 { positive as f64 / total as f64 } else { 0.0 },
            avg_probability: avg_prob,
            avg_confidence: avg_conf,
            num_samples: sample_predictions.len(),
            samples_with_ecdna,
        }
    }
    
    /// Print summary to stdout
    pub fn print(&self) {
        println!("\n=== Prediction Summary ===");
        println!("Total predictions: {}", self.total_predictions);
        println!("Positive predictions: {} ({:.2}%)", 
            self.positive_predictions, 
            self.positive_rate * 100.0
        );
        println!("Negative predictions: {}", self.negative_predictions);
        println!("Average probability: {:.4}", self.avg_probability);
        println!("Average confidence: {:.4}", self.avg_confidence);
        println!("Number of samples: {}", self.num_samples);
        println!("Samples with ecDNA: {}", self.samples_with_ecdna);
        println!("==========================\n");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_gene_prediction() {
        let pred = GenePrediction::new("S1".to_string(), "G1".to_string(), 0.8);
        
        assert_eq!(pred.sample, "S1");
        assert_eq!(pred.gene_id, "G1");
        assert_eq!(pred.probability, 0.8);
        assert_eq!(pred.prediction, 1);
        assert!(pred.is_positive());
        assert_eq!(pred.prediction_label(), "ecDNA_cargo");
    }
    
    #[test]
    fn test_focal_amplification_type() {
        assert_eq!(FocalAmplificationType::NoFocal.as_str(), "nofocal");
        assert_eq!(FocalAmplificationType::NonCircular.as_str(), "noncircular");
        assert_eq!(FocalAmplificationType::Circular.as_str(), "circular");
        
        assert_eq!(FocalAmplificationType::from_str("nofocal"), Some(FocalAmplificationType::NoFocal));
        assert_eq!(FocalAmplificationType::from_str("circular"), Some(FocalAmplificationType::Circular));
        assert_eq!(FocalAmplificationType::from_str("unknown"), None);
    }
    
    #[test]
    fn test_batch_prediction_result() {
        let predictions = vec![
            GenePrediction::new("S1".to_string(), "G1".to_string(), 0.8),
            GenePrediction::new("S1".to_string(), "G2".to_string(), 0.3),
            GenePrediction::new("S2".to_string(), "G1".to_string(), 0.9),
        ];
        
        let result = BatchPredictionResult::new(predictions);
        
        assert_eq!(result.gene_predictions.len(), 3);
        assert_eq!(result.sample_predictions.len(), 2);
        assert_eq!(result.summary.total_predictions, 3);
        assert_eq!(result.summary.positive_predictions, 2);
        
        let s1_preds = result.get_sample_predictions("S1");
        assert_eq!(s1_preds.len(), 2);
        
        let positive_preds = result.get_positive_predictions();
        assert_eq!(positive_preds.len(), 2);
    }
    
    #[test]
    fn test_csv_export() {
        let predictions = vec![
            GenePrediction::new("S1".to_string(), "G1".to_string(), 0.8),
        ];
        
        let result = BatchPredictionResult::new(predictions);
        let csv = result.to_csv();
        
        assert!(csv.contains("sample,gene_id,probability,prediction,confidence"));
        assert!(csv.contains("S1,G1"));
    }
}