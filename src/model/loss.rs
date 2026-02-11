use burn::tensor::activation::sigmoid;
use burn::nn::loss::*;
use burn::tensor::backend::Backend;

/// Focal loss for handling class imbalance
/// 
/// Focal loss down-weights easy examples and focuses on hard examples,
/// which is useful for highly imbalanced datasets like ecDNA prediction.
#[derive(Clone, Debug)]
pub struct FocalLoss<B: Backend> {
    /// Alpha weight for positive class
    alpha: f32,
    /// Focusing parameter (gamma)
    gamma: f32,
    /// BCE loss
    bce: BinaryCrossEntropyLoss<B>,
}

impl<B: Backend> FocalLoss<B> {
    /// Create new focal loss
    pub fn new(alpha: f32, gamma: f32, device: &B::Device) -> Self {
        Self {
            alpha,
            gamma,
            bce: BinaryCrossEntropyLossConfig::new()
                .init(device),
        }
    }
    
    /// Compute focal loss
    /// 
    /// # Arguments
    /// * `logits` - Model output logits [batch_size, num_classes]
    /// * `targets` - Target labels [batch_size]
    pub fn forward(
        &self,
        logits: Tensor<B, 2>,
        targets: Tensor<B, 1, Int>,
    ) -> Tensor<B, 1> {
        // Compute BCE loss
        let bce_loss = self.bce.forward(logits.clone(), targets.clone());
        
        // Compute probabilities
        let probs = sigmoid(logits);
        
        // Get target probabilities (p_t)
        let targets_float = targets.float();
        let p_t = probs.clone().mul(targets_float.clone())
            + probs.neg().add_scalar(1.0).mul(targets_float.neg().add_scalar(1.0));
        
        // Compute focal weight: (1 - p_t)^gamma
        let focal_weight = p_t.neg().add_scalar(1.0).powf_scalar(self.gamma);
        
        // Apply alpha weighting
        let alpha_t = targets_float.mul_scalar(self.alpha)
            + targets_float.neg().add_scalar(1.0).mul_scalar(1.0 - self.alpha);
        
        // Compute focal loss
        let loss = focal_weight.mul(alpha_t).mul(bce_loss);
        
        // Apply reduction
        loss.mean()
    }
}

/// Metrics for evaluation
pub mod metrics {
    use burn::tensor::backend::Backend;
    
    /// Compute accuracy
    pub fn accuracy<B: Backend>(
        predictions: Tensor<B, 1, Int>,
        targets: Tensor<B, 1, Int>,
    ) -> f32 {
        let correct = predictions.equal(targets).int().sum().into_scalar() as f32;
        let total = predictions.dims()[0] as f32;
        correct / total
    }
    
    /// Compute precision
    pub fn precision<B: Backend>(
        predictions: Tensor<B, 1, Int>,
        targets: Tensor<B, 1, Int>,
    ) -> f32 {
        let true_positives = predictions.clone().equal(targets.clone())
            .int()
            .mul(predictions.clone().equal_elem(1).int())
            .sum()
            .into_scalar() as f32;
        
        let predicted_positives = predictions.equal_elem(1).int().sum().into_scalar() as f32;
        
        if predicted_positives > 0.0 {
            true_positives / predicted_positives
        } else {
            0.0
        }
    }
    
    /// Compute recall
    pub fn recall<B: Backend>(
        predictions: Tensor<B, 1, Int>,
        targets: Tensor<B, 1, Int>,
    ) -> f32 {
        let true_positives = predictions.clone().equal(targets.clone())
            .int()
            .mul(targets.clone().equal_elem(1).int())
            .sum()
            .into_scalar() as f32;
        
        let actual_positives = targets.equal_elem(1).int().sum().into_scalar() as f32;
        
        if actual_positives > 0.0 {
            true_positives / actual_positives
        } else {
            0.0
        }
    }
    
    /// Compute F1 score
    pub fn f1_score<B: Backend>(
        predictions: Tensor<B, 1, Int>,
        targets: Tensor<B, 1, Int>,
    ) -> f32 {
        let p = precision(predictions.clone(), targets.clone());
        let r = recall(predictions, targets);
        
        if p + r > 0.0 {
            2.0 * p * r / (p + r)
        } else {
            0.0
        }
    }
    
    /// Compute area under precision-recall curve (approximation)
    pub fn auprc<B: Backend>(
        probabilities: Tensor<B, 2>,
        targets: Tensor<B, 1, Int>,
        num_thresholds: usize,
    ) -> f32 {
        let device = probabilities.device();
        let probs = probabilities.squeeze::<1>(1).to_device(&device);
        let targets_float = targets.float();
        
        let mut precisions = Vec::new();
        let mut recalls = Vec::new();
        
        for i in 0..=num_thresholds {
            let threshold = i as f32 / num_thresholds as f32;
            let predictions = probs.clone().greater_elem(threshold).int();
            
            let tp = predictions.clone().equal(targets.clone())
                .int()
                .mul(targets_float.clone().equal_elem(1.0).int())
                .sum()
                .into_scalar() as f32;
            
            let fp = predictions.clone().equal_elem(1).int()
                .mul(targets_float.clone().equal_elem(0.0).int())
                .sum()
                .into_scalar() as f32;
            
            let fn_ = predictions.equal_elem(0).int()
                .mul(targets_float.clone().equal_elem(1.0).int())
                .sum()
                .into_scalar() as f32;
            
            let precision = if tp + fp > 0.0 { tp / (tp + fp) } else { 0.0 };
            let recall = if tp + fn_ > 0.0 { tp / (tp + fn_) } else { 0.0 };
            
            precisions.push(precision);
            recalls.push(recall);
        }
        
        // Compute AUPRC using trapezoidal rule
        let mut auprc = 0.0;
        for i in 1..precisions.len() {
            let recall_diff = recalls[i] - recalls[i - 1];
            let avg_precision = (precisions[i] + precisions[i - 1]) / 2.0;
            auprc += recall_diff * avg_precision;
        }
        
        auprc.abs()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;
    
    type TestBackend = NdArray<f32>;
    
    #[test]
    fn test_focal_loss() {
        let device = <TestBackend as Backend>::Device::default();
        let focal_loss = FocalLoss::new(0.25, 2.0, &device);
        
        let logits = Tensor::<TestBackend, 2>::from_floats(&[0.5, -0.5, 1.0, -1.0], &device)
            .reshape([4, 1]);
        let targets = Tensor::<TestBackend, 1, Int>::from_data(&[1i64, 0, 1, 0], &device);
        
        let loss = focal_loss.forward(logits, targets);
        
        // Loss should be a scalar
        assert_eq!(loss.dims(), [1]);
        assert!(loss.into_scalar() > 0.0);
    }
    
    #[test]
    fn test_metrics() {
        let device = <TestBackend as Backend>::Device::default();
        
        let predictions = Tensor::<TestBackend, 1, Int>::from_data(&[1i64, 0, 1, 0, 1], &device);
        let targets = Tensor::<TestBackend, 1, Int>::from_data(&[1i64, 0, 0, 0, 1], &device);
        
        let acc = metrics::accuracy(predictions.clone(), targets.clone());
        assert_eq!(acc, 0.8); // 4 out of 5 correct
        
        let prec = metrics::precision(predictions.clone(), targets.clone());
        assert_eq!(prec, 2.0 / 3.0); // 2 true positives out of 3 predicted positives
        
        let rec = metrics::recall(predictions.clone(), targets.clone());
        assert_eq!(rec, 1.0); // 2 true positives out of 2 actual positives
        
        let f1 = metrics::f1_score(predictions, targets);
        assert!((f1 - 0.8).abs() < 0.01);
    }
}
