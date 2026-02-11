use crate::model::ModelConfig;
use burn::tensor::activation::{relu, sigmoid};
use burn::nn::*;
use burn::nn::loss::BinaryCrossEntropyLossConfig;
use burn::tensor::backend::Backend;
use burn::Tensor;
use burn::Int;

/// ecDNA prediction model
#[derive(Debug)]
pub struct EcDnaModel<B: Backend> {
    /// First fully connected layer
    fc1: Linear<B>,
    /// Second fully connected layer
    fc2: Linear<B>,
    /// Third fully connected layer
    fc3: Linear<B>,
    /// Output layer
    output: Linear<B>,
    /// Dropout layer
    dropout: Dropout,
}

/// ecDNA model output
#[derive(Debug, Clone)]
pub struct EcDnaOutput<B: Backend> {
    /// Predicted probabilities
    pub probabilities: Tensor<B, 2>,
    /// Binary predictions (0 or 1)
    pub predictions: Tensor<B, 1, Int>,
    /// Loss (if target provided)
    pub loss: Option<Tensor<B, 1>>,
}

impl<B: Backend> EcDnaModel<B> {
    /// Forward pass for inference
    pub fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
        // First block
        let x = self.fc1.forward(input);
        let x = relu(x);
        let x = self.dropout.forward(x);
        
        // Second block
        let x = self.fc2.forward(x);
        let x = relu(x);
        let x = self.dropout.forward(x);
        
        // Third block
        let x = self.fc3.forward(x);
        let x = relu(x);
        let x = self.dropout.forward(x);
        
        // Output layer (logits)
        self.output.forward(x)
    }
    
    /// Forward pass for training
    pub fn forward_training(
        &self,
        input: Tensor<B, 2>,
        targets: Tensor<B, 1, Int>,
    ) -> TrainingOutput<B> {
        let logits = self.forward(input);
        
        let loss = BinaryCrossEntropyLossConfig::new()
            .with_logits(true)
            .init(&logits.device())
            .forward(logits.clone(), targets.clone());
        
        TrainingOutput { loss, logits, targets }
    }
    
    /// Predict probabilities and binary labels
    pub fn predict(&self, input: Tensor<B, 2>) -> EcDnaOutput<B> {
        let logits = self.forward(input);
        let probabilities = sigmoid(logits.clone());
        
        // Convert to binary predictions (threshold at 0.5)
        let predictions = probabilities.clone().greater_elem(0.5).int();
        
        EcDnaOutput {
            probabilities,
            predictions: predictions.squeeze(1),
            loss: None,
        }
    }
    
    /// Predict with targets for evaluation
    pub fn predict_with_targets(
        &self,
        input: Tensor<B, 2>,
        targets: Tensor<B, 1, Int>,
    ) -> EcDnaOutput<B> {
        let logits = self.forward(input);
        let probabilities = sigmoid(logits.clone());
        
        let predictions = probabilities.clone().greater_elem(0.5).int();
        
        let loss = BinaryCrossEntropyLossConfig::new()
            .with_logits(true)
            .init(&logits.device())
            .forward(logits, targets.clone());
        
        EcDnaOutput {
            probabilities,
            predictions: predictions.squeeze(1),
            loss: Some(loss),
        }
    }
}

/// Training output
#[derive(Debug)]
pub struct TrainingOutput<B: Backend> {
    pub loss: Tensor<B, 1>,
    pub logits: Tensor<B, 2>,
    pub targets: Tensor<B, 1, Int>,
}

/// Initialize ecDNA model from configuration
pub fn init_model<B: Backend>(
    config: &ModelConfig,
    device: &B::Device,
) -> EcDnaModel<B> {
    // Initialize fully connected layers
    let fc1 = LinearConfig::new(config.input_size, config.hidden_size_1)
        .with_bias(true)
        .init(device);
    
    let fc2 = LinearConfig::new(config.hidden_size_1, config.hidden_size_2)
        .with_bias(true)
        .init(device);
    
    let fc3 = LinearConfig::new(config.hidden_size_2, config.hidden_size_3)
        .with_bias(true)
        .init(device);
    
    let output = LinearConfig::new(config.hidden_size_3, 1)
        .with_bias(true)
        .init(device);
    
    let dropout = DropoutConfig::new(config.dropout).init();
    
    EcDnaModel {
        fc1,
        fc2,
        fc3,
        output,
        dropout,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;
    
    type TestBackend = NdArray<f32>;
    
    #[test]
    fn test_model_forward() {
        let device = <TestBackend as Backend>::Device::default();
        let config = ModelConfig::ecdna_default();
        let model = init_model::<TestBackend>(&config, &device);
        
        // Create batch of 2 samples with 85 features
        let input = Tensor::<TestBackend, 2>::zeros([2, 85], &device);
        let output = model.forward(input);
        
        // Output should be [2, 1] (batch_size, 1 for binary classification)
        assert_eq!(output.dims(), [2, 1]);
    }
    
    #[test]
    fn test_model_predict() {
        let device = <TestBackend as Backend>::Device::default();
        let config = ModelConfig::ecdna_default();
        let model = init_model::<TestBackend>(&config, &device);
        
        let input = Tensor::<TestBackend, 2>::zeros([2, 85], &device);
        let prediction = model.predict(input);
        
        assert_eq!(prediction.probabilities.dims(), [2, 1]);
        assert_eq!(prediction.predictions.dims(), [2]);
    }
}