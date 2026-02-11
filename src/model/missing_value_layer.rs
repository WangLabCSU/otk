use burn::prelude::*;
use burn::nn::*;

/// Missing value handling layer
/// 
/// This layer learns to impute missing values based on observed features.
/// It uses a learned transformation to estimate missing values from available features.
#[derive(Module, Debug)]
pub struct MissingValueLayer<B: Backend> {
    /// Transformation network for imputation
    imputation_net: Linear<B>,
    /// Gate network to control imputation strength
    gate_net: Linear<B>,
    /// Output projection
    output_proj: Linear<B>,
}

/// Missing value layer configuration
#[derive(Config, Debug)]
pub struct MissingValueLayerConfig {
    /// Input feature dimension
    pub input_size: usize,
    /// Hidden dimension for imputation network
    pub hidden_size: usize,
}

impl MissingValueLayerConfig {
    /// Initialize missing value layer
    pub fn init<B: Backend>(&self, device: &B::Device) -> MissingValueLayer<B> {
        let imputation_net = LinearConfig::new(self.input_size, self.hidden_size)
            .with_bias(true)
            .init(device);
        
        let gate_net = LinearConfig::new(self.input_size, self.input_size)
            .with_bias(true)
            .init(device);
        
        let output_proj = LinearConfig::new(self.hidden_size, self.input_size)
            .with_bias(true)
            .init(device);
        
        MissingValueLayer {
            imputation_net,
            gate_net,
            output_proj,
        }
    }
}

impl<B: Backend> MissingValueLayer<B> {
    /// Forward pass with missing value handling
    /// 
    /// # Arguments
    /// * `x` - Input tensor of shape [batch_size, input_size]
    /// * `mask` - Binary mask tensor of shape [batch_size, input_size], 
    ///            where 1 indicates observed value, 0 indicates missing
    /// 
    /// # Returns
    /// * Tensor with imputed missing values
    pub fn forward(&self, x: Tensor<B, 2>, mask: Tensor<B, 2>) -> Tensor<B, 2> {
        // Create masked input (set missing values to 0)
        let masked_input = x.clone().mul(mask.clone());
        
        // Compute imputation values
        let imputation_hidden = self.imputation_net.forward(masked_input.clone());
        let imputation_hidden = relu(imputation_hidden);
        let imputation_values = self.output_proj.forward(imputation_hidden);
        
        // Compute gating weights
        let gate_logits = self.gate_net.forward(masked_input);
        let gates = sigmoid(gate_logits);
        
        // Combine observed values with imputed values
        // For observed positions: keep original values
        // For missing positions: use imputed values
        let inverse_mask = mask.clone().neg().add_scalar(1.0);
        let output = x.mul(mask).add(imputation_values.mul(inverse_mask));
        
        // Apply learned gating for smooth transition
        output.mul(gates).add(x.mul(mask).mul(gates.neg().add_scalar(1.0)))
    }
    
    /// Forward pass without explicit mask (assumes NaN for missing)
    pub fn forward_with_nan(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        // Create mask from NaN values
        let mask = x.clone().equal_elem(f32::NAN).neg().add_scalar(1.0);
        
        // Replace NaN with 0 for processing
        let x_clean = x.clone().mask_fill(x.equal_elem(f32::NAN), 0.0);
        
        self.forward(x_clean, mask)
    }
}

/// Utility function to create a mask tensor from feature vectors
pub fn create_mask_tensor<B: Backend>(
    vectors: &[crate::data::FeatureVector],
    device: &B::Device,
) -> Tensor<B, 2> {
    let batch_size = vectors.len();
    let n_features = vectors[0].features.len();
    
    let mask_data: Vec<f32> = vectors
        .iter()
        .flat_map(|v| v.mask.iter().map(|&m| if m { 1.0 } else { 0.0 }))
        .collect();
    
    Tensor::from_floats(mask_data.as_slice(), device)
        .reshape([batch_size, n_features])
}

/// Utility function to create feature tensor from feature vectors
pub fn create_feature_tensor<B: Backend>(
    vectors: &[crate::data::FeatureVector],
    device: &B::Device,
) -> Tensor<B, 2> {
    let batch_size = vectors.len();
    let n_features = vectors[0].features.len();
    
    let feature_data: Vec<f32> = vectors
        .iter()
        .flat_map(|v| v.features.clone())
        .collect();
    
    Tensor::from_floats(feature_data.as_slice(), device)
        .reshape([batch_size, n_features])
}

/// Utility function to create target tensor
pub fn create_target_tensor<B: Backend>(
    vectors: &[crate::data::FeatureVector],
    device: &B::Device,
) -> Tensor<B, 1, Int> {
    let targets: Vec<i64> = vectors
        .iter()
        .map(|v| v.target.unwrap_or(0) as i64)
        .collect();
    
    Tensor::from_data(targets.as_slice(), device)
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;
    
    type TestBackend = NdArray<f32>;
    
    #[test]
    fn test_missing_value_layer_forward() {
        let device = <TestBackend as Backend>::Device::default();
        let config = MissingValueLayerConfig {
            input_size: 10,
            hidden_size: 5,
        };
        
        let layer = config.init::<TestBackend>(&device);
        
        // Create input with some missing values
        let x = Tensor::<TestBackend, 2>::from_floats(
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
            &device,
        ).reshape([1, 10]);
        
        let mask = Tensor::<TestBackend, 2>::from_floats(
            &[1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            &device,
        ).reshape([1, 10]);
        
        let output = layer.forward(x, mask);
        
        // Output should have shape [1, 10]
        assert_eq!(output.dims(), [1, 10]);
    }
}