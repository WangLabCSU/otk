# otk: ecDNA Analysis Toolkit

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

otk (ecDNA Analysis Toolkit) is a deep learning-based tool for analyzing extrachromosomal DNA (ecDNA), predicting whether genes are detected as ecDNA cargo genes at the gene level, and classifying focal amplification types at the sample level.

## Core Features

- Deep learning-based ecDNA cargo gene prediction
- Sample-level focal amplification type classification
- Support for analysis from BAM files or processed copy number data
- Efficient command-line interface
- GPU acceleration support

## Technology Stack

- Python 3.8+
- PyTorch 2.0+
- NumPy
- Pandas
- scikit-learn
- Click (command-line interface)

## Installation Guide

### From Source

1. Clone the repository:

```bash
git clone https://github.com/WangLabCSU/otk.git
cd otk
```

2. Install with pip:

```bash
pip install -e .
```

### Dependencies

The following dependencies will be installed automatically:

- pandas>=2.0
- numpy>=1.24
- torch>=2.0
- scikit-learn>=1.3
- tqdm>=4.65
- click>=8.1
- matplotlib>=3.7
- seaborn>=0.12
- pyyaml>=6.0

## Usage

otk provides two main command-line subcommands: `train` and `predict`.

### Model Training

Use the `otk train` command to train the model:

```bash
otk train --config configs/model_config.yml --output models/ --gpu 0
```

Parameters:
- `--config, -c`: Path to configuration file (default: configs/model_config.yml)
- `--output, -o`: Output directory for trained models (default: models/)
- `--gpu, -g`: GPU device ID to use (default: 0)

### Model Prediction

Use the `otk predict` command for predictions:

```bash
otk predict --model models/best_model.pth --input data/test_data.csv --output predictions/ --gpu -1
```

Parameters:
- `--model, -m`: Path to trained model (required)
- `--input, -i`: Path to input data file (required)
- `--output, -o`: Output directory for predictions (default: predictions/)
- `--gpu, -g`: GPU device ID to use (default: -1, i.e., use CPU)

## Data Format

### Input Data Format

Input data should be in CSV format with the following columns:

- `sample`: Tumor sample ID
- `gene_id`: Gene ID
- `segVal`: Total gene copy number
- `minor_cn`: Minor gene copy number
- `age`: Patient age
- `gender`: Patient gender
- One-hot encoded columns for various tumor types (e.g., `type_BLCA`, `type_BRCA`, etc.)
- `freq_Linear`, `freq_BFB`, `freq_Circular`, `freq_HR`: Prior estimated frequencies of genes in different types of genomic focal amplifications

### Output Data Format

Prediction results include the following columns:

- `sample`: Tumor sample ID
- `gene_id`: Gene ID
- `prediction_prob`: Probability of being an ecDNA cargo gene
- `prediction`: Binary classification result (0 or 1)

Additionally, sample-level prediction results are generated:

- `sample`: Tumor sample ID
- `prediction_prob`: Maximum prediction probability in the sample
- `prediction`: Sample-level prediction result (0 or 1)
- `focal_amplification_type`: Sample's focal amplification type (circular or noncircular)

## Model Architecture

otk uses a Multi-Layer Perceptron (MLP) as the deep learning model architecture with the following default configuration:

- Input layer: 58 features
- Hidden layer 1: 128 neurons, ReLU activation, 20% dropout
- Hidden layer 2: 64 neurons, ReLU activation, 20% dropout
- Hidden layer 3: 32 neurons, ReLU activation, 10% dropout
- Output layer: 1 neuron, Sigmoid activation

The model uses BCEWithLogitsLoss as the loss function and Adam as the optimizer.

## Configuration File

Model configuration uses YAML format, with example configuration files located in `configs/`. You can modify parameters in the configuration files as needed, such as model architecture and training parameters.

## Examples

### Training Examples

```bash
# Train model with default configuration
otk train

# Train model with custom configuration file
otk train --config my_config.yml
```

### Prediction Examples

```bash
# Make predictions using a trained model
otk predict --model models/best_model.pth --input test_data.csv
```

## Performance Metrics

The following performance metrics are recorded during model training:

- auPRC (Area under Precision-Recall Curve)
- AUC (Area under ROC Curve)
- F1 Score
- Precision
- Recall

## Contribution Guide

We welcome community contributions! If you have any questions or suggestions, please submit them through GitHub Issues.

### Development Process

1. Fork the repository
2. Create a feature branch
3. Implement features or fix bugs
4. Run tests
5. Submit a Pull Request

## License

This project is licensed under the Apache License 2.0. See the [LICENSE](LICENSE) file for details.

## Citation

If you use otk in your research, please cite the following paper:

```
Wang, S., Wu, C. Y., He, M. M., Yong, J. X., Chen, Y. X., Qian, L. M., ... & Zhao, Q. (2024). Machine learning-based extrachromosomal DNA identification in large-scale cohorts reveals its clinical implications in cancer. Nature Communications, 15(1), 1-17.
```

## Contact

- Project homepage: https://github.com/WangLabCSU/otk
- Email: wangshx@csu.edu.cn
