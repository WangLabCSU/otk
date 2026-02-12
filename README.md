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

**Required identifier columns:**
- `sample`: Tumor sample ID
- `gene_id`: Gene ID

**Copy number features:**
- `segVal`: Total gene copy number
- `minor_cn`: Minor gene copy number
- `intersect_ratio`: Proportion of overlap between copy number detection segment and gene region

**Sample-level genomic features (same value for all genes in a sample):**
- `purity`: Tumor purity estimate
- `ploidy`: Tumor genome ploidy estimate
- `AScore`: Aneuploidy score
- `pLOH`: Proportion of genome with loss of heterozygosity (LOH)
- `cna_burden`: Proportion of genome with copy number alterations

**Copy number signature features:**
- `CN1` to `CN19`: 19 copy number signature activity estimates

**Clinical features:**
- `age`: Patient age
- `gender`: Patient gender (0/1 encoded)

**Tumor type features (one-hot encoded, 24 cancer types):**
- `type_BLCA`, `type_BRCA`, `type_CESC`, `type_COAD`, `type_DLBC`, `type_ESCA`, `type_GBM`, `type_HNSC`
- `type_KICH`, `type_KIRC`, `type_KIRP`, `type_LGG`, `type_LIHC`, `type_LUAD`, `type_LUSC`, `type_OV`
- `type_PRAD`, `type_READ`, `type_SARC`, `type_SKCM`, `type_STAD`, `type_THCA`, `type_UCEC`, `type_UVM`

**Gene frequency features:**
- `freq_Linear`: Prior estimated frequency of gene in linear focal amplifications
- `freq_BFB`: Prior estimated frequency of gene in breakage-fusion-bridge (BFB) events
- `freq_Circular`: Prior estimated frequency of gene in circular focal amplifications (ecDNA)
- `freq_HR`: Prior estimated frequency of gene in homologous recombination events

**Target column (for training data):**
- `y`: Binary label indicating whether the gene is detected as an ecDNA cargo gene (1) or not (0)

### Output Data Format

Prediction results are saved as a CSV file with the following columns:

**Gene-level predictions:**
- `sample`: Tumor sample ID
- `gene_id`: Gene ID
- `prediction_prob`: Probability of being an ecDNA cargo gene (0-1)
- `prediction`: Binary classification result (0 = not ecDNA cargo, 1 = ecDNA cargo)

**Sample-level predictions:**
- `sample_level_prediction_label`: Sample-level focal amplification type classification:
  - `nofocal`: No focal amplification detected
  - `noncircular`: Non-circular focal amplification detected
  - `circular`: Circular focal amplification (ecDNA) detected
- `sample_level_prediction`: Numerical encoding of sample-level classification (0 = nofocal, 1 = noncircular, 2 = circular)

Note: Sample-level classification follows these rules:
1. If any gene in the sample is predicted as ecDNA cargo (`prediction` = 1), the sample is classified as `circular`
2. If no ecDNA cargo genes but any gene has `segVal > ploidy + 2`, the sample is classified as `noncircular`
3. Otherwise, the sample is classified as `nofocal`

## Model Architecture

otk supports multiple deep learning model architectures (MLP, Transformer, MultiInputTransformer) with configurable parameters. The default MLP configuration is:

- Input layer: 57 features (matching the input data format)
- Hidden layer 1: 128 neurons, ReLU activation, 20% dropout
- Hidden layer 2: 64 neurons, ReLU activation, 20% dropout
- Hidden layer 3: 32 neurons, ReLU activation, 10% dropout
- Output layer: 1 neuron, Sigmoid activation

The model uses BCEWithLogitsLoss (or CombinedLoss with Focal Loss for imbalanced data) as the loss function and Adam as the optimizer.

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
