# OTK: ecDNA Prediction Tool

OTK is a deep learning-based tool for predicting extrachromosomal DNA (ecDNA) cargo genes from genomic data. It uses the [Burn](https://burn.dev/) deep learning framework to provide high-performance predictions with native support for missing values.

## Features

- **Gene-level ecDNA cargo prediction**: Predict whether a gene is an ecDNA cargo gene
- **Sample-level focal amplification classification**: Classify samples into nofocal, noncircular, or circular types
- **Missing value handling**: Learned imputation for missing features
- **Multiple input formats**: Support for CSV, TSV, and gzipped files
- **Efficient inference**: Batch processing with progress tracking
- **Model checkpointing**: Save and resume training from checkpoints

## Installation

### Prerequisites

- Rust 1.70+ 
- For GPU support: CUDA or compatible graphics drivers

### Build from Source

```bash
git clone https://github.com/yourusername/otk.git
cd otk
cargo build --release
```

The compiled binary will be available at `target/release/otk`.

### Install via Cargo

```bash
cargo install otk
```

## Quick Start

### Training a Model

```bash
otk train -i data.csv -o ./output --epochs 100 --batch-size 256
```

### Making Predictions

```bash
otk predict -i input.csv -m output/checkpoints/best_model.mpk -o predictions.csv
```

### Evaluating a Model

```bash
otk evaluate -i test.csv -m output/checkpoints/best_model.mpk
```

## Input Data Format

OTK expects input data in CSV or TSV format with the following columns:

### Required Columns

- `sample`: Sample identifier
- `gene_id`: Gene identifier

### Feature Columns

- `segVal`: Total copy number
- `minor_cn`: Minor copy number
- `intersect_ratio`: Intersection ratio
- `purity`: Tumor purity
- `ploidy`: Tumor ploidy
- `AScore`: Aneuploidy score
- `pLOH`: Proportion of LOH
- `cna_burden`: CNA burden
- `CN1` to `CN19`: Copy number signature activities
- `age`: Patient age
- `gender`: Patient gender (0=female, 1=male)
- `cancer_type`: Cancer type (will be one-hot encoded)
- `freq_Linear`: Linear amplification frequency
- `freq_BFB`: BFB frequency
- `freq_Circular`: Circular amplification frequency
- `freq_HR`: HR frequency

### Target Column (for training)

- `y`: Binary label (1=ecDNA cargo gene, 0=non-ecDNA)

## Command Reference

### `otk train`

Train a new ecDNA prediction model.

```
Options:
  -i, --input <FILE>              Input data file (CSV or TSV)
  -o, --output <DIR>              Output directory for model [default: ./output]
  -e, --epochs <NUM>              Number of training epochs [default: 100]
  -b, --batch-size <SIZE>         Batch size [default: 256]
      --learning-rate <LR>        Learning rate [default: 0.001]
      --dropout <RATE>            Dropout rate [default: 0.3]
      --seed <SEED>               Random seed [default: 2026]
      --val-ratio <RATIO>         Validation ratio [default: 0.1]
      --test-ratio <RATIO>        Test ratio [default: 0.2]
      --patience <NUM>            Early stopping patience [default: 15]
  -d, --device <DEVICE>           Device to use (cpu, cuda, wgpu) [default: cpu]
      --quick                     Quick test mode (5 epochs)
  -v, --verbose                   Enable verbose output
```

### `otk predict`

Make predictions using a trained model.

```
Options:
  -i, --input <FILE>              Input data file
  -m, --model <FILE>              Model checkpoint file
  -o, --output <FILE>             Output file [default: predictions.csv]
  -f, --format <FORMAT>           Output format (csv, json) [default: csv]
  -b, --batch-size <SIZE>         Batch size [default: 256]
      --threshold <THRESHOLD>     Probability threshold [default: 0.5]
  -d, --device <DEVICE>           Device to use [default: cpu]
      --sample-level              Save sample-level predictions
  -v, --verbose                   Enable verbose output
```

### `otk evaluate`

Evaluate model performance on labeled data.

```
Options:
  -i, --input <FILE>              Input data file with labels
  -m, --model <FILE>              Model checkpoint file
  -o, --output <FILE>             Output report file
  -b, --batch-size <SIZE>         Batch size [default: 256]
  -d, --device <DEVICE>           Device to use [default: cpu]
  -v, --verbose                   Enable verbose output
```

## Model Architecture

OTK uses a deep neural network with the following architecture:

```
Input (85 features)
    ↓
Missing Value Layer (optional)
    ↓
Dense Layer (256 units) + BatchNorm + ReLU + Dropout
    ↓
Dense Layer (128 units) + BatchNorm + ReLU + Dropout
    ↓
Dense Layer (64 units) + BatchNorm + ReLU + Dropout
    ↓
Output Layer (1 unit) + Sigmoid
```

### Feature Dimensions

- Base features: 25 (segVal, minor_cn, intersect_ratio, purity, ploidy, AScore, pLOH, cna_burden, CN1-CN19, age, gender)
- Cancer type one-hot: 24
- Prior frequencies: 4 (freq_Linear, freq_BFB, freq_Circular, freq_HR)
- **Total: 85 features**

## Performance

OTK is optimized for both training and inference:

- **Training**: ~0.5-1 hour per epoch on CPU for 100K samples
- **Inference**: ~1000 samples/second on CPU
- **Memory**: ~2GB RAM for batch size 256

## Development

### Running Tests

```bash
cargo test
```

### Building Documentation

```bash
cargo doc --open
```

### Code Formatting

```bash
cargo fmt
cargo clippy
```

## Citation

If you use OTK in your research, please cite:

```bibtex
@article{wang2024machine,
  title={Machine learning-based extrachromosomal DNA identification in large-scale cohorts reveals its clinical implications in cancer},
  author={Wang, S and Wu, CY and He, MM and Yong, JX and Chen, YX and Qian, LM and others},
  journal={Nature Communications},
  volume={15},
  number={1},
  pages={1--17},
  year={2024},
  publisher={Nature Publishing Group}
}
```

## License

OTK is licensed under the MIT License. See [LICENSE](LICENSE) for details.

## Acknowledgments

- Built with [Burn](https://burn.dev/) deep learning framework
- Inspired by [GCAP](https://github.com/shixiangwang/gcap) project
- Data format based on AmpliconArchitect output

## Contact

For questions or issues, please open an issue on GitHub or contact the maintainers.