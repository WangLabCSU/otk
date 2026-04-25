# otk: ecDNA Analysis Toolkit

[![PyPI version](https://badge.fury.io/py/otk-ecdna.svg)](https://pypi.org/project/otk-ecdna/)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.9+-green.svg)](https://www.python.org/)

**otk** (ecDNA Analysis Toolkit) is a machine learning toolkit for predicting extrachromosomal DNA (ecDNA) cargo genes. It classifies genes at the gene level (ecDNA cargo vs. non-ecDNA) and identifies focal amplification types at the sample level (nofocal, noncircular, circular/ecDNA).

Based on the paper: Wang, S., et al. (2024). Machine learning-based extrachromosomal DNA identification in large-scale cohorts reveals its clinical implications in cancer. Nature Communications.

## Core Features

- Deep learning-based ecDNA cargo gene prediction at gene level
- Sample-level focal amplification type classification (nofocal/noncircular/circular)
- Multiple pre-trained models (XGBoost, Neural Networks, TabPFN)
- Efficient command-line interface for training and prediction
- GPU acceleration support
- Pre-trained models ready to use after pip install
- RESTful API for web service deployment
- Chinese mirror support for large model downloads

## Installation

### From PyPI (Recommended)

```bash
pip install otk-ecdna
```

This installs the `otk` CLI command and all pre-trained models (except TabPFN which is ~275MB and needs separate download).

### Download Large Models

The TabPFN model (~275MB) is hosted on GitHub Release:

```bash
# List available large models
otk download --list

# Download TabPFN model
otk download --model tabpfn
```

### From Source

```bash
git clone https://github.com/WangLabCSU/otk.git
cd otk/otk
pip install -e .
```

## Quick Start

```bash
# Check installation
otk --version

# List available models
otk models

# Run prediction (example)
otk predict --input data.csv --output predictions.csv --model xgb_new

# Start API server
otk api --port 8000
```

## CLI Commands

### Model Management

```bash
# List all available models with performance metrics
otk models

# Analyze a specific model
otk analyze --model xgb_new

# Generate model configuration
otk config generate --model xgb_new
```

### Training

```bash
# Train single model
otk train --model xgb_new --gpu 0

# Train neural network model
otk train --model transformer --gpu 0

# Train all models sequentially
otk train --all --gpu 0

# Train all models in parallel on multiple GPUs
otk train --all --parallel --gpus 0,1,2,3

# CPU-only training
otk train --model xgb_new --gpu -1
```

### Prediction

```bash
# Basic prediction
otk predict --input data.csv --output predictions.csv --model xgb_new

# With GPU acceleration
otk predict -i data.csv -o results/ -m transformer --gpu 0

# With custom threshold
otk predict -i data.csv -o predictions.csv -m xgb_new --threshold 0.5
```

### API Server

```bash
# Start API with default settings (base path /otk)
otk api

# Custom port
otk api --port 8080

# Serve at root (no base path)
otk api --base-path ""

# Development mode with auto-reload
otk api --reload

# Multiple workers
otk api --workers 4
```

### Model Download

```bash
# List large models requiring download
otk download --list

# Download TabPFN model
otk download --model tabpfn

# Force re-download
otk download --model tabpfn --force
```

## Data Format

### Input Data Format

Input data should be in CSV format.

**Minimal required columns:**

| Column | Description |
|--------|-------------|
| `sample` | Tumor sample ID |
| `gene_id` | Gene ID (e.g., ENSG00000284662) |
| `segVal` | Total gene copy number |

**Auto-filled columns (defaults applied if missing):**

| Column | Default | Description |
|--------|---------|-------------|
| `minor_cn` | 0 | Minor copy number |
| `intersect_ratio` | 1.0 | Segment-gene overlap ratio |
| `purity` | 0.8 | Tumor purity |
| `ploidy` | 2.0 | Genome ploidy |
| `AScore` | 10.0 | Aneuploidy score |
| `pLOH` | 0.1 | LOH proportion |
| `cna_burden` | 0.2 | CNA burden |
| `CN1-CN19` | 0.05 each | Copy number signatures |
| `type` | - | Cancer type → auto-converts to `type_*` columns |

**Automatically generated features (from gene_id matching):**

| Column | Description |
|--------|-------------|
| `freq_Linear` | Prior frequency in linear amplifications |
| `freq_BFB` | Prior frequency in BFB events |
| `freq_Circular` | Prior frequency in ecDNA |
| `freq_HR` | Prior frequency in HR events |

**Training data requires:**

| Column | Description |
|--------|-------------|
| `y` | Binary label (1=ecDNA cargo gene, 0=not) |

**Supported cancer types (24):**
BLCA, BRCA, CESC, COAD, DLBC, ESCA, GBM, HNSC, KICH, KIRC, KIRP, LGG, LIHC, LUAD, LUSC, OV, PRAD, READ, SARC, SKCM, STAD, THCA, UCEC, UVM

### Output Format

| Column | Description |
|--------|-------------|
| `sample` | Sample ID |
| `gene_id` | Gene ID |
| `prediction_prob` | Probability of ecDNA (0-1) |
| `prediction` | Binary classification (0/1) |
| `sample_level_prediction_label` | Sample type: nofocal/noncircular/circular |
| `sample_level_prediction` | Sample type code (0/1/2) |

Sample classification rules:
- `circular` (2): Any gene predicted as ecDNA cargo
- `noncircular` (1): No ecDNA but segVal > ploidy + 2
- `nofocal` (0): Otherwise

## Available Models

| Model | Type | Test auPRC | Description |
|-------|------|------------|-------------|
| xgb_new | XGBoost | 0.8339 | Optimized with feature engineering |
| tabpfn | TabPFN | 0.8323 | TabPFN ensemble (~275MB, needs download) |
| deep_residual | Neural | 0.8132 | Deep residual network |
| xgb_tuned | XGBoost | 0.8065 | Hyperparameter tuned |
| optimized_residual | Neural | 0.7906 | Optimized residual network |
| baseline_mlp | Neural | 0.7663 | Simple MLP baseline |
| dgit_super | Neural | 0.7662 | Deep gated interaction transformer |
| xgb_paper | XGBoost | 0.7138 | Paper reproduction (11 features) |
| transformer | Neural | 0.6875 | Transformer architecture |

All models use unified 80/10/10 data split with seed=2026 for reproducibility.

## API Service

Start a RESTful API for web-based prediction:

```bash
# Start API (default base path /otk)
otk api

# Access points:
# - API docs: http://localhost:8000/otk/docs
# - Health: http://localhost:8000/otk/health
# - Web UI: http://localhost:8000/otk/
```

See [otk_api/README.md](otk_api/README.md) for full API documentation.

## Project Structure

```
otk/
├── src/otk/           # Core library
│   ├── data/          # Data handling
│   ├── models/        # Model implementations
│   ├── predict/       # Prediction utilities
│   └── cli.py         # Command-line interface
├── otk_api/           # FastAPI web service
│   ├── api/           # API implementation
│   ├── models/        # Pre-trained models
│   └── static/        # Performance charts
├── configs/           # Model configurations
└── tests/             # Unit tests
```

## Citation

If you use otk in your research, please cite:

```bibtex
Wang, S., et al. (2024). Machine learning-based extrachromosomal DNA 
identification in large-scale cohorts reveals its clinical implications 
in cancer. Nature Communications.
```

## License

MIT License. See [LICENSE](LICENSE) file for details.

## Contact

- **Homepage**: https://github.com/WangLabCSU/otk
- **PyPI**: https://pypi.org/project/otk-ecdna/
- **Email**: wangshx@csu.edu.cn
