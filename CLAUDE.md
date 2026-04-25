# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

OTK (ecDNA Analysis Toolkit) is a deep learning toolkit for extrachromosomal DNA (ecDNA) prediction. It predicts whether genes are detected as ecDNA cargo genes at the gene level and classifies focal amplification types at the sample level.

**Paper Citation**: Wang, S., et al. (2024). Machine learning-based extrachromosomal DNA identification in large-scale cohorts reveals its clinical implications in cancer. Nature Communications.

## Key Commands

### Training
```bash
# Install package
pip install -e .

# Train single model
otk train --model xgb_new --gpu 0
otk train --model transformer --gpu 0
otk train --model tabpfn --gpu 0

# Train all models sequentially
otk train --all --gpu 0

# Train all models in parallel on multiple GPUs
otk train --all --parallel --gpus 0,1,2,3

# CPU-only training
otk train --model xgb_new --gpu -1
```

### Prediction
```bash
otk predict --input data.csv --output predictions.csv --model xgb_new
otk predict -i data.csv -o results/ -m transformer --gpu 0
```

### Model Management
```bash
otk models              # List available models
otk analyze --model xgb_new  # Analyze trained model performance
otk config generate --model xgb_new  # Generate config.yml for model
otk config generate --all    # Generate configs for all models
```

### API Server
```bash
# Using CLI (recommended)
otk api                           # Start with base path /otk (default)
otk api --port 8080               # Custom port, base path /otk
otk api --base-path ""            # Serve at root (no base path)
otk api --base-path /myapp        # Custom base path
otk api --reload                  # Development mode with auto-reload

# Using shell script (legacy)
cd otk_api
./start_api.sh                         # Start with base path /otk
OTK_BASE_PATH="" ./start_api.sh        # Serve at root
API_HOST=0.0.0.0 API_PORT=8080 ./start_api.sh  # Custom host/port
```

**Note**: Default base path is `/otk` for reverse proxy deployment (e.g., http://biotree.top:38123/otk). Local users can use `--base-path ""` to serve at root.

### Model Download
```bash
otk download --list               # List large models requiring download
otk download --model tabpfn       # Download TabPFN model (~275MB)
otk download --model tabpfn --force  # Force re-download
```

### Testing
```bash
cd otk_api/tests
python test_predict_api.py  # Run API tests
```

## Unified Data Split (CRITICAL)

All models must use the same data split for reproducibility:
- **Ratio**: 80/10/10 (train/val/test)
- **Random seed**: 2026 (fixed, do not change)
- **Split file**: `src/otk/data/split_2026.json`
- **Implementation**: `src/otk/data/data_split.py`

Usage:
```python
from otk.data import load_split
train_df, val_df, test_df = load_split()
```

## Model Architecture

All models inherit from `BaseEcDNAModel` (`src/otk/models/base_model.py`) with unified interface:
- `fit(X_train, y_train, X_val, y_val)` - Training
- `predict_proba(X)` - Probability prediction
- `predict(X)` - Binary prediction
- `save(path)` / `load(path)` - Persistence
- `evaluate_gene_level()` / `evaluate_sample_level()` - Metrics

### Available Models

| Model | Type | File | Description |
|-------|------|------|-------------|
| xgb_new | XGBoost | `xgb11_model.py` | Optimized with feature engineering |
| xgb_paper | XGBoost | `xgb11_model.py` | Paper reproduction (11 features) |
| baseline_mlp | Neural | `neural_models.py` | Simple MLP baseline |
| transformer | Neural | `neural_models.py` | Transformer architecture |
| deep_residual | Neural | `neural_models.py` | Deep residual network |
| optimized_residual | Neural | `neural_models.py` | Optimized residual network |
| dgit_super | Neural | `neural_models.py` | Deep gated interaction transformer |
| tabpfn | TabPFN | `tabpfn_model.py` | TabPFN ensemble |

## Project Structure

```
otk/
├── src/otk/              # Core library
│   ├── data/             # Data handling
│   │   ├── data_split.py     # Unified split (seed=2026)
│   │   ├── data_processor.py # Preprocessing
│   │   └── sorted_modeling_data.csv.gz  # Main dataset (~52MB)
│   ├── models/           # Model implementations
│   │   ├── base_model.py     # Abstract base class
│   │   ├── xgb11_model.py    # XGBoost models
│   │   ├── neural_models.py  # Neural networks
│   │   ├── tabpfn_model.py   # TabPFN
│   │   └── custom_losses.py  # Loss functions
│   ├── cli.py            # Command-line interface
│   └── predict/          # Prediction utilities
├── otk_api/              # FastAPI prediction service
│   ├── api/              # API implementation
│   │   ├── main.py           # FastAPI app
│   │   ├── predictor_wrapper.py
│   │   └── data_validator.py
│   ├── models/           # Trained model storage
│   ├── tests/            # API tests
│   └── start_api.sh      # Startup script
├── configs/              # Model configuration files (YAML)
├── train_unified.py      # Training wrapper (calls otk CLI)
└── tests/                # Unit tests (placeholder)
```

## Input Data Format

Required columns in CSV:
- `sample`: Sample ID
- `gene_id`: Gene identifier (e.g., ENSG00000284662)
- `segVal`: Gene total copy number
- `y`: Binary label (for training only)

Auto-filled columns with defaults:
- `minor_cn`: 0, `purity`: 0.8, `ploidy`: 2.0, `AScore`: 10.0
- `pLOH`: 0.1, `cna_burden`: 0.2, `intersect_ratio`: 1.0
- `CN1-CN19`: 0.05 each (copy number signatures)
- Cancer type columns (`type_*`): converted from `type` column or 0

## Important Notes

1. **Never change random seed 2026** - all models depend on consistent splits
2. **Convert numpy types to Python floats** before saving to YAML files to avoid serialization errors
3. **Use unified data split** via `load_split()` - never create custom splits
4. **Models are stored in `otk_api/models/{model_name}/`** with `best_model.pkl` or `best_model.pth` and `training_summary.yml`
5. **API runs on port 8000 by default** - set `OTK_BASE_PATH` for reverse proxy

## Performance Targets

- Gene-level auPRC: ≥ 0.85
- Gene-level Precision: ≥ 0.8
- Sample-level auROC: ≥ 0.9
- Sample-level auPRC: ≥ 0.99