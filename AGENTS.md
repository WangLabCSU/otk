# AGENTS.md - AI Assistant Guide for OTK Project

This document provides guidance for AI assistants working on the OTK (ecDNA prediction) project.

## Project Overview

OTK is a machine learning toolkit for extrachromosomal DNA (ecDNA) prediction. The project uses deep learning and gradient boosting models to predict ecDNA presence in cancer samples.

## Key Design Principles

### 1. Unified Data Split (CRITICAL)
- **All models must use the same data split**: 80/10/10 (train/val/test)
- **Random seed**: 2026 (fixed across all models)
- **Implementation**: `src/otk/data/data_split.py`
- **Usage**: `from otk.data import load_split`

```python
train_df, val_df, test_df = load_split()
# Returns: train=5808151, val=715972, test=755098 rows
```

### 2. Unified Model Interface
All models inherit from `BaseEcDNAModel` and implement:
- `fit(X_train, y_train, X_val, y_val)` - Training
- `predict_proba(X)` - Probability prediction
- `predict(X)` - Binary prediction
- `save(path)` / `load(path)` - Persistence
- `evaluate_gene_level()` - Gene-level metrics
- `evaluate_sample_level()` - Sample-level metrics

### 3. Configuration Files
Each model has a `config.yml` in its directory:
```
otk_api/models/{model_name}/config.yml
```

Models should read configuration from this file for:
- Hyperparameters
- Architecture settings
- Training configuration

### 4. Training Summary Format
After training, models save `training_summary.yml` with unified format:
```yaml
model_name: xgb_new
gene_level:
  train:
    auPRC: 0.9519
    AUC: 0.9993
    Precision: 0.7987
    Recall: 0.9299
    F1: 0.8593
  val:
    auPRC: 0.6838
    ...
  test:
    auPRC: 0.8339
    ...
sample_level:
  train:
    auPRC: 0.9906
    AUC: 0.9670
    ...
  val:
    ...
  test:
    ...
```

**IMPORTANT**: Never use numpy objects in YAML files. Always convert to Python floats:
```python
# WRONG
metrics = {'auPRC': np.float64(0.85)}  # Will cause YAML serialization issues

# CORRECT
metrics = {'auPRC': float(0.85)}
```

## Project Structure

```
otk/
├── src/otk/
│   ├── data/
│   │   ├── data_split.py      # Unified data split (seed=2026)
│   │   ├── data_processor.py  # Data preprocessing
│   │   └── sorted_modeling_data.csv.gz  # Main dataset
│   ├── models/
│   │   ├── base_model.py      # Base class for all models
│   │   ├── xgb11_model.py     # XGBoost models (XGB11, XGBNew)
│   │   ├── neural_models.py   # Neural network models
│   │   ├── tabpfn_model.py    # TabPFN model
│   │   ├── custom_losses.py   # Loss functions
│   │   └── config_generator.py # Config file generator
│   ├── train/
│   │   └── trainer.py         # Training utilities
│   ├── predict/
│   │   └── predictor.py       # Prediction utilities
│   └── utils/
│       └── __init__.py
├── otk_api/
│   ├── models/                # Trained models directory
│   │   ├── xgb_new/
│   │   ├── xgb_paper/
│   │   ├── baseline_mlp/
│   │   ├── transformer/
│   │   ├── deep_residual/
│   │   ├── optimized_residual/
│   │   ├── dgit_super/
│   │   └── tabpfn/
│   └── model_analyzer.py      # Model analysis and reporting
├── train_unified.py           # Unified training script
└── README.md
```

## Model List

| Model | Type | File | Description |
|-------|------|------|-------------|
| xgb_new | XGBoost | xgb11_model.py | Optimized with feature engineering |
| xgb_paper | XGBoost | xgb11_model.py | Paper reproduction (11 features) |
| baseline_mlp | Neural | neural_models.py | Simple MLP baseline |
| transformer | Neural | neural_models.py | Transformer architecture |
| deep_residual | Neural | neural_models.py | Deep residual network |
| optimized_residual | Neural | neural_models.py | Optimized residual network |
| dgit_super | Neural | neural_models.py | Deep gated interaction transformer |
| tabpfn | TabPFN | tabpfn_model.py | TabPFN ensemble |

## Performance Targets

- **Gene-level auPRC**: ≥ 0.85
- **Gene-level Precision**: ≥ 0.8
- **Sample-level auROC**: ≥ 0.9
- **Sample-level auPRC**: ≥ 0.99

## Common Tasks

### Training a Model
```bash
# Train single model
python train_unified.py --model xgb_new

# Train all models
python train_unified.py --all
```

### Analyzing Models
```bash
cd otk_api
python model_analyzer.py
```

### Using the API
```bash
cd otk_api
OTK_BASE_PATH=/otk bash start_api.sh
```

## Important Notes

1. **Never change the random seed** (2026) without updating all models
2. **Always use unified data split** from `data_split.py`
3. **Convert numpy types to Python types** before saving to YAML
4. **Keep config.yml synchronized** with model implementation
5. **Test model loading** after training to ensure persistence works

## Troubleshooting

### YAML Serialization Error
If you see `could not determine a constructor for the tag 'tag:yaml.org,2002:python/object/apply:numpy...'`:
- The training_summary.yml contains numpy objects
- Solution: Re-train the model with proper float conversion

### Data Split Mismatch
If models have different sample counts:
- Check that all models use `load_split()` from `data_split.py`
- Verify the split file exists: `src/otk/data/split_2026.json`

### Model Import Error
If model cannot be imported:
- Check `src/otk/models/__init__.py` includes the model
- Verify the model class inherits from `BaseEcDNAModel`
