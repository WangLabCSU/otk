# Model Performance Analysis Report

**Generated**: 2026-02-12 22:34:16
**Models Directory**: `/data/home/wsx/Projects/otk/otk/otk_api/models`
**Total Models**: 4

## Abstract

This report presents a comprehensive analysis of multiple deep learning models
developed for extrachromosomal DNA (ecDNA) prediction. The models were trained on
a large-scale dataset with severe class imbalance and evaluated using multiple
performance metrics including auPRC, AUC, Precision, Recall, and F1-score.

## Dataset Description

### Sample Distribution

| Dataset | Total Samples | Positive Samples | Positive Rate |
|---------|---------------|------------------|---------------|
| Training | 5,109,832 | 17,925 | 0.3508% |
| Validation | 716,303 | 2,681 | 0.3743% |
| Test | 1,453,086 | 5,118 | 0.3522% |

**Total**: 7,279,221 samples, 25,724 positive (0.3534%)

**Note**: The dataset exhibits severe class imbalance with only ~0.35% positive samples,
which presents significant challenges for model training and evaluation.

## Model Architecture Comparison

### Overview

| Model | Architecture | Network Structure | Loss Function | Optimizer |
|-------|--------------|-------------------|---------------|-----------|
| baseline_mlp | Baseline | 57→256→128→64→1 | BCEWithLogitsLoss | Adam |
| deep_residual | PrecisionFocusedEcDNA | 57→512→256→128→64→32→1 | auPRCOptimizedLoss | AdamW |
| optimized_residual | OptimizedEcDNA | 57→128→64→32→16→1 | CombinedLoss | AdamW |
| transformer | TransformerEcDNA | 57→128(embedding)→Attention→64→32→1 | auPRCOptimizedLoss | Adam |

### Training Configuration

| Model | Learning Rate | Weight Decay | Batch Size | Epochs | Best Epoch | Early Stop |
|-------|---------------|--------------|------------|--------|------------|------------|
| baseline_mlp | 0.001000 | 0.0001 | 512 | 16 | 6 | Yes |
| deep_residual | 0.000100 | 0.0100 | 1024 | 17 | 2 | Yes |
| optimized_residual | 0.001000 | 0.1000 | 1024 | 38 | 3 | Yes |
| transformer | 0.000100 | 0.0001 | 1024 | 20 | 10 | Yes |

## Performance Metrics

### Test Set Performance (Primary Evaluation)

| Model | auPRC | AUC | Precision | Recall | F1-Score |
|-------|-------|-----|-----------|--------|----------|
| deep_residual | **0.7505** | 0.9780 | 0.7153 | 0.7141 | 0.7147 |
| transformer | **0.7207** | 0.9726 | 0.7654 | 0.5883 | 0.6653 |
| baseline_mlp | **0.7031** | 0.9719 | 0.8929 | 0.5686 | 0.6948 |
| optimized_residual | **0.7004** | 0.9733 | 0.8528 | 0.5748 | 0.6867 |

### Complete Performance Comparison

#### Training Set Performance

| Model | auPRC | AUC | Precision | Recall | F1-Score |
|-------|-------|-----|-----------|--------|----------|
| baseline_mlp | 0.8495 | 0.9767 | 0.9422 | 0.7564 | 0.8391 |
| deep_residual | 0.8581 | 0.9977 | 0.8183 | 0.7857 | 0.8017 |
| optimized_residual | 0.4764 | 0.9514 | 0.5812 | 0.4335 | 0.4966 |
| transformer | 0.8864 | 0.9985 | 0.8429 | 0.8305 | 0.8366 |

#### Validation Set Performance

| Model | auPRC | AUC | Precision | Recall | F1-Score |
|-------|-------|-----|-----------|--------|----------|
| baseline_mlp | 0.8164 | 0.9499 | 0.9213 | 0.6986 | 0.7947 |
| deep_residual | 0.7592 | 0.9882 | 0.7703 | 0.7128 | 0.7404 |
| optimized_residual | 0.7542 | 0.9895 | 0.6537 | 0.8060 | 0.7219 |
| transformer | 0.7710 | 0.9781 | 0.7618 | 0.7490 | 0.7553 |

#### Test Set Performance

| Model | auPRC | AUC | Precision | Recall | F1-Score |
|-------|-------|-----|-----------|--------|----------|
| baseline_mlp | 0.7031 | 0.9719 | 0.8929 | 0.5686 | 0.6948 |
| deep_residual | 0.7505 | 0.9780 | 0.7153 | 0.7141 | 0.7147 |
| optimized_residual | 0.7004 | 0.9733 | 0.8528 | 0.5748 | 0.6867 |
| transformer | 0.7207 | 0.9726 | 0.7654 | 0.5883 | 0.6653 |

### Overfitting Analysis

| Model | Train-Val auPRC Gap | Severity | Precision Gap | Recall Gap |
|-------|---------------------|----------|---------------|------------|
| baseline_mlp | 0.0330 | ✅ low | 0.0209 | 0.0578 |
| deep_residual | 0.0989 | ⚠️ medium | 0.0480 | 0.0729 |
| optimized_residual | -0.2778 | ✅ low | -0.0724 | -0.3726 |
| transformer | 0.1154 | ⚠️ medium | 0.0812 | 0.0815 |

## Best Model Recommendations

| Metric | Best Model | Value |
|--------|------------|-------|
| **Best auPRC** | deep_residual | 0.7505 |
| **Best AUC** | deep_residual | 0.9780 |
| **Best F1-Score** | deep_residual | 0.7147 |
| **Best Precision** | baseline_mlp | 0.8929 |
| **Best Recall** | deep_residual | 0.7141 |
| **Best Generalization** | optimized_residual | Gap: -0.2778 |

## Architecture Details

### baseline_mlp

- **Type**: `Baseline`
- **Description**: Simple MLP Network
- **Structure**: `57→256→128→64→1`
- **Key Features**: ReLU activation, Sigmoid output, No regularization
- **Suitable For**: Baseline model, high precision low recall scenarios
- **Loss Function**: `BCEWithLogitsLoss`
- **Optimizer**: `Adam` (lr=0.001, weight_decay=0.0001)

### deep_residual

- **Type**: `PrecisionFocusedEcDNA`
- **Description**: Deep Residual Network
- **Structure**: `57→512→256→128→64→32→1`
- **Key Features**: Residual connections, LayerNorm, GELU activation, Progressive dimension reduction
- **Suitable For**: Deep feature learning, high precision scenarios
- **Loss Function**: `auPRCOptimizedLoss`
- **Optimizer**: `AdamW` (lr=0.0001, weight_decay=0.01)

### optimized_residual

- **Type**: `OptimizedEcDNA`
- **Description**: Optimized Residual Network
- **Structure**: `57→128→64→32→16→1`
- **Key Features**: Residual blocks, BatchNorm, Combined loss function
- **Suitable For**: Balanced training, stable convergence
- **Loss Function**: `CombinedLoss`
- **Optimizer**: `AdamW` (lr=0.001, weight_decay=0.1)

### transformer

- **Type**: `TransformerEcDNA`
- **Description**: Transformer Attention Model
- **Structure**: `57→128(embedding)→Attention→64→32→1`
- **Key Features**: Self-attention mechanism, LayerNorm, GELU activation, Dropout regularization
- **Suitable For**: Feature interaction learning, balanced precision-recall
- **Loss Function**: `auPRCOptimizedLoss`
- **Optimizer**: `Adam` (lr=0.0001, weight_decay=0.0001)

## Statistical Considerations

### Evaluation Metrics

- **auPRC (Area under Precision-Recall Curve)**: Primary metric for imbalanced classification.
  More informative than AUC when positive class is rare (~0.35% in this dataset).
- **AUC (Area under ROC Curve)**: Measures overall discriminative ability.
- **Precision**: Proportion of predicted positives that are true positives.
- **Recall (Sensitivity)**: Proportion of actual positives correctly identified.
- **F1-Score**: Harmonic mean of Precision and Recall.

### Class Imbalance

The dataset exhibits severe class imbalance (positive rate ~0.35%). This presents
significant challenges for model training and evaluation. Models were trained using
specialized loss functions and techniques to handle this imbalance effectively.

## Conclusions

Among the 4 models evaluated, **deep_residual** achieved the highest
test auPRC of **0.7505**, demonstrating superior performance
for ecDNA prediction on this challenging imbalanced dataset.


## Methods

### Data Splitting

Samples were stratified by positive sample count per patient to ensure balanced
distribution across training, validation, and test sets. The splitting was performed
at the sample level (not gene level) to prevent data leakage.

### Model Training

All models were trained using PyTorch with the following common practices:
- Early stopping based on validation auPRC with patience of 5-35 epochs
- Learning rate scheduling (ReduceLROnPlateau or CosineAnnealingWarmRestarts)
- Gradient clipping for training stability
- Model checkpointing to save best performing weights

---

*Report generated by OTK Model Analyzer*
