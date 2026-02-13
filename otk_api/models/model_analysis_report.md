# Model Performance Analysis Report

**Generated**: 2026-02-13 04:57:31
**Total Models**: 4 trained models

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
| baseline_mlp | 0.001000 | 0.0001 | 512 | 11 | 1 | Yes |
| deep_residual | 0.000100 | 0.0100 | 1024 | 17 | 2 | Yes |
| optimized_residual | 0.001000 | 0.1000 | 1024 | 98 | 63 | Yes |
| transformer | 0.000100 | 0.0001 | 1024 | 19 | 9 | Yes |

## Performance Metrics

### Test Set Performance (Primary Evaluation)

| Model | auPRC | AUC | Precision | Recall | F1-Score |
|-------|-------|-----|-----------|--------|----------|
| transformer | **0.7484** | 0.9746 | 0.9150 | 0.5868 | 0.7150 |
| deep_residual | **0.7384** | 0.9797 | 0.7685 | 0.6862 | 0.7250 |
| baseline_mlp | **0.7208** | 0.9690 | 0.8860 | 0.5860 | 0.7054 |
| optimized_residual | **0.6218** | 0.9729 | 0.8713 | 0.4869 | 0.6247 |

### Complete Performance Comparison

#### Training Set Performance

| Model | auPRC | AUC | Precision | Recall | F1-Score |
|-------|-------|-----|-----------|--------|----------|
| baseline_mlp | 0.6541 | 0.9462 | 0.8463 | 0.4833 | 0.6153 |
| deep_residual | 0.8583 | 0.9978 | 0.8368 | 0.7765 | 0.8055 |
| optimized_residual | 0.5715 | 0.9475 | 0.7058 | 0.5305 | 0.6057 |
| transformer | 0.8849 | 0.9985 | 0.8405 | 0.8254 | 0.8329 |

#### Validation Set Performance

| Model | auPRC | AUC | Precision | Recall | F1-Score |
|-------|-------|-----|-----------|--------|----------|
| baseline_mlp | 0.7949 | 0.9434 | 0.8964 | 0.7195 | 0.7983 |
| deep_residual | 0.7260 | 0.9885 | 0.7060 | 0.7147 | 0.7103 |
| optimized_residual | 0.7684 | 0.9820 | 0.9140 | 0.6736 | 0.7756 |
| transformer | 0.7272 | 0.9832 | 0.7328 | 0.6904 | 0.7110 |

#### Test Set Performance

| Model | auPRC | AUC | Precision | Recall | F1-Score |
|-------|-------|-----|-----------|--------|----------|
| baseline_mlp | 0.7208 | 0.9690 | 0.8860 | 0.5860 | 0.7054 |
| deep_residual | 0.7384 | 0.9797 | 0.7685 | 0.6862 | 0.7250 |
| optimized_residual | 0.6218 | 0.9729 | 0.8713 | 0.4869 | 0.6247 |
| transformer | 0.7484 | 0.9746 | 0.9150 | 0.5868 | 0.7150 |

## Sample-Level Performance (Circular Detection)

Sample-level evaluation determines whether a sample contains circular ecDNA.
A sample is predicted as circular if any gene in the sample is predicted positive.

### Test Set Sample-Level Performance

| Model | auPRC | AUC | Accuracy | Precision | Recall | F1 | Samples |
|-------|-------|-----|----------|-----------|--------|-----|---------|
| transformer | **0.9560** | 0.8568 | 0.6000 | 1.0000 | 0.4615 | 0.6316 | 70 |
| deep_residual | **0.9480** | 0.8344 | 0.5571 | 1.0000 | 0.4038 | 0.5753 | 70 |
| baseline_mlp | **0.9000** | 0.7201 | 0.6000 | 0.7609 | 0.6731 | 0.7143 | 70 |

### Validation Set Sample-Level Performance

| Model | auPRC | AUC | Accuracy | Precision | Recall | F1 | Samples |
|-------|-------|-----|----------|-----------|--------|-----|---------|
| baseline_mlp | 0.9339 | 0.7553 | 0.5652 | 0.8696 | 0.5405 | 0.6667 | 46 |
| deep_residual | 0.9777 | 0.9099 | 0.5217 | 1.0000 | 0.4054 | 0.5769 | 46 |
| transformer | 0.9720 | 0.8889 | 0.5217 | 1.0000 | 0.4054 | 0.5769 | 46 |

### Training Set Sample-Level Performance

| Model | auPRC | AUC | Accuracy | Precision | Recall | F1 | Samples |
|-------|-------|-----|----------|-----------|--------|-----|---------|
| baseline_mlp | 0.8902 | 0.6613 | 0.5778 | 0.8054 | 0.5854 | 0.6780 | 270 |
| deep_residual | 0.9333 | 0.7886 | 0.5148 | 1.0000 | 0.3610 | 0.5305 | 270 |
| transformer | 0.9439 | 0.8146 | 0.5037 | 1.0000 | 0.3463 | 0.5145 | 270 |

### Overfitting Analysis

| Model | Train-Val auPRC Gap | Severity | Precision Gap | Recall Gap |
|-------|---------------------|----------|---------------|------------|
| baseline_mlp | -0.1408 | ✅ low | -0.0501 | -0.2362 |
| deep_residual | 0.1323 | ⚠️ medium | 0.1309 | 0.0619 |
| optimized_residual | -0.1969 | ✅ low | -0.2082 | -0.1431 |
| transformer | 0.1576 | ❌ high | 0.1077 | 0.1350 |

## Best Model Recommendations

| Metric | Best Model | Value |
|--------|------------|-------|
| **Best auPRC** | transformer | 0.7484 |
| **Best AUC** | deep_residual | 0.9797 |
| **Best F1-Score** | deep_residual | 0.7250 |
| **Best Precision** | transformer | 0.9150 |
| **Best Recall** | deep_residual | 0.6862 |
| **Best Generalization** | optimized_residual | Gap: -0.1969 |
| **Best Sample-Level auPRC** | transformer | 0.9560 |

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

### Sample-Level vs Gene-Level Evaluation

- **Gene-Level**: Each gene is evaluated independently for ecDNA presence.
- **Sample-Level**: A sample is predicted as circular if any gene is predicted positive.
  This reflects the clinical question: 'Does this sample contain circular ecDNA?'

### Class Imbalance

The dataset exhibits severe class imbalance (positive rate ~0.35%). This presents
significant challenges for model training and evaluation. Models were trained using
specialized loss functions and techniques to handle this imbalance effectively.

## Conclusions

Among the 4 models evaluated, **transformer** achieved the highest
test auPRC of **0.7484**, demonstrating superior performance
for ecDNA prediction on this challenging imbalanced dataset.

The model achieved a precision of **0.9150**, indicating that
over 80% of predicted positive samples are true positives, which is crucial for
reducing false positives in clinical applications.

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
