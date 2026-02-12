================================================================================
OTK API 模型分析报告
生成时间: 2026-02-12 11:49:59
模型目录: /data/home/wsx/Projects/otk/otk/otk_api/models
================================================================================

## 模型概览

共发现 4 个模型:
  - baseline_mlp (Baseline)
  - deep_residual (PrecisionFocusedEcDNA)
  - optimized_residual (OptimizedEcDNA)
  - transformer (TransformerEcDNA)

## 架构对比

| 模型名称 | 架构类型 | 网络结构 | 损失函数 | 优化器 |
|----------|----------|----------|----------|--------|
| baseline_mlp | Baseline | 57→256→128→64→1 | BCEWithLogitsLoss | Adam |
| deep_residual | PrecisionFocusedEcDNA | 57→512→256→128→64→32→1 | auPRCOptimizedLoss | AdamW |
| optimized_residual | OptimizedEcDNA | 57→128→64→32→16→1 | CombinedLoss | AdamW |
| transformer | TransformerEcDNA | 57→128(embedding)→Attention→64→32→1 | auPRCOptimizedLoss | Adam |

## 训练配置

| 模型名称 | 学习率 | Weight Decay | Batch Size | 训练轮数 | 早停 |
|----------|--------|--------------|------------|----------|------|
| baseline_mlp | 0.001 | 0.0001 | 512 | 16 | 是 |
| deep_residual | 0.0001 | 0.01 | 1024 | 19 | 是 |
| optimized_residual | 0.001 | 0.1 | 1024 | 43 | 是 |
| transformer | 0.0001 | 0.0001 | 1024 | 17 | 是 |

## 性能指标

| 模型名称 | AUC | val_auPRC | test_auPRC | F1 | Precision | Recall |
|----------|-----|-----------|------------|-----|-----------|--------|
| baseline_mlp | 0.9737 | 0.8189 | 0.7328 | 0.6824 | 0.9242 | 0.5408 |
| deep_residual | 0.9813 | 0.7422 | 0.7363 | 0.6888 | 0.7513 | 0.6358 |
| optimized_residual | 0.9842 | 0.7374 | 0.7393 | 0.7186 | 0.8583 | 0.6180 |
| transformer | 0.9795 | 0.7599 | 0.7484 | 0.7264 | 0.7570 | 0.6981 |

## 最佳模型推荐

- **最佳 test_auPRC**: transformer (0.7484)
- **最佳 val_auPRC**: baseline_mlp (0.8189)
- **最佳 AUC**: optimized_residual (0.9842)
- **最佳 F1**: transformer (0.7264)
- **最佳 Precision**: baseline_mlp (0.9242)
- **最佳 Recall**: transformer (0.6981)

## 架构详情

### baseline_mlp

- **类型**: Baseline
- **描述**: 简单MLP网络
- **结构**: 57→256→128→64→1
- **特性**: ReLU激活, Sigmoid输出, 无正则化
- **适用场景**: 基准模型, 高精度低召回场景

### deep_residual

- **类型**: PrecisionFocusedEcDNA
- **描述**: 深度残差网络
- **结构**: 57→512→256→128→64→32→1
- **特性**: 残差连接, LayerNorm, GELU激活, 渐进降维
- **适用场景**: 深度特征学习, 高精度场景

### optimized_residual

- **类型**: OptimizedEcDNA
- **描述**: 优化残差网络
- **结构**: 57→128→64→32→16→1
- **特性**: 残差块, BatchNorm, 组合损失函数
- **适用场景**: 平衡训练, 稳定收敛

### transformer

- **类型**: TransformerEcDNA
- **描述**: Transformer注意力模型
- **结构**: 57→128(embedding)→Attention→64→32→1
- **特性**: 自注意力机制, LayerNorm, GELU激活, Dropout正则化
- **适用场景**: 特征交互学习, 平衡精度召回

================================================================================
报告结束
================================================================================