#!/usr/bin/env python3
"""
Model Analyzer - 自动分析 otk_api/models 下的模型架构和性能

功能:
1. 扫描所有模型目录
2. 解析 config.yml 和 training_summary.yml
3. 生成架构对比和性能汇总报告 (SCI论文级别)
"""

import yaml
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class DatasetStatistics:
    train_samples: int = 0
    val_samples: int = 0
    test_samples: int = 0
    train_positive: int = 0
    val_positive: int = 0
    test_positive: int = 0
    train_positive_rate: float = 0.0
    val_positive_rate: float = 0.0
    test_positive_rate: float = 0.0


@dataclass
class PerformanceMetrics:
    auPRC: float = 0.0
    AUC: float = 0.0
    F1: float = 0.0
    Precision: float = 0.0
    Recall: float = 0.0
    optimal_threshold: float = 0.5


@dataclass
class OverfittingAnalysis:
    train_val_auPRC_gap: float = 0.0
    train_val_precision_gap: float = 0.0
    train_val_recall_gap: float = 0.0
    overfitting_severity: str = "unknown"


@dataclass
class ModelInfo:
    name: str
    model_type: str
    input_dim: int
    architecture_summary: str
    layers: List[Dict] = field(default_factory=list)
    loss_function: str = ""
    optimizer: str = ""
    learning_rate: float = 0.0
    weight_decay: float = 0.0
    batch_size: int = 0
    epochs_trained: int = 0
    best_epoch: int = 0
    early_stopped: bool = False
    training_time: float = 0.0
    config_path: str = ""
    model_path: str = ""
    dataset_stats: DatasetStatistics = field(default_factory=DatasetStatistics)
    train_metrics: PerformanceMetrics = field(default_factory=PerformanceMetrics)
    val_metrics: PerformanceMetrics = field(default_factory=PerformanceMetrics)
    test_metrics: PerformanceMetrics = field(default_factory=PerformanceMetrics)
    overfitting: OverfittingAnalysis = field(default_factory=OverfittingAnalysis)


class ModelAnalyzer:
    MODEL_ARCHITECTURE_INFO = {
        "Baseline": {
            "description": "Simple MLP Network",
            "structure": "57→256→128→64→1",
            "features": ["ReLU activation", "Sigmoid output", "No regularization"],
            "suitable_for": "Baseline model, high precision low recall scenarios"
        },
        "TransformerEcDNA": {
            "description": "Transformer Attention Model",
            "structure": "57→128(embedding)→Attention→64→32→1",
            "features": ["Self-attention mechanism", "LayerNorm", "GELU activation", "Dropout regularization"],
            "suitable_for": "Feature interaction learning, balanced precision-recall"
        },
        "PrecisionFocusedEcDNA": {
            "description": "Deep Residual Network",
            "structure": "57→512→256→128→64→32→1",
            "features": ["Residual connections", "LayerNorm", "GELU activation", "Progressive dimension reduction"],
            "suitable_for": "Deep feature learning, high precision scenarios"
        },
        "AdvancedEcDNA": {
            "description": "CNN-Transformer Hybrid Model",
            "structure": "CNN→Transformer→FPN→Classification Head",
            "features": ["CNN local features", "Transformer global interactions", "Feature Pyramid Network"],
            "suitable_for": "Complex feature learning, multi-scale representation"
        },
        "OptimizedEcDNA": {
            "description": "Optimized Residual Network",
            "structure": "57→128→64→32→16→1",
            "features": ["Residual blocks", "BatchNorm", "Combined loss function"],
            "suitable_for": "Balanced training, stable convergence"
        },
        "EnsembleEcDNA": {
            "description": "Ensemble Model",
            "structure": "Advanced + Precision → Meta classifier",
            "features": ["Multi-model fusion", "Weighted voting", "Meta-learning"],
            "suitable_for": "Best performance, production deployment"
        },
        "EnsembleOptimizedEcDNA": {
            "description": "Optimized Ensemble Model",
            "structure": "3×OptimizedEcDNA → Meta classifier",
            "features": ["Multi-model ensemble", "Meta-classifier fusion"],
            "suitable_for": "Robust prediction, reduced overfitting"
        },
        "ImprovedV2": {
            "description": "Improved V2 Model",
            "structure": "57→256→128→64→1",
            "features": ["BatchNorm", "Dropout", "Residual connections"],
            "suitable_for": "Improved baseline, better generalization"
        },
        "ImprovedV2_Deep": {
            "description": "Deep Improved V2 Model",
            "structure": "57→512→256→128→64→1",
            "features": ["Deeper network", "Residual connections", "BatchNorm"],
            "suitable_for": "Complex pattern learning"
        },
        "EnhancedTransformerEcDNA": {
            "description": "Enhanced Transformer Model",
            "structure": "57→256(embedding)→4-layer Transformer→64→1",
            "features": ["Deep Transformer", "Multi-head attention", "Gradient checkpointing"],
            "suitable_for": "Large-scale feature interaction"
        },
        "LightweightTransformerEcDNA": {
            "description": "Lightweight Transformer Model",
            "structure": "57→64(embedding)→2-layer Transformer→1",
            "features": ["Lightweight design", "Fast inference", "Low memory footprint"],
            "suitable_for": "Fast training, resource-constrained scenarios"
        },
    }

    def __init__(self, models_dir: Optional[Path] = None):
        if models_dir is None:
            self.models_dir = Path(__file__).parent / "models"
        else:
            self.models_dir = Path(models_dir)
        
        self.models: List[ModelInfo] = []
    
    def scan_models(self) -> List[str]:
        """扫描模型目录,返回所有模型名称"""
        model_names = []
        if not self.models_dir.exists():
            print(f"模型目录不存在: {self.models_dir}")
            return model_names
        
        for item in self.models_dir.iterdir():
            if item.is_dir() and not item.name.startswith('.'):
                config_file = item / "config.yml"
                if config_file.exists():
                    model_names.append(item.name)
        
        return sorted(model_names)
    
    def load_yaml(self, file_path: Path) -> Dict[str, Any]:
        """加载YAML文件"""
        if not file_path.exists():
            return {}
        
        with open(file_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f) or {}
    
    def parse_config(self, config: Dict) -> Dict[str, Any]:
        """解析配置文件"""
        result = {}
        
        model_config = config.get('model', {})
        arch = model_config.get('architecture', {})
        
        result['model_type'] = arch.get('type', 'Unknown')
        result['layers'] = arch.get('layers', [])
        result['hidden_dims'] = arch.get('hidden_dims', [])
        result['dropout_rate'] = arch.get('dropout_rate', None)
        
        if result['layers']:
            result['input_dim'] = result['layers'][0].get('input_dim', 57)
        elif arch.get('input_dim'):
            result['input_dim'] = arch.get('input_dim', 57)
        else:
            result['input_dim'] = 57
        
        loss_config = model_config.get('loss_function', {})
        result['loss_function'] = loss_config.get('type', 'Unknown')
        
        opt_config = model_config.get('optimizer', {})
        result['optimizer'] = opt_config.get('type', 'Unknown')
        result['learning_rate'] = opt_config.get('lr', 0.0)
        result['weight_decay'] = opt_config.get('weight_decay', 0.0)
        
        training_config = config.get('training', {})
        result['batch_size'] = training_config.get('batch_size', 0)
        
        return result
    
    def parse_performance_metrics(self, metrics_dict: Dict) -> PerformanceMetrics:
        """解析性能指标"""
        return PerformanceMetrics(
            auPRC=metrics_dict.get('auPRC', 0.0),
            AUC=metrics_dict.get('AUC', 0.0),
            F1=metrics_dict.get('F1', 0.0),
            Precision=metrics_dict.get('Precision', 0.0),
            Recall=metrics_dict.get('Recall', 0.0),
            optimal_threshold=metrics_dict.get('optimal_threshold', 0.5)
        )
    
    def parse_dataset_statistics(self, stats_dict: Dict) -> DatasetStatistics:
        """解析数据集统计信息"""
        return DatasetStatistics(
            train_samples=stats_dict.get('train_samples', 0),
            val_samples=stats_dict.get('val_samples', 0),
            test_samples=stats_dict.get('test_samples', 0),
            train_positive=stats_dict.get('train_positive', 0),
            val_positive=stats_dict.get('val_positive', 0),
            test_positive=stats_dict.get('test_positive', 0),
            train_positive_rate=stats_dict.get('train_positive_rate', 0.0),
            val_positive_rate=stats_dict.get('val_positive_rate', 0.0),
            test_positive_rate=stats_dict.get('test_positive_rate', 0.0)
        )
    
    def parse_overfitting_analysis(self, overfitting_dict: Dict) -> OverfittingAnalysis:
        """解析过拟合分析"""
        return OverfittingAnalysis(
            train_val_auPRC_gap=overfitting_dict.get('train_val_auPRC_gap', 0.0),
            train_val_precision_gap=overfitting_dict.get('train_val_precision_gap', 0.0),
            train_val_recall_gap=overfitting_dict.get('train_val_recall_gap', 0.0),
            overfitting_severity=overfitting_dict.get('overfitting_severity', 'unknown')
        )
    
    def parse_training_summary(self, summary: Dict) -> Dict[str, Any]:
        """解析训练摘要 - 支持新旧格式"""
        result = {}
        
        # 新格式: performance包含training_set, validation_set, test_set
        performance = summary.get('performance', {})
        if performance:
            result['train_metrics'] = self.parse_performance_metrics(
                performance.get('training_set', {})
            )
            result['val_metrics'] = self.parse_performance_metrics(
                performance.get('validation_set', {})
            )
            result['test_metrics'] = self.parse_performance_metrics(
                performance.get('test_set', {})
            )
        else:
            # 旧格式兼容
            test_metrics = summary.get('test_metrics', {})
            result['test_metrics'] = self.parse_performance_metrics(test_metrics)
            result['train_metrics'] = PerformanceMetrics()
            result['val_metrics'] = PerformanceMetrics()
        
        # 数据集统计
        dataset_stats = summary.get('dataset_statistics', {})
        if dataset_stats:
            result['dataset_stats'] = self.parse_dataset_statistics(dataset_stats)
        else:
            result['dataset_stats'] = DatasetStatistics()
        
        # 过拟合分析
        overfitting = summary.get('overfitting_analysis', {})
        if overfitting:
            result['overfitting'] = self.parse_overfitting_analysis(overfitting)
        else:
            result['overfitting'] = OverfittingAnalysis()
        
        # 训练进度
        training_progress = summary.get('training_progress', {})
        result['best_val_auPRC'] = summary.get('best_val_auPRC', 
            result['val_metrics'].auPRC if result['val_metrics'].auPRC > 0 else 0.0)
        result['epochs_trained'] = training_progress.get('epochs_trained', 
            summary.get('epochs_trained', 0))
        result['best_epoch'] = training_progress.get('best_epoch', 0)
        result['early_stopped'] = training_progress.get('early_stopped', 
            summary.get('early_stopped', False))
        result['training_time'] = training_progress.get('total_training_time_seconds', 
            summary.get('total_training_time', 0.0))
        
        return result
    
    def analyze_model(self, model_name: str) -> Optional[ModelInfo]:
        """分析单个模型"""
        model_dir = self.models_dir / model_name
        config_path = model_dir / "config.yml"
        summary_path = model_dir / "training_summary.yml"
        model_path = model_dir / "best_model.pth"
        
        if not config_path.exists():
            print(f"跳过 {model_name}: 缺少 config.yml")
            return None
        
        config = self.load_yaml(config_path)
        config_info = self.parse_config(config)
        
        summary = self.load_yaml(summary_path)
        summary_info = self.parse_training_summary(summary)
        
        model_type = config_info['model_type']
        arch_info = self.MODEL_ARCHITECTURE_INFO.get(model_type, {
            "description": "Unknown architecture",
            "structure": "N/A",
            "features": [],
            "suitable_for": "N/A"
        })
        
        return ModelInfo(
            name=model_name,
            model_type=model_type,
            input_dim=config_info['input_dim'],
            architecture_summary=arch_info['structure'],
            layers=config_info['layers'],
            loss_function=config_info['loss_function'],
            optimizer=config_info['optimizer'],
            learning_rate=config_info['learning_rate'],
            weight_decay=config_info['weight_decay'],
            batch_size=config_info['batch_size'],
            epochs_trained=summary_info['epochs_trained'],
            best_epoch=summary_info['best_epoch'],
            early_stopped=summary_info['early_stopped'],
            training_time=summary_info['training_time'],
            config_path=str(config_path),
            model_path=str(model_path) if model_path.exists() else "",
            dataset_stats=summary_info['dataset_stats'],
            train_metrics=summary_info['train_metrics'],
            val_metrics=summary_info['val_metrics'],
            test_metrics=summary_info['test_metrics'],
            overfitting=summary_info['overfitting'],
        )
    
    def analyze_all(self) -> List[ModelInfo]:
        """分析所有模型"""
        self.models = []
        model_names = self.scan_models()
        
        for name in model_names:
            model_info = self.analyze_model(name)
            if model_info:
                self.models.append(model_info)
        
        return self.models
    
    def generate_sci_report(self) -> str:
        """生成SCI论文级别的分析报告"""
        if not self.models:
            self.analyze_all()
        
        lines = []
        
        # 标题
        lines.append("# Model Performance Analysis Report")
        lines.append("")
        lines.append(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"**Total Models**: {len(self.models)}")
        lines.append("")
        
        # 摘要
        lines.append("## Abstract")
        lines.append("")
        lines.append("This report presents a comprehensive analysis of multiple deep learning models")
        lines.append("developed for extrachromosomal DNA (ecDNA) prediction. The models were trained on")
        lines.append("a large-scale dataset with severe class imbalance and evaluated using multiple")
        lines.append("performance metrics including auPRC, AUC, Precision, Recall, and F1-score.")
        lines.append("")
        
        # 数据集描述
        lines.append("## Dataset Description")
        lines.append("")
        if self.models and self.models[0].dataset_stats.train_samples > 0:
            stats = self.models[0].dataset_stats
            lines.append("### Sample Distribution")
            lines.append("")
            lines.append("| Dataset | Total Samples | Positive Samples | Positive Rate |")
            lines.append("|---------|---------------|------------------|---------------|")
            lines.append(f"| Training | {stats.train_samples:,} | {stats.train_positive:,} | {stats.train_positive_rate:.4%} |")
            lines.append(f"| Validation | {stats.val_samples:,} | {stats.val_positive:,} | {stats.val_positive_rate:.4%} |")
            lines.append(f"| Test | {stats.test_samples:,} | {stats.test_positive:,} | {stats.test_positive_rate:.4%} |")
            lines.append("")
            total_samples = stats.train_samples + stats.val_samples + stats.test_samples
            total_positive = stats.train_positive + stats.val_positive + stats.test_positive
            lines.append(f"**Total**: {total_samples:,} samples, {total_positive:,} positive ({total_positive/total_samples:.4%})")
            lines.append("")
            lines.append("**Note**: The dataset exhibits severe class imbalance with only ~0.35% positive samples,")
            lines.append("which presents significant challenges for model training and evaluation.")
            lines.append("")
        
        # 模型架构对比
        lines.append("## Model Architecture Comparison")
        lines.append("")
        lines.append("### Overview")
        lines.append("")
        lines.append("| Model | Architecture | Network Structure | Loss Function | Optimizer |")
        lines.append("|-------|--------------|-------------------|---------------|-----------|")
        for m in self.models:
            lines.append(f"| {m.name} | {m.model_type} | {m.architecture_summary} | {m.loss_function} | {m.optimizer} |")
        lines.append("")
        
        # 训练配置
        lines.append("### Training Configuration")
        lines.append("")
        lines.append("| Model | Learning Rate | Weight Decay | Batch Size | Epochs | Best Epoch | Early Stop |")
        lines.append("|-------|---------------|--------------|------------|--------|------------|------------|")
        for m in self.models:
            early_stop_str = "Yes" if m.early_stopped else "No"
            lines.append(f"| {m.name} | {m.learning_rate:.6f} | {m.weight_decay:.4f} | {m.batch_size} | {m.epochs_trained} | {m.best_epoch} | {early_stop_str} |")
        lines.append("")
        
        # 性能指标
        lines.append("## Performance Metrics")
        lines.append("")
        lines.append("### Test Set Performance (Primary Evaluation)")
        lines.append("")
        lines.append("| Model | auPRC | AUC | Precision | Recall | F1-Score |")
        lines.append("|-------|-------|-----|-----------|--------|----------|")
        for m in sorted(self.models, key=lambda x: x.test_metrics.auPRC, reverse=True):
            tm = m.test_metrics
            lines.append(f"| {m.name} | **{tm.auPRC:.4f}** | {tm.AUC:.4f} | {tm.Precision:.4f} | {tm.Recall:.4f} | {tm.F1:.4f} |")
        lines.append("")
        
        # 完整性能对比表 (Train/Val/Test)
        lines.append("### Complete Performance Comparison")
        lines.append("")
        lines.append("#### Training Set Performance")
        lines.append("")
        lines.append("| Model | auPRC | AUC | Precision | Recall | F1-Score |")
        lines.append("|-------|-------|-----|-----------|--------|----------|")
        for m in self.models:
            tm = m.train_metrics
            if tm.auPRC > 0:
                lines.append(f"| {m.name} | {tm.auPRC:.4f} | {tm.AUC:.4f} | {tm.Precision:.4f} | {tm.Recall:.4f} | {tm.F1:.4f} |")
        lines.append("")
        
        lines.append("#### Validation Set Performance")
        lines.append("")
        lines.append("| Model | auPRC | AUC | Precision | Recall | F1-Score |")
        lines.append("|-------|-------|-----|-----------|--------|----------|")
        for m in self.models:
            vm = m.val_metrics
            if vm.auPRC > 0:
                lines.append(f"| {m.name} | {vm.auPRC:.4f} | {vm.AUC:.4f} | {vm.Precision:.4f} | {vm.Recall:.4f} | {vm.F1:.4f} |")
        lines.append("")
        
        lines.append("#### Test Set Performance")
        lines.append("")
        lines.append("| Model | auPRC | AUC | Precision | Recall | F1-Score |")
        lines.append("|-------|-------|-----|-----------|--------|----------|")
        for m in self.models:
            tm = m.test_metrics
            lines.append(f"| {m.name} | {tm.auPRC:.4f} | {tm.AUC:.4f} | {tm.Precision:.4f} | {tm.Recall:.4f} | {tm.F1:.4f} |")
        lines.append("")
        
        # 过拟合分析
        lines.append("### Overfitting Analysis")
        lines.append("")
        lines.append("| Model | Train-Val auPRC Gap | Severity | Precision Gap | Recall Gap |")
        lines.append("|-------|---------------------|----------|---------------|------------|")
        for m in self.models:
            of = m.overfitting
            severity_emoji = {"low": "✅", "medium": "⚠️", "high": "❌"}.get(of.overfitting_severity, "❓")
            lines.append(f"| {m.name} | {of.train_val_auPRC_gap:.4f} | {severity_emoji} {of.overfitting_severity} | {of.train_val_precision_gap:.4f} | {of.train_val_recall_gap:.4f} |")
        lines.append("")
        
        # 最佳模型推荐
        lines.append("## Best Model Recommendations")
        lines.append("")
        
        if self.models:
            best_auprc = max(self.models, key=lambda m: m.test_metrics.auPRC)
            best_auc = max(self.models, key=lambda m: m.test_metrics.AUC)
            best_f1 = max(self.models, key=lambda m: m.test_metrics.F1)
            best_precision = max(self.models, key=lambda m: m.test_metrics.Precision)
            best_recall = max(self.models, key=lambda m: m.test_metrics.Recall)
            best_generalization = min(self.models, key=lambda m: m.overfitting.train_val_auPRC_gap)
            
            lines.append("| Metric | Best Model | Value |")
            lines.append("|--------|------------|-------|")
            lines.append(f"| **Best auPRC** | {best_auprc.name} | {best_auprc.test_metrics.auPRC:.4f} |")
            lines.append(f"| **Best AUC** | {best_auc.name} | {best_auc.test_metrics.AUC:.4f} |")
            lines.append(f"| **Best F1-Score** | {best_f1.name} | {best_f1.test_metrics.F1:.4f} |")
            lines.append(f"| **Best Precision** | {best_precision.name} | {best_precision.test_metrics.Precision:.4f} |")
            lines.append(f"| **Best Recall** | {best_recall.name} | {best_recall.test_metrics.Recall:.4f} |")
            lines.append(f"| **Best Generalization** | {best_generalization.name} | Gap: {best_generalization.overfitting.train_val_auPRC_gap:.4f} |")
        lines.append("")
        
        # 架构详情
        lines.append("## Architecture Details")
        lines.append("")
        for m in self.models:
            arch_info = self.MODEL_ARCHITECTURE_INFO.get(m.model_type, {})
            lines.append(f"### {m.name}")
            lines.append("")
            features_str = ", ".join(arch_info.get('features', []))
            lines.append(f"- **Type**: `{m.model_type}`")
            lines.append("")
            lines.append(f"- **Description**: {arch_info.get('description', 'N/A')}")
            lines.append("")
            lines.append(f"- **Structure**: `{m.architecture_summary}`")
            lines.append("")
            lines.append(f"- **Key Features**: {features_str}")
            lines.append("")
            lines.append(f"- **Suitable For**: {arch_info.get('suitable_for', 'N/A')}")
            lines.append("")
            lines.append(f"- **Loss Function**: `{m.loss_function}`")
            lines.append("")
            lines.append(f"- **Optimizer**: `{m.optimizer}` (lr={m.learning_rate}, weight_decay={m.weight_decay})")
            lines.append("")
        
        # 统计显著性说明
        lines.append("## Statistical Considerations")
        lines.append("")
        lines.append("### Evaluation Metrics")
        lines.append("")
        lines.append("- **auPRC (Area under Precision-Recall Curve)**: Primary metric for imbalanced classification.")
        lines.append("  More informative than AUC when positive class is rare (~0.35% in this dataset).")
        lines.append("")
        lines.append("- **AUC (Area under ROC Curve)**: Measures overall discriminative ability.")
        lines.append("")
        lines.append("- **Precision**: Proportion of predicted positives that are true positives.")
        lines.append("")
        lines.append("- **Recall (Sensitivity)**: Proportion of actual positives correctly identified.")
        lines.append("")
        lines.append("- **F1-Score**: Harmonic mean of Precision and Recall.")
        lines.append("")
        lines.append("### Class Imbalance")
        lines.append("")
        lines.append("The dataset exhibits severe class imbalance (positive rate ~0.35%). This presents")
        lines.append("significant challenges for model training and evaluation. Models were trained using")
        lines.append("specialized loss functions and techniques to handle this imbalance effectively.")
        lines.append("")
        
        # 结论
        lines.append("## Conclusions")
        lines.append("")
        if self.models:
            best = max(self.models, key=lambda m: m.test_metrics.auPRC)
            lines.append(f"Among the {len(self.models)} models evaluated, **{best.name}** achieved the highest")
            lines.append(f"test auPRC of **{best.test_metrics.auPRC:.4f}**, demonstrating superior performance")
            lines.append("for ecDNA prediction on this challenging imbalanced dataset.")
            lines.append("")
            if best.test_metrics.Precision >= 0.8:
                lines.append(f"The model achieved a precision of **{best.test_metrics.Precision:.4f}**, indicating that")
                lines.append("over 80% of predicted positive samples are true positives, which is crucial for")
                lines.append("reducing false positives in clinical applications.")
        lines.append("")
        
        # 方法论
        lines.append("## Methods")
        lines.append("")
        lines.append("### Data Splitting")
        lines.append("")
        lines.append("Samples were stratified by positive sample count per patient to ensure balanced")
        lines.append("distribution across training, validation, and test sets. The splitting was performed")
        lines.append("at the sample level (not gene level) to prevent data leakage.")
        lines.append("")
        lines.append("### Model Training")
        lines.append("")
        lines.append("All models were trained using PyTorch with the following common practices:")
        lines.append("")
        lines.append("- Early stopping based on validation auPRC with patience of 5-35 epochs")
        lines.append("- Learning rate scheduling (ReduceLROnPlateau or CosineAnnealingWarmRestarts)")
        lines.append("- Gradient clipping for training stability")
        lines.append("- Model checkpointing to save best performing weights")
        lines.append("")
        
        lines.append("---")
        lines.append("")
        lines.append("*Report generated by OTK Model Analyzer*")
        lines.append("")
        
        return "\n".join(lines)
    
    def generate_report(self) -> str:
        """生成分析报告 (兼容旧接口)"""
        return self.generate_sci_report()
    
    def print_comparison_table(self):
        """打印对比表格"""
        if not self.models:
            self.analyze_all()
        
        print("\n" + "=" * 120)
        print("Model Performance Comparison Table")
        print("=" * 120)
        
        header = f"{'Model':<20} {'Type':<25} {'auPRC':>8} {'AUC':>8} {'Prec':>8} {'Recall':>8} {'F1':>8}"
        print(header)
        print("-" * 120)
        
        for m in sorted(self.models, key=lambda x: x.test_metrics.auPRC, reverse=True):
            tm = m.test_metrics
            row = f"{m.name:<20} {m.model_type:<25} {tm.auPRC:>8.4f} {tm.AUC:>8.4f} {tm.Precision:>8.4f} {tm.Recall:>8.4f} {tm.F1:>8.4f}"
            print(row)
        
        print("=" * 120 + "\n")


def main():
    """主函数"""
    analyzer = ModelAnalyzer()
    
    print("\nAnalyzing models...")
    models = analyzer.analyze_all()
    
    if not models:
        print("No models found!")
        return
    
    print(f"\nFound {len(models)} models\n")
    
    analyzer.print_comparison_table()
    
    report = analyzer.generate_sci_report()
    
    report_path = analyzer.models_dir / "model_analysis_report.md"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"\nReport saved to: {report_path}")


if __name__ == "__main__":
    main()
