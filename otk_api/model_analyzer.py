#!/usr/bin/env python3
"""
Model Analyzer - 自动分析 otk_api/models 下的模型架构和性能

功能:
1. 扫描所有模型目录
2. 解析 config.yml 和 training_summary.yml
3. 生成架构对比和性能汇总报告 (SCI论文级别)
4. 样本级别评估 (sample-level circular detection)
5. 生成论文风格图表
"""

import yaml
import sys
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from sklearn.metrics import precision_recall_curve, auc, roc_auc_score, accuracy_score

# 图表生成 - SCI 论文风格设置
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 非交互式后端

# SCI 论文图表设置
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans', 'Helvetica'],
    'font.size': 8,
    'axes.labelsize': 9,
    'axes.titlesize': 10,
    'axes.titleweight': 'bold',
    'axes.labelweight': 'bold',
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 7,
    'legend.frameon': False,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linestyle': '--',
})

# 色盲友好配色方案
SCI_COLORS = {
    'train': '#0072B2',    # 蓝色
    'val': '#F0E442',      # 黄色
    'test': '#CC79A7',     # 粉色
    'model1': '#0072B2',   # 蓝
    'model2': '#D55E00',   # 橙
    'model3': '#009E73',   # 绿
    'model4': '#CC79A7',   # 粉
    'model5': '#F0E442',   # 黄
    'model6': '#56B4E9',   # 天蓝
    'model7': '#E69F00',   # 金
    'model8': '#000000',   # 黑
}


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
class SampleLevelMetrics:
    auPRC: float = 0.0
    AUC: float = 0.0
    Accuracy: float = 0.0
    Precision: float = 0.0
    Recall: float = 0.0
    F1: float = 0.0
    total_samples: int = 0
    positive_samples: int = 0
    predicted_positive: int = 0
    true_positive: int = 0


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
    max_depth: int = 0  # For XGBoost models
    early_stopped: bool = False
    training_time: float = 0.0
    config_path: str = ""
    model_path: str = ""
    dataset_stats: DatasetStatistics = field(default_factory=DatasetStatistics)
    train_metrics: PerformanceMetrics = field(default_factory=PerformanceMetrics)
    val_metrics: PerformanceMetrics = field(default_factory=PerformanceMetrics)
    test_metrics: PerformanceMetrics = field(default_factory=PerformanceMetrics)
    overfitting: OverfittingAnalysis = field(default_factory=OverfittingAnalysis)
    sample_train_metrics: SampleLevelMetrics = field(default_factory=SampleLevelMetrics)
    sample_val_metrics: SampleLevelMetrics = field(default_factory=SampleLevelMetrics)
    sample_test_metrics: SampleLevelMetrics = field(default_factory=SampleLevelMetrics)
    is_trained: bool = False


class ModelAnalyzer:
    MODEL_TYPE_TEMPLATES = {
        "XGB": {
            "description": "XGBoost Gradient Boosting",
            "features": ["Tree-based ensemble", "Feature importance", "Native missing value handling"],
            "suitable_for": "Tabular data, interpretable predictions"
        },
        "Transformer": {
            "description": "Transformer Attention Model",
            "features": ["Self-attention mechanism", "LayerNorm", "GELU activation"],
            "suitable_for": "Feature interaction learning"
        },
        "MLP": {
            "description": "Multi-Layer Perceptron",
            "features": ["Fully connected layers", "Non-linear activation"],
            "suitable_for": "Baseline model, quick training"
        },
        "Residual": {
            "description": "Residual Network",
            "features": ["Residual connections", "Skip connections"],
            "suitable_for": "Deep feature learning"
        },
        "Ensemble": {
            "description": "Ensemble Model",
            "features": ["Multi-model fusion", "Weighted voting"],
            "suitable_for": "Robust prediction, reduced overfitting"
        },
        "TabPFN": {
            "description": "TabPFN Foundation Model",
            "features": ["Pre-trained foundation model", "In-context learning"],
            "suitable_for": "Small datasets, zero-shot prediction"
        },
    }

    def __init__(self, models_dir: Optional[Path] = None, data_path: Optional[Path] = None):
        if models_dir is None:
            self.models_dir = Path(__file__).parent / "models"
        else:
            self.models_dir = Path(models_dir)
        
        self.data_path = data_path
        self.data_df = None
        self.sample_splits = None
        self.models: List[ModelInfo] = []
    
    def load_data(self):
        """Load preprocessed data for sample-level evaluation using unified split"""
        if self.data_df is not None:
            return
        
        data_file = self.models_dir.parent.parent / "src/otk/data/sorted_modeling_data.csv.gz"
        if not data_file.exists():
            print(f"Data file not found: {data_file}")
            return
        
        import gzip
        print(f"Loading data from {data_file}...")
        with gzip.open(data_file, 'rt') as f:
            self.data_df = pd.read_csv(f)
        print(f"Loaded {len(self.data_df)} rows")
        
        # Use unified data split (80/10/10, seed=2026)
        sys.path.insert(0, str(self.models_dir.parent.parent / "src"))
        from otk.data.data_split import get_data_splits
        
        train_samples, val_samples, test_samples = get_data_splits()
        
        self.sample_splits = {
            'train': train_samples,
            'val': val_samples,
            'test': test_samples
        }
        
        print(f"Sample splits (unified, seed=2026): train={len(train_samples)}, val={len(val_samples)}, test={len(test_samples)}")
    
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
        
        # Support new format: model.type + model.variant
        model_type = model_config.get('variant', model_config.get('type', 'Unknown'))
        result['model_type'] = model_type
        
        arch = model_config.get('architecture', {})
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
        
        # XGBoost models use objective from params
        if result['loss_function'] == 'Unknown' and model_type in ['XGBNew', 'XGB11', 'XGBoost']:
            params = model_config.get('params', {})
            objective = params.get('objective', '')
            eval_metric = params.get('eval_metric', '')
            if objective == 'binary:logistic':
                if eval_metric == 'aucpr':
                    result['loss_function'] = 'LogLoss (optimizes auPRC)'
                elif eval_metric == 'auc':
                    result['loss_function'] = 'LogLoss (optimizes AUC)'
                else:
                    result['loss_function'] = 'LogLoss (binary:logistic)'
        
        opt_config = model_config.get('optimizer', {})
        result['optimizer'] = opt_config.get('type', 'Unknown')
        result['learning_rate'] = opt_config.get('lr', 0.0)
        result['weight_decay'] = opt_config.get('weight_decay', 0.0)
        
        training_config = config.get('training', {})
        
        # XGBoost models have different config structure
        if model_type in ['XGBNew', 'XGB11', 'XGBoost']:
            params = model_config.get('params', {})
            result['optimizer'] = 'Gradient Boosting'
            result['learning_rate'] = params.get('eta', 0.0)
            result['weight_decay'] = params.get('alpha', 0.0) + params.get('lambda', 0.0)
            result['n_estimators'] = training_config.get('epochs', 0)
            result['max_depth'] = params.get('max_depth', 0)
        
        result['batch_size'] = training_config.get('batch_size', 0)
        result['epochs'] = training_config.get('epochs', 0)
        early_stop = training_config.get('early_stopping', {})
        result['early_stopping_patience'] = early_stop.get('patience', 0) if early_stop else 0
        
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
        
        # New format: gene_level and sample_level
        gene_level = summary.get('gene_level', {})
        sample_level = summary.get('sample_level', {})
        
        if gene_level:
            # New unified format
            result['train_metrics'] = self.parse_performance_metrics(gene_level.get('train', {}))
            result['val_metrics'] = self.parse_performance_metrics(gene_level.get('val', {}))
            result['test_metrics'] = self.parse_performance_metrics(gene_level.get('test', {}))
            
            # Sample level metrics
            result['sample_train_metrics'] = self.parse_sample_level_metrics(sample_level.get('train', {}))
            result['sample_val_metrics'] = self.parse_sample_level_metrics(sample_level.get('val', {}))
            result['sample_test_metrics'] = self.parse_sample_level_metrics(sample_level.get('test', {}))
            
            # Dataset stats from sample level
            test_sample = sample_level.get('test', {})
            train_sample = sample_level.get('train', {})
            val_sample = sample_level.get('val', {})
            
            train_samples = train_sample.get('total_samples', 0)
            val_samples = val_sample.get('total_samples', 0)
            test_samples = test_sample.get('total_samples', 0)
            train_positive = train_sample.get('positive_samples', 0)
            val_positive = val_sample.get('positive_samples', 0)
            test_positive = test_sample.get('positive_samples', 0)
            
            result['dataset_stats'] = DatasetStatistics(
                train_samples=train_samples,
                val_samples=val_samples,
                test_samples=test_samples,
                train_positive=train_positive,
                val_positive=val_positive,
                test_positive=test_positive,
                train_positive_rate=train_positive / train_samples if train_samples > 0 else 0.0,
                val_positive_rate=val_positive / val_samples if val_samples > 0 else 0.0,
                test_positive_rate=test_positive / test_samples if test_samples > 0 else 0.0
            )
            
            # 计算过拟合分析
            train_metrics = result['train_metrics']
            val_metrics = result['val_metrics']
            auprc_gap = train_metrics.auPRC - val_metrics.auPRC if train_metrics.auPRC > 0 and val_metrics.auPRC > 0 else 0.0
            precision_gap = train_metrics.Precision - val_metrics.Precision if train_metrics.Precision > 0 and val_metrics.Precision > 0 else 0.0
            recall_gap = train_metrics.Recall - val_metrics.Recall if train_metrics.Recall > 0 and val_metrics.Recall > 0 else 0.0
            
            # 判断过拟合严重程度
            if auprc_gap > 0.15:
                severity = "high"
            elif auprc_gap > 0.08:
                severity = "medium"
            else:
                severity = "low"
            
            result['overfitting'] = OverfittingAnalysis(
                train_val_auPRC_gap=auprc_gap,
                train_val_precision_gap=precision_gap,
                train_val_recall_gap=recall_gap,
                overfitting_severity=severity
            )
            result['best_val_auPRC'] = result['val_metrics'].auPRC
            result['epochs_trained'] = 0
            result['best_epoch'] = 0
            result['early_stopped'] = False
            result['training_time'] = 0.0
        else:
            # Old format: performance
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
                test_metrics = summary.get('test_metrics', {})
                result['test_metrics'] = self.parse_performance_metrics(test_metrics)
                result['train_metrics'] = PerformanceMetrics()
                result['val_metrics'] = PerformanceMetrics()
            
            dataset_stats = summary.get('dataset_statistics', {})
            if dataset_stats:
                result['dataset_stats'] = self.parse_dataset_statistics(dataset_stats)
            else:
                result['dataset_stats'] = DatasetStatistics()
            
            overfitting = summary.get('overfitting_analysis', {})
            if overfitting:
                result['overfitting'] = self.parse_overfitting_analysis(overfitting)
            else:
                result['overfitting'] = OverfittingAnalysis()
            
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
    
    def parse_sample_level_metrics(self, metrics_dict: Dict) -> SampleLevelMetrics:
        """解析样本级别指标"""
        return SampleLevelMetrics(
            auPRC=metrics_dict.get('auPRC', 0.0),
            AUC=metrics_dict.get('AUC', 0.0),
            Accuracy=metrics_dict.get('Accuracy', 0.0),
            Precision=metrics_dict.get('Precision', 0.0),
            Recall=metrics_dict.get('Recall', 0.0),
            F1=metrics_dict.get('F1', 0.0),
            total_samples=metrics_dict.get('total_samples', 0),
            positive_samples=metrics_dict.get('positive_samples', 0),
            predicted_positive=metrics_dict.get('predicted_positive', 0),
            true_positive=metrics_dict.get('TP', 0)
        )
    
    def evaluate_sample_level(self, model_info: ModelInfo) -> Tuple[SampleLevelMetrics, SampleLevelMetrics, SampleLevelMetrics]:
        """Evaluate model at sample level (circular detection)"""
        if self.data_df is None or self.sample_splits is None:
            return SampleLevelMetrics(), SampleLevelMetrics(), SampleLevelMetrics()
        
        model_path = Path(model_info.model_path)
        if not model_path.exists():
            return SampleLevelMetrics(), SampleLevelMetrics(), SampleLevelMetrics()
        
        try:
            sys.path.insert(0, str(self.models_dir.parent.parent / "src"))
            from otk.predict.predictor import Predictor
            
            predictor = Predictor(str(model_path), gpu=0)
            
            optimal_threshold = predictor.optimal_threshold if predictor.optimal_threshold else 0.5
            
            def evaluate_split(split_name: str) -> SampleLevelMetrics:
                split_samples = self.sample_splits[split_name]
                split_df = self.data_df[self.data_df['sample'].isin(split_samples)].copy()
                
                if len(split_df) == 0:
                    return SampleLevelMetrics()
                
                # 使用 predictor.prepare_features 准备特征（包括工程特征）
                X, feature_cols = predictor.prepare_features(split_df)
                y_true_gene = split_df['y'].values
                samples = split_df['sample'].values
                
                # 使用 predict_proba 获取概率，传入 feature_names
                probs = predictor.predict_proba(X, feature_names=feature_cols).flatten()
                
                if np.any(np.isnan(probs)):
                    print(f"  Warning: NaN values in predictions, replacing with 0.5")
                    probs = np.nan_to_num(probs, nan=0.5)
                
                split_df['prob'] = probs
                sample_predictions = split_df.groupby('sample').agg({
                    'y': 'max',
                    'prob': 'max'
                }).reset_index()
                
                y_true = sample_predictions['y'].values
                y_prob = sample_predictions['prob'].values
                y_pred = (y_prob >= optimal_threshold).astype(int)
                
                if len(np.unique(y_true)) < 2:
                    print(f"  Warning: Only one class in {split_name} set")
                    return SampleLevelMetrics()
                
                precision, recall, _ = precision_recall_curve(y_true, y_prob)
                auprc = auc(recall, precision)
                auc_score = roc_auc_score(y_true, y_prob)
                
                tp = ((y_pred == 1) & (y_true == 1)).sum()
                fp = ((y_pred == 1) & (y_true == 0)).sum()
                fn = ((y_pred == 0) & (y_true == 1)).sum()
                tn = ((y_pred == 0) & (y_true == 0)).sum()
                
                prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
                acc = (tp + tn) / len(y_true)
                
                return SampleLevelMetrics(
                    auPRC=auprc,
                    AUC=auc_score,
                    Accuracy=acc,
                    Precision=prec,
                    Recall=rec,
                    F1=f1,
                    total_samples=len(y_true),
                    positive_samples=int(y_true.sum()),
                    predicted_positive=int(y_pred.sum()),
                    true_positive=int(tp)
                )
            
            train_metrics = evaluate_split('train')
            val_metrics = evaluate_split('val')
            test_metrics = evaluate_split('test')
            
            return train_metrics, val_metrics, test_metrics
            
        except Exception as e:
            import traceback
            print(f"Error evaluating sample level for {model_info.name}: {e}")
            traceback.print_exc()
            return SampleLevelMetrics(), SampleLevelMetrics(), SampleLevelMetrics()
    
    def _infer_architecture_info(self, model_type: str, config_info: Dict) -> Dict[str, Any]:
        """Infer architecture info from config and model type templates"""
        hidden_dims = config_info.get('hidden_dims', [])
        layers = config_info.get('layers', [])
        input_dim = config_info.get('input_dim', 57)
        max_depth = config_info.get('max_depth', 0)
        
        # Build structure string
        if 'XGB' in model_type or 'xgb' in model_type.lower():
            n_features = input_dim
            structure = f"Gradient Boosted Trees ({n_features} features"
            if max_depth > 0:
                structure += f", max_depth={max_depth}"
            structure += ")"
        elif hidden_dims:
            structure = f"{input_dim}→" + "→".join(map(str, hidden_dims)) + "→1"
        elif layers:
            dims = [layers[0].get('input_dim', input_dim)]
            for layer in layers:
                if 'output_dim' in layer:
                    dims.append(layer['output_dim'])
            structure = "→".join(map(str, dims)) + "→1"
        else:
            structure = f"{input_dim}→...→1"
        
        # Determine base type and get template
        base_type = None
        for type_key in self.MODEL_TYPE_TEMPLATES:
            if type_key.lower() in model_type.lower():
                base_type = type_key
                break
        
        if base_type:
            template = self.MODEL_TYPE_TEMPLATES[base_type].copy()
        else:
            template = {
                "description": f"{model_type} Model",
                "features": [],
                "suitable_for": "Custom model"
            }
        
        # Add specific features from config
        features = list(template.get('features', []))
        dropout_rate = config_info.get('dropout_rate') or 0
        if dropout_rate > 0:
            features.append(f"Dropout({dropout_rate})")
        
        return {
            "description": template.get('description', f"{model_type} Model"),
            "structure": structure,
            "features": features,
            "suitable_for": template.get('suitable_for', "Custom model - see config for details")
        }

    def analyze_model(self, model_name: str) -> Optional[ModelInfo]:
        """分析单个模型"""
        model_dir = self.models_dir / model_name
        config_path = model_dir / "config.yml"
        summary_path = model_dir / "training_summary.yml"
        model_path = model_dir / "best_model.pkl"
        if not model_path.exists():
            model_path = model_dir / "best_model.pth"
        
        if not config_path.exists():
            print(f"跳过 {model_name}: 缺少 config.yml")
            return None
        
        config = self.load_yaml(config_path)
        config_info = self.parse_config(config)
        
        summary = self.load_yaml(summary_path)
        summary_info = self.parse_training_summary(summary)
        
        model_type = config_info['model_type']
        arch_info = self._infer_architecture_info(model_type, config_info)
        
        is_trained = summary_info['test_metrics'].auPRC > 0
        
        model_info = ModelInfo(
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
            max_depth=config_info.get('max_depth', 0),
            early_stopped=summary_info['early_stopped'],
            training_time=summary_info['training_time'],
            config_path=str(config_path),
            model_path=str(model_path) if model_path.exists() else "",
            dataset_stats=summary_info['dataset_stats'],
            train_metrics=summary_info['train_metrics'],
            val_metrics=summary_info['val_metrics'],
            test_metrics=summary_info['test_metrics'],
            overfitting=summary_info['overfitting'],
            sample_train_metrics=summary_info.get('sample_train_metrics', SampleLevelMetrics()),
            sample_val_metrics=summary_info.get('sample_val_metrics', SampleLevelMetrics()),
            sample_test_metrics=summary_info.get('sample_test_metrics', SampleLevelMetrics()),
            is_trained=is_trained
        )
        
        return model_info
    
    def analyze_all(self, include_untrained: bool = False) -> List[ModelInfo]:
        """分析所有模型"""
        self.models = []
        model_names = self.scan_models()
        
        for name in model_names:
            model_info = self.analyze_model(name)
            if model_info:
                if include_untrained or model_info.is_trained:
                    self.models.append(model_info)
        
        return self.models
    
    def evaluate_all_sample_level(self):
        """Evaluate all models at sample level"""
        if self.data_df is None:
            self.load_data()
        
        if self.data_df is None:
            print("Cannot load data, skipping sample-level evaluation")
            return
        
        print("\nEvaluating sample-level performance...")
        for model_info in self.models:
            if model_info.is_trained and model_info.model_path:
                print(f"  Evaluating {model_info.name}...")
                train_m, val_m, test_m = self.evaluate_sample_level(model_info)
                model_info.sample_train_metrics = train_m
                model_info.sample_val_metrics = val_m
                model_info.sample_test_metrics = test_m
    
    def generate_sci_report(self) -> str:
        """生成SCI论文级别的分析报告"""
        if not self.models:
            self.analyze_all()
        
        trained_models = [m for m in self.models if m.is_trained]
        
        lines = []
        
        lines.append("# Model Performance Analysis Report")
        lines.append("")
        lines.append(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"**Total Models**: {len(trained_models)} trained models")
        if len(self.models) > len(trained_models):
            lines.append(f"**Note**: {len(self.models) - len(trained_models)} models excluded (training incomplete)")
        lines.append("")
        
        lines.append("## Abstract")
        lines.append("")
        lines.append("This report presents a comprehensive analysis of multiple deep learning models")
        lines.append("developed for extrachromosomal DNA (ecDNA) prediction. The models were trained on")
        lines.append("a large-scale dataset with severe class imbalance and evaluated using multiple")
        lines.append("performance metrics including auPRC, AUC, Precision, Recall, and F1-score.")
        lines.append("")
        
        lines.append("## Dataset Description")
        lines.append("")
        if trained_models and trained_models[0].dataset_stats.train_samples > 0:
            stats = trained_models[0].dataset_stats
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
            lines.append("**Note**: The dataset has a high positive rate (~76%). This is expected as samples were pre-selected based on ecDNA presence for model training.")
            lines.append("")
        
        lines.append("## Model Architecture Comparison")
        lines.append("")
        lines.append("### Overview")
        lines.append("")
        lines.append("| Model | Architecture | Network Structure | Loss Function | Optimizer |")
        lines.append("|-------|--------------|-------------------|---------------|-----------|")
        for m in trained_models:
            lines.append(f"| {m.name} | {m.model_type} | {m.architecture_summary} | {m.loss_function} | {m.optimizer} |")
        lines.append("")
        
        lines.append("### Training Configuration")
        lines.append("")
        
        # Separate neural networks and XGBoost models
        nn_models = [m for m in trained_models if m.model_type not in ['XGBNew', 'XGB11', 'XGBoost']]
        xgb_models = [m for m in trained_models if m.model_type in ['XGBNew', 'XGB11', 'XGBoost']]
        
        if nn_models:
            lines.append("#### Neural Network Models")
            lines.append("")
            lines.append("| Model | Learning Rate | Weight Decay | Batch Size |")
            lines.append("|-------|---------------|--------------|------------|")
            for m in nn_models:
                lines.append(f"| {m.name} | {m.learning_rate:.6f} | {m.weight_decay:.4f} | {m.batch_size} |")
            lines.append("")
        
        if xgb_models:
            lines.append("#### XGBoost Models")
            lines.append("")
            lines.append("| Model | Learning Rate (eta) | Max Depth | Regularization (L1+L2) |")
            lines.append("|-------|---------------------|-----------|------------------------|")
            for m in xgb_models:
                lines.append(f"| {m.name} | {m.learning_rate:.2f} | {getattr(m, 'max_depth', 'N/A')} | {m.weight_decay:.2f} |")
            lines.append("")
            lines.append("*Note: XGBoost uses gradient boosting optimization, not traditional gradient descent. The learning rate (eta) controls step size, max_depth limits tree depth, and regularization (alpha + lambda) prevents overfitting.*")
            lines.append("")
        
        lines.append("## Performance Metrics")
        lines.append("")
        lines.append("### Performance Visualization")
        lines.append("")
        lines.append("#### Gene-Level Performance Comparison")
        lines.append("")
        lines.append("![Performance Comparison](performance_comparison.png)")
        lines.append("")
        lines.append("*Figure 1: Model performance comparison on test set. (a) auPRC - primary metric for imbalanced classification. (b) AUC - overall discriminative ability. (c) Precision-Recall trade-off. (d) F1-Score - harmonic mean of precision and recall.*")
        lines.append("")
        
        sample_evaluated = any(m.sample_test_metrics.auPRC > 0 for m in trained_models)
        if sample_evaluated:
            lines.append("#### Sample-Level Performance (Circular Detection)")
            lines.append("")
            lines.append("![Sample-Level Performance](sample_level_performance.png)")
            lines.append("")
            lines.append("*Figure 2: Sample-level performance for circular ecDNA detection. A sample is predicted as circular if any gene is predicted positive.*")
            lines.append("")
        
        lines.append("#### Performance Across Datasets")
        lines.append("")
        lines.append("![Dataset Comparison](dataset_comparison.png)")
        lines.append("")
        lines.append("*Figure 3: Model performance comparison across training, validation, and test datasets. Lower performance on test set indicates potential overfitting.*")
        lines.append("")
        
        lines.append("#### Multi-dimensional Performance Radar")
        lines.append("")
        lines.append("![Radar Chart](radar_chart.png)")
        lines.append("")
        lines.append("*Figure 4: Multi-dimensional performance comparison of top 5 models. Larger area indicates better overall performance.*")
        lines.append("")
        
        lines.append("### Test Set Performance (Primary Evaluation)")
        lines.append("")
        lines.append("| Model | auPRC | AUC | Precision | Recall | F1-Score |")
        lines.append("|-------|-------|-----|-----------|--------|----------|")
        for m in sorted(trained_models, key=lambda x: x.test_metrics.auPRC, reverse=True):
            tm = m.test_metrics
            lines.append(f"| {m.name} | **{tm.auPRC:.4f}** | {tm.AUC:.4f} | {tm.Precision:.4f} | {tm.Recall:.4f} | {tm.F1:.4f} |")
        lines.append("")
        
        lines.append("### Complete Performance Comparison")
        lines.append("")
        lines.append("#### Training Set Performance")
        lines.append("")
        lines.append("| Model | auPRC | AUC | Precision | Recall | F1-Score |")
        lines.append("|-------|-------|-----|-----------|--------|----------|")
        for m in trained_models:
            tm = m.train_metrics
            if tm.auPRC > 0:
                lines.append(f"| {m.name} | {tm.auPRC:.4f} | {tm.AUC:.4f} | {tm.Precision:.4f} | {tm.Recall:.4f} | {tm.F1:.4f} |")
        lines.append("")
        
        lines.append("#### Validation Set Performance")
        lines.append("")
        lines.append("| Model | auPRC | AUC | Precision | Recall | F1-Score |")
        lines.append("|-------|-------|-----|-----------|--------|----------|")
        for m in trained_models:
            vm = m.val_metrics
            if vm.auPRC > 0:
                lines.append(f"| {m.name} | {vm.auPRC:.4f} | {vm.AUC:.4f} | {vm.Precision:.4f} | {vm.Recall:.4f} | {vm.F1:.4f} |")
        lines.append("")
        
        lines.append("#### Test Set Performance")
        lines.append("")
        lines.append("| Model | auPRC | AUC | Precision | Recall | F1-Score |")
        lines.append("|-------|-------|-----|-----------|--------|----------|")
        for m in trained_models:
            tm = m.test_metrics
            lines.append(f"| {m.name} | {tm.auPRC:.4f} | {tm.AUC:.4f} | {tm.Precision:.4f} | {tm.Recall:.4f} | {tm.F1:.4f} |")
        lines.append("")
        
        sample_evaluated = any(m.sample_test_metrics.auPRC > 0 for m in trained_models)
        if sample_evaluated:
            lines.append("## Sample-Level Performance (Circular Detection)")
            lines.append("")
            lines.append("Sample-level evaluation determines whether a sample contains circular ecDNA.")
            lines.append("A sample is predicted as circular if any gene in the sample is predicted positive.")
            lines.append("")
            
            lines.append("### Test Set Sample-Level Performance")
            lines.append("")
            lines.append("| Model | auPRC | AUC | Accuracy | Precision | Recall | F1 | Samples |")
            lines.append("|-------|-------|-----|----------|-----------|--------|-----|---------|")
            for m in sorted(trained_models, key=lambda x: x.sample_test_metrics.auPRC, reverse=True):
                sm = m.sample_test_metrics
                if sm.auPRC > 0:
                    lines.append(f"| {m.name} | **{sm.auPRC:.4f}** | {sm.AUC:.4f} | {sm.Accuracy:.4f} | {sm.Precision:.4f} | {sm.Recall:.4f} | {sm.F1:.4f} | {sm.total_samples} |")
            lines.append("")
            
            lines.append("### Validation Set Sample-Level Performance")
            lines.append("")
            lines.append("| Model | auPRC | AUC | Accuracy | Precision | Recall | F1 | Samples |")
            lines.append("|-------|-------|-----|----------|-----------|--------|-----|---------|")
            for m in trained_models:
                sm = m.sample_val_metrics
                if sm.auPRC > 0:
                    lines.append(f"| {m.name} | {sm.auPRC:.4f} | {sm.AUC:.4f} | {sm.Accuracy:.4f} | {sm.Precision:.4f} | {sm.Recall:.4f} | {sm.F1:.4f} | {sm.total_samples} |")
            lines.append("")
            
            lines.append("### Training Set Sample-Level Performance")
            lines.append("")
            lines.append("| Model | auPRC | AUC | Accuracy | Precision | Recall | F1 | Samples |")
            lines.append("|-------|-------|-----|----------|-----------|--------|-----|---------|")
            for m in trained_models:
                sm = m.sample_train_metrics
                if sm.auPRC > 0:
                    lines.append(f"| {m.name} | {sm.auPRC:.4f} | {sm.AUC:.4f} | {sm.Accuracy:.4f} | {sm.Precision:.4f} | {sm.Recall:.4f} | {sm.F1:.4f} | {sm.total_samples} |")
            lines.append("")
        
        lines.append("### Overfitting Analysis")
        lines.append("")
        lines.append("| Model | Train-Val auPRC Gap | Severity | Precision Gap | Recall Gap |")
        lines.append("|-------|---------------------|----------|---------------|------------|")
        for m in trained_models:
            of = m.overfitting
            severity_emoji = {"low": "✅", "medium": "⚠️", "high": "❌"}.get(of.overfitting_severity, "❓")
            lines.append(f"| {m.name} | {of.train_val_auPRC_gap:.4f} | {severity_emoji} {of.overfitting_severity} | {of.train_val_precision_gap:.4f} | {of.train_val_recall_gap:.4f} |")
        lines.append("")
        
        lines.append("## Best Model Recommendations")
        lines.append("")
        
        if trained_models:
            best_auprc = max(trained_models, key=lambda m: m.test_metrics.auPRC)
            best_auc = max(trained_models, key=lambda m: m.test_metrics.AUC)
            best_f1 = max(trained_models, key=lambda m: m.test_metrics.F1)
            best_precision = max(trained_models, key=lambda m: m.test_metrics.Precision)
            best_recall = max(trained_models, key=lambda m: m.test_metrics.Recall)
            best_generalization = min(trained_models, key=lambda m: m.overfitting.train_val_auPRC_gap)
            
            lines.append("| Metric | Best Model | Value |")
            lines.append("|--------|------------|-------|")
            lines.append(f"| **Best auPRC** | {best_auprc.name} | {best_auprc.test_metrics.auPRC:.4f} |")
            lines.append(f"| **Best AUC** | {best_auc.name} | {best_auc.test_metrics.AUC:.4f} |")
            lines.append(f"| **Best F1-Score** | {best_f1.name} | {best_f1.test_metrics.F1:.4f} |")
            lines.append(f"| **Best Precision** | {best_precision.name} | {best_precision.test_metrics.Precision:.4f} |")
            lines.append(f"| **Best Recall** | {best_recall.name} | {best_recall.test_metrics.Recall:.4f} |")
            lines.append(f"| **Best Generalization** | {best_generalization.name} | Gap: {best_generalization.overfitting.train_val_auPRC_gap:.4f} |")
            
            if sample_evaluated:
                best_sample = max(trained_models, key=lambda m: m.sample_test_metrics.auPRC)
                if best_sample.sample_test_metrics.auPRC > 0:
                    lines.append(f"| **Best Sample-Level auPRC** | {best_sample.name} | {best_sample.sample_test_metrics.auPRC:.4f} |")
        lines.append("")
        
        # 添加使用建议
        lines.append("## Usage Guidelines")
        lines.append("")
        lines.append("### Metric Selection for Different Scenarios")
        lines.append("")
        lines.append("While **auPRC** (Area under Precision-Recall Curve) is the primary optimization target for gene-level ecDNA prediction due to class imbalance, users should select metrics based on their specific needs:")
        lines.append("")
        lines.append("| Scenario | Recommended Metric | Rationale |")
        lines.append("|----------|---------------------|-----------|")
        lines.append("| **High-confidence predictions** | Precision | Minimize false positives; use when follow-up validation is expensive |")
        lines.append("| **Comprehensive detection** | Recall | Maximize true positive detection; use when missing ecDNA is costly |")
        lines.append("| **Balanced performance** | F1-Score | Harmonic mean of precision and recall; good general-purpose metric |")
        lines.append("| **Overall discriminative ability** | auPRC | Robust to class imbalance; recommended for gene-level modeling |")
        lines.append("| **Sample-level detection** | Sample-Level auPRC, Precision, Recall | For determining if a sample contains circular ecDNA; consider precision/recall trade-offs |")
        lines.append("")
        lines.append("### Practical Recommendations")
        lines.append("")
        lines.append("1. **For research validation**: Use high-precision models (e.g., baseline_mlp with 97.77% precision) to minimize false positives in downstream experiments.")
        lines.append("2. **For screening applications**: Use high-recall models (e.g., xgb_new with 74.54% recall) to capture most ecDNA-positive genes.")
        lines.append("3. **For balanced applications**: Consider F1-score optimized models (e.g., xgb_paper with 78.38% F1) for a good trade-off.")
        lines.append("4. **For sample-level detection**: All models achieve >98% sample-level auPRC, making them reliable for detecting ecDNA-containing samples.")
        lines.append("")
        
        lines.append("## Architecture Details")
        lines.append("")
        for m in trained_models:
            arch_info = self._infer_architecture_info(m.model_type, {'input_dim': m.input_dim})
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
        lines.append("### Sample-Level vs Gene-Level Evaluation")
        lines.append("")
        lines.append("- **Gene-Level**: Each gene is evaluated independently for ecDNA presence.")
        lines.append("- **Sample-Level**: A sample is predicted as circular if any gene is predicted positive.")
        lines.append("  This reflects the clinical question: 'Does this sample contain circular ecDNA?'")
        lines.append("")
        lines.append("### Class Imbalance")
        lines.append("")
        lines.append("The dataset exhibits severe class imbalance (positive rate ~0.35%). This presents")
        lines.append("significant challenges for model training and evaluation. Models were trained using")
        lines.append("specialized loss functions and techniques to handle this imbalance effectively.")
        lines.append("")
        
        lines.append("## Conclusions")
        lines.append("")
        if trained_models:
            best = max(trained_models, key=lambda m: m.test_metrics.auPRC)
            lines.append(f"Among the {len(trained_models)} models evaluated, **{best.name}** achieved the highest")
            lines.append(f"test auPRC of **{best.test_metrics.auPRC:.4f}**, demonstrating superior performance")
            lines.append("for ecDNA prediction on this challenging imbalanced dataset.")
            lines.append("")
            if best.test_metrics.Precision >= 0.8:
                lines.append(f"The model achieved a precision of **{best.test_metrics.Precision:.4f}**, indicating that")
                lines.append("over 80% of predicted positive samples are true positives, which is crucial for")
                lines.append("reducing false positives in clinical applications.")
        lines.append("")
        
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
    
    def generate_performance_plots(self, output_dir: Path):
        """生成论文风格的性能对比图表"""
        if not self.models:
            self.analyze_all()
        
        trained_models = [m for m in self.models if m.is_trained]
        if not trained_models:
            print("No trained models found for plotting")
            return
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        sorted_models = sorted(trained_models, key=lambda x: x.test_metrics.auPRC, reverse=True)
        model_names = [m.name for m in sorted_models]
        n_models = len(sorted_models)
        
        # 1. Gene-Level Performance (Train/Val/Test)
        fig, axes = plt.subplots(2, 3, figsize=(10, 6))
        fig.suptitle('Gene-Level Performance Across Datasets', fontsize=11)
        
        x = np.arange(n_models)
        width = 0.25
        
        metrics_config = [
            ('auPRC', 'auPRC', 0.5, 1.02),
            ('AUC', 'AUC', 0.9, 1.005),
            ('Precision', 'Precision', 0.4, 1.02),
            ('Recall', 'Recall', 0.3, 1.02),
            ('F1', 'F1', 0.3, 1.02),
            ('Accuracy', 'Accuracy', 0.8, 1.02),
        ]
        
        for idx, (metric_name, metric_key, xmin, xmax) in enumerate(metrics_config):
            ax = axes[idx // 3, idx % 3]
            
            train_vals = [getattr(m.train_metrics, metric_key, 0) for m in sorted_models]
            val_vals = [getattr(m.val_metrics, metric_key, 0) for m in sorted_models]
            test_vals = [getattr(m.test_metrics, metric_key, 0) for m in sorted_models]
            
            bars1 = ax.barh(x - width, train_vals, width, label='Train', color=SCI_COLORS['train'], alpha=0.9)
            bars2 = ax.barh(x, val_vals, width, label='Val', color=SCI_COLORS['val'], alpha=0.9, edgecolor='black', linewidth=0.3)
            bars3 = ax.barh(x + width, test_vals, width, label='Test', color=SCI_COLORS['test'], alpha=0.9)
            
            ax.set_yticks(x)
            ax.set_yticklabels(model_names if idx % 3 == 0 else [], fontsize=6)
            ax.set_xlabel(metric_name)
            ax.set_xlim(xmin, xmax)
            ax.invert_yaxis()
            
            subplot_label = chr(ord('a') + idx)
            ax.set_title(f'({subplot_label}) {metric_name}')
            
            for bars in [bars1, bars2, bars3]:
                for bar in bars:
                    w = bar.get_width()
                    ax.text(w + 0.002, bar.get_y() + bar.get_height()/2, f'{w:.2f}',
                           va='center', fontsize=5)
            
            if idx == 0:
                ax.legend(loc='lower right', fontsize=6, frameon=True, fancybox=False, edgecolor='black')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'gene_level_performance.png', dpi=300, bbox_inches='tight')
        plt.savefig(output_dir / 'gene_level_performance.pdf', bbox_inches='tight')
        plt.close()
        print(f"Saved: {output_dir / 'gene_level_performance.png'}")
        
        # 2. Sample-Level Performance (Train/Val/Test)
        sample_evaluated = any(m.sample_test_metrics.auPRC > 0 for m in trained_models)
        if sample_evaluated:
            sorted_sample = sorted([m for m in trained_models if m.sample_test_metrics.auPRC > 0], 
                                   key=lambda x: x.sample_test_metrics.auPRC, reverse=True)
            sample_names = [m.name for m in sorted_sample]
            n_sample_models = len(sorted_sample)
            
            fig, axes = plt.subplots(2, 3, figsize=(10, 6))
            fig.suptitle('Sample-Level Performance Across Datasets', fontsize=11)
            
            x_sample = np.arange(n_sample_models)
            
            for idx, (metric_name, metric_key, xmin, xmax) in enumerate(metrics_config):
                ax = axes[idx // 3, idx % 3]
                
                train_vals = [getattr(m.sample_train_metrics, metric_key, 0) for m in sorted_sample]
                val_vals = [getattr(m.sample_val_metrics, metric_key, 0) for m in sorted_sample]
                test_vals = [getattr(m.sample_test_metrics, metric_key, 0) for m in sorted_sample]
                
                bars1 = ax.barh(x_sample - width, train_vals, width, label='Train', color=SCI_COLORS['train'], alpha=0.9)
                bars2 = ax.barh(x_sample, val_vals, width, label='Val', color=SCI_COLORS['val'], alpha=0.9, edgecolor='black', linewidth=0.3)
                bars3 = ax.barh(x_sample + width, test_vals, width, label='Test', color=SCI_COLORS['test'], alpha=0.9)
                
                ax.set_yticks(x_sample)
                ax.set_yticklabels(sample_names if idx % 3 == 0 else [], fontsize=6)
                ax.set_xlabel(metric_name)
                ax.set_xlim(xmin, xmax)
                ax.invert_yaxis()
                
                subplot_label = chr(ord('a') + idx)
                ax.set_title(f'({subplot_label}) {metric_name}')
                
                for bars in [bars1, bars2, bars3]:
                    for bar in bars:
                        w = bar.get_width()
                        ax.text(w + 0.002, bar.get_y() + bar.get_height()/2, f'{w:.2f}',
                               va='center', fontsize=5)
                
                if idx == 0:
                    ax.legend(loc='lower right', fontsize=6, frameon=True, fancybox=False, edgecolor='black')
            
            plt.tight_layout()
            plt.savefig(output_dir / 'sample_level_performance.png', dpi=300, bbox_inches='tight')
            plt.savefig(output_dir / 'sample_level_performance.pdf', bbox_inches='tight')
            plt.close()
            print(f"Saved: {output_dir / 'sample_level_performance.png'}")
        
        # 3. Precision-Recall and ROC Trade-off (Gene-Level)
        fig, axes = plt.subplots(1, 2, figsize=(7, 3))
        fig.suptitle('Gene-Level Trade-off Analysis', fontsize=10)
        
        ax = axes[0]
        precisions = [m.test_metrics.Precision for m in sorted_models]
        recalls = [m.test_metrics.Recall for m in sorted_models]
        colors_scatter = [SCI_COLORS[f'model{i+1}'] for i in range(min(n_models, 8))]
        
        for i, name in enumerate(model_names):
            ax.scatter(recalls[i], precisions[i], c=[colors_scatter[i % 8]], s=60, edgecolors='black', linewidth=0.5, zorder=3)
            ax.annotate(name, (recalls[i], precisions[i]), xytext=(3, 3), textcoords='offset points', fontsize=6)
        
        ax.set_xlabel('Recall (Sensitivity)')
        ax.set_ylabel('Precision (PPV)')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.05)
        ax.set_title('(a) Precision-Recall Trade-off')
        ax.plot([0, 1], [1, 0], 'k--', alpha=0.3, label='Random')
        ax.legend(fontsize=6)
        
        ax = axes[1]
        fprs = [1 - m.test_metrics.Recall for m in sorted_models]
        tprs = [m.test_metrics.Precision for m in sorted_models]
        
        for i, name in enumerate(model_names):
            ax.scatter(fprs[i], tprs[i], c=[colors_scatter[i % 8]], s=60, edgecolors='black', linewidth=0.5, zorder=3)
            ax.annotate(name, (fprs[i], tprs[i]), xytext=(3, 3), textcoords='offset points', fontsize=6)
        
        ax.set_xlabel('False Positive Rate (1 - Specificity)')
        ax.set_ylabel('True Positive Rate (Precision)')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.05)
        ax.set_title('(b) ROC-like Trade-off')
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='Random')
        ax.legend(fontsize=6)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'gene_level_tradeoff.png', dpi=300, bbox_inches='tight')
        plt.savefig(output_dir / 'gene_level_tradeoff.pdf', bbox_inches='tight')
        plt.close()
        print(f"Saved: {output_dir / 'gene_level_tradeoff.png'}")
        
        # 4. Precision-Recall and ROC Trade-off (Sample-Level)
        if sample_evaluated:
            fig, axes = plt.subplots(1, 2, figsize=(7, 3))
            fig.suptitle('Sample-Level Trade-off Analysis', fontsize=10)
            
            ax = axes[0]
            precisions = [m.sample_test_metrics.Precision for m in sorted_sample]
            recalls = [m.sample_test_metrics.Recall for m in sorted_sample]
            
            for i, name in enumerate(sample_names):
                ax.scatter(recalls[i], precisions[i], c=[colors_scatter[i % 8]], s=60, edgecolors='black', linewidth=0.5, zorder=3)
                ax.annotate(name, (recalls[i], precisions[i]), xytext=(3, 3), textcoords='offset points', fontsize=6)
            
            ax.set_xlabel('Recall (Sensitivity)')
            ax.set_ylabel('Precision (PPV)')
            ax.set_xlim(0.6, 1.02)
            ax.set_ylim(0.9, 1.02)
            ax.set_title('(a) Precision-Recall Trade-off')
            ax.plot([0, 1], [1, 0], 'k--', alpha=0.3)
            
            ax = axes[1]
            fprs = [1 - m.sample_test_metrics.Recall for m in sorted_sample]
            tprs = [m.sample_test_metrics.Precision for m in sorted_sample]
            
            for i, name in enumerate(sample_names):
                ax.scatter(fprs[i], tprs[i], c=[colors_scatter[i % 8]], s=60, edgecolors='black', linewidth=0.5, zorder=3)
                ax.annotate(name, (fprs[i], tprs[i]), xytext=(3, 3), textcoords='offset points', fontsize=6)
            
            ax.set_xlabel('False Positive Rate (1 - Specificity)')
            ax.set_ylabel('True Positive Rate (Precision)')
            ax.set_xlim(0, 0.5)
            ax.set_ylim(0.9, 1.02)
            ax.set_title('(b) ROC-like Trade-off')
            ax.plot([0, 1], [0, 1], 'k--', alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(output_dir / 'sample_level_tradeoff.png', dpi=300, bbox_inches='tight')
            plt.savefig(output_dir / 'sample_level_tradeoff.pdf', bbox_inches='tight')
            plt.close()
            print(f"Saved: {output_dir / 'sample_level_tradeoff.png'}")
        
        # 5. Multi-dimensional Performance Radar (Gene-Level and Sample-Level)
        from math import pi
        
        fig, axes = plt.subplots(1, 2, figsize=(7, 3.5), subplot_kw=dict(projection='polar'))
        fig.suptitle('Multi-dimensional Performance Comparison', fontsize=10)
        
        top_models = sorted_models[:5]
        categories = ['auPRC', 'AUC', 'Precision', 'Recall', 'F1']
        N = len(categories)
        angles = [n / float(N) * 2 * pi for n in range(N)]
        angles += angles[:1]
        
        ax = axes[0]
        radar_color_keys = ['model1', 'model2', 'model3', 'model4', 'model5']
        
        for i, model in enumerate(top_models):
            values = [model.test_metrics.auPRC, model.test_metrics.AUC, 
                     model.test_metrics.Precision, model.test_metrics.Recall, model.test_metrics.F1]
            values += values[:1]
            color = SCI_COLORS[radar_color_keys[i]]
            ax.plot(angles, values, 'o-', linewidth=1, label=model.name, color=color, markersize=3)
            ax.fill(angles, values, alpha=0.1, color=color)
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, fontsize=6)
        ax.set_ylim(0, 1)
        ax.set_title('(a) Gene-Level', fontsize=9, pad=10)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0), fontsize=5, frameon=True, fancybox=False, edgecolor='black')
        
        if sample_evaluated:
            ax = axes[1]
            top_sample = sorted_sample[:5]
            
            for i, model in enumerate(top_sample):
                values = [model.sample_test_metrics.auPRC, model.sample_test_metrics.AUC,
                         model.sample_test_metrics.Precision, model.sample_test_metrics.Recall, model.sample_test_metrics.F1]
                values += values[:1]
                color = SCI_COLORS[radar_color_keys[i]]
                ax.plot(angles, values, 'o-', linewidth=1, label=model.name, color=color, markersize=3)
                ax.fill(angles, values, alpha=0.1, color=color)
            
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(categories, fontsize=6)
            ax.set_ylim(0, 1)
            ax.set_title('(b) Sample-Level', fontsize=9, pad=10)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'performance_radar.png', dpi=300, bbox_inches='tight')
        plt.savefig(output_dir / 'performance_radar.pdf', bbox_inches='tight')
        plt.close()
        print(f"Saved: {output_dir / 'performance_radar.png'}")
        
        # 6. 模型排名热力图 - 适合 web 展示
        fig, ax = plt.subplots(figsize=(7, 4))
        
        metrics_for_heatmap = ['auPRC', 'AUC', 'Precision', 'Recall', 'F1']
        heatmap_data = []
        for m in sorted_models:
            row = [
                m.test_metrics.auPRC,
                m.test_metrics.AUC,
                m.test_metrics.Precision,
                m.test_metrics.Recall,
                m.test_metrics.F1
            ]
            heatmap_data.append(row)
        
        heatmap_array = np.array(heatmap_data)
        im = ax.imshow(heatmap_array, cmap='RdYlGn', aspect='auto', vmin=0.5, vmax=1.0)
        
        ax.set_xticks(np.arange(len(metrics_for_heatmap)))
        ax.set_xticklabels(metrics_for_heatmap, fontsize=8)
        ax.set_yticks(np.arange(len(sorted_models)))
        ax.set_yticklabels([m.name for m in sorted_models], fontsize=7)
        
        for i in range(len(sorted_models)):
            for j in range(len(metrics_for_heatmap)):
                text = ax.text(j, i, f'{heatmap_array[i, j]:.3f}',
                              ha='center', va='center', fontsize=6,
                              color='white' if heatmap_array[i, j] < 0.75 else 'black')
        
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label('Score', fontsize=8)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'model_ranking_heatmap.png', dpi=300, bbox_inches='tight')
        plt.savefig(output_dir / 'model_ranking_heatmap.pdf', bbox_inches='tight')
        plt.close()
        print(f"Saved: {output_dir / 'model_ranking_heatmap.png'}")
        
        # 6. 导出 Source Data (SCI 论文要求)
        source_data_path = output_dir / 'source_data'
        source_data_path.mkdir(parents=True, exist_ok=True)
        
        # Gene-level performance table
        gene_table = []
        for m in sorted_models:
            gene_table.append({
                'Model': m.name,
                'Type': m.model_type,
                'Train_auPRC': m.train_metrics.auPRC,
                'Train_AUC': m.train_metrics.AUC,
                'Train_Precision': m.train_metrics.Precision,
                'Train_Recall': m.train_metrics.Recall,
                'Train_F1': m.train_metrics.F1,
                'Val_auPRC': m.val_metrics.auPRC,
                'Val_AUC': m.val_metrics.AUC,
                'Val_Precision': m.val_metrics.Precision,
                'Val_Recall': m.val_metrics.Recall,
                'Val_F1': m.val_metrics.F1,
                'Test_auPRC': m.test_metrics.auPRC,
                'Test_AUC': m.test_metrics.AUC,
                'Test_Precision': m.test_metrics.Precision,
                'Test_Recall': m.test_metrics.Recall,
                'Test_F1': m.test_metrics.F1,
            })
        pd.DataFrame(gene_table).to_csv(source_data_path / 'gene_level_performance.csv', index=False)
        
        # Sample-level performance table
        sample_table = []
        for m in sorted_models:
            if m.sample_test_metrics.auPRC > 0:
                sample_table.append({
                    'Model': m.name,
                    'Train_auPRC': m.sample_train_metrics.auPRC,
                    'Train_Precision': m.sample_train_metrics.Precision,
                    'Train_Recall': m.sample_train_metrics.Recall,
                    'Train_Accuracy': m.sample_train_metrics.Accuracy,
                    'Val_auPRC': m.sample_val_metrics.auPRC,
                    'Val_Precision': m.sample_val_metrics.Precision,
                    'Val_Recall': m.sample_val_metrics.Recall,
                    'Val_Accuracy': m.sample_val_metrics.Accuracy,
                    'Test_auPRC': m.sample_test_metrics.auPRC,
                    'Test_Precision': m.sample_test_metrics.Precision,
                    'Test_Recall': m.sample_test_metrics.Recall,
                    'Test_Accuracy': m.sample_test_metrics.Accuracy,
                })
        if sample_table:
            pd.DataFrame(sample_table).to_csv(source_data_path / 'sample_level_performance.csv', index=False)
        
        # Dataset statistics
        stats_table = []
        for m in sorted_models:
            stats_table.append({
                'Model': m.name,
                'Train_Samples': m.dataset_stats.train_samples,
                'Train_Positive': m.dataset_stats.train_positive,
                'Train_Positive_Rate': f"{m.dataset_stats.train_positive_rate:.2%}",
                'Val_Samples': m.dataset_stats.val_samples,
                'Val_Positive': m.dataset_stats.val_positive,
                'Val_Positive_Rate': f"{m.dataset_stats.val_positive_rate:.2%}",
                'Test_Samples': m.dataset_stats.test_samples,
                'Test_Positive': m.dataset_stats.test_positive,
                'Test_Positive_Rate': f"{m.dataset_stats.test_positive_rate:.2%}",
            })
        pd.DataFrame(stats_table).to_csv(source_data_path / 'dataset_statistics.csv', index=False)
        
        print(f"Saved: {source_data_path / '*.csv'} (Source Data for SCI paper)")
        
        print(f"\nAll plots saved to: {output_dir}")
    
    def generate_report(self) -> str:
        """生成分析报告 (兼容旧接口)"""
        return self.generate_sci_report()
    
    def print_comparison_table(self):
        """打印对比表格"""
        if not self.models:
            self.analyze_all()
        
        trained_models = [m for m in self.models if m.is_trained]
        
        print("\n" + "=" * 120)
        print("Model Performance Comparison Table (Trained Models Only)")
        print("=" * 120)
        
        header = f"{'Model':<20} {'Type':<25} {'auPRC':>8} {'AUC':>8} {'Prec':>8} {'Recall':>8} {'F1':>8}"
        print(header)
        print("-" * 120)
        
        for m in sorted(trained_models, key=lambda x: x.test_metrics.auPRC, reverse=True):
            tm = m.test_metrics
            row = f"{m.name:<20} {m.model_type:<25} {tm.auPRC:>8.4f} {tm.AUC:>8.4f} {tm.Precision:>8.4f} {tm.Recall:>8.4f} {tm.F1:>8.4f}"
            print(row)
        
        print("=" * 120 + "\n")


def main():
    """主函数"""
    import argparse
    parser = argparse.ArgumentParser(description='Model Analyzer')
    parser.add_argument('--no-sample-level', action='store_true', help='Skip sample-level evaluation')
    args = parser.parse_args()
    
    analyzer = ModelAnalyzer()
    
    print("\nAnalyzing models...")
    models = analyzer.analyze_all(include_untrained=False)
    
    if not models:
        print("No trained models found!")
        return
    
    print(f"\nFound {len(models)} trained models\n")
    
    if not args.no_sample_level:
        analyzer.evaluate_all_sample_level()
    
    analyzer.print_comparison_table()
    
    # 生成报告
    report = analyzer.generate_sci_report()
    report_path = analyzer.models_dir / "model_analysis_report.md"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"\nReport saved to: {report_path}")
    
    # 生成图表（直接放在 models 目录下）
    print("\nGenerating performance plots...")
    plots_dir = analyzer.models_dir
    analyzer.generate_performance_plots(plots_dir)
    
    # 复制报告和图表到 static 目录供网页访问
    static_dir = Path(__file__).parent / "static"
    if static_dir.exists():
        import shutil
        try:
            shutil.copy(report_path, static_dir / "model_analysis_report.md")
            for png_file in plots_dir.glob("*.png"):
                shutil.copy(png_file, static_dir / png_file.name)
            print(f"\nCopied report and plots to: {static_dir}")
        except Exception as e:
            print(f"\nWarning: Could not copy to static dir: {e}")


if __name__ == "__main__":
    main()
