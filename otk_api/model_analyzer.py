#!/usr/bin/env python3
"""
Model Analyzer - 自动分析 otk_api/models 下的模型架构和性能

功能:
1. 扫描所有模型目录
2. 解析 config.yml 和 training_summary.yml
3. 生成架构对比和性能汇总报告
"""

import yaml
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime


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
    early_stopped: bool = False
    val_auPRC: float = 0.0
    metrics: Dict[str, float] = field(default_factory=dict)
    training_time: float = 0.0
    config_path: str = ""
    model_path: str = ""


class ModelAnalyzer:
    MODEL_ARCHITECTURE_INFO = {
        "Baseline": {
            "description": "简单MLP网络",
            "structure": "57→256→128→64→1",
            "features": ["ReLU激活", "Sigmoid输出", "无正则化"],
            "suitable_for": "基准模型, 高精度低召回场景"
        },
        "TransformerEcDNA": {
            "description": "Transformer注意力模型",
            "structure": "57→128(embedding)→Attention→64→32→1",
            "features": ["自注意力机制", "LayerNorm", "GELU激活", "Dropout正则化"],
            "suitable_for": "特征交互学习, 平衡精度召回"
        },
        "PrecisionFocusedEcDNA": {
            "description": "深度残差网络",
            "structure": "57→512→256→128→64→32→1",
            "features": ["残差连接", "LayerNorm", "GELU激活", "渐进降维"],
            "suitable_for": "深度特征学习, 高精度场景"
        },
        "AdvancedEcDNA": {
            "description": "CNN-Transformer混合模型",
            "structure": "CNN→Transformer→FPN→分类头",
            "features": ["CNN局部特征", "Transformer全局交互", "特征金字塔网络"],
            "suitable_for": "复杂特征学习, 多尺度表示"
        },
        "OptimizedEcDNA": {
            "description": "优化残差网络",
            "structure": "57→128→64→32→16→1",
            "features": ["残差块", "BatchNorm", "组合损失函数"],
            "suitable_for": "平衡训练, 稳定收敛"
        },
        "EnsembleEcDNA": {
            "description": "集成模型",
            "structure": "Advanced + Precision → Meta分类器",
            "features": ["多模型融合", "加权投票", "元学习"],
            "suitable_for": "最佳性能, 生产部署"
        },
        "EnsembleOptimizedEcDNA": {
            "description": "优化集成模型",
            "structure": "3×OptimizedEcDNA → Meta分类器",
            "features": ["多模型集成", "元分类器融合"],
            "suitable_for": "鲁棒预测, 减少过拟合"
        },
        "ImprovedV2": {
            "description": "改进V2模型",
            "structure": "57→256→128→64→1",
            "features": ["BatchNorm", "Dropout", "残差连接"],
            "suitable_for": "改进基准, 更好泛化"
        },
        "ImprovedV2_Deep": {
            "description": "深度改进V2模型",
            "structure": "57→512→256→128→64→1",
            "features": ["更深网络", "残差连接", "BatchNorm"],
            "suitable_for": "复杂模式学习"
        },
        "EnhancedTransformerEcDNA": {
            "description": "增强Transformer模型",
            "structure": "57→256(embedding)→4层Transformer→64→1",
            "features": ["深层Transformer", "多头注意力", "梯度检查点"],
            "suitable_for": "大规模特征交互"
        },
        "LightweightTransformerEcDNA": {
            "description": "轻量Transformer模型",
            "structure": "57→64(embedding)→2层Transformer→1",
            "features": ["轻量设计", "快速推理", "低内存占用"],
            "suitable_for": "快速训练, 资源受限场景"
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
        
        if result['layers']:
            result['input_dim'] = result['layers'][0].get('input_dim', 57)
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
    
    def parse_training_summary(self, summary: Dict) -> Dict[str, Any]:
        """解析训练摘要"""
        result = {}
        
        result['best_val_auPRC'] = summary.get('best_val_auPRC', 0.0)
        result['epochs_trained'] = summary.get('epochs_trained', 0)
        result['early_stopped'] = summary.get('early_stopped', False)
        result['training_time'] = summary.get('total_training_time', 0.0)
        
        test_metrics = summary.get('test_metrics', {})
        result['metrics'] = {
            'AUC': test_metrics.get('AUC', 0.0),
            'auPRC': test_metrics.get('auPRC', 0.0),
            'F1': test_metrics.get('F1', 0.0),
            'Precision': test_metrics.get('Precision', 0.0),
            'Recall': test_metrics.get('Recall', 0.0),
        }
        
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
            "description": "未知架构",
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
            early_stopped=summary_info['early_stopped'],
            val_auPRC=summary_info['best_val_auPRC'],
            metrics=summary_info['metrics'],
            training_time=summary_info['training_time'],
            config_path=str(config_path),
            model_path=str(model_path) if model_path.exists() else "",
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
    
    def generate_report(self) -> str:
        """生成分析报告"""
        if not self.models:
            self.analyze_all()
        
        lines = []
        lines.append("=" * 80)
        lines.append("OTK API 模型分析报告")
        lines.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"模型目录: {self.models_dir}")
        lines.append("=" * 80)
        lines.append("")
        
        lines.append("## 模型概览")
        lines.append("")
        lines.append(f"共发现 {len(self.models)} 个模型:")
        lines.append("")
        for m in self.models:
            lines.append(f"  - {m.name} ({m.model_type})")
        lines.append("")
        
        lines.append("## 架构对比")
        lines.append("")
        lines.append("| 模型名称 | 架构类型 | 网络结构 | 损失函数 | 优化器 |")
        lines.append("|----------|----------|----------|----------|--------|")
        for m in self.models:
            lines.append(f"| {m.name} | {m.model_type} | {m.architecture_summary} | {m.loss_function} | {m.optimizer} |")
        lines.append("")
        
        lines.append("## 训练配置")
        lines.append("")
        lines.append("| 模型名称 | 学习率 | Weight Decay | Batch Size | 训练轮数 | 早停 |")
        lines.append("|----------|--------|--------------|------------|----------|------|")
        for m in self.models:
            early_stop_str = "是" if m.early_stopped else "否"
            lines.append(f"| {m.name} | {m.learning_rate} | {m.weight_decay} | {m.batch_size} | {m.epochs_trained} | {early_stop_str} |")
        lines.append("")
        
        lines.append("## 性能指标")
        lines.append("")
        lines.append("| 模型名称 | AUC | val_auPRC | test_auPRC | F1 | Precision | Recall |")
        lines.append("|----------|-----|-----------|------------|-----|-----------|--------|")
        for m in self.models:
            metrics = m.metrics
            lines.append(f"| {m.name} | {metrics['AUC']:.4f} | {m.val_auPRC:.4f} | {metrics['auPRC']:.4f} | {metrics['F1']:.4f} | {metrics['Precision']:.4f} | {metrics['Recall']:.4f} |")
        lines.append("")
        
        lines.append("## 最佳模型推荐")
        lines.append("")
        
        if self.models:
            best_auprc = max(self.models, key=lambda m: m.metrics['auPRC'])
            best_val_auprc = max(self.models, key=lambda m: m.val_auPRC)
            best_auc = max(self.models, key=lambda m: m.metrics['AUC'])
            best_f1 = max(self.models, key=lambda m: m.metrics['F1'])
            best_precision = max(self.models, key=lambda m: m.metrics['Precision'])
            best_recall = max(self.models, key=lambda m: m.metrics['Recall'])
            
            lines.append("")
            lines.append(f"- **最佳 test_auPRC**: {best_auprc.name} ({best_auprc.metrics['auPRC']:.4f})")
            lines.append(f"- **最佳 val_auPRC**: {best_val_auprc.name} ({best_val_auprc.val_auPRC:.4f})")
            lines.append(f"- **最佳 AUC**: {best_auc.name} ({best_auc.metrics['AUC']:.4f})")
            lines.append(f"- **最佳 F1**: {best_f1.name} ({best_f1.metrics['F1']:.4f})")
            lines.append(f"- **最佳 Precision**: {best_precision.name} ({best_precision.metrics['Precision']:.4f})")
            lines.append(f"- **最佳 Recall**: {best_recall.name} ({best_recall.metrics['Recall']:.4f})")
        lines.append("")
        
        lines.append("## 架构详情")
        lines.append("")
        for m in self.models:
            arch_info = self.MODEL_ARCHITECTURE_INFO.get(m.model_type, {})
            lines.append(f"### {m.name}")
            lines.append("")
            lines.append("")
            lines.append(f"- **类型**: {m.model_type}")
            lines.append(f"- **描述**: {arch_info.get('description', 'N/A')}")
            lines.append(f"- **结构**: {m.architecture_summary}")
            lines.append(f"- **特性**: {', '.join(arch_info.get('features', []))}")
            lines.append(f"- **适用场景**: {arch_info.get('suitable_for', 'N/A')}")
            lines.append("")
        
        lines.append("=" * 80)
        lines.append("报告结束")
        lines.append("=" * 80)
        
        return "\n".join(lines)
    
    def print_comparison_table(self):
        """打印对比表格"""
        if not self.models:
            self.analyze_all()
        
        print("\n" + "=" * 120)
        print("模型性能对比表")
        print("=" * 120)
        
        header = f"{'模型名称':<20} {'类型':<25} {'AUC':>8} {'val_auPRC':>10} {'test_auPRC':>10} {'F1':>8} {'Prec':>8} {'Recall':>8}"
        print(header)
        print("-" * 120)
        
        for m in sorted(self.models, key=lambda x: x.metrics['auPRC'], reverse=True):
            row = f"{m.name:<20} {m.model_type:<25} {m.metrics['AUC']:>8.4f} {m.val_auPRC:>10.4f} {m.metrics['auPRC']:>10.4f} {m.metrics['F1']:>8.4f} {m.metrics['Precision']:>8.4f} {m.metrics['Recall']:>8.4f}"
            print(row)
        
        print("=" * 120 + "\n")


def main():
    """主函数"""
    analyzer = ModelAnalyzer()
    
    print("\n正在分析模型...")
    models = analyzer.analyze_all()
    
    if not models:
        print("未找到任何模型!")
        return
    
    print(f"\n发现 {len(models)} 个模型\n")
    
    analyzer.print_comparison_table()
    
    report = analyzer.generate_report()
    print(report)
    
    report_path = analyzer.models_dir / "model_analysis_report.md"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"\n报告已保存至: {report_path}")


if __name__ == "__main__":
    main()
