# otk: ecDNA Analysis Toolkit

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

otk (ecDNA Analysis Toolkit) 是一个基于深度学习的ecDNA分析工具，用于预测基因是否在某样本中被检测为ecDNA cargo gene（gene level），以及样本的focal amplification类型（sample level）。

## 核心功能

- 基于深度学习的ecDNA cargo gene预测
- 样本级别的focal amplification类型分类
- 支持从BAM文件或处理后的拷贝数数据进行分析
- 高效的命令行界面
- 支持GPU加速

## 技术栈

- Python 3.8+
- PyTorch 2.0+
- NumPy
- Pandas
- scikit-learn
- Click (命令行工具)

## 安装指南

### 从源码安装

1. 克隆项目仓库：

```bash
git clone https://github.com/yourusername/otk.git
cd otk
```

2. 使用pip安装：

```bash
pip install -e .
```

### 依赖项

安装过程中会自动安装以下依赖项：

- pandas>=2.0
- numpy>=1.24
- torch>=2.0
- scikit-learn>=1.3
- tqdm>=4.65
- click>=8.1
- matplotlib>=3.7
- seaborn>=0.12
- pyyaml>=6.0

## 使用方法

otk提供了两个主要的命令行子命令：`train`和`predict`。

### 模型训练

使用`otk train`命令训练模型：

```bash
otk train --config configs/model_config.yml --output models/ --gpu 0
```

参数说明：
- `--config, -c`: 配置文件路径（默认：configs/model_config.yml）
- `--output, -o`: 训练模型的输出目录（默认：models/）
- `--gpu, -g`: 使用的GPU设备ID（默认：0）

### 模型预测

使用`otk predict`命令进行预测：

```bash
otk predict --model models/best_model.pth --input data/test_data.csv --output predictions/ --gpu -1
```

参数说明：
- `--model, -m`: 训练好的模型路径（必需）
- `--input, -i`: 输入数据文件路径（必需）
- `--output, -o`: 预测结果的输出目录（默认：predictions/）
- `--gpu, -g`: 使用的GPU设备ID（默认：-1，即使用CPU）

## 数据格式

### 输入数据格式

输入数据应为CSV格式，包含以下列：

- `sample`: 肿瘤样本ID
- `gene_id`: 基因ID
- `segVal`: 基因总拷贝数
- `minor_cn`: 基因小拷贝数
- `age`: 患者年龄
- `gender`: 患者性别
- 各种肿瘤类型的one-hot编码列（如`type_BLCA`, `type_BRCA`等）
- `freq_Linear`, `freq_BFB`, `freq_Circular`, `freq_HR`: 基因在不同类型基因组focal amplification中的先验估计频率

### 输出数据格式

预测结果包含以下列：

- `sample`: 肿瘤样本ID
- `gene_id`: 基因ID
- `prediction_prob`: 预测为ecDNA cargo gene的概率
- `prediction`: 二分类预测结果（0或1）

此外，还会生成样本级别的预测结果：

- `sample`: 肿瘤样本ID
- `prediction_prob`: 样本中最大的预测概率
- `prediction`: 样本级别的预测结果（0或1）
- `focal_amplification_type`: 样本的focal amplification类型（circular或noncircular）

## 模型架构

otk使用多层感知器（MLP）作为深度学习模型架构，默认配置如下：

- 输入层：58个特征
- 隐藏层1：128个神经元，ReLU激活，20% dropout
- 隐藏层2：64个神经元，ReLU激活，20% dropout
- 隐藏层3：32个神经元，ReLU激活，10% dropout
- 输出层：1个神经元，Sigmoid激活

模型使用BCEWithLogitsLoss作为损失函数，Adam作为优化器。

## 配置文件

模型配置使用YAML格式，示例配置文件位于`configs/model_config.yml`。你可以根据需要修改配置文件中的参数，如模型架构、训练参数等。

## 示例

### 训练示例

```bash
# 使用默认配置训练模型
otk train

# 使用自定义配置文件
otk train --config my_config.yml
```

### 预测示例

```bash
# 使用训练好的模型进行预测
otk predict --model models/best_model.pth --input test_data.csv
```

## 性能指标

模型训练过程中会记录以下性能指标：

- auPRC (Area under Precision-Recall Curve)
- AUC (Area under ROC Curve)
- F1 Score
- Precision
- Recall

## 贡献指南

我们欢迎社区贡献！如果你有任何问题或建议，请通过GitHub Issues提交。

### 开发流程

1. Fork仓库
2. 创建功能分支
3. 实现功能或修复bug
4. 运行测试
5. 提交Pull Request

## 许可证

本项目采用MIT许可证。详见[LICENSE](LICENSE)文件。

## 引用

如果您在研究中使用了otk，请引用以下论文：

```
Wang, S., Wu, C. Y., He, M. M., Yong, J. X., Chen, Y. X., Qian, L. M., ... & Zhao, Q. (2024). Machine learning-based extrachromosomal DNA identification in large-scale cohorts reveals its clinical implications in cancer. Nature Communications, 15(1), 1-17.
```

## 联系我们

- 项目主页：https://github.com/yourusername/otk
- 邮箱：your.email@example.com
