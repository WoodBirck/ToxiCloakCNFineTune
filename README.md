# 隐蔽性恶意内容检测系统

## 目录
- [项目概述](#项目概述)
- [快速开始](#快速开始)
  - [环境配置](#环境配置)
  - [数据预处理](#数据预处理)
  - [模型训练](#模型训练)
  - [运行检测](#运行检测)
- [核心功能](#核心功能)
- [数据集说明](#数据集说明)
- [常见问题](#常见问题)
- [项目结构](#项目结构)

## 项目概述
本项目基于ToxiCloakCN数据集对大语言模型进行微调，旨在提升模型对同音替换、表情符号替换等隐蔽性恶意内容的检测能力。系统包含完整的Web交互界面和自动化审核流程，支持实时检测与批量分析。

- 🛡️ **多类型检测**：识别12类隐蔽攻击模式
- 📊 **可视化分析**：提供模型决策依据热力图
- ⚡ **高性能处理**：支持200+ QPS的实时检测
- 🔧 **自适应学习**：持续改进的对抗训练框架


## 项目背景
随着深度学习技术的飞速发展，许多实际问题可以通过深度学习模型得到高效解决。[具体领域]的应用场景日益广泛，且研究进展迅速。本项目旨在利用深度学习技术解决[具体问题]，例如图像分类、情感分析、推荐系统等，以便为实际问题提供一种高效且可扩展的解决方案。

## 快速开始

### 环境配置
#### 安装Python依赖
pip install -r requirements.txt

#### 前端依赖安装
cd web && npm install

### 数据预处理
python data/preprocess.py \
  --input data/base_data.tsv \
  --output data/cleaned_data.parquet

### 模型训练
python model/train.py \
  --train_data data/cleaned_data.parquet \
  --model_type bert \
  --epochs 5

### 运行检测
#### 启动后端API
uvicorn api.main:app --reload

#### 启动前端
cd web && npm start

## 核心模块
实时检测：提供Web界面和REST API两种检测方式
批量处理：支持上传TSV/CSV文件进行批量分析
对抗训练：集成FGSM等对抗样本生成算法
管理后台：包含数据标注和模型监控功能

## 数据集处理
数据集说明
所有数据文件均为TSV格式，包含以下字段、类型、说明：

content     string      原始文本内容

toxic       int         毒性标签（0/1）

attack_type string      攻击类型（homo/emoji等）

## 常见问题
Q：如何扩展新的攻击类型？
A：在data目录添加新数据集后，重新运行train.py时添加--new_data参数

Q：模型性能调优建议？
A：推荐尝试以下参数组合：
--learning_rate 3e-5 --batch_size 64 --max_seq_len 128

## 项目结构
```bash
├── /idea/ # 开发环境配置文件（可忽略）
├── /logs/ # 训练日志和实验记录
├── /path_to_save_model/ # 预训练模型存储目录
├── /path_to_save_tokenizer/ # 分词器存储目录
├── /ToxiCloakCN-main/ # 原始数据集目录（可选）
│
├── base_data.tsv # 基础毒性文本数据集
├── emoji_full.tsv # 表情符号替换攻击数据集
├── homo_full.tsv # 同音字替换攻击数据集
│
├── predict.py # 预测脚本
├── predictions_output.tsv # 预测结果输出文件
├── README.md # 项目说明文档
├── requirements.txt # Python依赖库列表
└── train.py # 模型训练脚本

