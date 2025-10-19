# 数据并行训练基线 (CA Backends)

## 项目概述

这是RingGNN项目的数据并行训练基线实现，提供了一个简化但功能完整的分布式GNN训练方案。该实现专注于**缓存感知的数据并行训练**（Cache-Aware Backends），通过图数据分区和参数同步实现高效的分布式训练。

## 核心特性

- **图数据分区**: 使用`hybrid_fixed_random`策略实现训练和评估子图的分离
- **批模式训练**: 每个Worker使用完整的训练子集，无复杂的batch缓存
- **参数同步**: 基于FedAvg策略的参数聚合
- **多架构支持**: 支持GCN、GAT、SAGE等多种GNN架构的动态构建
- **设备灵活性**: 支持CPU和多GPU训练

## 目录结构

```
baseline/gnas_ca_rl/
├── config.yaml              # 配置文件
├── run.py                    # 主运行脚本
├── test_ca_backends.py       # 测试脚本
└── logs/                     # 日志输出目录（自动创建）
└── models/                   # 模型保存目录（自动创建）

baseline/components/backends/
└── ca_backends.py            # 核心实现
```

## 快速开始

### 1. 环境准备

确保已安装以下依赖：
```bash
# PyTorch和相关库
torch
torch-geometric

# 项目依赖
PyYAML
numpy

# 切换到conda环境
conda activate FastGNAS
```

### 2. 基本使用

#### 运行默认实验
```bash
cd /path/to/RingGNN
python baseline/gnas_ca_rl/run.py
```

#### 指定配置参数
```bash
# 使用不同数据集
python baseline/gnas_ca_rl/run.py --dataset PubMed

# 调整Worker数量
python baseline/gnas_ca_rl/run.py --workers 4

# 测试特定架构
python baseline/gnas_ca_rl/run.py --arch simple_gcn
```

#### 自定义配置文件
```bash
python baseline/gnas_ca_rl/run.py --config path/to/your/config.yaml
```

### 3. 运行测试

```bash
# 运行功能测试
python baseline/gnas_ca_rl/test_ca_backends.py

# 检查基本功能
cd baseline/gnas_ca_rl
python -c "from test_ca_backends import run_integration_test; run_integration_test()"
```

## 配置说明

### 主要配置项

```yaml
ca_backends:
  num_workers: 2              # Worker数量
  dataset: "Cora"             # 数据集 (Cora/PubMed/Computers)
  device: "cuda"              # 设备类型 (cuda/cpu)
  learning_rate: 0.01         # 学习率
  num_classes: 7              # 分类数量

training:
  epochs_per_round: 5         # 每轮训练的epoch数
  num_rounds: 10              # 总训练轮数
  early_stopping_patience: 5 # 早停耐心值

architecture:
  test_architectures:         # 测试架构列表
    - name: "simple_gcn"
      layers:
        - type: "gcn"
          hidden_dim: 64
          act: "relu"
```

### 支持的架构类型

- **GCN**: 图卷积网络
- **GAT**: 图注意力网络
- **SAGE**: GraphSAGE
- **Cheb**: 切比雪夫图卷积
- **ARMA**: ARMA图卷积
- **SG**: 简化图卷积

### 支持的激活函数

- relu, elu, tanh, sigmoid
- leaky_relu, relu6, softplus
- linear (恒等映射)

## 核心组件

### CABackends 类

主要的数据并行训练后端，提供以下功能：
- 自动设备配置和数据分区
- 分布式训练协调
- 参数同步和模型评估

```python
from baseline.components.backends.ca_backends import CABackends

# 初始化
ca_backends = CABackends(config)
ca_backends.setup_distributed_training()

# 训练
model_state, loss = ca_backends.train_epoch(arch_spec, model_state, epochs=5)

# 评估
accuracy = ca_backends.evaluate_model(arch_spec, model_state)
```

### SimpleWorker 类

单个Worker的训练逻辑：
- 在本地数据分区上训练模型
- 支持模型参数的加载和保存
- 提供本地评估功能

## 实验结果

运行后会自动生成以下输出：

1. **控制台日志**: 实时训练进度和结果
2. **日志文件**: 保存在`logs/run_TIMESTAMP/`目录
3. **结果文件**: JSON格式的详细实验结果
4. **最佳模型**: 可选保存最佳模型参数

### 典型输出示例

```
================================================================================
数据并行训练基线实验开始
================================================================================
数据集: Cora
Worker数量: 2
设备类型: cuda

📊 数据分区信息:
分区数量: 2
总节点数: 2708
  分区 0: 1544 节点, 训练=108, 验证=67, 测试=200
  分区 1: 1164 节点, 训练=32, 验证=33, 测试=300

🏆 最佳架构: simple_gcn
🎯 最佳准确率: 0.8150
⚡ 训练时间: 45.23 秒
```

## 性能对比

相比单机训练的理论优势：
- **训练加速**: 近似线性加速比（取决于Worker数量）
- **内存效率**: 分布式内存使用，支持更大的模型
- **扩展性**: 支持多GPU和多节点扩展

## 设计理念

### 简洁性原则

遵循Linus的"好品味"设计哲学：
- **消除特殊情况**: 统一的训练和评估接口
- **简化数据结构**: 清晰的配置和状态管理
- **最小化复杂度**: 避免过度设计和不必要的抽象

### 实用主义导向

- **解决实际问题**: 专注于分布式训练的核心需求
- **可靠性优先**: 确保在各种环境下都能稳定运行
- **易于扩展**: 为未来功能扩展预留清晰接口

## 常见问题

### Q: 如何添加新的GNN层类型？
A: 在`trainer/builder.py`的`gnn_map`函数中添加新的映射。

### Q: 如何调整分区策略？
A: 修改`CABackends`中的`PartitionGraph.get_partitions`调用参数。

### Q: 如何支持更多数据集？
A: 确保数据集在`data/data_utils.py`中已注册，然后在配置文件中指定。

### Q: GPU内存不足怎么办？
A: 减少`num_workers`数量或降低模型的`hidden_dim`参数。

## 扩展开发

### 添加新的参数同步策略

可以在`CABackends.synchronize_parameters`中实现新的聚合算法：
```python
def synchronize_parameters_weighted(self, model_states, weights):
    # 实现加权平均等高级同步策略
    pass
```

### 支持异步训练

可以通过修改训练循环实现异步参数更新：
```python
async def async_train_epoch(self, ...):
    # 异步训练实现
    pass
```

## 许可证

本项目遵循与主项目相同的开源许可证。