# 架构并行训练基线 (Parallel Backends)

## 项目概述

`baseline/gnas_parallel_rl` 提供了一个面向 RingGNN 的 **架构并行 (Architecture-Parallel)** 训练基线，实现一次性在多台设备上评估不同 GNN 架构的流程，可用于与 `gnas_ca_rl`、`gnas_co_rl` 做基线对比。

## 核心特性

- **完整图副本**：每个 Worker 拥有完整图数据，适合架构层面的并行探索
- **Ray 统一调度**：自动根据 GPU/CPU 情况初始化 Ray 并创建 `ArchParallelWorker`
- **动态模型构建**：复用 `ModelBuilder` 支持 GCN、GAT、SAGE 等架构
- **RL 控制器接入**：默认提供 `RLBalancedSearcher` 以层级策略采样架构
- **日志与结果归档**：与其它基线一致，输出结构化 JSON 记录

## 目录结构

```
baseline/gnas_parallel_rl/
├── config.yaml          # 默认配置
├── run.py               # 主运行脚本
├── README.md            # 使用说明
└── logs/                # 日志输出目录（自动创建）

baseline/components/backends/
└── parallel_backends.py # ArchParallelBackend 实现
```

## 快速开始

### 1. 环境准备

确保安装项目依赖，建议使用作者提供的 Conda 环境：

```bash
conda activate FastGNAS
pip install -r requirements.txt
```

### 2. 基本使用

在仓库根目录执行：

```bash
python baseline/gnas_parallel_rl/run.py
```

常见参数：

```bash
# 指定数据集
python baseline/gnas_parallel_rl/run.py --dataset PubMed

# 指定 CPU Worker 数量（仅在无 GPU 时生效）
python baseline/gnas_parallel_rl/run.py --workers 4

# 限定测试某个已命名架构
python baseline/gnas_parallel_rl/run.py --arch parallel_gat_gcn

# 使用自定义配置文件
python baseline/gnas_parallel_rl/run.py --config path/to/your.yaml
```

### 3. 运行测试

```bash
python test_parallel_backends.py
```

该脚本会构造一个人工图并验证 `ArchParallelBackend` 的训练、评估接口。

## 配置说明

`config.yaml` 与 CA/CO 基线保持一致，关键区别如下：

```yaml
parallel_backends:
  dataset: "Cora"
  num_workers: 2         # CPU 模式有效，GPU 环境自动等于可用 GPU 数
  ray:
    init_kwargs:
      include_dashboard: false

training:
  epochs_per_round: 1
  num_rounds: 5

architecture:
  test_architectures:    # 预设对比架构
    - name: parallel_simple_gcn
      layers:
        - type: gcn
          hidden_dim: 64
          act: relu
```

## 支持的架构与激活函数

- 图层：GCN、GAT、GraphSAGE、Cheb、ARMA、SG
- 激活：relu、elu、tanh、sigmoid、leaky_relu、relu6、softplus、linear

## 输出结果

运行完成后将得到：

- 控制台与日志文件：位于 `logs/parallel_run_*` 目录
- `experiment_results.json`：记录每个架构的最佳精度、训练时间等信息

## 注意事项

- 默认会尝试初始化 Ray，如需复用已有 Ray 集群，可在配置文件中设置 `parallel_backends.ray.address`
- 与 CA/CO 流程一致：采样架构 → 多轮训练 → 评估 → 回传奖励
- 结束时会自动调用 `Ray.shutdown()` 和 `ArchParallelBackend.cleanup()` 释放资源

祝实验顺利！
