# FastGNAS-RL 实验

完整的 FastGNAS 方案实验，包含：
- **负载均衡策略**：使用 `BalanceController` 和 `MloadPredictor` 进行智能负载预测和架构分配
- **桶式轮转调度**：通过 `Dispatcher` 实现高效的模型轮转和资源利用
- **RL 控制器**：使用强化学习搜索最优 GNN 架构
- **分布式训练**：基于 Ray 的多 GPU 并行训练

## 快速开始

### 1. 基础用法

```bash
# 使用 Cora 数据集运行
python experiment/FastGNAS-rl/run.py --dataset Cora

# 使用 Reddit 数据集运行
python experiment/FastGNAS-rl/run.py --dataset Reddit

# 使用 OGB 数据集运行
python experiment/FastGNAS-rl/run.py --dataset ogbn-arxiv
```

### 2. 自定义配置

```bash
# 指定配置文件
python experiment/FastGNAS-rl/run.py \
    --dataset Cora \
    --config experiment/FastGNAS-rl/config.yaml

# 覆盖 worker 数量
python experiment/FastGNAS-rl/run.py \
    --dataset Cora \
    --workers 4
```

## 配置说明

### 数据集配置

所有数据集的参数统一在 `experiment/dataset.yaml` 中配置，包括：
- 批大小 (`train_batch_size`, `eval_batch_size`)
- 采样策略 (`fan_out`)
- 训练参数 (`epochs_per_round`, `num_rounds`, `steps_per_epoch`)
- GNN 层数 (`gnn_layers`)
- 类别数 (`num_classes`)

运行时只需指定 `--dataset` 参数，相关配置会自动加载。

### FastGNAS 核心配置

在 `config.yaml` 中的 `nas.balancing_strategy` 部分：

```yaml
balancing_strategy:
  enable: true                    # 启用负载均衡
  rotation_mode: "subtask"        # 桶式轮转模式
  batch_size: 10                  # 批量生成架构数
  tolerance: 0.1                  # 负载容忍度
  model_path: "mload_checkpoint.pth"  # 负载预测模型
```

#### 轮转模式说明

- **subtask**: 子任务动态 batch 调度（推荐）
  - 细粒度负载分配
  - 动态桶轮转
  - 适合负载差异大的场景

- **bucket**: 固定桶轮转
  - 按预定义桶分配
  - 适合负载相对均衡的场景

- **auto**: 自动选择
  - 启用负载均衡时使用 subtask
  - 未启用时使用简单轮转

### GPU 配置

在 `config.yaml` 中的 `fastgnas_backends` 部分：

```yaml
fastgnas_backends:
  num_workers: 8                  # Worker 数量
  device_GPU: 8                   # 可用 GPU 数量
  workers: [0,1,2,3,4,5,6,7]      # GPU 设备编号
```

## 与其他方案的对比

| 方案 | 负载均衡 | 桶式轮转 | RL搜索 | 适用场景 |
|------|---------|---------|--------|---------|
| **FastGNAS-RL** | ✅ | ✅ | ✅ | 完整方案，生产环境 |
| **Ca-RL** | ❌ | ❌ | ✅ | 数据并行基线 |
| **Co-RL** | ❌ | ❌ | ✅ | 计算并行基线 |

## 输出说明

实验完成后会生成：

1. **日志目录**: `experiment/FastGNAS-rl/log/run_YYYYMMDD_HHMMSS/`
   - 详细的训练日志
   - 指标 CSV 文件
   - 实验配置快照

2. **控制台输出**:
   - 每轮搜索的架构评估结果
   - Top K 模型排名
   - 最佳架构详情
   - 时间统计信息

## 关键特性

### 1. 负载预测
使用训练好的 `MloadPredictor` 预测每个架构的计算负载，实现智能分配。

### 2. 桶式轮转
将架构分组到桶中，按桶轮转训练，减少架构切换开销：
- Worker 1: [Bucket 0] → [Bucket 1] → [Bucket 0]
- Worker 2: [Bucket 1] → [Bucket 0] → [Bucket 1]

### 3. 动态调度
`Dispatcher` 根据负载实时调整架构分配，保证各 Worker 负载均衡。

### 4. RL 优化
控制器根据评估结果不断优化采样策略，逐步收敛到高性能架构。

## 常见问题

### Q1: 负载预测模型不存在怎么办？

**A**: 如果 `mload_checkpoint.pth` 不存在，系统会警告但继续运行，不使用负载预测功能。此时等同于关闭负载均衡。

### Q2: 如何禁用负载均衡？

**A**: 在 `config.yaml` 中设置：
```yaml
nas:
  balancing_strategy:
    enable: false
```

### Q3: 如何调整搜索轮次？

**A**: 修改 `dataset.yaml` 中对应数据集的 `num_search_rounds` 参数，或在 `config.yaml` 中设置默认值。

### Q4: 内存不足怎么办？

**A**:
1. 减少 `batch_size`
2. 减少 `samples_per_round`
3. 调整 `fan_out` 采样数量

## 实验最佳实践

1. **小数据集**（Cora, CiteSeer）:
   ```bash
   workers: 2-4
   samples_per_round: 2
   num_search_rounds: 10-20
   ```

2. **中等数据集**（Reddit, Flickr）:
   ```bash
   workers: 4-8
   samples_per_round: 3
   num_search_rounds: 30-50
   ```

3. **大数据集**（ogbn-products）:
   ```bash
   workers: 8
   samples_per_round: 2
   num_search_rounds: 40-60
   ```

## 相关文档

- 主项目文档：[../../README.md](../../README.md)
- 数据集配置：[../dataset.yaml](../dataset.yaml)
- CA 基线方案：[../Ca-rl/README.md](../Ca-rl/README.md)
