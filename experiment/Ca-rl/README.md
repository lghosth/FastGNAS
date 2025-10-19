# CA-RL 实验

基于数据并行（Communication Avoidance）和强化学习控制器的图神经网络架构搜索实验。

## 核心特性

- ✅ 使用 `dataset.yaml` 统一管理数据集配置
- ✅ 自动从数据集配置读取训练参数（batch_size、fan_out、epochs等）
- ✅ 支持 RL 控制器采样架构
- ✅ 支持 Top-K 重训练

## 使用方法

### 基本运行

```bash
# 使用 Cora 数据集（默认）
python experiment/Ca-rl/run.py --dataset Cora

# 使用 PubMed 数据集
python experiment/Ca-rl/run.py --dataset PubMed

# 使用 Flickr 数据集
python experiment/Ca-rl/run.py --dataset Flickr

# 使用 ogbn-arxiv 数据集
python experiment/Ca-rl/run.py --dataset ogbn-arxiv
```

### 高级选项

```bash
# 指定 worker 数量
python experiment/Ca-rl/run.py --dataset Cora --workers 4

# 使用自定义配置文件
python experiment/Ca-rl/run.py --config my_config.yaml --dataset Cora
```

## 配置文件

### experiment/Ca-rl/config.yaml

主配置文件，包含：
- Worker 配置（数量、GPU 编号）
- 学习率、正则化参数
- RL 采样器配置（total_samples、reward_times）
- 日志配置

### experiment/dataset.yaml

数据集配置文件（项目级共享），包含：
- 数据集统计信息（节点数、边数、类别数等）
- 训练参数（batch_size、fan_out、steps_per_epoch等）
- 搜索配置（GNN 层数等）

## 配置说明

### 数据集配置自动合并

运行时，脚本会自动从 `dataset.yaml` 读取指定数据集的配置并合并到训练配置中：

```python
# dataset.yaml 中的配置
Cora:
  train_batch_size: 15
  eval_batch_size: 15
  fan_out: [-1, -1]
  num_rounds: 2
  epochs_per_round: 5
  num_classes: 7
```

会自动映射到：

```python
config['training'] = {
    'batch_size': 15,
    'eval_batch_size': 15,
    'fan_out': [-1, -1],
    'num_rounds': 2,
    'epochs_per_round': 5,
    'num_classes': 7
}
```

### RL 采样配置

在 `config.yaml` 中配置：

```yaml
ca_backends:
  sampler:
    total_samples: 610      # 总共采样的架构数量
    reward_times: 30        # 奖励回传次数（决定批次数）
    controller:
      search_strategy: "layer_wise"
      num_layers: 2
      learning_rate: 0.00035
      entropy_weight: 0.0001
```

## 输出结果

### 日志目录结构

```
experiment/Ca-rl/logs/
└── run_20250104_223000/
    ├── experiment_results.json   # 实验结果JSON
    ├── search_timeline.csv       # 搜索时间线数据（新增）
    └── graphnas.log               # 详细日志
```

### 搜索时间线数据 (search_timeline.csv)

记录整个架构搜索过程中的时间和准确率变化，用于绘制搜索结果图。

**CSV字段说明：**
- `cumulative_time`: 累计训练时间（秒）
- `global_best_accuracy`: 当前时刻的全局历史最优准确率
- `sampled_arch_count`: 当前已采样的架构数量
- `best_arch_index`: 产生历史最优准确率的架构序号
- `current_arch_name`: 当前训练的架构名称
- `current_round`: 当前架构的训练轮次
- `current_accuracy`: 当前轮次的准确率

**数据特点：**
- 横坐标：累计训练时间，每训练完一轮就记录一个时间点
- 纵坐标：全局历史最优准确率（只增不减，表示搜索过程中找到的最好结果）
- 采样轮次：记录产生最优结果的架构是第几个被采样的

**示例数据：**
```csv
cumulative_time,global_best_accuracy,sampled_arch_count,best_arch_index,current_arch_name,current_round,current_accuracy
5.23,0.7845,1,1,rl_arch_001,1,0.7845
10.45,0.7845,1,1,rl_arch_001,2,0.7823
15.67,0.8012,2,2,rl_arch_002,1,0.8012
20.89,0.8012,2,2,rl_arch_002,2,0.7998
```

### 实验结果

结果包含：
- 每个架构的训练历史（包含每轮的累计时间和历史最优）
- 最佳准确率和对应轮次
- 训练时间统计（详细的各阶段耗时分析）
- 搜索时间线数据（CSV格式，便于绘图）
- Top-K 重训练结果对比

## 注意事项

1. **数据集配置优先级**：dataset.yaml 中的参数会覆盖 config.yaml 中的对应参数
2. **GPU 设置**：通过 `workers` 字段指定使用的 GPU 编号
3. **随机种子**：通过 `experiment.random_seed` 设置，确保可复现性
4. **重训练轮数**：Top-K 重训练固定为 200 轮

## 添加新数据集

在 `experiment/dataset.yaml` 中添加新数据集配置：

```yaml
MyDataset:
  name: "MyDataset"
  num_classes: 10
  num_nodes: 50000
  train_batch_size: 256
  eval_batch_size: 512
  fan_out: [15, 10]
  num_rounds: 2
  epochs_per_round: 5
  steps_per_epoch: 10
  gnn_layers: 2
```

然后直接运行：

```bash
python experiment/Ca-rl/run.py --dataset MyDataset
```
