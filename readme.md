# 🚀 GraphNAS Distributed Training Platform

A modular and scalable distributed training platform for Graph Neural Architecture Search (GraphNAS). Designed to support CUDA-accelerated multi-GPU execution, architecture-aware batch caching, and pluggable RL-based NAS controllers.

当前阶段：Phase 1 - GNN 分布式训练组件构建  
后续阶段：Phase 2 - 搜索策略集成（RL Controller、Search Space 等）

---

## 📌 项目亮点特性

- ✅ 多 GPU 同步训练（支持任意数量 GPU）
- ✅ 每个子模型按其层数建构专属训练子图（层敏感采样）
- ✅ GPU 侧 batch 缓存池（SimpleBatchManager），高速复用采样结构
- ✅ 模型 spec + state 参数轮转机制，避免显存冲突
- ✅ 控制器/调度器与训练解耦，方便扩展强化学习策略
- ✅ 结构清晰，兼容 GraphNAS 与 GraphNAS++ 分布式建模思想

---

## 🏗️ 项目结构概览

```
.
├── batching                      # 批处理(Batch)与数据采样模块
│   ├── sampler.py                # ▶ GNN子图采样器 (例如实现k-hop邻居采样)
│   └── simple_batch_manager.py   # ▶ 架构感知的批处理管理器 (含GPU缓存池)
│
├── configs                       # 配置文件目录
│   └── train.yaml                # ▶ 训练运行参数 (如学习率, epoch, GPU数量等)
│
├── core                          # 核心控制与调度模块
│   ├── controller.py             # ▶ 架构搜索控制器 (Phase 2实现RL策略, 当前为模拟)
│   └── dispatcher.py             # ▶ 分布式任务调度器 (分发任务, 管理模型轮转)
│
├── data                          # 数据加载与预处理工具
│   ├── data_utils.py             # ▶ 数据集加载器 (自动下载Cora, Pubmed等)
│   └── partition.py              # ▶ 图数据划分工具 (未来使用)
│
├── dataset                       # 本地数据集存储目录
│   ├── Cora                      # ▶ Cora数据集文件
│   └── PubMed                    # ▶ PubMed数据集文件
│
├── environment.yml               # Conda环境定义文件 (用于一键创建环境)
├── readme.md                     # 项目说明文档
├── requirements.txt              # Pip依赖库列表文件
├── run.py                        # ▶ 主运行入口脚本
├── test-run.py                   # ▶ 功能测试或快速验证脚本
│
├── trainer                       # 训练执行器模块
│   ├── builder.py                # ▶ 模型构建器 (根据架构描述动态生成GNN模型)
│   └── worker.py                 # ▶ GPU训练工作单元 (Ray Actor, 负责在单卡上训练)
│
└── utils                         # 通用工具模块
    └── logger.py                 # ▶ 日志记录工具
```

```text
┌───────────────┐      1. 采样架构 Spec        ┌───────────────┐
│   Controller  │────────────────────────────▶│  Dispatcher   │
└───────────────┘                            └───────┬────────┘
                                                    │
                  ┌─────────────────────────────────┘
                  │ 2. 分发训练任务 (spec + 子图)
                  ▼
        ┌────────────────────┐       3. 获取Batch      ┌──────────────────────┐
        │ TrainerWorker-0    │◀────────────────────────│ BatchManager-0       │
        │ (绑定 GPU-0)       │                         │ (内含Batch缓存)       │
        └──────────┬─────────┘                         └──────────────────────┘
                   │
                   ▼ 4. 训练并返回结果（val acc、参数）
        ┌──────────┴─────────┐
        │ TrainerWorker-1    │
        │ (绑定 GPU-1)       │
        └──────────┬─────────┘
                   │
                   ▼ 5. 模型轮转（结构 + 参数）
        ┌──────────┴─────────┐
        │ TrainerWorker-2    │
        │ (绑定 GPU-2)       │
        └────────────────────┘
```
📘 说明：

- Controller：生成待训练的子模型结构 arch_spec  
- Dispatcher：根据架构层数构造子图，并调度给各 GPU Worker  
- TrainerWorker：加载子图+模型结构，自主构建 mini-batch 并训练  
- BatchManager：维护以架构规格为 key 的 batch 数据显存缓存  
- 轮转机制：Dispatcher 每轮同步模型结构和参数，在 GPU 间轮换执行  

---

## 🚀 快速开始（Quick Start）

### 🧱 构建依赖环境（推荐）

使用 Conda 创建环境 + pip 安装依赖：

```bash
conda create -n RTO python=3.11.13 -y
conda activate RTO
pip install -r requirements.txt
```

或使用一键脚本：

```bash
conda env create -f environment.yml
conda activate RTO
```

▶️ 验证安装：

```bash
python -c "import torch; import torch_geometric; print(f'PyTorch: {torch.__version__}'); print(f'PyG: {torch_geometric.__version__}')"
```

---

## ⚙️ 技术依赖描述

| 组件             | 版本               |
|------------------|--------------------|
| Python           | 3.11.13            |
| PyTorch          | 2.5.1 + cu121      |
| Torch Geometric  | 2.6.1              |
| torch-scatter    | 2.1.2 + pt25cu121  |
| pyg-lib          | 0.4.0 + pt25cu121  |
| Ray              | 2.47.1             |
| CUDA Toolkit     | 12.1               |
| PyYAML           | 6.0.2              |
| Pydantic         | 2.11.7             |
| tqdm             | 4.67.1             |

🔗 安装 PyG 依赖需使用官方 index:

https://data.pyg.org/whl/torch-2.5.1+cu121.html

---

## 🛠️ 项目目标与开发阶段

### ✅ Phase 1：训练执行后端（进行中）

核心目标：完成分布式训练组件，支持模型轮转 + 层敏感 batch 构建 + 缓存提速：

- [x] 基于架构 spec 按需构建多跳子图
- [x] Dispatcher 同步分配任务，支持 ring 模型轮转策略
- [x] Worker 支持显存 batch 缓存（使用 hash 识别复用）
- [x] 每轮训练完后轮换模型到下一个 GPU，并 warm-load 参数
- [x] 记录训练状态（val acc、缓存命中率）

### 🔬 Phase 2：搜索控制器及策略集成

模块任务包括：

- Controller：强化学习采样器（如 LSTM / PPO）
- Search Space：聚合函数/采样策略/激活函数等空间定义
- NSGA/RL/Proxyless-GNN 模型集成（可选接口）
- 支持 GraphNAS / GraphNAS++ / GraphNAP 等策略（兼容 Cell-based）

---

## 🧪 数据集处理

默认支持 PyG Planetoid 数据集：

```python
from data.data_utils import get_dataset
dataset = get_dataset("Cora", root="./data")
```

自动下载并解析 {Cora, Citeseer, Pubmed...} 数据。

---

## 📔 示例运行

运行最简训练流程（mock 环境）：

```bash
python run.py --config configs/train.yaml
```

支持参数：

- --config: 配置文件路径
- --test: 启用测试模式
- --gpus: 使用 GPU 数量

---

## 📂 文件说明

| 文件           | 功能                     |
|----------------|--------------------------|
| run.py         | 项目主调度入口           |
| dispatcher.py  | 同步调度器（轮训 + 分发） |
| worker.py      | 单卡 Actor 模型训练器     |
| batch_manager.py| 根据架构信息构建 batch   |
| controller.py  | 强化学习控制器（后续阶段）|

---

## 🧠 相关项目与参考

参考论文：

- GraphNAS: https://arxiv.org/abs/1904.09981
- GraphNAS++: https://ieeexplore.ieee.org/document/9815284
- torch-geometric: https://pytorch-geometric.readthedocs.io

原始 GraphNAS 项目仓库（手工架构搜索 + RL 控制器）：  
https://github.com/GraphNAS/GraphNAS-simple

---

## 📜 License

开源协议：Apache License 2.0  
可广泛用于科研、教学与商业应用，欢迎扩展！

---

