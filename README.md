# 🚀 FastGNAS: Accelerating and Scaling Graph Neural Architecture Search with Ring-Based Model Migration

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.5.1+-red.svg)](https://pytorch.org/)
[![Ray](https://img.shields.io/badge/Ray-2.47.1+-green.svg)](https://ray.io/)

FastGNAS is a fast and scalable multi-GPU framework for Graph Neural Architecture Search (GNAS). It combines data and architecture parallelism via a novel **ring-based model migration** mechanism to enable synchronization-free training, provides a caching layer with incremental storage and probabilistic batch reuse to remove data redundancy, and applies fine-grained load balancing through exploratory task generation and predictive workload estimation. It accelerates state-of-the-art baselines by an average of **3.14×** (up to **6.18×**) on diverse benchmarks while achieving competitive accuracy.

## ✨ Key Features

- **Ring-based model migration** — hybrid data + architecture parallelism with zero synchronization overhead
- **Intelligent batch management** — incremental batch storage, probabilistic reuse, and hierarchical caching
- **Fine-grained load balancing** — exploratory task generation and ML-based predictive load estimation
- **Multiple search strategies** — reinforcement learning (RL), Shapley value evaluation, and more
- **Modular design** — clean layering that is easy to extend and maintain

## 📦 Installation

Requirements: Python 3.11+, CUDA 12.1+, NVIDIA GPU (multi-GPU recommended), 32GB+ RAM.

```bash
conda create -n RTO python=3.11
conda activate RTO

pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
pip install torch_geometric==2.6.1 ray==2.47.1 pyyaml==6.0.2 pydantic==2.11.7

# Verify
python -c "import torch; import torch_geometric; import ray; print('OK')"
```

## 🎯 Usage

```bash
# Main entry
python run.py --config configs/train.yaml
python run.py --config configs/train.yaml --dataset Cora --test

# Complete FastGNAS solution
python experiment/FastGNAS-rl/run.py --dataset Cora
python experiment/FastGNAS-rl/run.py --dataset ogbn-arxiv   # requires more GPUs

# Train the Mload load-prediction model
python train_mload.py --data dataset/Mload_dataset/fastgnas-a6000-batch.yaml
```

### Experiment Types

| Experiment | Description |
|------------|-------------|
| `FastGNAS-rl` | Complete FastGNAS solution (best performance) |
| `Ca-rl` / `Co-rl` / `Ad-rl` | Centralized / collaborative / adaptive search variants |
| `*-shapley` | Shapley value-based evaluation |

### Configuration (`configs/train.yaml`)

```yaml
training:
  dataset: "ogbn-products"
  batch_size: 256
  epochs_per_rotation: 5
  fan_out: [15, 10]

nas:
  num_search_rounds: 5
  samples_per_round: 2
  gnn_layers: 2
  search_strategy: "layer_wise"
  balancing_strategy:
    enable: true
    batch_size: 24
    tolerance: 0.1
    rotation_mode: "subtask"
```

Supported datasets: Cora, PubMed, Citeseer (small); ogbn-arxiv, ogbn-products, Flickr (large); any PyG-format graph.

## 📁 Project Structure

```
├── run.py / train_mload.py   # Main entry & load-prediction training
├── configs/                  # train.yaml
├── core/                     # Controller, load balancing, dispatching & scheduling
├── trainer/                  # Ray GPU workers, dynamic GNN builder
├── batching/                 # Batch management & caching
├── baseline/                 # Baseline methods
├── data/                     # Dataset utilities & partitioning
├── experiment/               # Experiment entry scripts (FastGNAS-rl, Ca-rl, ...)
├── tools/                    # Monitoring & analysis scripts
└── utils/                    # Common utilities
```

## 📊 Monitoring & Testing

```bash
# Monitoring & visualization
python tools/gpu_log_summary.py logs/console_out/
python utils/plot_search_timeline.py logs/logings/run_*/

# Unit tests
python -m pytest test_utils/
```

## 🤝 Contributing

1. Fork and create a feature branch
2. Format with `black . && isort .`, then open a Pull Request

## 📖 Citation

If you use FastGNAS in your research, please cite our paper:

```bibtex
@article{10.1145/3802059,
author = {Song, Zhen and Li, Hao and Li, Tianyi and Gu, Yu and Li, Yushuai and Zhang, Yanfeng and Jensen, Christian S. and Cui, Lizhen and Yu, Ge},
title = {FastGNAS: Accelerating and Scaling Graph Neural Architecture Search on Multi-GPUs via Ring-Based Model Migration},
year = {2026},
issue_date = {June 2026},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
volume = {4},
number = {3},
url = {https://doi.org/10.1145/3802059},
doi = {10.1145/3802059},
abstract = {Graph Neural Architecture Search (GNAS), a subdomain of Automated Machine Learning (AutoML), aims to automate the design of graph neural network (GNN) architectures. However, GNAS is an inherently data-intensive process, as it calls for the training and evaluation of numerous candidate GNN architectures, leading to massive data access redundancy. Existing multi-GPU GNAS frameworks typically use data parallelism, which incurs high synchronization costs, or architecture parallelism, which struggles to scale to large datasets. We propose FastGNAS, a fast and scalable multi-GPU framework designed to tackle the data management challenges underlying GNAS.First, FastGNAS introduces a hybrid parallel framework that combines data and architecture parallelism, enabled by a novel ring-based model migration. Specifically, it offers an efficient data flow that exploits scalable data parallelism and isolated architecture parallelism to enable synchronization-free processing. Next, to address data redundancy, FastGNAS offers a specialized caching layer for intermediate data products, implemented through advanced batch management strategies including incremental storage and a probabilistic reuse policy. Finally, FastGNAS employs fine-grained load balancing and scheduling via exploratory task generation and predictive workload estimation, enabling purposeful resource allocation across candidate architectures. Extensive experiments show that FastGNAS can accelerate state-of-the-art baselines by an average of 3.14{\texttimes} (up to 6.18{\texttimes}) on diverse benchmarks, while achieving competitive accuracy.},
journal = {Proc. ACM Manag. Data},
month = may,
articleno = {182},
numpages = {27},
keywords = {graph neural architecture search, multi-gpu training, task decomposition, synchronization-free framework}
}
```

## 📄 License

This project is licensed under the Apache License 2.0. See the [LICENSE](LICENSE) file for details.
