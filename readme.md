# 🚀 GraphNAS Distributed Training Platform

A modular and scalable distributed training platform for Graph Neural Architecture Search (GraphNAS). Designed to support CUDA-accelerated multi-GPU execution, architecture-aware batch caching, and pluggable RL-based NAS controllers.

---

## 📌 Key Features

- ✅ Synchronized multi-GPU training (supports any number of GPUs)
- ✅ Layer-sensitive subgraph sampling for each model
- ✅ GPU-side batch caching pool for efficient reuse
- ✅ Model spec + state rotation mechanism to avoid memory conflicts
- ✅ Decoupled controller/dispatcher for easy extension of RL strategies
- ✅ Clear structure, compatible with GraphNAS & GraphNAS++ ideas

---

## 🏗️ Project Structure

```
.
├── batching                      # Batching and data sampling modules
│   ├── sampler.py                # ▶ GNN subgraph sampler (e.g., k-hop)
│   └── simple_batch_manager.py   # ▶ Architecture-aware batch manager with GPU cache
│
├── configs                       # Configuration files
│   └── train.yaml                # ▶ Training parameters (lr, epoch, gpus, etc.)
│
├── core                          # Core control and scheduling modules
│   ├── controller.py             # ▶ Architecture search controller
│   └── dispatcher.py             # ▶ Distributed task dispatcher
│
├── data                          # Data loading and preprocessing tools
│   ├── data_utils.py             # ▶ Dataset loader (Cora, Pubmed, etc.)
│   └── partition.py              # ▶ Graph partitioning tool
│
├── dataset                       # Local dataset storage
│
├── environment.yml               # Conda environment definition file
├── readme.md                     # Project documentation
├── requirements.txt              # Pip requirements list
├── run.py                        # ▶ Main execution script
│
├── trainer                       # Training executor modules
│   ├── builder.py                # ▶ Dynamic GNN model builder
│   └── worker.py                 # ▶ GPU training worker (Ray Actor)
│
└── utils                         # Utility modules
    └── logger.py                 # ▶ Logging utility
```

```text
┌───────────────┐      1. Sample Arch Spec       ┌───────────────┐
│  Controller   │────────────────────────────▶│  Dispatcher   │
└───────────────┘                            └───────┬────────┘
                                                    │
                  ┌─────────────────────────────────┘
                  │ 2. Dispatch Task (spec + subgraph)
                  ▼
        ┌────────────────────┐       3. Get Batch        ┌──────────────────────┐
        │ TrainerWorker-0    │◀────────────────────────│ BatchManager-0       │
        │ (on GPU-0)         │                         │ (with Batch Cache)   │
        └──────────┬─────────┘                         └──────────────────────┘
                   │
                   ▼ 4. Train & Return Results
        ┌──────────┴─────────┐
        │ TrainerWorker-1    │
        │ (on GPU-1)         │
        └──────────┬─────────┘
                   │
                   ▼ 5. Rotate Model (spec + params)
        ┌──────────┴─────────┐
        │ TrainerWorker-2    │
        │ (on GPU-2)         │
        └────────────────────┘
```
**Description:**

-   **Controller**: Generates `arch_spec` for models to be trained.
-   **Dispatcher**: Constructs subgraphs based on architecture layers and schedules tasks to GPU workers.
-   **TrainerWorker**: Loads subgraph and model, builds mini-batches, and performs training.
-   **BatchManager**: Caches batch data in GPU memory, keyed by architecture spec.
-   **Rotation**: Dispatcher synchronizes and rotates model specs and parameters between GPUs after each round.

---

## 🚀 Quick Start

### 🧱 Environment Setup

Using Conda and Pip:

```bash
conda create -n graphnas python=3.11 -y
conda activate graphnas
pip install -r requirements.txt
```

Alternatively, from the environment file:

```bash
conda env create -f environment.yml
conda activate graphnas
```

▶️ Verify installation:

```bash
python -c "import torch; import torch_geometric; print(f'PyTorch: {torch.__version__}'); print(f'PyG: {torch_geometric.__version__}')"
```

---

## ⚙️ Dependencies

| Component         | Version             |
|-------------------|---------------------|
| Python            | 3.11.13             |
| PyTorch           | 2.5.1+cu121         |
| Torch Geometric   | 2.6.1               |
| Ray               | 2.47.1              |
| CUDA Toolkit      | 12.1                |

---

## 🧪 Datasets

Supports PyG's Planetoid datasets by default:

```python
from data.data_utils import get_dataset
dataset = get_dataset("Cora", root="./data")
```

This will automatically download and process datasets like {Cora, Citeseer, Pubmed}.

---

## 📔 Example Usage

Run the training process with a mock environment:

```bash
python run.py --config configs/train.yaml
```

**Arguments:**

-   `--config`: Path to the configuration file.

---

## 📜 License

Apache License 2.0
