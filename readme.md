# FastGNAS: A Fast and Scalable Multi-GPU Framework for GNAS

This is the official implementation of our paper, **"FastGNAS: Accelerating and Scaling Graph Neural Architecture Search on Multi-GPUs via Ring-Based Model Migration"**.

Graph Neural Architecture Search (GNAS) automates the design of optimal Graph Neural Network (GNN) architectures but introduces significant computational overhead. Existing multi-GPU frameworks, which typically rely on data parallelism (incurring high synchronization costs) or architecture parallelism (struggling to scale with large datasets), are ill-suited for the parallel, multi-architecture workloads inherent to GNAS.

`FastGNAS` is a high-performance and scalable multi-GPU framework engineered to address these challenges. By introducing a novel training paradigm and advanced scheduling strategies, FastGNAS accelerates state-of-the-art baselines by an average of **3.14×** (up to **6.18×**) across diverse benchmarks, while maintaining competitive accuracy.

---

##  Key Features

-   **Hybrid Training Paradigm**: Innovatively integrates **data parallelism** and **architecture parallelism** with a **ring-based model migration** mechanism, enabling highly efficient and nearly synchronization-free training.
-   **Advanced Batch Management**: Implements **incremental batch storage** and **probabilistic batch reuse** strategies to minimize redundant batch generation and significantly reduce data loading overhead.
-   **Fine-grained Load Balancing**: Employs **exploratory task generation** and **predictive workload estimation** to achieve precise resource allocation and dynamic task scheduling across all candidate architectures.
-   **Modular and Extensible Design**: Features a **decoupled controller and dispatcher** architecture, allowing for the easy integration and extension of custom Reinforcement Learning (RL) search strategies.

---

## Project Structure

```
.
├── batching                      # Batching and data sampling modules
│   ├── sampler.py                # GNN subgraph sampler (e.g., k-hop)
│   └── simple_batch_manager.py   # Architecture-aware batch manager with GPU cache
│
├── configs                       # Configuration files
│   └── train.yaml                # Training parameters (lr, epoch, gpus, etc.)
│
├── core                          # Core control and scheduling modules
│   ├── controller.py             # Architecture search controller
│   └── dispatcher.py             # Distributed task dispatcher
│
├── data                          # Data loading and preprocessing tools
│   ├── data_utils.py             # Dataset loader (Cora, Pubmed, etc.)
│   └── partition.py              # Graph partitioning tool
│
├── dataset                       # Local dataset storage
│
├── environment.yml               # Conda environment definition file
├── readme.md                     # Project documentation
├── requirements.txt              # Pip requirements list
├── run.py                        # Main execution script
│
├── trainer                       # Training executor modules
│   ├── builder.py                # Dynamic GNN model builder
│   └── worker.py                 # GPU training worker (Ray Actor)
│
└── utils                         # Utility modules
    └── logger.py                 # Logging utility
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

## Quick Start

### Environment Setup

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

Verify installation:

```bash
python -c "import torch; import torch_geometric; print(f'PyTorch: {torch.__version__}'); print(f'PyG: {torch_geometric.__version__}')"
```

---

## Dependencies

| Component         | Version             |
|-------------------|---------------------|
| Python            | 3.11.13             |
| PyTorch           | 2.5.1+cu121         |
| Torch Geometric   | 2.6.1               |
| Ray               | 2.47.1              |
| CUDA Toolkit      | 12.1                |

---

## Datasets

Supports PyG's Planetoid datasets by default:

```python
from data.data_utils import get_dataset
dataset = get_dataset("Cora", root="./data")
```

This will automatically download and process datasets like {Cora, Citeseer, Pubmed}.

---

## Example Usage

Run the training process with a mock environment:

```bash
python run.py --config configs/train.yaml
```

**Arguments:**

-   `--config`: Path to the configuration file.

---

## License

Apache License 2.0
