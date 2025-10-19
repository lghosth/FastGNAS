# Ad_OD-RL Experiment

Architecture-Parallel experiment using `parallel_v2` backend.

## Features

### Backend: ArchParallelBackendV2
- **Parallel Strategy**: Architecture-parallel (multiple architectures trained simultaneously)
- **Data Loading**: Automatic dataset loading
- **Setup**: No data parameter required for `setup_distributed_training()`

### Time Statistics
Implements comprehensive time tracking similar to Ca-rl:

```
Time Statistics Summary
================================================================================
Total elapsed time: X.XXs
  |-- Architecture sampling time: X.XXs (XX.X%)
  |-- Architecture training time: X.XXs (XX.X%)
  `-- Architecture evaluation time: X.XXs (XX.X%)
```

### Output Files
- `search_timeline.csv`: Time series of search progress
- `experiment_results.json`: Complete experiment results

## Usage

```bash
# Basic usage
python experiment/Ad_OD-rl/run.py --dataset Cora

# With custom workers
python experiment/Ad_OD-rl/run.py --dataset Flickr --workers 4

# Large dataset
python experiment/Ad_OD-rl/run.py --dataset Reddit --workers 8
```

## Configuration

Edit `config.yaml` to customize:
- `total_samples`: Number of architectures to evaluate
- `epochs_per_round`: Training epochs per round
- `num_rounds`: Total training rounds
- `workers`: GPU indices to use

## Time Breakdown

The experiment tracks three main time components:

1. **Sampling Time**: RL controller architecture sampling
2. **Training Time**: Pure model training (forward + backward)
3. **Evaluation Time**: Model validation on eval set

Total time may include additional overhead from:
- Ray initialization
- Data loading
- Inter-process communication
