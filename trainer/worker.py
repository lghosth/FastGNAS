# graphnas_dist/trainer/worker.py (modified)

import torch
import torch.nn.functional as F
from typing import Dict, Any, Tuple
import ray
import os

# Project module imports
from trainer.builder import ModelBuilder
from batching.hierarchical_cache_manager import HierarchicalCacheManager
from utils.logger import GraphNASLogger
from torch_geometric.data import Data


@ray.remote(num_gpus=1)
class TrainerWorker:
    """
    A stateful data owner.
    - Receives a fixed data partition upon initialization.
    - Trains a model on its local partition and returns updated parameters.
    """
    
    def __init__(self, worker_id: int, gpu_id: int, data_partition: Data):
        self.worker_id = worker_id
        self.gpu_id = gpu_id
        
        print(f"[Worker {worker_id}] CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES')}")
        print(f"[Worker {worker_id}] torch.cuda.is_available(): {torch.cuda.is_available()}")
        print(f"[Worker {worker_id}] ray.get_gpu_ids(): {ray.get_gpu_ids() if hasattr(ray, 'get_gpu_ids') else 'N/A'}")
        
        # Use the GPU device assigned by Ray
        if torch.cuda.is_available():
            self.device = torch.device('cuda:0')
            torch.cuda.set_device(self.device) # Ensure all CUDA ops are on this device
        else:
            self.device = torch.device('cpu')
        
        # Load data partition onto the assigned device
        self.data_partition = data_partition.to(self.device)
        
        # Initialize a batch manager for the data partition
        self.batch_manager = HierarchicalCacheManager(
            subgraph=data_partition,
            device=self.device,
            max_cache_pools=10, # Number of cache pools
            max_pool_size=50    # Size of each cache pool
        )
        
        self.logger = GraphNASLogger(name=f"worker_{worker_id}", track_gpu=True)
        
        # Reusable model placeholder to avoid re-creation
        self.model = None 
        self.current_arch_spec = None

        self.logger.logger.info(f"Worker {worker_id} on GPU {gpu_id} initialized with data partition: {self.data_partition}")

    def train_model_on_partition(self, 
                                 arch_spec: Dict[str, Any], 
                                 state_dict: Dict, 
                                 config: Dict[str, Any]
                                ) -> Tuple[float, Dict]:
        """
        Receives a model, trains it on the local data partition, and returns results.
        """
        # 1. Rebuild model if architecture has changed
        if self.model is None or str(arch_spec) != str(self.current_arch_spec):
            if not isinstance(self.data_partition.y, torch.Tensor):
                raise TypeError("data_partition.y must be a torch.Tensor to determine output_dim")
            output_dim = int(self.data_partition.y.max().item()) + 1

            self.model = ModelBuilder.build_model(
                arch_spec, 
                self.data_partition.num_node_features,
                output_dim,
                config
            ).to(self.device)
            self.current_arch_spec = arch_spec

        # 2. Load state_dict if provided (model transfer)
        if state_dict:
            self.model.load_state_dict(state_dict)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=config.get('lr', 0.01))
        
        # 3. Training loop
        epochs = config.get('epochs_per_rotation', 5)
        requested_bs = config.get('batch_size', 64)
        steps_per_epoch = config.get('steps_per_epoch', 1) 
        requested_layers = self._get_num_layers_from_arch(arch_spec)

        for epoch in range(1, epochs + 1):
            self.model.train()
            for step in range(steps_per_epoch):
                # Get a batch from the manager in each step
                batch = self.batch_manager.get_batch(
                    requested_bs=requested_bs,
                    requested_layers=requested_layers
                )
                if batch is None:
                    print(f"Warning: Failed to get batch at epoch {epoch}, step {step+1}.")
                    continue

                # Standard training logic
                optimizer.zero_grad()
                out = self.model(batch.x, batch.edge_index)

                if isinstance(out, torch.Tensor) and isinstance(batch.y, torch.Tensor):
                    loss = F.cross_entropy(out[:batch.batch_size], batch.y[:batch.batch_size])
                    loss.backward()
                    optimizer.step()
                else:
                    print(f"Warning: Output or label is not a tensor at epoch {epoch}, step {step+1}.")

        # 4. Evaluate and return results
        val_acc, _ = self._evaluate(self.model)
        # Move updated model parameters to CPU for transfer
        updated_state_dict = {k: v.cpu() for k, v in self.model.state_dict().items()}
        
        return val_acc, updated_state_dict

    def _get_num_layers_from_arch(self, arch_spec: Dict[str, Any]) -> int:
        """
        Extracts number of layers from arch_spec for both layer-wise and cell-based formats.
        """
        if 'layers' in arch_spec:
            return len(arch_spec['layers'])
        elif 'cell_spec' in arch_spec:
            return arch_spec.get('num_cells_to_stack', 2)
        else:
            print(f"Warning: Unknown arch format {arch_spec.keys()}, using default 2 layers.")
            return 2

    @torch.no_grad()
    def _evaluate(self, model) -> Tuple[float, float]:
        """Evaluates on the local validation set."""
        model.eval()
        # Note: Evaluation is on the entire local partition
        out = model(self.data_partition.x, self.data_partition.edge_index)
        pred = out.argmax(dim=1)

        if isinstance(self.data_partition.val_mask, torch.Tensor) and \
           isinstance(self.data_partition.y, torch.Tensor):
            val_mask_sum = self.data_partition.val_mask.sum().item()
            if val_mask_sum == 0: return 0.0, 0.0
            val_correct = (pred[self.data_partition.val_mask] == self.data_partition.y[self.data_partition.val_mask]).sum()
            val_acc = int(val_correct) / val_mask_sum
            return val_acc, 0.0 
        else:
            return 0.0, 0.0
