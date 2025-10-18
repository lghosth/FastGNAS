# graphnas_dist/batching/simple_batch_manager.py

import torch
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader
from collections import OrderedDict
from typing import Dict, Any
import sys
import os

class SimpleBatchManager:
    """
    A batch manager that follows a strict two-level sampling process.
    Features:
    1.  Receives a pre-constructed, smaller k-hop subgraph as its operational "world".
    2.  Uses PyG's NeighborLoader for efficient mini-batch sampling on this subgraph.
    3.  (Future Extension) Caching can be refined here. Current focus is on reusing loaders.
    """

    def __init__(self, 
                 k_hop_subgraph: Data, 
                 device: torch.device, 
                 cache_size: int = 5):
        """
        Args:
            k_hop_subgraph (Data): The k-hop subgraph extracted from the global graph, expected on the target device.
            device (torch.device): The GPU device bound to the worker.
            cache_size (int): Cache size for Loader instances.
        """
        # Note: self.graph is the k-hop subgraph, not the full graph.
        self.graph = k_hop_subgraph
        self.device = device
        self.loader_cache = OrderedDict()
        self.cache_size = cache_size

    def get_loader(self, batch_config: Dict[str, Any], num_layers: int) -> NeighborLoader:
        """
        Gets an iterable mini-batch loader for the current k-hop subgraph.

        Args:
            batch_config (Dict[str, Any]): Config containing 'batch_size', etc.
            num_layers (int): Number of model layers, determining sampling hops.

        Returns:
            NeighborLoader: A configured loader for generating mini-batches.
        """
        batch_size = batch_config.get("batch_size", 512)
        
        # Cache key is determined by batch_size and number of layers.
        cache_key = f"bs_{batch_size}-layers_{num_layers}"

        if cache_key in self.loader_cache:
            self.loader_cache.move_to_end(cache_key)
            return self.loader_cache[cache_key]

        # Sample within this k-hop subgraph.
        # Note: input_nodes is the train_mask within the subgraph.
        loader = NeighborLoader(
            data=self.graph,
            num_neighbors=[-1] * num_layers,
            batch_size=batch_size,
            input_nodes=self.graph.train_mask, # Use subgraph's train mask
            shuffle=True
        )

        self.loader_cache[cache_key] = loader
        if len(self.loader_cache) > self.cache_size:
            self.loader_cache.popitem(last=False)
            
        return loader

# ====================
#  Independent Test Module
# ====================
if __name__ == '__main__':
    print("===== Running Test for simple_batch_manager.py =====")
    
    # Import necessary modules
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from data.data_utils import get_dataset
    
    # --- Test Case 1: Creating a simulated k-hop subgraph ---
    print("\n[Test Case 1] Creating a simulated k-hop subgraph...")
    try:
        # Create simple mock graph data
        num_nodes, num_features, num_classes = 100, 16, 3
        x = torch.randn(num_nodes, num_features)
        edge_index = torch.randint(0, num_nodes, (2, num_nodes * 2))
        edge_index = edge_index[:, edge_index[0] != edge_index[1]] # Remove self-loops
        y = torch.randint(0, num_classes, (num_nodes,))
        
        # Create masks
        train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        train_mask[:30] = True
        val_mask = torch.zeros(num_nodes, dtype=torch.bool)
        val_mask[30:50] = True
        test_mask = torch.zeros(num_nodes, dtype=torch.bool)
        test_mask[50:70] = True
        
        mock_subgraph = Data(x=x, edge_index=edge_index, y=y, train_mask=train_mask, val_mask=val_mask, test_mask=test_mask)
        
        print(f"  - Mock subgraph created successfully:")
        print(f"    - Nodes: {mock_subgraph.num_nodes}, Edges: {mock_subgraph.num_edges}")
        print(f"    - Features: {mock_subgraph.x.shape[1] if mock_subgraph.x is not None else 'None'}")
        print(f"    - Train nodes: {mock_subgraph.train_mask.sum().item() if mock_subgraph.train_mask is not None else 0}")
        
    except Exception as e:
        print(f"  Failed to create mock subgraph: {e}")
        sys.exit(1)
    
    # --- Test Case 2: Initializing SimpleBatchManager ---
    print("\n[Test Case 2] Initializing SimpleBatchManager...")
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"  - Using device: {device}")
        
        mock_subgraph = mock_subgraph.to(device)
            
        batch_manager = SimpleBatchManager(k_hop_subgraph=mock_subgraph, device=device, cache_size=3)
        
        print(f"  - BatchManager initialized successfully.")
        print(f"    - Cache size: {batch_manager.cache_size}")
        
    except Exception as e:
        print(f"  Failed to initialize BatchManager: {e}")
        sys.exit(1)
    
    # --- Test Case 3: Testing loader creation and caching ---
    print("\n[Test Case 3] Testing loader creation and caching...")
    try:
        test_configs = [
            ({'batch_size': 16}, 2, "Small batch, 2 layers"),
            ({'batch_size': 32}, 2, "Medium batch, 2 layers"),
            ({'batch_size': 16}, 3, "Small batch, 3 layers"),
        ]
        
        for i, (batch_config, num_layers, description) in enumerate(test_configs):
            print(f"  - Testing {description}...")
            loader1 = batch_manager.get_loader(batch_config, num_layers)
            print(f"    - Created, cache items: {len(batch_manager.loader_cache)}")
            loader2 = batch_manager.get_loader(batch_config, num_layers)
            if loader1 is loader2:
                print(f"    - Cache hit successful.")
            else:
                print(f"    - Cache miss.")
        
        print(f"  - Final cache items: {len(batch_manager.loader_cache)}")
        print(f"  - Cache keys: {list(batch_manager.loader_cache.keys())}")
        
    except Exception as e:
        print(f"  Loader creation test failed: {e}")
        sys.exit(1)
    
    # --- Test Case 4: Testing cache size limit ---
    print("\n[Test Case 4] Testing cache size limit...")
    try:
        for i in range(5):
            batch_config = {'batch_size': 8 + i * 4}
            num_layers = 2 + (i % 2)
            loader = batch_manager.get_loader(batch_config, num_layers)
            print(f"  - Added config {i+1}: batch_size={batch_config['batch_size']}, layers={num_layers}. Cache items: {len(batch_manager.loader_cache)}")
        
        assert len(batch_manager.loader_cache) <= batch_manager.cache_size
        print(f"  - Cache size limit test passed.")
        
    except Exception as e:
        print(f"  Cache size limit test failed: {e}")
        sys.exit(1)
    
    # --- Test Case 5: Testing actual data iteration ---
    print("\n[Test Case 5] Testing actual data iteration...")
    try:
        loader = batch_manager.get_loader({'batch_size': 8}, 2)
        batch_count = 0
        for batch in loader:
            batch_count += 1
            print(f"  - Batch {batch_count}: Nodes={batch.num_nodes}, Edges={batch.num_edges}")
            if batch_count >= 3:
                break
        print(f"  - Data iteration test passed.")
        
    except Exception as e:
        print(f"  Data iteration test failed: {e}")
        sys.exit(1)
    
    print("\n===== All tests for simple_batch_manager.py completed successfully! =====")
