import random
import torch
from typing import Dict
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader

from batching.cache_pool import CachePool

class HierarchicalCacheManager:
    """
    Implements a hierarchical cache with dynamic and incremental sampling.

    This manager uses a probabilistic model to balance cache utilization and new sampling,
    and supports incremental sampling on cached lower-layer batches to meet higher-layer requests.
    """
    def __init__(self,
                 subgraph: Data,
                 device: torch.device,
                 max_cache_pools: int = 10,
                 max_pool_size: int = 50):
        """
        Initializes the hierarchical cache manager.

        Args:
            subgraph (Data): The graph partition held by the current worker for sampling.
            device (torch.device): The GPU device.
            max_cache_pools (int): Max number of CachePool types the global cache can hold.
            max_pool_size (int): Max number of batches each CachePool can hold.
        """
        # TODO: Review/Refine
        self.subgraph = subgraph
        self.device = device
        self.max_cache_pools = max_cache_pools
        self.max_pool_size = max_pool_size
        
        # Global cache pool, keys are batch_size, values are CachePool objects
        self.global_cache: Dict[int, CachePool] = {}
        
        # Pre-calculate average degree of the subgraph for the probability model
        if (self.subgraph.edge_index is not None and 
            self.subgraph.num_nodes is not None and 
            self.subgraph.num_nodes > 0):
            self.avg_degree = self.subgraph.num_edges / self.subgraph.num_nodes
        else:
            self.avg_degree = 1.0  # Avoid division by zero

    def get_batch(self, requested_bs: int, requested_layers: int) -> Data:
        """
        Gets a mini-batch based on the requested batch size and number of layers.
        This is the main entry point for this class.
        """

        cache_pool = self.global_cache.get(requested_bs)

        # Case A: Batch size miss
        if cache_pool is None:
            return self._handle_bs_miss(requested_bs, requested_layers)
        # Case B: Batch size hit
        else:
            return self._handle_bs_hit(cache_pool, requested_bs, requested_layers)

    def _handle_bs_miss(self, req_bs: int, req_layers: int) -> Data:
        """Handles a cache miss for a given batch size."""

        if len(self.global_cache) >= self.max_cache_pools:
            self._evict_pool()
        

        new_pool = CachePool(batch_size=req_bs, max_pool_size=self.max_pool_size)
        self.global_cache[req_bs] = new_pool
        

        return self._full_sample_and_cache(new_pool, req_bs, req_layers)

    def _handle_bs_hit(self, pool: CachePool, req_bs: int, req_layers: int) -> Data:
        """Handles a cache hit for a given batch size."""
        pool.touch() # Update timestamp
        

        # Ensure denominator is not zero and probability does not exceed 1.0
        p_hit = min(len(pool) / self.avg_degree, 1.0) if self.avg_degree > 0 else 0.0


        if random.random() < p_hit:
            # If hit, proceed to hierarchical matching
            return self._hierarchical_match(pool, req_layers)
        else:
            # If probabilistic miss, perform a full sample
            return self._full_sample_and_cache(pool, req_bs, req_layers)

    def _hierarchical_match(self, pool: CachePool, req_layers: int) -> Data:
        """Hierarchical matching and incremental sampling logic."""
        cached_batch = pool.get_random_batch()
        
        # If pool is empty, fall back to full sampling
        if cached_batch is None:
            return self._full_sample_and_cache(pool, pool.batch_size, req_layers)

        # Assume the batch object stores its number of layers
        # Note: PyG batches do not have this by default; it must be added during sampling
        cached_layers = getattr(cached_batch, 'num_layers', 1)

        # Case 1: L_req <= L_cache (downgrade or same-level use)
        if req_layers <= cached_layers:
            return cached_batch
        
        # Case 2: L_req > L_cache (upgrade use)
        else:
            # Trigger incremental sampling
            return self._incremental_sample_and_cache(pool, cached_batch, req_layers)

    def _full_sample_and_cache(self, pool: CachePool, bs: int, layers: int) -> Data:
        """Performs a full neighbor sample and stores the result in the cache pool."""
        # The length of num_neighbors list for NeighborLoader should match the number of layers
        num_neighbors = [-1] * layers
        
        # Create loader and get one batch
        loader = NeighborLoader(
            self.subgraph,
            num_neighbors=num_neighbors,
            batch_size=bs,
            shuffle=True
        )
        
        try:
            new_batch = next(iter(loader))
            # IMPORTANT: Attach layer info to the batch object for hierarchical matching
            new_batch.num_layers = layers
            new_batch.to(self.device)
            pool.add(new_batch)
            return new_batch
        except StopIteration:
            # If the graph is too small for sampling, raise an error
            raise ValueError(f"Could not sample a batch of size {bs} from the subgraph.")

    def _incremental_sample_and_cache(self, pool: CachePool, old_batch: Data, new_layers: int) -> Data:
        """Performs incremental sampling and replaces the old batch with the new one."""
        # NOTE: This is a complex process; the following is a simplified implementation.
        # 1. Get original seed nodes (must be saved on initial sample)
        # The first batch_size nodes in a NeighborLoader batch are the seed nodes.
        seed_nodes = old_batch.n_id[:old_batch.batch_size]
        
        # 2. Resample for these specific nodes with the new number of layers
        num_neighbors = [-1] * new_layers
        loader = NeighborLoader(
            self.subgraph,
            num_neighbors=num_neighbors,
            input_nodes=seed_nodes, # Specify seed nodes for sampling
            batch_size=len(seed_nodes)
        )
        
        try:
            new_batch = next(iter(loader))
            new_batch.num_layers = new_layers
            new_batch.to(self.device)
            
            # 3. Replace the old batch in the cache pool
            pool.replace_batch(old_batch, new_batch)
            return new_batch
        except StopIteration:
            # If incremental sampling fails, return the old batch as a fallback
            return old_batch

    def _evict_pool(self):
        """Smart eviction policy: Evict the least recently used pool with the fewest batches."""
        if not self.global_cache:
            return

        # Find all pools and sort them by eviction criteria
        # Criteria: Sort by last used time (asc), then by number of batches (asc)
        pools_to_evict = sorted(
            self.global_cache.items(),
            key=lambda item: (item[1].last_used_timestamp, len(item[1]))
        )
        
        # Get the key of the least important pool and remove it from the global cache
        eviction_key = pools_to_evict[0][0]
        del self.global_cache[eviction_key]
        print(f"Cache Eviction: Removed pool for batch_size={eviction_key}")
