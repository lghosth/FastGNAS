import time
from typing import List, Deque
from collections import deque
from torch_geometric.data import Data
import random

class CachePool:
    """
    Manages a cache pool for a specific batch size.
    
    This pool contains a list of mini-batches with varying numbers of layers
    generated for this batch_size and maintains metadata for LRU eviction.
    """
    def __init__(self, batch_size: int, max_pool_size: int = 50):
        """
        Initializes a cache pool.

        Args:
            batch_size (int): The batch size this cache pool corresponds to.
            max_pool_size (int): The maximum number of batches the pool can store.
        """
        self.batch_size = batch_size
        self.max_pool_size = max_pool_size
        
        # Use a deque for efficient appends and pops from both ends
        self.batch_list: Deque[Data] = deque(maxlen=self.max_pool_size)
        
        # Initialize metadata
        self.last_used_timestamp: float = time.time()
        self.batch_count: int = 0
        self.hits: int = 0
        self.misses: int = 0

    def add(self, batch: Data):
        """Adds a new mini-batch to the pool."""
        # If the deque is full, the oldest item is automatically removed
        self.batch_list.append(batch)
        self.batch_count = len(self.batch_list)
        self.touch()

    def get_random_batch(self) -> Data | None:
        """Gets a random batch from the pool and updates the timestamp."""
        if not self.batch_list:
            return None
        
        self.touch()
        return random.choice(self.batch_list)

    def replace_batch(self, old_batch: Data, new_batch: Data):
        """Replaces an old batch with a new, higher-layer one."""
        try:
            index = self.batch_list.index(old_batch)
            self.batch_list[index] = new_batch
            self.touch()
        except ValueError:
            # If the old batch doesn't exist (e.g., evicted), just add the new one
            self.add(new_batch)

    def touch(self):
        """Updates the last used timestamp."""
        self.last_used_timestamp = time.time()

    def __len__(self) -> int:
        """Returns the number of cached batches in the pool."""
        return len(self.batch_list)

    def __repr__(self) -> str:
        return (
            f"CachePool(batch_size={self.batch_size}, "
            f"count={self.batch_count}, "
            f"last_used={self.last_used_timestamp:.0f})"
        )
