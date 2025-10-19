import time
import torch
from typing import List, Deque
from collections import deque
from torch_geometric.data import Data

def _format_bytes(num_bytes) -> str:
    try:
        size = float(num_bytes)
    except (TypeError, ValueError):
        return str(num_bytes)
    sign = '-' if size < 0 else ''
    size = abs(size)
    units = ('B', 'KB', 'MB', 'GB', 'TB', 'PB')
    for unit in units:
        if size < 1024.0 or unit == units[-1]:
            value_str = f'{size:.2f}'.rstrip('0').rstrip('.')
            return f"{sign}{value_str or '0'}{unit}"
        size /= 1024.0
    return f'{sign}{num_bytes}B'

def _tensor_nbytes(tensor: torch.Tensor) -> int:
    if not isinstance(tensor, torch.Tensor):
        return 0
    return tensor.element_size() * tensor.numel()

class EdgeListUnit:

    def __init__(self, edge_index: torch.Tensor, n_id: torch.Tensor, batch_size: int, num_layers: int):
        self.edge_index = edge_index
        self.n_id = n_id
        self.batch_size = batch_size
        self.num_layers = num_layers

    def __repr__(self) -> str:
        return f'EdgeListUnit(num_layers={self.num_layers}, edge_index_shape={self.edge_index.shape}, n_id_shape={self.n_id.shape}, batch_size={self.batch_size})'

class MiniBatchCacheObject:

    def __init__(self, target_vertices: torch.Tensor):
        self.target_vertices = target_vertices
        self.elus: List[EdgeListUnit] = []

    def add_elu(self, elu: EdgeListUnit):
        self.elus.append(elu)
        self.elus.sort(key=lambda x: x.num_layers)

    def get_elu(self, layer_idx: int) -> EdgeListUnit | None:
        for elu in self.elus:
            if elu.num_layers == layer_idx:
                return elu
        return None

    @property
    def num_elus(self) -> int:
        return len(self.elus)

    @property
    def max_cached_layers(self) -> int:
        if not self.elus:
            return 0
        return self.elus[-1].num_layers

    @property
    def batch_size(self) -> int:
        return self.target_vertices.size(0)

    def __repr__(self) -> str:
        return f'MiniBatchCacheObject(batch_size={self.batch_size}, num_elus={self.num_elus}, max_cached_layers={self.max_cached_layers})'

class CachePool:

    def __init__(self, batch_size: int, max_pool_size: int=50):
        self.batch_size = batch_size
        self.max_pool_size = max_pool_size
        self.batch_list: Deque[MiniBatchCacheObject] = deque(maxlen=self.max_pool_size)
        self.last_used_timestamp: float = time.time()
        self.batch_count: int = 0
        self.hits: int = 0
        self.misses: int = 0

    def add(self, mco: MiniBatchCacheObject):
        self.batch_list.append(mco)
        self.batch_count = len(self.batch_list)
        self.touch()

    def get_random_mco(self) -> MiniBatchCacheObject | None:
        if not self.batch_list:
            return None
        import random
        self.touch()
        return random.choice(self.batch_list)

    def replace_mco(self, old_mco: MiniBatchCacheObject, new_mco: MiniBatchCacheObject):
        try:
            index = self.batch_list.index(old_mco)
            self.batch_list[index] = new_mco
            self.touch()
        except ValueError:
            self.add(new_mco)

    def touch(self):
        self.last_used_timestamp = time.time()

    def __len__(self) -> int:
        return len(self.batch_list)

    def __repr__(self) -> str:
        return f'CachePool(batch_size={self.batch_size}, count={self.batch_count}, last_used={self.last_used_timestamp:.0f})'