import time
from typing import List, Deque
from collections import deque
from torch_geometric.data import Data

class CachePool:
    """
    为一个特定的采样数 (batch_size) 管理一个缓存池。

    这个池包含为该 batch_size 生成的、不同层数的 mini-batch 列表，
    并维护用于LRU驱逐策略的元数据。
    """
    def __init__(self, batch_size: int, max_pool_size: int = 50):
        """
        初始化一个缓存池。

        Args:
            batch_size (int): 此缓存池对应的批处理大小。
            max_pool_size (int): 池内能存储的最大 batch 数量。
        """
        self.batch_size = batch_size
        self.max_pool_size = max_pool_size
        
        # 使用双端队列存储缓存的mini-batch，方便快速地在两端增删
        self.batch_list: Deque[Data] = deque(maxlen=self.max_pool_size)
        
        # 初始化元数据
        self.last_used_timestamp: float = time.time()
        self.batch_count: int = 0
        self.hits: int = 0
        self.misses: int = 0

    def add(self, batch: Data):
        """向缓存池中添加一个新的mini-batch。"""
        # 如果队列已满，最旧的元素会自动被移除
        self.batch_list.append(batch)
        self.batch_count = len(self.batch_list)
        self.touch()

    def get_random_batch(self) -> Data | None:
        """从池中随机获取一个batch，并更新时间戳。"""
        if not self.batch_list:
            return None
        
        import random
        self.touch()
        return random.choice(self.batch_list)

    def replace_batch(self, old_batch: Data, new_batch: Data):
        """用一个新的、更高层的batch替换一个旧的。"""
        try:
            index = self.batch_list.index(old_batch)
            self.batch_list[index] = new_batch
            self.touch()
        except ValueError:
            # 如果旧batch不存在（可能已被驱逐），则直接添加新的
            self.add(new_batch)

    def touch(self):
        """更新最后使用时间戳。"""
        self.last_used_timestamp = time.time()

    def __len__(self) -> int:
        """返回池中缓存的batch数量。"""
        return len(self.batch_list)

    def __repr__(self) -> str:
        return (
            f"CachePool(batch_size={self.batch_size}, "
            f"count={self.batch_count}, "
            f"last_used={self.last_used_timestamp:.0f})"
        )