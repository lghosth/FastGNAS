import random
import torch
from typing import Dict
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader

from batching.cache_pool import CachePool

class HierarchicalCacheManager:
    """
    实现动态分层与增量采样缓存策略。

    该管理器通过一个概率模型来平衡缓存利用和新采样，并支持对已缓存的
    低层级batch进行增量采样以满足更高层级的需求。
    """
    def __init__(self,
                 subgraph: Data,
                 device: torch.device,
                 max_cache_pools: int = 10,
                 max_pool_size: int = 50):
        """
        初始化分层缓存管理器。

        Args:
            subgraph (Data): 当前Worker持有的、用于采样的图数据分区。
            device (torch.device): GPU设备。
            max_cache_pools (int): 全局缓存能容纳的最大CachePool类别数量。
            max_pool_size (int): 每个CachePool内部能容纳的最大batch数量。
        """
        # TODO： 修改
        self.subgraph = subgraph
        self.device = device
        self.max_cache_pools = max_cache_pools
        self.max_pool_size = max_pool_size
        
        # 全局缓存池，键是batch_size，值是CachePool对象
        self.global_cache: Dict[int, CachePool] = {}
        
        # 预计算子图的平均度数，用于概率模型
        if (self.subgraph.edge_index is not None and 
            self.subgraph.num_nodes is not None and 
            self.subgraph.num_nodes > 0):
            self.avg_degree = self.subgraph.num_edges / self.subgraph.num_nodes
        else:
            self.avg_degree = 1.0  # 避免除以零

    def get_batch(self, requested_bs: int, requested_layers: int) -> Data:
        """
        根据请求的采样数和层数，获取一个mini-batch。
        这是该类的主要入口方法。
        """
        # 1. 查找缓存类别 (按采样数)
        cache_pool = self.global_cache.get(requested_bs)

        # 情况A：未匹配到采样数
        if cache_pool is None:
            return self._handle_bs_miss(requested_bs, requested_layers)

        # 情况B：匹配到采样数
        else:
            return self._handle_bs_hit(cache_pool, requested_bs, requested_layers)

    def _handle_bs_miss(self, req_bs: int, req_layers: int) -> Data:
        """处理按batch_size查找缓存未命中的情况。"""
        # 3.A.1. 驱逐检查
        if len(self.global_cache) >= self.max_cache_pools:
            self._evict_pool()
        
        # 3.A.2. 创建新池
        new_pool = CachePool(batch_size=req_bs, max_pool_size=self.max_pool_size)
        self.global_cache[req_bs] = new_pool
        
        # 3.A.3. 完全采样, 并将结果存入新池
        return self._full_sample_and_cache(new_pool, req_bs, req_layers)

    def _handle_bs_hit(self, pool: CachePool, req_bs: int, req_layers: int) -> Data:
        """处理按batch_size查找缓存命中的情况。"""
        pool.touch() # 更新时间戳
        
        # 3.B.1. 计算命中概率
        # 确保分母不为0，且概率不超过1.0
        p_hit = min(len(pool) / self.avg_degree, 1.0) if self.avg_degree > 0 else 0.0

        # 3.B.2. 随机决策
        if random.random() < p_hit:
            # 判定为“缓存命中”，进入分层匹配
            return self._hierarchical_match(pool, req_layers)
        else:
            # 判定为“概率性未命中”，进行完全采样
            return self._full_sample_and_cache(pool, req_bs, req_layers)

    def _hierarchical_match(self, pool: CachePool, req_layers: int) -> Data:
        """3.B.3. 分层匹配与增量采样逻辑。"""
        cached_batch = pool.get_random_batch()
        
        # 如果池为空，则回退到完全采样
        if cached_batch is None:
            return self._full_sample_and_cache(pool, pool.batch_size, req_layers)

        # 假设batch对象中存储了其被采样时的层数信息
        # 注意: PyG的batch默认不带此信息，需在采样时手动附加
        cached_layers = getattr(cached_batch, 'num_layers', 1)

        # ① 如果 L_req <= L_cache (降级或平级使用)
        if req_layers <= cached_layers:
            return cached_batch
        
        # ② 如果 L_req > L_cache (升级使用)
        else:
            # 触发增量采样
            return self._incremental_sample_and_cache(pool, cached_batch, req_layers)

    def _full_sample_and_cache(self, pool: CachePool, bs: int, layers: int) -> Data:
        """执行一次完整的邻居采样，并将结果存入缓存池。"""
        # NeighborLoader的num_neighbors列表长度应等于层数
        num_neighbors = [-1] * layers
        
        # 创建加载器并获取一个batch
        loader = NeighborLoader(
            self.subgraph,
            num_neighbors=num_neighbors,
            batch_size=bs,
            shuffle=True
        )
        
        try:
            new_batch = next(iter(loader))
            # 【重要】将层数信息附加到batch对象上，以备后续分层匹配使用
            new_batch.num_layers = layers
            new_batch.to(self.device)
            pool.add(new_batch)
            return new_batch
        except StopIteration:
            # 如果图太小无法采样，返回None或抛出异常
            raise ValueError(f"无法从子图中采样一个大小为 {bs} 的batch。")

    def _incremental_sample_and_cache(self, pool: CachePool, old_batch: Data, new_layers: int) -> Data:
        """执行增量采样，并用新batch替换旧batch。"""
        # 【注意】这是一个复杂的过程，以下为简化实现
        # 1. 获取原始种子节点 (需在初次采样时保存)
        # NeighborLoader返回的batch中，前batch_size个节点是种子节点
        seed_nodes = old_batch.n_id[:old_batch.batch_size]
        
        # 2. 以新层数重新对这些特定节点进行采样
        num_neighbors = [-1] * new_layers
        loader = NeighborLoader(
            self.subgraph,
            num_neighbors=num_neighbors,
            input_nodes=seed_nodes, # 指定种子节点进行采样
            batch_size=len(seed_nodes)
        )
        
        try:
            new_batch = next(iter(loader))
            new_batch.num_layers = new_layers
            new_batch.to(self.device)
            
            # 3. 在缓存池中替换掉旧的batch
            pool.replace_batch(old_batch, new_batch)
            return new_batch
        except StopIteration:
            # 如果增量采样失败，则返回旧的batch作为降级方案
            return old_batch

    def _evict_pool(self):
        """智能驱逐策略：驱逐最久未使用且包含batch数量最少的池。"""
        if not self.global_cache:
            return

        # 找到所有池，并根据驱逐标准进行排序
        # 标准：首先按最后使用时间升序（越小越旧），然后按batch数量升序
        pools_to_evict = sorted(
            self.global_cache.items(),
            key=lambda item: (item[1].last_used_timestamp, len(item[1]))
        )
        
        # 获取最不重要的池的键（batch_size）并将其从全局缓存中删除
        eviction_key = pools_to_evict[0][0]
        del self.global_cache[eviction_key]
        print(f"Cache Eviction: Removed pool for batch_size={eviction_key}")