import random
import torch
from typing import Dict, List, Tuple, Any
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader
from utils.utils import normalize_fan_out_for_layers
from batching.cache_pool import CachePool, MiniBatchCacheObject, EdgeListUnit

class HierarchicalCacheManager:

    def __init__(self, subgraph: Data, gpu_subgraph: Data, device: torch.device, max_cache_pools: int=10, max_pool_size: int=50, use_fixed_loader: bool=False, fixed_batch_size: int=None, fixed_fan_out: List[int]=None, fixed_num_layers: int=None):
        self.subgraph = subgraph
        self.gpu_subgraph = gpu_subgraph
        self.device = device
        if self.subgraph.edge_index is not None and self.subgraph.edge_index.is_cuda:
            raise ValueError('采样用的subgraph应该在CPU上以避免隐式的GPU↔CPU传输')
        if device.type == 'cuda' and self.gpu_subgraph.x is not None and (not self.gpu_subgraph.x.is_cuda):
            raise ValueError('gpu_subgraph应该在GPU上以实现高效的特征提取')
        self.use_fixed_loader = use_fixed_loader
        self.fixed_loader = None
        self.fixed_loader_iterator = None
        self.fixed_batch_size = fixed_batch_size
        self.fixed_num_layers = fixed_num_layers
        if self.use_fixed_loader:
            if not all([fixed_batch_size is not None, fixed_fan_out is not None, fixed_num_layers is not None]):
                raise ValueError('固定Loader模式需要指定 fixed_batch_size, fixed_fan_out, fixed_num_layers')
            print(f'🔧 [CacheManager] 启用固定NeighborLoader优化:')
            print(f'   - batch_size: {fixed_batch_size}')
            print(f'   - num_layers: {fixed_num_layers}')
            print(f'   - fan_out: {fixed_fan_out}')
            self._init_fixed_loader(fixed_batch_size, fixed_fan_out, fixed_num_layers)
        else:
            self.max_cache_pools = max_cache_pools
            self.max_pool_size = max_pool_size
        self.global_cache: Dict[Tuple[int, Tuple[int, ...]], CachePool] = {}
        self.current_model_id = None
        self.model_training_requests = 0
        if self.subgraph.edge_index is not None and self.subgraph.num_nodes is not None and (self.subgraph.num_nodes > 0):
            self.avg_degree = self.subgraph.num_edges / self.subgraph.num_nodes
        else:
            self.avg_degree = 1.0
        if hasattr(self.subgraph, 'train_mask') and self.subgraph.train_mask is not None:
            self.num_train_nodes = self.subgraph.train_mask.sum().item()
        else:
            self.num_train_nodes = self.subgraph.num_nodes if self.subgraph.num_nodes is not None else 1
        if self.num_train_nodes == 0:
            self.num_train_nodes = 1
        self.fan_out: List[int] = []
        self.probability_cache: Dict[Tuple[int, int, Tuple[int, ...]], float] = {}
        self.prob_cache_hits = 0
        self.prob_cache_misses = 0

    def set_current_model(self, model_id: str):
        if self.current_model_id != model_id:
            self.current_model_id = model_id
            self.model_training_requests = 0

    def _init_fixed_loader(self, batch_size: int, fan_out: List[int], num_layers: int):
        fan_out_list = normalize_fan_out_for_layers(fan_out, num_layers)
        train_nodes = torch.where(self.subgraph.train_mask)[0]
        if len(train_nodes) == 0:
            raise ValueError('数据分区中没有可用的训练节点进行采样')
        print(f'   - 训练节点数: {len(train_nodes)}')
        self.fixed_loader = NeighborLoader(self.subgraph, num_neighbors=fan_out_list, input_nodes=train_nodes, batch_size=batch_size, shuffle=True, drop_last=False)
        self.fixed_loader_iterator = iter(self.fixed_loader)
        print(f'   ✅ 固定Loader初始化完成')

    def get_batch(self, requested_bs: int, requested_layers: int, fan_out) -> Data:
        self.model_training_requests += 1
        if isinstance(fan_out, list):
            fan_out_list = fan_out
        else:
            from utils.utils import parse_fan_out
            fan_out_list = normalize_fan_out_for_layers(parse_fan_out(fan_out), requested_layers)
        self.fan_out = fan_out_list
        cache_key = (requested_bs, tuple(fan_out_list))
        cache_pool = self.global_cache.get(cache_key)
        target_mco: MiniBatchCacheObject
        if cache_pool is None:
            target_mco = self._handle_bs_miss(cache_key, requested_layers, fan_out_list)
        else:
            target_mco = self._handle_bs_hit(cache_pool, requested_layers, fan_out_list)
        if target_mco is None:
            raise RuntimeError('未能获取有效的MiniBatchCacheObject。')
        elu_to_use = target_mco.get_elu(requested_layers)
        if elu_to_use is None:
            max_cached_layers = target_mco.max_cached_layers
            if max_cached_layers > 0:
                elu_to_use = target_mco.get_elu(max_cached_layers)
        if elu_to_use is None:
            raise RuntimeError(f'无法从MiniBatchCacheObject获取层数 {requested_layers} 或最大缓存层对应的ELU。')
        x = self.gpu_subgraph.x[elu_to_use.n_id] if self.gpu_subgraph.x is not None else None
        y = self.gpu_subgraph.y[elu_to_use.n_id] if self.gpu_subgraph.y is not None else None
        data_batch = Data(x=x, edge_index=elu_to_use.edge_index, y=y, num_nodes=len(elu_to_use.n_id))
        data_batch.batch_size = elu_to_use.batch_size
        data_batch.num_layers = elu_to_use.num_layers
        return data_batch

    def _handle_bs_miss(self, cache_key: Tuple[int, Tuple[int, ...]], req_layers: int, fan_out_list: List[int]) -> MiniBatchCacheObject:
        if len(self.global_cache) >= self.max_cache_pools:
            self._evict_pool()
        req_bs = cache_key[0]
        new_pool = CachePool(batch_size=req_bs, max_pool_size=self.max_pool_size)
        self.global_cache[cache_key] = new_pool
        return self._full_sample_and_cache(new_pool, req_bs, req_layers, fan_out_list)

    def _handle_bs_hit(self, pool: CachePool, req_layers: int, fan_out_list: List[int]) -> MiniBatchCacheObject:
        pool.touch()
        prob_cache_key = (len(pool), pool.batch_size, tuple(fan_out_list))
        if prob_cache_key in self.probability_cache:
            p_hit = self.probability_cache[prob_cache_key]
            self.prob_cache_hits += 1
        else:
            self.prob_cache_misses += 1
            mapped_fan_outs = []
            for value in fan_out_list:
                if value is None:
                    continue
                if value == -1:
                    mapped_fan_outs.append(self.avg_degree)
                elif value > 0:
                    mapped_fan_outs.append(value)
            if not mapped_fan_outs:
                raise ValueError('fan_out 列表映射后为空，可能包含无效值')
            n_fo = sum(mapped_fan_outs) / len(mapped_fan_outs)
            num_mcos = len(pool)
            numerator = num_mcos * pool.batch_size * n_fo
            if self.num_train_nodes > 0:
                denominator = self.num_train_nodes * self.avg_degree
            else:
                raise ValueError('训练节点数量不能为0')
            if denominator > 0:
                p_hit = float(numerator) / denominator
                if p_hit >= 1.0:
                    p_hit = 0.95
                elif p_hit <= 0.0:
                    p_hit = 0.0
            else:
                p_hit = 0.0
            self.probability_cache[prob_cache_key] = p_hit
        if random.random() < p_hit:
            return self._hierarchical_match(pool, req_layers, fan_out_list)
        else:
            return self._full_sample_and_cache(pool, pool.batch_size, req_layers, fan_out_list)

    def _hierarchical_match(self, pool: CachePool, req_layers: int, fan_out_list: List[int]) -> MiniBatchCacheObject:
        if pool.batch_list:
            cached_mco = pool.get_random_mco()
            if cached_mco is not None:
                return cached_mco
        return self._full_sample_and_cache(pool, pool.batch_size, req_layers, fan_out_list)

    def _full_sample_and_cache(self, pool: CachePool, bs: int, layers: int, fan_out_list: List[int]) -> MiniBatchCacheObject:
        if len(fan_out_list) < layers:
            fan_out_list = normalize_fan_out_for_layers(fan_out_list, layers)
        if self.use_fixed_loader:
            return self._sample_with_fixed_loader(pool, bs)
        return self._sample_dynamically(pool, bs, layers, fan_out_list)

    def _sample_with_fixed_loader(self, pool: CachePool, bs: int) -> MiniBatchCacheObject:
        if self.fixed_loader is None or self.fixed_loader_iterator is None:
            raise RuntimeError('固定Loader未初始化，请先调用 _init_fixed_loader()')
        try:
            sampled_data = next(self.fixed_loader_iterator)
        except StopIteration:
            self.fixed_loader_iterator = iter(self.fixed_loader)
            sampled_data = next(self.fixed_loader_iterator)
        edge_index = sampled_data.edge_index.to(self.device)
        n_id = sampled_data.n_id.to(self.device)
        batch_size = sampled_data.batch_size
        sampled_seeds = n_id[:batch_size]
        new_mco = MiniBatchCacheObject(target_vertices=sampled_seeds)
        elu = EdgeListUnit(edge_index=edge_index, n_id=n_id, batch_size=batch_size, num_layers=self.fixed_num_layers)
        new_mco.add_elu(elu)
        pool.add(new_mco)
        return new_mco

    def _sample_dynamically(self, pool: CachePool, bs: int, layers: int, fan_out_list: List[int]) -> MiniBatchCacheObject:
        train_nodes = torch.where(self.subgraph.train_mask)[0]
        if len(train_nodes) == 0:
            raise ValueError('数据分区中没有可用的训练节点进行采样。')
        sampled_seed_indices = torch.randperm(len(train_nodes))[:bs]
        sampled_seeds = train_nodes[sampled_seed_indices]
        new_mco = MiniBatchCacheObject(target_vertices=sampled_seeds.to(self.device))
        current_input_nodes = sampled_seeds
        for l in range(1, layers + 1):
            layer_fan_out = fan_out_list[l - 1]
            num_neighbors = [layer_fan_out]
            loader = NeighborLoader(self.subgraph, num_neighbors=num_neighbors, input_nodes=current_input_nodes, batch_size=len(current_input_nodes), shuffle=False, drop_last=False)
            try:
                sampled_data = next(iter(loader))
                edge_index_gpu = sampled_data.edge_index.to(self.device)
                n_id_gpu = sampled_data.n_id.to(self.device)
                elu = EdgeListUnit(edge_index=edge_index_gpu, n_id=n_id_gpu, batch_size=bs, num_layers=l)
                new_mco.add_elu(elu)
                current_input_nodes = sampled_data.n_id
            except StopIteration:
                break
        pool.add(new_mco)
        return new_mco

    def _incremental_sample_and_cache(self, pool: CachePool, old_mco: MiniBatchCacheObject, new_layers: int, fan_out_list: List[int]) -> MiniBatchCacheObject:
        fan_out_list = normalize_fan_out_for_layers(fan_out_list, new_layers)
        cloned_mco = MiniBatchCacheObject(target_vertices=old_mco.target_vertices.clone())
        for elu in old_mco.elus:
            cloned_mco.add_elu(EdgeListUnit(elu.edge_index.clone(), elu.n_id.clone(), elu.batch_size, elu.num_layers))
        last_elu = old_mco.get_elu(old_mco.max_cached_layers)
        if last_elu is not None:
            current_input_nodes = last_elu.n_id.cpu()
        else:
            current_input_nodes = old_mco.target_vertices.cpu()
        for l in range(old_mco.max_cached_layers + 1, new_layers + 1):
            layer_fan_out = fan_out_list[l - 1]
            num_neighbors = [layer_fan_out]
            loader = NeighborLoader(self.subgraph, num_neighbors=num_neighbors, input_nodes=current_input_nodes, batch_size=len(current_input_nodes), shuffle=False, drop_last=False)
            try:
                sampled_data = next(iter(loader))
                edge_index_gpu = sampled_data.edge_index.to(self.device)
                n_id_gpu = sampled_data.n_id.to(self.device)
                elu = EdgeListUnit(edge_index=edge_index_gpu, n_id=n_id_gpu, batch_size=cloned_mco.batch_size, num_layers=l)
                cloned_mco.add_elu(elu)
                current_input_nodes = sampled_data.n_id
            except StopIteration:
                break
        pool.replace_mco(old_mco, cloned_mco)
        return cloned_mco

    def get_probability_cache_stats(self) -> Dict[str, Any]:
        total = self.prob_cache_hits + self.prob_cache_misses
        hit_rate = self.prob_cache_hits / total * 100 if total > 0 else 0.0
        return {'total_requests': total, 'cache_hits': self.prob_cache_hits, 'cache_misses': self.prob_cache_misses, 'hit_rate': hit_rate, 'cache_size': len(self.probability_cache)}

    def _evict_pool(self):
        if not self.global_cache:
            return
        pools_to_evict = sorted(self.global_cache.items(), key=lambda item: (item[1].last_used_timestamp, len(item[1].batch_list)))
        eviction_key = pools_to_evict[0][0]
        del self.global_cache[eviction_key]