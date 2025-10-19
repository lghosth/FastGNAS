import os
import sys
import torch
import torch.nn.functional as F
from typing import Dict, Any, List, Tuple, Optional, Set
import ray
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader
from torch_geometric.loader import NeighborLoader
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
from data.partition import PartitionGraph
from trainer.builder import ModelBuilder
from utils.logger import GraphNASLogger

class COCommunicator:

    def __init__(self, worker_id: int, num_workers: int, device: torch.device):
        self.worker_id = worker_id
        self.num_workers = num_workers
        self.device = device
        self.logger = GraphNASLogger(name=f'co_communicator_{worker_id}')

    def identify_remote_neighbors(self, local_nodes: torch.Tensor, edge_index: torch.Tensor, global_node_ids: Optional[torch.Tensor]=None) -> Set[int]:
        local_global_set = set(local_nodes.cpu().tolist())
        if edge_index.numel() == 0:
            return set()
        idx_to_global: Optional[torch.Tensor] = None
        if global_node_ids is not None and global_node_ids.numel() > 0:
            idx_to_global = global_node_ids.detach().cpu()
        edge_src = edge_index[0].detach().cpu()
        edge_dst = edge_index[1].detach().cpu()
        remote_neighbors: Set[int] = set()
        num_nodes_mapped = idx_to_global.numel() if idx_to_global is not None else 0
        for src_local, dst_local in zip(edge_src.tolist(), edge_dst.tolist()):
            if idx_to_global is not None and 0 <= src_local < num_nodes_mapped:
                src_global = int(idx_to_global[src_local].item())
            else:
                src_global = int(src_local)
            if idx_to_global is not None and 0 <= dst_local < num_nodes_mapped:
                dst_global = int(idx_to_global[dst_local].item())
            else:
                dst_global = int(dst_local)
            if src_global in local_global_set and dst_global not in local_global_set:
                remote_neighbors.add(dst_global)
            if dst_global in local_global_set and src_global not in local_global_set:
                remote_neighbors.add(src_global)
        return remote_neighbors

    def gather_remote_features(self, remote_node_ids: Set[int], global_node_features: Optional[torch.Tensor]) -> Dict[int, torch.Tensor]:
        remote_features = {}
        if global_node_features is None:
            return remote_features
        for node_id in remote_node_ids:
            if node_id < global_node_features.size(0):
                remote_features[node_id] = global_node_features[node_id].to(self.device)
        return remote_features

@ray.remote(num_gpus=1)
class CORayWorker:

    def __init__(self, worker_id: int, data_partition: Data, device: torch.device, config: Dict[str, Any], global_feature_ref: Optional[object]=None, physical_gpu_id: Optional[int]=None) -> None:
        self.worker_id = worker_id
        self.physical_gpu_id = physical_gpu_id
        self.config = config
        self.co_config = config['co_backends']
        self.train_config = config['training']
        log_dir_cfg = config.get('logging', {})
        worker_log_dir = os.path.join(log_dir_cfg.get('log_dir', 'logs'), 'worker')
        os.makedirs(worker_log_dir, exist_ok=True)
        self.logger = GraphNASLogger(name=f'co_worker_{worker_id}', log_dir=worker_log_dir)
        if device.type == 'cuda':
            torch.cuda.set_device(0)
            runtime_device = torch.device('cuda:0')
            self.logger.logger.info(f'当前worker {self.worker_id} 使用第一个GPU')
        else:
            runtime_device = torch.device('cpu')
            raise ValueError(f'当前worker {self.worker_id} 没有可用的GPU')
        self.device = runtime_device
        self.data_partition = data_partition.to(self.device)
        self.global_node_features_cpu: Optional[torch.Tensor] = None
        if global_feature_ref is not None:
            cpu_tensor: Optional[torch.Tensor] = None
            if isinstance(global_feature_ref, ray.ObjectRef):
                cpu_tensor = ray.get(global_feature_ref)
            elif isinstance(global_feature_ref, torch.Tensor):
                cpu_tensor = global_feature_ref
            else:
                self.logger.logger.warning(f'global_feature_ref 类型 {type(global_feature_ref)} 非预期，将忽略全局特征')
            if isinstance(cpu_tensor, torch.Tensor):
                self.global_node_features_cpu = cpu_tensor if cpu_tensor.device.type == 'cpu' else cpu_tensor.cpu()
            else:
                self.logger.logger.warning('未能获取CPU全局特征，远程邻居特征将不可用')
        else:
            self.logger.logger.warning('未提供全局特征引用，远程邻居特征将不可用')
        num_workers = self.co_config.get('num_workers', 2)
        self.communicator = COCommunicator(worker_id, num_workers, self.device)
        if hasattr(data_partition, 'global_node_ids'):
            self.logger.logger.info('成功获取全局节点ID')
            self.global_node_ids = data_partition.global_node_ids.detach().cpu()
        else:
            self.logger.logger.info('未能获取全局节点ID，将使用默认范围')
            raise ValueError('未能获取全局节点ID')
            self.global_node_ids = torch.arange(end=self.data_partition.num_nodes, dtype=torch.long)

    def _merge_local_and_remote_features(self, local_x: torch.Tensor, remote_features: Optional[Dict[int, torch.Tensor]], remote_ids: Set[int]) -> Tuple[torch.Tensor, torch.Tensor]:
        local_count = local_x.size(0)
        feature_dim = local_x.size(1)
        local_global_ids = self.global_node_ids.detach()
        local_global_ids_cpu = local_global_ids.cpu()
        local_id_set = set(local_global_ids_cpu.tolist())
        remote_only_sorted = sorted((int(rid) for rid in remote_ids if rid not in local_id_set))
        remote_count = len(remote_only_sorted)
        combined_ids = torch.empty(local_count + remote_count, dtype=local_global_ids.dtype, device=self.device)
        if local_count > 0:
            combined_ids[:local_count] = local_global_ids
        combined_x = torch.zeros((local_count + remote_count, feature_dim), dtype=local_x.dtype, device=self.device)
        if local_count > 0:
            combined_x[:local_count] = local_x
        if remote_count > 0:
            remote_tensor_list: List[torch.Tensor] = []
            for rid in remote_only_sorted:
                feature = None
                if remote_features is not None:
                    feature = remote_features.get(rid)
                if feature is None:
                    feature = torch.zeros(feature_dim, dtype=local_x.dtype, device=self.device)
                else:
                    feature = feature.to(self.device)
                remote_tensor_list.append(feature)
            if remote_tensor_list:
                remote_tensor = torch.stack(remote_tensor_list, dim=0)
                combined_x[local_count:] = remote_tensor
                combined_ids[local_count:] = torch.tensor(remote_only_sorted, dtype=local_global_ids.dtype, device=self.device)
        return (combined_x, combined_ids)

    def _build_enhanced_edge_index(self, original_edge_index: torch.Tensor, remote_ids: Set[int]) -> torch.Tensor:
        return original_edge_index

    def train_on_partition_with_communication(self, arch_spec: Dict[str, Any], model_state: Optional[Dict], epochs: int, batch_size: Optional[int]=None) -> Tuple[Dict, float]:
        input_dim = self.data_partition.num_node_features
        output_dim = self.co_config.get('num_classes', 7)
        model = ModelBuilder.build_model(arch_spec, input_dim, output_dim, self.config).to(self.device)
        if model_state is not None:
            model.load_state_dict(model_state)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.config.get('learning_rate', 0.01))
        remote_neighbors = self.communicator.identify_remote_neighbors(self.global_node_ids, self.data_partition.edge_index, self.global_node_ids)
        enhanced_x_device, combined_global_ids = self._merge_local_and_remote_features(self.data_partition.x, remote_features=None, remote_ids=remote_neighbors)
        local_count = int(self.data_partition.num_nodes)
        enhanced_x_cpu = enhanced_x_device.detach().cpu()
        combined_global_ids_cpu = combined_global_ids.cpu()
        enhanced_edge_index_cpu = self._build_enhanced_edge_index(self.data_partition.edge_index, remote_neighbors).detach().cpu()
        self.logger.logger.debug(f'Worker {self.worker_id} 通信统计: 本地节点={local_count}, 远程邻居={len(remote_neighbors)}, 增强特征矩阵={enhanced_x_cpu.shape}')
        train_mask = self.data_partition.train_mask
        if train_mask.sum() == 0:
            raise ValueError('train_mask为空')
            return (model.state_dict(), 0.0)
        train_local_indices = torch.where(train_mask)[0]
        if train_local_indices.numel() == 0:
            raise ValueError('train_local_indices.numel()为空')
            return (model.state_dict(), 0.0)
        labels_cpu = torch.full((enhanced_x_cpu.size(0),), -1, dtype=self.data_partition.y.dtype)
        labels_cpu[:local_count] = self.data_partition.y.cpu()
        enhanced_data = Data(x=enhanced_x_cpu, edge_index=enhanced_edge_index_cpu, y=labels_cpu, global_node_ids=combined_global_ids_cpu)
        enhanced_data.num_nodes = enhanced_x_cpu.size(0)
        num_neighbors_cfg = self.train_config.get('fan_out', [-1])
        if isinstance(num_neighbors_cfg, int):
            num_neighbors = [num_neighbors_cfg]
        elif isinstance(num_neighbors_cfg, (list, tuple)) and len(num_neighbors_cfg) > 0:
            num_neighbors = list(num_neighbors_cfg)
        else:
            num_neighbors = [-1]
        effective_batch_size = batch_size if batch_size and batch_size > 0 else self.train_config.get('batch_size', 1024)
        if not effective_batch_size or effective_batch_size <= 0:
            effective_batch_size = 1024
        train_loader = NeighborLoader(enhanced_data, input_nodes=train_local_indices, num_neighbors=num_neighbors, batch_size=effective_batch_size, shuffle=True, num_workers=0, pin_memory=self.device.type == 'cuda')
        local_global_ids_cpu = self.global_node_ids.cpu()
        local_global_id_set = set(local_global_ids_cpu.tolist())
        remote_global_id_set = set(combined_global_ids_cpu[local_count:].tolist())
        model.train()
        total_loss = 0.0
        total_steps = 0
        for epoch in range(epochs):
            self.logger.logger.info(f'Worker {self.worker_id} 开始第 {epoch} 轮训练，共{epochs}轮')
            for batch in train_loader:
                current_batch_size = getattr(batch, 'batch_size', 0)
                if current_batch_size == 0:
                    continue
                batch_n_id = batch.n_id.clone()
                batch_global_ids = enhanced_data.global_node_ids[batch_n_id.cpu()]
                remote_batch_ids = [int(gid) for gid in batch_global_ids.tolist() if gid in remote_global_id_set and gid not in local_global_id_set]
                batch = batch.to(self.device, non_blocking=True)
                if remote_batch_ids:
                    remote_features_batch = self.communicator.gather_remote_features(set(remote_batch_ids), self.global_node_features_cpu)
                    if remote_features_batch:
                        id_to_pos = {int(gid): idx for idx, gid in enumerate(batch_global_ids.tolist())}
                        for remote_id, feature in remote_features_batch.items():
                            pos = id_to_pos.get(int(remote_id))
                            if pos is not None:
                                batch.x[pos] = feature.to(self.device)
                optimizer.zero_grad()
                out = model(batch.x, batch.edge_index)
                logits = out[:current_batch_size]
                labels_batch = batch.y[:current_batch_size].long()
                valid_idx = (labels_batch >= 0).nonzero(as_tuple=False).view(-1)
                if valid_idx.numel() == 0:
                    del batch, batch_n_id, batch_global_ids, out, logits, labels_batch, valid_idx
                    continue
                logits = logits.index_select(0, valid_idx)
                labels_batch = labels_batch.index_select(0, valid_idx)
                loss = F.cross_entropy(logits, labels_batch)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                total_steps += 1
                del batch, batch_n_id, batch_global_ids, out, logits, labels_batch, valid_idx, loss
                if self.device.type == 'cuda' and torch.cuda.is_available():
                    torch.cuda.empty_cache()
        avg_loss = total_loss / total_steps if total_steps > 0 else 0.0
        return (model.state_dict(), avg_loss)

    def train_on_partition_with_communication_optimized(self, arch_spec: Dict[str, Any], model_state: Optional[Dict], epochs: int, batch_size: Optional[int]=None, steps_per_epoch: Optional[int]=None) -> Tuple[Dict, float, Dict[str, float]]:
        import time
        model_build_start = time.perf_counter()
        input_dim = self.data_partition.num_node_features
        output_dim = self.co_config.get('num_classes', 7)
        model = ModelBuilder.build_model(arch_spec, input_dim, output_dim, self.co_config).to(self.device)
        model_build_time = time.perf_counter() - model_build_start
        self.logger.logger.debug(f'[Worker {self.worker_id}] 模型构建耗时: {model_build_time * 1000:.2f}ms')
        model_migration_time = 0.0
        if model_state is not None:
            t_migration_start = time.perf_counter()
            model.load_state_dict(model_state)
            t_migration_end = time.perf_counter()
            model_migration_time = t_migration_end - t_migration_start
            self.logger.logger.debug(f'[Worker {self.worker_id}] 模型迁移耗时: {model_migration_time * 1000:.2f}ms')
        optimizer = torch.optim.Adam(model.parameters(), lr=self.co_config.get('learning_rate', 0.01))
        local_x_gpu = self.data_partition.x.to(self.device)
        local_y_gpu = self.data_partition.y.to(self.device)
        local_count = int(self.data_partition.num_nodes)
        remote_neighbors = self.communicator.identify_remote_neighbors(self.global_node_ids, self.data_partition.edge_index, self.global_node_ids)
        enhanced_edge_index_cpu = self._build_enhanced_edge_index(self.data_partition.edge_index, remote_neighbors).cpu()
        combined_global_ids_cpu = torch.cat([self.global_node_ids.cpu(), torch.tensor(sorted(remote_neighbors), dtype=torch.long)])
        local_global_id_set = set(self.global_node_ids.cpu().tolist())
        self.logger.logger.info(f'[Optimized] Worker {self.worker_id} - 本地节点={local_count}, 远程邻居={len(remote_neighbors)}, 本地特征GPU显存={local_x_gpu.element_size() * local_x_gpu.nelement() / 1024 / 1024:.2f}MB')
        train_mask = self.data_partition.train_mask
        if train_mask.sum() == 0:
            return (model.state_dict(), 0.0)
        train_local_indices = torch.where(train_mask)[0].cpu()
        if train_local_indices.numel() == 0:
            return (model.state_dict(), 0.0)
        sampling_data = Data(edge_index=enhanced_edge_index_cpu, num_nodes=local_count + len(remote_neighbors))
        num_neighbors_cfg = self.config.get('num_neighbors', self.config.get('fan_out', [-1]))
        if isinstance(num_neighbors_cfg, int):
            num_neighbors = [num_neighbors_cfg]
        elif isinstance(num_neighbors_cfg, (list, tuple)) and len(num_neighbors_cfg) > 0:
            num_neighbors = list(num_neighbors_cfg)
        else:
            num_neighbors = [-1]
        effective_batch_size = batch_size if batch_size and batch_size > 0 else self.train_config.get('batch_size', 1024)
        if not effective_batch_size or effective_batch_size <= 0:
            effective_batch_size = 1024
        model.train()
        total_loss = 0.0
        total_steps = 0
        timing_stats = {'model_build': model_build_time, 'model_migration': model_migration_time, 'sampling': 0.0, 'feature_assembly': 0.0, 'forward': 0.0, 'backward': 0.0, 'optimizer_step': 0.0}
        for epoch in range(epochs):
            epoch_steps = 0
            max_steps = steps_per_epoch if steps_per_epoch is not None else float('inf')
            while epoch_steps < max_steps:
                t0 = time.perf_counter()
                try:
                    train_loader = NeighborLoader(sampling_data, input_nodes=train_local_indices, num_neighbors=num_neighbors, batch_size=effective_batch_size, shuffle=True)
                    train_iter = iter(train_loader)
                    batch_cpu = next(train_iter)
                except StopIteration:
                    break
                t1 = time.perf_counter()
                timing_stats['sampling'] += t1 - t0
                current_batch_size = getattr(batch_cpu, 'batch_size', 0)
                if current_batch_size == 0:
                    continue
                batch_n_id_cpu = batch_cpu.n_id
                batch_global_ids = combined_global_ids_cpu[batch_n_id_cpu]
                t_feature_start = time.perf_counter()
                local_mask_cpu = batch_n_id_cpu < local_count
                remote_mask_cpu = batch_n_id_cpu >= local_count
                local_indices_in_batch_cpu = batch_n_id_cpu[local_mask_cpu]
                remote_global_ids = batch_global_ids[remote_mask_cpu].tolist()
                batch_num_nodes = batch_n_id_cpu.size(0)
                batch_x_gpu = torch.zeros(batch_num_nodes, input_dim, dtype=local_x_gpu.dtype, device=self.device)
                if local_mask_cpu.any():
                    local_positions_gpu = torch.where(local_mask_cpu.to(self.device))[0]
                    local_indices_gpu = local_indices_in_batch_cpu.to(self.device)
                    batch_x_gpu[local_positions_gpu] = local_x_gpu[local_indices_gpu]
                if len(remote_global_ids) > 0:
                    remote_features_batch = self.communicator.gather_remote_features(set(remote_global_ids), self.global_node_features_cpu)
                    if remote_features_batch:
                        remote_positions_gpu = torch.where(remote_mask_cpu.to(self.device))[0]
                        for idx, remote_gid in enumerate(remote_global_ids):
                            if idx >= len(remote_positions_gpu):
                                break
                            remote_feat = remote_features_batch.get(remote_gid)
                            if remote_feat is not None:
                                batch_x_gpu[remote_positions_gpu[idx]] = remote_feat.to(self.device)
                batch_edge_index_gpu = batch_cpu.edge_index.to(self.device)
                batch_y_gpu = torch.full((current_batch_size,), -1, dtype=local_y_gpu.dtype, device=self.device)
                seed_n_id = batch_n_id_cpu[:current_batch_size]
                local_seed_mask = seed_n_id < local_count
                if local_seed_mask.any():
                    local_seed_indices = seed_n_id[local_seed_mask].to(self.device)
                    local_positions_in_batch = torch.where(local_seed_mask)[0].to(self.device)
                    batch_y_gpu[local_positions_in_batch] = local_y_gpu[local_seed_indices]
                if self.device.type == 'cuda':
                    torch.cuda.synchronize(self.device)
                t_feature_end = time.perf_counter()
                timing_stats['feature_assembly'] += t_feature_end - t_feature_start
                t_forward_start = time.perf_counter()
                optimizer.zero_grad()
                out = model(batch_x_gpu, batch_edge_index_gpu)
                logits = out[:current_batch_size]
                valid_idx = (batch_y_gpu >= 0).nonzero(as_tuple=False).view(-1)
                if valid_idx.numel() == 0:
                    continue
                loss = F.cross_entropy(logits.index_select(0, valid_idx), batch_y_gpu.index_select(0, valid_idx))
                t_forward_end = time.perf_counter()
                timing_stats['forward'] += t_forward_end - t_forward_start
                t_backward_start = time.perf_counter()
                loss.backward()
                t_backward_end = time.perf_counter()
                timing_stats['backward'] += t_backward_end - t_backward_start
                t_optimizer_start = time.perf_counter()
                optimizer.step()
                t_optimizer_end = time.perf_counter()
                timing_stats['optimizer_step'] += t_optimizer_end - t_optimizer_start
                self.logger.logger.info('第{}步训练完成，损失：{:.4f}'.format(total_steps + 1, loss.item()))
                total_loss += loss.item()
                total_steps += 1
                epoch_steps += 1
                del train_loader, train_iter, batch_cpu, batch_n_id_cpu, batch_global_ids
                del batch_x_gpu, batch_edge_index_gpu, batch_y_gpu, out, loss
                if self.device.type == 'cuda':
                    torch.cuda.empty_cache()
        total_time = sum(timing_stats.values())
        if total_time > 0:
            compute_time = timing_stats['forward'] + timing_stats['backward']
            self.logger.logger.info(f"[性能分析-CO] Worker {self.worker_id} - 总时间={total_time:.2f}s, 模型构建={timing_stats['model_build']:.4f}s ({timing_stats['model_build'] / total_time * 100:.1f}%), 模型迁移={timing_stats['model_migration']:.4f}s ({timing_stats['model_migration'] / total_time * 100:.1f}%), 采样={timing_stats['sampling']:.2f}s ({timing_stats['sampling'] / total_time * 100:.1f}%), 特征组装={timing_stats['feature_assembly']:.2f}s ({timing_stats['feature_assembly'] / total_time * 100:.1f}%), 计算时间={compute_time:.2f}s ({compute_time / total_time * 100:.1f}%), [前向={timing_stats['forward']:.2f}s, 反向={timing_stats['backward']:.2f}s], 参数更新={timing_stats['optimizer_step']:.2f}s ({timing_stats['optimizer_step'] / total_time * 100:.1f}%)")
        avg_loss = total_loss / max(1, total_steps)
        return (model.state_dict(), avg_loss, timing_stats)

    def evaluate_on_partition_with_communication(self, arch_spec: Dict[str, Any], model_state: Dict) -> Dict[str, Any]:
        input_dim = self.data_partition.num_node_features
        output_dim = self.co_config.get('num_classes', 7)
        model = ModelBuilder.build_model(arch_spec, input_dim, output_dim, self.config).to(self.device)
        model.load_state_dict(model_state)
        remote_neighbors = self.communicator.identify_remote_neighbors(self.global_node_ids, self.data_partition.edge_index, self.global_node_ids)
        remote_features = self.communicator.gather_remote_features(remote_neighbors, self.global_node_features_cpu)
        enhanced_x_device, combined_global_ids = self._merge_local_and_remote_features(self.data_partition.x, remote_features, remote_neighbors)
        local_count = int(self.data_partition.num_nodes)
        enhanced_x_cpu = enhanced_x_device.detach().cpu()
        enhanced_edge_index_cpu = self._build_enhanced_edge_index(self.data_partition.edge_index, remote_neighbors).detach().cpu()
        combined_global_ids_cpu = combined_global_ids.cpu()
        model.eval()
        assert hasattr(self.data_partition, 'eval_mask'), f'COWorker {self.worker_id} 分区必须包含eval_mask'
        eval_mask = self.data_partition.eval_mask
        if eval_mask.sum() == 0:
            return {'worker_id': self.worker_id, 'total_eval_nodes': 0, 'correct_eval_nodes': 0, 'correct_node_ids': [], 'incorrect_node_ids': []}
        eval_indices = torch.where(eval_mask)[0]
        if eval_indices.numel() == 0:
            return {'worker_id': self.worker_id, 'total_eval_nodes': 0, 'correct_eval_nodes': 0, 'correct_node_ids': [], 'incorrect_node_ids': []}
        labels_cpu = torch.full((enhanced_x_cpu.size(0),), -1, dtype=self.data_partition.y.dtype)
        labels_cpu[:local_count] = self.data_partition.y.cpu()
        enhanced_data = Data(x=enhanced_x_cpu, edge_index=enhanced_edge_index_cpu, y=labels_cpu, global_node_ids=combined_global_ids_cpu)
        enhanced_data.num_nodes = enhanced_x_cpu.size(0)
        num_neighbors_cfg = self.train_config.get('fan_out', [-1])
        if isinstance(num_neighbors_cfg, int):
            num_neighbors = [num_neighbors_cfg]
        elif isinstance(num_neighbors_cfg, (list, tuple)) and len(num_neighbors_cfg) > 0:
            num_neighbors = list(num_neighbors_cfg)
        else:
            num_neighbors = [-1]
        eval_batch_size = self.train_config.get('eval_batch_size', self.train_config.get('batch_size', 1024))
        if not eval_batch_size or eval_batch_size <= 0:
            eval_batch_size = 1024
        eval_loader = NeighborLoader(enhanced_data, input_nodes=eval_indices, num_neighbors=num_neighbors, batch_size=eval_batch_size, shuffle=False, num_workers=0, pin_memory=self.device.type == 'cuda')
        correct_ids: List[int] = []
        incorrect_ids: List[int] = []
        total_eval_nodes = 0
        correct_eval_nodes = 0
        enhanced_global_ids_cpu = enhanced_data.global_node_ids
        local_global_ids_set = set(self.global_node_ids.cpu().tolist())
        remote_global_id_set = set(enhanced_global_ids_cpu[local_count:].tolist())
        prefetched_remote = bool(remote_neighbors) and bool(remote_features)
        with torch.no_grad():
            for batch in eval_loader:
                batch_size = getattr(batch, 'batch_size', 0)
                if batch_size == 0:
                    continue
                input_local_ids = batch.n_id[:batch_size].clone()
                batch_global_ids = enhanced_global_ids_cpu[input_local_ids.cpu()]
                batch = batch.to(self.device, non_blocking=True)
                if remote_global_id_set and (not prefetched_remote):
                    remote_batch_ids = [int(gid) for gid in batch_global_ids.tolist() if gid in remote_global_id_set and gid not in local_global_ids_set]
                    if remote_batch_ids:
                        remote_features_batch = self.communicator.gather_remote_features(set(remote_batch_ids), self.global_node_features_cpu)
                        if remote_features_batch:
                            id_to_pos = {int(gid): idx for idx, gid in enumerate(batch_global_ids.tolist())}
                            for remote_id, feature in remote_features_batch.items():
                                pos = id_to_pos.get(int(remote_id))
                                if pos is not None:
                                    batch.x[pos] = feature.to(self.device)
                out = model(batch.x, batch.edge_index)
                logits = out[:batch_size]
                labels = batch.y[:batch_size].long()
                valid_idx = (labels >= 0).nonzero(as_tuple=False).view(-1)
                if valid_idx.numel() == 0:
                    del batch, input_local_ids, batch_global_ids, out, logits, labels, valid_idx
                    continue
                preds = logits.argmax(dim=1)
                preds_valid = preds.index_select(0, valid_idx)
                labels_valid = labels.index_select(0, valid_idx)
                ids_valid_local = input_local_ids.index_select(0, valid_idx.cpu())
                ids_valid_global = enhanced_global_ids_cpu[ids_valid_local.cpu()]
                correct_mask = preds_valid.eq(labels_valid)
                correct_eval_nodes += int(correct_mask.sum().item())
                total_eval_nodes += int(labels_valid.numel())
                correct_ids.extend((int(gid) for gid in ids_valid_global[correct_mask.cpu()].tolist()))
                incorrect_ids.extend((int(gid) for gid in ids_valid_global[~correct_mask.cpu()].tolist()))
                del batch, input_local_ids, batch_global_ids, out, logits, labels, valid_idx
                del preds, preds_valid, labels_valid, ids_valid_local, ids_valid_global, correct_mask
            del eval_loader
        return {'worker_id': self.worker_id, 'total_eval_nodes': int(total_eval_nodes), 'correct_eval_nodes': int(correct_eval_nodes), 'correct_node_ids': correct_ids, 'incorrect_node_ids': incorrect_ids}

class COBackends:

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.co_config = config['co_backends']
        self.train_config = config['training']
        self.logger = GraphNASLogger(name='co_backends')
        self.num_workers = self.co_config.get('num_workers', 2)
        self.dataset_name = self.co_config.get('dataset', 'Cora')
        self.device_type = self.co_config.get('device', 'cuda')
        self.partition_strategy = self.co_config.get('partition_strategy', 'balanced_communication_only')
        self.devices = self._setup_devices()
        self.partitions: List[Data] = []
        self.workers: List[ray.actor.ActorHandle] = []
        self.global_node_features_cpu: Optional[torch.Tensor] = None
        self.global_feature_ref: Optional[ray.ObjectRef] = None
        self._initialize_ray_cluster()

    def _initialize_ray_cluster(self) -> None:
        if not ray.is_initialized():
            ray_config = {}
            if self.device_type == 'cuda' and torch.cuda.is_available():
                ray_config['num_gpus'] = torch.cuda.device_count()
            ray.init(**ray_config)

    def _setup_devices(self) -> List[torch.device]:
        self.assigned_gpu_ids = [None for _ in range(self.num_workers)]
        devices: List[torch.device] = [torch.device('cpu') for _ in range(self.num_workers)]
        if self.device_type != 'cuda' or not torch.cuda.is_available():
            return devices
        available_gpu_count = torch.cuda.device_count()
        chosen_gpu_ids: List[int] = []
        worker_gpu_cfg = self.co_config.get('workers')
        if isinstance(worker_gpu_cfg, (list, tuple)):
            for idx, entry in enumerate(worker_gpu_cfg):
                if len(chosen_gpu_ids) >= self.num_workers:
                    break
                try:
                    gpu_id = int(entry)
                except (TypeError, ValueError):
                    self.logger.logger.warning(f'workers[{idx}] = {entry} 无法解析为GPU编号，跳过')
                    continue
                if gpu_id < 0 or gpu_id >= available_gpu_count:
                    self.logger.logger.warning(f'workers[{idx}] = {gpu_id} 超出可用GPU范围 [0, {available_gpu_count - 1}]')
                    continue
                chosen_gpu_ids.append(gpu_id)
        if not chosen_gpu_ids:
            for worker_idx in range(self.num_workers):
                chosen_gpu_ids.append(worker_idx % available_gpu_count)
        for worker_idx in range(self.num_workers):
            if worker_idx < len(chosen_gpu_ids):
                self.assigned_gpu_ids[worker_idx] = chosen_gpu_ids[worker_idx]
                devices[worker_idx] = torch.device('cuda:0')
            else:
                devices[worker_idx] = torch.device('cpu')
        return devices

    def setup_distributed_training(self, num_workers: Optional[int]=None) -> None:
        if num_workers is not None:
            self.num_workers = num_workers
            self.devices = self._setup_devices()
        self.partitions = PartitionGraph.get_partitions(dataset_name=self.dataset_name, num_partitions=self.num_workers, max_hops=2, strategy=self.partition_strategy)
        from data.data_utils import get_dataset
        _, full_graph = get_dataset(self.dataset_name, root='dataset')
        self.global_node_features_cpu = full_graph.x.cpu()
        self.global_feature_ref = ray.put(self.global_node_features_cpu)
        self.workers = []
        for i in range(self.num_workers):
            assigned_gpu = self.assigned_gpu_ids[i] if i < len(self.assigned_gpu_ids) else None
            target_device = self.devices[i]
            if self.device_type == 'cuda' and torch.cuda.is_available():
                worker = CORayWorker.options(num_gpus=1).remote(worker_id=i, data_partition=self.partitions[i], device=target_device, config=self.config, global_feature_ref=self.global_feature_ref or self.global_node_features_cpu, physical_gpu_id=assigned_gpu)
            else:
                worker = CORayWorker.options(num_gpus=0, num_cpus=2).remote(worker_id=i, data_partition=self.partitions[i], device=target_device, config=self.config, global_feature_ref=self.global_feature_ref or self.global_node_features_cpu, physical_gpu_id=None)
            self.workers.append(worker)

    def train_epoch(self, arch_spec: Dict[str, Any], model_state: Optional[Dict]=None, epochs: int=1, batch_size: Optional[int]=None, steps_per_epoch: Optional[int]=None) -> Tuple[Dict, float, Dict[str, float]]:
        if not self.workers:
            raise RuntimeError('请先调用 setup_distributed_training() 初始化环境')
        total_steps = epochs * (steps_per_epoch if steps_per_epoch and steps_per_epoch > 0 else 1)
        use_optimized = self.co_config.get('use_optimized_communication', True)
        self.logger.logger.info(f"开始CO batch级同步训练，{len(self.workers)} 个worker，总steps={total_steps}, batch_size={batch_size}, 使用{('优化版' if use_optimized else '原始版')}通信")
        current_state = model_state
        accumulated_loss = 0.0
        completed_steps = 0
        import time
        total_sync_time = 0.0
        total_param_transfer_time = 0.0
        total_param_aggregate_time = 0.0
        aggregated_timing = {'model_build': 0.0, 'model_migration': 0.0, 'sampling': 0.0, 'feature_assembly': 0.0, 'forward': 0.0, 'backward': 0.0, 'optimizer_step': 0.0}
        for step in range(total_steps):
            training_futures = []
            for worker in self.workers:
                if use_optimized:
                    future = worker.train_on_partition_with_communication_optimized.remote(arch_spec, current_state, epochs=1, batch_size=batch_size, steps_per_epoch=1)
                else:
                    future = worker.train_on_partition_with_communication.remote(arch_spec, current_state, epochs=1, batch_size=batch_size)
                training_futures.append(future)
            ray_get_start = time.perf_counter()
            training_results = ray.get(training_futures)
            ray_get_end = time.perf_counter()
            step_ray_get_time = ray_get_end - ray_get_start
            worker_states = []
            batch_loss = 0.0
            worker_total_times = []
            for worker_state, worker_loss, worker_timing in training_results:
                worker_states.append(worker_state)
                batch_loss += worker_loss
                for key in aggregated_timing:
                    aggregated_timing[key] += worker_timing.get(key, 0.0)
                worker_total_times.append(sum(worker_timing.values()))
            max_worker_time = max(worker_total_times) if worker_total_times else 0.0
            step_ray_comm = max(0.0, step_ray_get_time - max_worker_time)
            total_param_transfer_time += step_ray_comm
            sync_start = time.perf_counter()
            current_state = self.synchronize_parameters(worker_states)
            sync_end = time.perf_counter()
            step_aggregate_time = sync_end - sync_start
            total_param_aggregate_time += step_aggregate_time
            step_sync_time = step_ray_comm + step_aggregate_time
            total_sync_time += step_sync_time
            accumulated_loss += batch_loss / len(self.workers)
            completed_steps += 1
            if (step + 1) % 10 == 0 or step + 1 == total_steps:
                self.logger.logger.info(f'CO Step {step + 1}/{total_steps} 完成，当前batch平均loss: {batch_loss / len(self.workers):.4f}, 同步时间: {step_sync_time:.4f}s (Ray通信={step_ray_comm:.4f}s, 参数聚合={step_aggregate_time:.4f}s)')
        avg_loss = accumulated_loss / completed_steps if completed_steps > 0 else 0.0
        self.logger.logger.info(f'CO Batch级同步训练完成，平均损失: {avg_loss:.4f}')
        timing_summary = {'model_migration': aggregated_timing['model_migration'], 'sampling': aggregated_timing['sampling'], 'data_transfer': aggregated_timing['feature_assembly'], 'forward': aggregated_timing['forward'], 'backward': aggregated_timing['backward'], 'optimizer_step': aggregated_timing['optimizer_step'], 'sync_time': total_sync_time}
        self.logger.logger.debug(f'CO同步时间分解: 总计={total_sync_time:.3f}s (Ray通信={total_param_transfer_time:.3f}s, 参数聚合={total_param_aggregate_time:.3f}s)')
        return (current_state, avg_loss, timing_summary)

    def evaluate_model(self, arch_spec: Dict[str, Any], model_state: Dict) -> Dict[str, Any]:
        if not self.workers:
            raise RuntimeError('请先调用 setup_distributed_training() 初始化环境')
        evaluation_futures = [worker.evaluate_on_partition_with_communication.remote(arch_spec, model_state) for worker in self.workers]
        evaluation_results = ray.get(evaluation_futures)
        total_correct = sum((result.get('correct_eval_nodes', 0) for result in evaluation_results))
        total_nodes = sum((result.get('total_eval_nodes', 0) for result in evaluation_results))
        if total_nodes == 0:
            return {'global_accuracy': 0.0, 'total_unique_nodes': 0, 'correct_predictions': 0, 'worker_stats': [], 'evaluation_successful': False}
        global_accuracy = total_correct / total_nodes
        worker_stats = []
        for result in evaluation_results:
            worker_id = result.get('worker_id')
            total_eval_nodes = result.get('total_eval_nodes', 0)
            correct_eval_nodes = result.get('correct_eval_nodes', 0)
            worker_accuracy = correct_eval_nodes / total_eval_nodes if total_eval_nodes > 0 else 0.0
            worker_stats.append({'worker_id': worker_id, 'assigned_eval_nodes': total_eval_nodes, 'correct_eval_nodes': correct_eval_nodes, 'accuracy': worker_accuracy})
        return {'global_accuracy': global_accuracy, 'total_unique_nodes': total_nodes, 'correct_predictions': total_correct, 'incorrect_predictions': total_nodes - total_correct, 'worker_stats': worker_stats, 'evaluation_successful': True}

    def synchronize_parameters(self, model_states: List[Dict]) -> Dict:
        if not model_states:
            raise ValueError('模型参数列表不能为空')
        param_names = model_states[0].keys()
        synchronized_state = {}
        for param_name in param_names:
            param_tensors = [state[param_name] for state in model_states]
            target_device = param_tensors[0].device
            param_tensors = [tensor.to(target_device) for tensor in param_tensors]
            if len(param_tensors) > 1:
                stacked_params = torch.stack(param_tensors, dim=0)
                avg_param = torch.mean(stacked_params, dim=0)
            else:
                avg_param = param_tensors[0]
            synchronized_state[param_name] = avg_param
        return synchronized_state

    def get_partition_info(self) -> Dict[str, Any]:
        if not self.partitions:
            return {}
        info = {'num_partitions': len(self.partitions), 'total_nodes': sum((p.num_nodes for p in self.partitions)), 'total_edges': sum((p.num_edges for p in self.partitions)), 'strategy': 'communication_only', 'partitions': []}
        for i, partition in enumerate(self.partitions):
            partition_info = {'id': i, 'nodes': partition.num_nodes, 'edges': partition.num_edges, 'train_nodes': partition.train_mask.sum().item(), 'val_nodes': partition.val_mask.sum().item() if hasattr(partition, 'val_mask') else 0, 'test_nodes': partition.test_mask.sum().item() if hasattr(partition, 'test_mask') else 0}
            info['partitions'].append(partition_info)
        return info