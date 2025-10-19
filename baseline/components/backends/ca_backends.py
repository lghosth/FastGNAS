import os
import sys
import torch
import torch.nn.functional as F
from typing import Dict, Any, List, Tuple, Optional
import ray
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
from data.partition import PartitionGraph
from trainer.builder import ModelBuilder
from utils.logger import GraphNASLogger

@ray.remote(num_gpus=1)
class RayWorker:

    def __init__(self, worker_id: int, data_partition: Data, device: torch.device, config: Dict[str, Any]):
        self.worker_id = worker_id
        self.device = device
        self.config = config
        self.logger = GraphNASLogger(name=f'simple_worker_{worker_id}')
        self.data_partition_cpu = data_partition.clone().cpu()
        self.data_partition = data_partition.clone().to(device)
        self.sampling_data_cpu = Data(edge_index=self.data_partition_cpu.edge_index.clone() if self.data_partition_cpu.edge_index is not None else None, num_nodes=self.data_partition_cpu.num_nodes, train_mask=self.data_partition_cpu.train_mask.clone() if hasattr(self.data_partition_cpu, 'train_mask') and self.data_partition_cpu.train_mask is not None else None, val_mask=self.data_partition_cpu.val_mask.clone() if hasattr(self.data_partition_cpu, 'val_mask') and self.data_partition_cpu.val_mask is not None else None, test_mask=self.data_partition_cpu.test_mask.clone() if hasattr(self.data_partition_cpu, 'test_mask') and self.data_partition_cpu.test_mask is not None else None)
        self.logger.logger.info(f'SimpleWorker {worker_id} 创建轻量级采样数据对象（仅边关系），与CO后端对齐')
        if hasattr(self.data_partition_cpu, 'global_node_ids'):
            self.global_node_ids_cpu = self.data_partition_cpu.global_node_ids
        else:
            self.global_node_ids_cpu = None
        if hasattr(self.data_partition, 'global_node_ids'):
            self.global_node_ids = self.data_partition.global_node_ids
            if hasattr(self.global_node_ids, 'to'):
                self.global_node_ids = self.global_node_ids.to(device)
        else:
            self.global_node_ids = None
        self.logger.logger.info(f'SimpleWorker {worker_id} 初始化完成，数据分区包含 {self.data_partition.num_nodes} 个节点')

    def update_training_config(self, config_updates: Dict[str, Any]) -> None:
        if not isinstance(config_updates, dict):
            return
        self.config.update(config_updates)

    def _resolve_neighbor_sizes(self, arch_spec: Dict[str, Any]) -> List[int]:
        layer_count = 1
        if isinstance(arch_spec, dict):
            if isinstance(arch_spec.get('layers'), list) and arch_spec['layers']:
                layer_count = len(arch_spec['layers'])
            elif 'cell_spec' in arch_spec and arch_spec.get('num_cells_to_stack'):
                layer_count = int(arch_spec.get('num_cells_to_stack', 1))
        layer_count = max(layer_count, 1)
        configured_sizes = self.config.get('neighbor_sizes')
        if configured_sizes is None:
            configured_sizes = self.config.get('fan_out')
        default_size = int(self.config.get('default_neighbor_size', 15))
        if configured_sizes is None:
            return [default_size for _ in range(layer_count)]
        if isinstance(configured_sizes, int):
            return [configured_sizes for _ in range(layer_count)]
        configured_list = list(configured_sizes)
        if not configured_list:
            return [default_size for _ in range(layer_count)]
        if len(configured_list) < layer_count:
            configured_list.extend([configured_list[-1]] * (layer_count - len(configured_list)))
        elif len(configured_list) > layer_count:
            configured_list = configured_list[:layer_count]
        return configured_list

    def train_on_partition(self, arch_spec: Dict[str, Any], model_state: Optional[Dict], epochs: int, batch_size: Optional[int]=None, steps_per_epoch: Optional[int]=None) -> Tuple[Dict, float, Dict[str, float]]:
        import time
        model_build_start = time.perf_counter()
        input_dim = self.data_partition.num_node_features
        output_dim = self.config.get('num_classes', 7)
        model = ModelBuilder.build_model(arch_spec, input_dim, output_dim, self.config).to(self.device)
        model_build_time = time.perf_counter() - model_build_start
        self.logger.logger.debug(f'[Worker {self.worker_id}] 模型构建耗时: {model_build_time * 1000:.2f}ms')
        model_migration_time = 0.0
        if model_state is not None:
            t_migration_start = time.perf_counter()
            model.load_state_dict(model_state)
            t_migration_end = time.perf_counter()
            model_migration_time = t_migration_end - t_migration_start
            self.logger.logger.debug(f'[Worker {self.worker_id}] 模型迁移耗时: {model_migration_time * 1000:.2f}ms')
        optimizer = torch.optim.Adam(model.parameters(), lr=self.config.get('learning_rate', 0.01), weight_decay=self.config.get('weight_decay', 0.0005))
        model.train()
        total_loss = 0.0
        total_steps = 0
        train_mask = self.data_partition_cpu.train_mask
        train_indices = torch.where(train_mask)[0]
        if train_indices.numel() == 0:
            self.logger.logger.warning(f'Worker {self.worker_id} 训练集为空')
            return (model.state_dict(), 0.0)
        effective_batch_size = batch_size if batch_size and batch_size > 0 else train_indices.numel()
        effective_batch_size = max(1, min(effective_batch_size, train_indices.numel()))
        neighbor_sizes = self._resolve_neighbor_sizes(arch_spec)
        timing_stats = {'model_build': model_build_time, 'model_migration': model_migration_time, 'sampling': 0.0, 'data_transfer': 0.0, 'forward': 0.0, 'backward': 0.0, 'optimizer_step': 0.0}
        for _ in range(epochs):
            step_count = 0
            max_steps = steps_per_epoch if steps_per_epoch and steps_per_epoch > 0 else float('inf')
            while step_count < max_steps:
                t0 = time.perf_counter()
                try:
                    train_loader = NeighborLoader(self.sampling_data_cpu, num_neighbors=neighbor_sizes, input_nodes=train_mask, batch_size=effective_batch_size, shuffle=True)
                    train_iter = iter(train_loader)
                    batch_cpu = next(train_iter)
                except StopIteration:
                    break
                t1 = time.perf_counter()
                timing_stats['sampling'] += t1 - t0
                batch_size_local = int(getattr(batch_cpu, 'batch_size', 0))
                if batch_size_local == 0:
                    continue
                t_transfer_start = time.perf_counter()
                n_id_gpu = batch_cpu.n_id.to(self.device)
                edge_index_gpu = batch_cpu.edge_index.to(self.device)
                batch_x_gpu = self.data_partition.x[n_id_gpu]
                batch_y_gpu = self.data_partition.y[n_id_gpu[:batch_size_local]]
                if self.device.type == 'cuda':
                    torch.cuda.synchronize(self.device)
                t_transfer_end = time.perf_counter()
                timing_stats['data_transfer'] += t_transfer_end - t_transfer_start
                t_forward_start = time.perf_counter()
                optimizer.zero_grad()
                out = model(batch_x_gpu, edge_index_gpu)
                loss = F.cross_entropy(out[:batch_size_local], batch_y_gpu)
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
                total_loss += loss.item()
                total_steps += 1
                step_count += 1
                del train_loader, train_iter, batch_cpu, n_id_gpu, edge_index_gpu
                del batch_x_gpu, batch_y_gpu, out, loss
                if self.device.type == 'cuda' and torch.cuda.is_available():
                    torch.cuda.empty_cache()
        total_time = sum(timing_stats.values())
        if total_time > 0:
            compute_time = timing_stats['forward'] + timing_stats['backward']
            self.logger.logger.info(f"[性能分析-CA] Worker {self.worker_id} - 总时间={total_time:.2f}s, 模型构建={timing_stats['model_build']:.4f}s ({timing_stats['model_build'] / total_time * 100:.1f}%), 模型迁移={timing_stats['model_migration']:.4f}s ({timing_stats['model_migration'] / total_time * 100:.1f}%), 采样={timing_stats['sampling']:.2f}s ({timing_stats['sampling'] / total_time * 100:.1f}%), 数据传输={timing_stats['data_transfer']:.2f}s ({timing_stats['data_transfer'] / total_time * 100:.1f}%), 计算时间={compute_time:.2f}s ({compute_time / total_time * 100:.1f}%), [前向={timing_stats['forward']:.2f}s, 反向={timing_stats['backward']:.2f}s], 参数更新={timing_stats['optimizer_step']:.2f}s ({timing_stats['optimizer_step'] / total_time * 100:.1f}%)")
        avg_loss = total_loss / total_steps if total_steps > 0 else 0.0
        self.logger.logger.debug(f'Worker {self.worker_id} 训练完成，平均损失: {avg_loss:.4f}')
        return (model.state_dict(), avg_loss, timing_stats)

    def evaluate_on_partition(self, arch_spec: Dict[str, Any], model_state: Dict) -> Dict[str, Any]:
        input_dim = self.data_partition.num_node_features
        output_dim = self.config.get('num_classes', 7)
        model = ModelBuilder.build_model(arch_spec, input_dim, output_dim, self.config).to(self.device)
        model.load_state_dict(model_state)
        model.eval()
        with torch.no_grad():
            assert hasattr(self.data_partition_cpu, 'eval_mask'), f'Worker {self.worker_id} 分区必须包含eval_mask'
            eval_mask = self.data_partition_cpu.eval_mask
            assert isinstance(eval_mask, torch.Tensor), f'Worker {self.worker_id} eval_mask必须是torch.Tensor'
            if eval_mask.sum() == 0:
                self.logger.logger.debug(f'Worker {self.worker_id} 未分配评估节点')
                return {'worker_id': self.worker_id, 'total_eval_nodes': 0, 'correct_eval_nodes': 0, 'correct_node_ids': [], 'incorrect_node_ids': []}
            assert self.global_node_ids is not None, f'Worker {self.worker_id} 必须有global_node_ids映射'
            neighbor_sizes = self._resolve_neighbor_sizes(arch_spec)
            eval_batch_size = self.config.get('eval_batch_size', 1024)
            eval_loader = NeighborLoader(self.sampling_data_cpu, num_neighbors=neighbor_sizes, input_nodes=eval_mask, batch_size=max(1, int(eval_batch_size)), shuffle=False)
            total_eval_nodes = 0
            correct_eval_nodes = 0
            correct_ids: List[int] = []
            incorrect_ids: List[int] = []
            for batch in eval_loader:
                batch_size_local = int(getattr(batch, 'batch_size', 0))
                if batch_size_local == 0:
                    continue
                seed_local_ids = batch.n_id[:batch_size_local]
                n_id_gpu = batch.n_id.to(self.device)
                edge_index_gpu = batch.edge_index.to(self.device)
                batch_x_gpu = self.data_partition.x[n_id_gpu]
                batch_y_gpu = self.data_partition.y[n_id_gpu[:batch_size_local]]
                out = model(batch_x_gpu, edge_index_gpu)
                predictions = out[:batch_size_local].argmax(dim=1)
                ground_truth = batch_y_gpu
                correct_mask = predictions == ground_truth
                total_eval_nodes += batch_size_local
                correct_eval_nodes += int(correct_mask.sum().item())
                global_ids = self.global_node_ids[seed_local_ids.to(self.global_node_ids.device)].cpu()
                for gid, is_correct in zip(global_ids.tolist(), correct_mask.cpu().tolist()):
                    if is_correct:
                        correct_ids.append(gid)
                    else:
                        incorrect_ids.append(gid)
                del seed_local_ids, n_id_gpu, edge_index_gpu, batch_x_gpu, batch_y_gpu, out, predictions, ground_truth, correct_mask, global_ids
            del eval_loader
            evaluation_result = {'worker_id': self.worker_id, 'total_eval_nodes': int(total_eval_nodes), 'correct_eval_nodes': int(correct_eval_nodes), 'correct_node_ids': correct_ids, 'incorrect_node_ids': incorrect_ids}
            self.logger.logger.debug(f"Worker {self.worker_id} 评估完成: {evaluation_result['total_eval_nodes']}个节点, 正确 {evaluation_result['correct_eval_nodes']}")
            return evaluation_result

    def get_val_nodes(self) -> Dict[str, Any]:
        val_nodes = self.data_partition.val_mask.sum().item()
        if val_nodes == 0:
            val_nodes = self.data_partition.test_mask.sum().item()
            mask_type = 'test'
        else:
            mask_type = 'val'
        val_mask = self.data_partition.val_mask if mask_type == 'val' else self.data_partition.test_mask
        val_indices = torch.where(val_mask)[0]
        if self.global_node_ids is not None:
            global_ids = self.global_node_ids[val_indices].cpu().tolist()
        else:
            global_ids = val_indices.cpu().tolist()
        return {'worker_id': self.worker_id, 'num_val_nodes': val_nodes, 'mask_type': mask_type, 'global_node_ids': global_ids}

    def warmup_worker(self, warmup_epochs: int=3) -> Dict[str, Any]:
        warmup_result = {'worker_id': self.worker_id, 'device': str(self.device), 'gpu_warmed': False, 'data_warmed': False, 'model_warmed': False, 'warmup_time': 0.0}
        import time
        start_time = time.time()
        if self.device.type == 'cuda':
            dummy_tensor = torch.randn(1000, 1000, device=self.device)
            for _ in range(5):
                result = torch.matmul(dummy_tensor, dummy_tensor.T)
                result.sum().backward() if result.requires_grad else None
            torch.cuda.synchronize(self.device)
            warmup_result['gpu_warmed'] = True
            self.logger.logger.debug(f'Worker {self.worker_id} GPU预热完成')
        _ = self.data_partition.x.to(self.device)
        _ = self.data_partition.edge_index.to(self.device)
        _ = self.data_partition.y.to(self.device)
        if self.device.type == 'cuda':
            torch.cuda.synchronize(self.device)
        warmup_result['data_warmed'] = True
        self.logger.logger.debug(f'Worker {self.worker_id} 数据预热完成')
        warmup_arch = {'layers': [{'type': 'gcn', 'hidden_dim': 32, 'act': 'relu'}, {'type': 'gcn', 'hidden_dim': 16, 'act': 'relu'}]}
        input_dim = self.data_partition.num_node_features
        output_dim = self.config.get('num_classes', 7)
        warmup_model = ModelBuilder.build_model(warmup_arch, input_dim, output_dim, self.config).to(self.device)
        warmup_optimizer = torch.optim.Adam(warmup_model.parameters(), lr=0.01)
        warmup_model.train()
        for _ in range(warmup_epochs):
            warmup_optimizer.zero_grad()
            out = warmup_model(self.data_partition.x, self.data_partition.edge_index)
            train_mask = self.data_partition.train_mask
            if train_mask.sum() > 0:
                loss = F.cross_entropy(out[train_mask], self.data_partition.y[train_mask])
                loss.backward()
                warmup_optimizer.step()
        if self.device.type == 'cuda':
            torch.cuda.synchronize(self.device)
        warmup_result['model_warmed'] = True
        self.logger.logger.debug(f'Worker {self.worker_id} 模型预热完成')
        warmup_result['warmup_time'] = time.time() - start_time
        self.logger.logger.info(f"Worker {self.worker_id} 预热完成，耗时 {warmup_result['warmup_time']:.3f}s")
        return warmup_result

class CABackends:

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = GraphNASLogger(name='ca_backends')
        self.num_workers = config.get('num_workers', 2)
        self.dataset_name = config.get('dataset', 'Cora')
        self.device_type = config.get('device', 'cuda')
        self.ray_address = config.get('ray_address', None)
        self.worker_gpu_ids: List[int] = self._normalize_worker_gpu_ids()
        if self.worker_gpu_ids:
            if len(self.worker_gpu_ids) != self.num_workers:
                self.logger.logger.warning(f'配置的GPU列表长度({len(self.worker_gpu_ids)})与worker数量({self.num_workers})不一致，将按num_workers调整GPU列表（命令行--workers参数优先）')
                if len(self.worker_gpu_ids) > self.num_workers:
                    self.worker_gpu_ids = self.worker_gpu_ids[:self.num_workers]
                    self.logger.logger.info(f'GPU列表已截取为前{self.num_workers}个: {self.worker_gpu_ids}')
                else:
                    original_list = self.worker_gpu_ids.copy()
                    while len(self.worker_gpu_ids) < self.num_workers:
                        idx = len(self.worker_gpu_ids)
                        self.worker_gpu_ids.append(original_list[idx % len(original_list)])
                    self.logger.logger.info(f'GPU列表已循环扩展为{self.num_workers}个: {self.worker_gpu_ids}')
        self._apply_sampling_config()
        self.assigned_gpu_ids: List[int] = []
        self.ray_initialized = False
        self.cluster_resources = {}
        self.devices = self._setup_devices()
        self.partitions = []
        self.workers = []
        self._initialize_ray_cluster()
        self.logger.logger.info(f'CABackends 初始化完成，使用 {self.num_workers} 个worker，数据集: {self.dataset_name}')

    def _initialize_ray_cluster(self) -> None:
        if not ray.is_initialized():
            if self.ray_address:
                ray.init(address=self.ray_address)
                self.logger.logger.info(f'连接到Ray集群: {self.ray_address}')
            else:
                ray_config = {}
                if self.device_type == 'cuda' and torch.cuda.is_available():
                    ray_config['num_gpus'] = torch.cuda.device_count()
                ray.init(**ray_config)
                self.logger.logger.info('初始化本地Ray集群')
        self.ray_initialized = True
        self.cluster_resources = ray.cluster_resources()
        self.logger.logger.info(f'Ray集群资源: {self.cluster_resources}')

    def _normalize_worker_gpu_ids(self) -> List[int]:
        workers_cfg = self.config.get('workers')
        if not isinstance(workers_cfg, (list, tuple)):
            return []
        normalized: List[int] = []
        for idx, gpu_id in enumerate(workers_cfg):
            try:
                normalized.append(int(gpu_id))
            except (TypeError, ValueError):
                self.logger.logger.warning(f'忽略无效的GPU编号配置 workers[{idx}]={gpu_id}')
        return normalized

    def _apply_sampling_config(self) -> None:
        if 'neighbor_sizes' not in self.config:
            fan_out = self.config.get('fan_out')
            if fan_out is not None:
                self.config['neighbor_sizes'] = fan_out
        neighbor_sizes = self.config.get('neighbor_sizes')
        if isinstance(neighbor_sizes, (list, tuple)) and neighbor_sizes:
            default_size = neighbor_sizes[-1]
            self.config.setdefault('default_neighbor_size', default_size)

    def _setup_devices(self) -> List[torch.device]:
        self.assigned_gpu_ids = []
        if self.device_type == 'cuda':
            if torch.cuda.is_available():
                visible_gpu_total = torch.cuda.device_count()
                if visible_gpu_total < self.num_workers:
                    self.logger.logger.warning(f'可见GPU数量({visible_gpu_total}) 少于 worker 数({self.num_workers})，Ray可能无法分配全部worker')
                devices = [torch.device('cuda:0') for _ in range(self.num_workers)]
                if self.worker_gpu_ids:
                    self.assigned_gpu_ids = self.worker_gpu_ids[:self.num_workers]
                    self.logger.logger.info(f'使用配置的GPU列表: {self.worker_gpu_ids}')
                else:
                    self.assigned_gpu_ids = list(range(self.num_workers))
                    self.logger.logger.info('未指定GPU列表，按Ray分配顺序使用可见GPU')
                self.logger.logger.info(f'可见GPU数量: {visible_gpu_total}')
                self.logger.logger.info(f'设备配置: {devices}')
                return devices
            self.logger.logger.warning('未检测到可用GPU，自动切换到CPU设备')
            return self._setup_cpu_devices()
        return self._setup_cpu_devices()

    def _setup_cpu_devices(self) -> List[torch.device]:
        devices = [torch.device('cpu') for _ in range(self.num_workers)]
        self.assigned_gpu_ids = []
        self.logger.logger.info('使用CPU设备')
        self.logger.logger.info(f'设备配置: {devices}')
        return devices

    def setup_distributed_training(self, num_workers: Optional[int]=None) -> None:
        if num_workers is not None:
            self.num_workers = num_workers
            self.devices = self._setup_devices()
        self.logger.logger.info('开始生成数据分区...')
        partition_strategy = self.config.get('partition_strategy', 'balanced_stratified_khop')
        self.partitions = PartitionGraph.get_partitions(dataset_name=self.dataset_name, num_partitions=self.num_workers, max_hops=2, strategy=partition_strategy)
        for i, partition in enumerate(self.partitions):
            train_nodes = partition.train_mask.sum().item()
            val_nodes = partition.val_mask.sum().item()
            test_nodes = partition.test_mask.sum().item()
            self.logger.logger.info(f'分区 {i}: {partition.num_nodes} 节点, 训练={train_nodes}, 验证={val_nodes}, 测试={test_nodes}')
        self.workers = []
        for i in range(self.num_workers):
            if self.device_type == 'cuda' and torch.cuda.is_available():
                worker = RayWorker.options(num_gpus=1).remote(worker_id=i, data_partition=self.partitions[i], device=self.devices[i], config=self.config)
            else:
                worker = RayWorker.options(num_gpus=0, num_cpus=2).remote(worker_id=i, data_partition=self.partitions[i], device=self.devices[i], config=self.config)
            self.workers.append(worker)
            if self.device_type == 'cuda' and torch.cuda.is_available():
                physical_gpu = None
                if self.assigned_gpu_ids and i < len(self.assigned_gpu_ids):
                    physical_gpu = self.assigned_gpu_ids[i]
                if physical_gpu is not None:
                    bind_desc = f'cuda:0 (physical gpu {physical_gpu})'
                else:
                    bind_desc = str(self.devices[i])
            else:
                bind_desc = str(self.devices[i])
            self.logger.logger.info(f'Worker {i} 绑定到 {bind_desc}')
        self.logger.logger.info('分布式训练环境设置完成')

    def warmup_system(self, warmup_epochs: int=3, warmup_architectures: int=2) -> Dict[str, Any]:
        if not self.workers:
            raise RuntimeError('请先调用 setup_distributed_training() 初始化环境')
        import time
        total_start_time = time.time()
        warmup_result = {'total_workers': len(self.workers), 'worker_warmup_time': 0.0, 'architecture_warmup_time': 0.0, 'total_warmup_time': 0.0, 'workers_warmed': 0, 'architectures_tested': 0, 'warmup_successful': False}
        self.logger.logger.info(f'开始系统预热：{len(self.workers)}个worker，{warmup_architectures}个测试架构')
        worker_warmup_start = time.time()
        self.logger.logger.info('阶段1：Worker预热...')
        worker_warmup_futures = []
        for worker in self.workers:
            future = worker.warmup_worker.remote(warmup_epochs)
            worker_warmup_futures.append(future)
        worker_warmup_results = ray.get(worker_warmup_futures)
        successful_workers = 0
        for result in worker_warmup_results:
            if result.get('gpu_warmed', False) and result.get('model_warmed', False):
                successful_workers += 1
                self.logger.logger.debug(f"Worker {result['worker_id']} 预热成功，耗时 {result['warmup_time']:.3f}s")
            else:
                self.logger.logger.warning(f"Worker {result['worker_id']} 预热失败")
        warmup_result['workers_warmed'] = successful_workers
        warmup_result['worker_warmup_time'] = time.time() - worker_warmup_start
        self.logger.logger.info(f"Worker预热完成：{successful_workers}/{len(self.workers)} 成功，耗时 {warmup_result['worker_warmup_time']:.3f}s")
        arch_warmup_start = time.time()
        self.logger.logger.info('阶段2：架构预热...')
        test_architectures = [{'layers': [{'type': 'gcn', 'hidden_dim': 64, 'act': 'relu'}, {'type': 'gcn', 'hidden_dim': 32, 'act': 'relu'}]}, {'layers': [{'type': 'gat', 'hidden_dim': 32, 'heads': 2, 'act': 'elu'}, {'type': 'gcn', 'hidden_dim': 16, 'act': 'relu'}]}]
        architectures_tested = 0
        for i, arch_spec in enumerate(test_architectures[:warmup_architectures]):
            try:
                self.logger.logger.debug(f'测试架构 {i + 1}/{warmup_architectures}')
                _, warmup_loss, _ = self.train_epoch(arch_spec, epochs=1)
                architectures_tested += 1
                self.logger.logger.debug(f'架构 {i + 1} 预热成功，损失: {warmup_loss:.4f}')
            except Exception as e:
                self.logger.logger.warning(f'架构 {i + 1} 预热失败: {e}')
        warmup_result['architectures_tested'] = architectures_tested
        warmup_result['architecture_warmup_time'] = time.time() - arch_warmup_start
        self.logger.logger.info(f"架构预热完成：{architectures_tested}/{warmup_architectures} 成功，耗时 {warmup_result['architecture_warmup_time']:.3f}s")
        warmup_result['total_warmup_time'] = time.time() - total_start_time
        warmup_result['warmup_successful'] = successful_workers >= len(self.workers) * 0.8 and architectures_tested > 0
        if warmup_result['warmup_successful']:
            self.logger.logger.info(f"系统预热成功完成！总耗时: {warmup_result['total_warmup_time']:.3f}s, Worker成功率: {successful_workers}/{len(self.workers)}, 架构测试: {architectures_tested}/{warmup_architectures}")
        else:
            self.logger.logger.warning(f"系统预热部分失败！总耗时: {warmup_result['total_warmup_time']:.3f}s, Worker成功率: {successful_workers}/{len(self.workers)}, 架构测试: {architectures_tested}/{warmup_architectures}")
        return warmup_result

    def update_training_config(self, config_updates: Dict[str, Any]) -> None:
        if not isinstance(config_updates, dict) or not config_updates:
            return
        self.config.update(config_updates)
        if not self.workers:
            return
        update_futures = [worker.update_training_config.remote(config_updates) for worker in self.workers]
        ray.get(update_futures)

    def train_epoch(self, arch_spec: Dict[str, Any], model_state: Optional[Dict]=None, epochs: int=1, batch_size: Optional[int]=None, steps_per_epoch: Optional[int]=None) -> Tuple[Dict, float, Dict[str, float]]:
        if not self.workers:
            raise RuntimeError('请先调用 setup_distributed_training() 初始化环境')
        total_steps = epochs * (steps_per_epoch if steps_per_epoch and steps_per_epoch > 0 else 1)
        self.logger.logger.info(f'开始batch级同步训练，{len(self.workers)} 个worker，总steps={total_steps}, batch_size={batch_size}')
        current_state = model_state
        accumulated_loss = 0.0
        completed_steps = 0
        import time
        total_sync_time = 0.0
        total_transfer_time = 0.0
        total_compute_time = 0.0
        aggregated_timing = {'model_build': 0.0, 'model_migration': 0.0, 'sampling': 0.0, 'data_transfer': 0.0, 'forward': 0.0, 'backward': 0.0, 'optimizer_step': 0.0}
        for step in range(total_steps):
            training_futures = []
            for worker in self.workers:
                future = worker.train_on_partition.remote(arch_spec, current_state, epochs=1, batch_size=batch_size, steps_per_epoch=1)
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
            total_transfer_time += step_ray_comm
            sync_start = time.perf_counter()
            current_state = self.synchronize_parameters(worker_states)
            sync_end = time.perf_counter()
            step_aggregate_time = sync_end - sync_start
            total_compute_time += step_aggregate_time
            step_sync_time = step_ray_comm + step_aggregate_time
            total_sync_time += step_sync_time
            accumulated_loss += batch_loss / len(self.workers)
            completed_steps += 1
            if (step + 1) % 10 == 0 or step + 1 == total_steps:
                self.logger.logger.info(f'Step {step + 1}/{total_steps} 完成，当前batch平均loss: {batch_loss / len(self.workers):.4f}, 同步时间: {step_sync_time:.4f}s (Ray通信={step_ray_comm:.4f}s, 参数聚合={step_aggregate_time:.4f}s)')
        avg_loss = accumulated_loss / completed_steps if completed_steps > 0 else 0.0
        self.logger.logger.info(f'Batch级同步训练完成，平均损失: {avg_loss:.4f}')
        timing_summary = {**aggregated_timing, 'sync_time': total_sync_time}
        self.logger.logger.debug(f'CA同步时间分解: 总计={total_sync_time:.3f}s (Ray通信={total_transfer_time:.3f}s, 参数聚合={total_compute_time:.3f}s)')
        return (current_state, avg_loss, timing_summary)

    def evaluate_model(self, arch_spec: Dict[str, Any], model_state: Dict) -> Dict[str, Any]:
        if not self.workers:
            raise RuntimeError('请先调用 setup_distributed_training() 初始化环境')
        self.logger.logger.info('开始分布式模型评估...')
        evaluation_futures = [worker.evaluate_on_partition.remote(arch_spec, model_state) for worker in self.workers]
        evaluation_results = ray.get(evaluation_futures)
        total_correct = sum((result.get('correct_eval_nodes', 0) for result in evaluation_results))
        total_nodes = sum((result.get('total_eval_nodes', 0) for result in evaluation_results))
        if total_nodes == 0:
            self.logger.logger.warning('没有找到任何验证节点')
            return {'global_accuracy': 0.0, 'total_unique_nodes': 0, 'correct_predictions': 0, 'worker_stats': [], 'evaluation_successful': False}
        global_accuracy = total_correct / total_nodes
        worker_stats = []
        for result in evaluation_results:
            worker_id = result.get('worker_id')
            total_eval_nodes = result.get('total_eval_nodes', 0)
            correct_eval_nodes = result.get('correct_eval_nodes', 0)
            worker_accuracy = correct_eval_nodes / total_eval_nodes if total_eval_nodes > 0 else 0.0
            worker_stats.append({'worker_id': worker_id, 'assigned_eval_nodes': total_eval_nodes, 'correct_eval_nodes': correct_eval_nodes, 'accuracy': worker_accuracy})
        evaluation_result = {'global_accuracy': global_accuracy, 'total_unique_nodes': total_nodes, 'correct_predictions': total_correct, 'incorrect_predictions': total_nodes - total_correct, 'worker_stats': worker_stats, 'evaluation_successful': True}
        self.logger.logger.info(f'分布式评估完成 - 全局准确率: {global_accuracy:.4f}, 总节点: {total_nodes}')
        return evaluation_result

    def analyze_evaluation_result(self, eval_result: Dict[str, Any]) -> str:
        if not eval_result.get('evaluation_successful', False):
            return '评估失败：没有找到有效的验证节点'
        lines = []
        lines.append('=' * 60)
        lines.append('分布式评估结果报告')
        lines.append('=' * 60)
        lines.append(f"全局准确率: {eval_result['global_accuracy']:.4f}")
        lines.append(f"正确预测: {eval_result['correct_predictions']}/{eval_result['total_unique_nodes']}")
        lines.append(f'\n数据统计:')
        lines.append(f"  唯一验证节点: {eval_result['total_unique_nodes']}")
        lines.append(f"  原始验证节点: {eval_result.get('total_raw_nodes', eval_result['total_unique_nodes'])}")
        duplicate_nodes = eval_result.get('duplicate_nodes', 0)
        lines.append(f'  重复节点数: {duplicate_nodes}')
        total_raw_nodes = eval_result.get('total_raw_nodes', eval_result['total_unique_nodes'])
        if duplicate_nodes > 0 and total_raw_nodes > 0:
            overlap_rate = duplicate_nodes / total_raw_nodes * 100
            lines.append(f'  重复率: {overlap_rate:.2f}%')
        lines.append(f'\nWorker统计:')
        for worker_stat in eval_result['worker_stats']:
            assigned_nodes = worker_stat.get('assigned_eval_nodes', worker_stat.get('num_eval_nodes', 0))
            accuracy = worker_stat.get('accuracy')
            if accuracy is None and assigned_nodes:
                accuracy = worker_stat.get('correct_eval_nodes', 0) / assigned_nodes
            elif accuracy is None:
                accuracy = 0.0
            lines.append(f"  Worker {worker_stat['worker_id']}: 本地准确率={accuracy:.4f}, 节点数={assigned_nodes}")
        class_accuracies = eval_result.get('class_accuracies', {})
        if class_accuracies:
            lines.append(f'\n各类别准确率:')
            class_counts = eval_result.get('class_counts', {})
            for class_id, accuracy in class_accuracies.items():
                count = class_counts.get(class_id, 0)
                lines.append(f'  类别 {class_id}: {accuracy:.4f} ({count}个节点)')
        lines.append('=' * 60)
        return '\n'.join(lines)

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
        self.logger.logger.debug(f'参数同步完成，同步了{len(param_names)}个参数')
        return synchronized_state

    def get_partition_info(self) -> Dict[str, Any]:
        if not self.partitions:
            return {}
        info = {'num_partitions': len(self.partitions), 'total_nodes': sum((p.num_nodes for p in self.partitions)), 'total_edges': sum((p.num_edges for p in self.partitions)), 'partitions': []}
        for i, partition in enumerate(self.partitions):
            partition_info = {'id': i, 'nodes': partition.num_nodes, 'edges': partition.num_edges, 'train_nodes': partition.train_mask.sum().item(), 'val_nodes': partition.val_mask.sum().item(), 'test_nodes': partition.test_mask.sum().item()}
            info['partitions'].append(partition_info)
        return info