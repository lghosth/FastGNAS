import os
import sys
import torch
import torch.nn.functional as F
from typing import Dict, Any, List, Tuple, Optional
import ray
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader
import time
import numpy as np
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
from trainer.builder import ModelBuilder
from utils.logger import GraphNASLogger

@ray.remote(num_gpus=1)
class ArchParallelWorkerV2:

    def __init__(self, worker_id: int, full_data: Data, use_gpu: bool, config: Dict[str, Any]):
        self.worker_id = worker_id
        self.config = config
        self.logger = GraphNASLogger(name=f'arch_parallel_worker_v2_{worker_id}')
        if use_gpu and torch.cuda.is_available():
            self.device = torch.device('cuda:0')
            self.logger.logger.info(f'Worker {worker_id} 使用GPU (Ray内部编号: cuda:0)')
        else:
            self.device = torch.device('cpu')
            self.logger.logger.info(f'Worker {worker_id} 使用CPU')
        self.full_data = full_data.to('cpu')
        self.cpu_data = self.full_data
        training_cfg = config.get('training', {})
        self.default_batch_size = training_cfg.get('batch_size', 1024)
        fan_out = training_cfg.get('fan_out', [25, 10])
        self.num_neighbors = fan_out
        self.logger.logger.info(f'ArchParallelWorkerV2 {worker_id} 初始化完成，数据包含 {self.full_data.num_nodes} 个节点，常驻CPU内存，batch_size={self.default_batch_size}, num_neighbors={self.num_neighbors}')
        self._last_data_transfer_sync_time = 0.0
        self._last_checkpoint_sync_time = 0.0

    def _create_neighbor_loader(self, train_mask: torch.Tensor, batch_size: int) -> NeighborLoader:
        train_indices = torch.where(train_mask)[0]
        loader = NeighborLoader(self.cpu_data, num_neighbors=self.num_neighbors, batch_size=batch_size, input_nodes=train_indices, shuffle=True, num_workers=0, pin_memory=False)
        return loader

    def _prepare_batch_for_gpu(self, batch: Data) -> Data:
        gpu_batch = batch.to(self.device, non_blocking=True)
        sync_time = 0.0
        if self.device.type == 'cuda':
            sync_start = time.perf_counter()
            torch.cuda.synchronize(self.device)
            sync_time = time.perf_counter() - sync_start
        self._last_data_transfer_sync_time = sync_time
        return gpu_batch

    def train_architecture(self, arch_spec: Dict[str, Any], model_state: Optional[Dict]=None, epochs: int=1, batch_size: Optional[int]=None, steps_per_epoch: Optional[int]=None) -> Tuple[Dict, float, Dict[str, Any]]:
        start_time = time.time()
        input_dim = self.full_data.num_node_features
        output_dim = self.config.get('num_classes', 7)
        model = ModelBuilder.build_model(arch_spec, input_dim, output_dim, self.config).to(self.device)
        model_to_device_sync_time = 0.0
        if self.device.type == 'cuda':
            sync_start = time.perf_counter()
            torch.cuda.synchronize(self.device)
            model_to_device_sync_time = time.perf_counter() - sync_start
        model_migration_time = 0.0
        load_state_dict_sync_time = 0.0
        if model_state is not None:
            t_migration_start = time.perf_counter()
            model.load_state_dict(model_state)
            if self.device.type == 'cuda':
                sync_start = time.perf_counter()
                torch.cuda.synchronize(self.device)
                load_state_dict_sync_time = time.perf_counter() - sync_start
            t_migration_end = time.perf_counter()
            model_migration_time = t_migration_end - t_migration_start
            self.logger.logger.debug(f'[Worker {self.worker_id}] 模型迁移耗时: {model_migration_time * 1000:.2f}ms (load_state_dict_sync={load_state_dict_sync_time * 1000:.2f}ms)')
        optimizer = torch.optim.Adam(model.parameters(), lr=self.config.get('learning_rate', 0.01), weight_decay=self.config.get('weight_decay', 0.0005))
        effective_batch_size = batch_size if batch_size and batch_size > 0 else self.default_batch_size
        train_mask = self.full_data.train_mask
        train_indices = torch.where(train_mask)[0]
        if train_indices.numel() == 0:
            self.logger.logger.warning(f'Worker {self.worker_id} 训练集为空')
            return (model.state_dict(), 0.0, {'epochs_completed': 0, 'steps_completed': 0})
        model.train()
        total_loss = 0.0
        total_steps = 0
        epoch_losses = []
        timing_stats = {'model_migration': model_migration_time, 'sampling': 0.0, 'data_transfer': 0.0, 'forward': 0.0, 'backward': 0.0, 'optimizer_step': 0.0, 'model_to_device_sync': model_to_device_sync_time, 'load_state_dict_sync': load_state_dict_sync_time, 'batch_to_device_sync': 0.0}
        for epoch in range(epochs):
            epoch_loss = 0.0
            epoch_steps = 0
            if steps_per_epoch and steps_per_epoch > 0:
                max_steps = steps_per_epoch
            else:
                max_steps = max(1, len(train_indices) // effective_batch_size)
            for step in range(max_steps):
                t_sampling_start = time.perf_counter()
                train_loader = self._create_neighbor_loader(train_mask, effective_batch_size)
                loader_iterator = iter(train_loader)
                try:
                    cpu_batch = next(loader_iterator)
                except StopIteration:
                    self.logger.logger.warning(f'Worker {self.worker_id} 在step {step} 无法获取batch')
                    continue
                t_sampling_end = time.perf_counter()
                timing_stats['sampling'] += t_sampling_end - t_sampling_start
                optimizer.zero_grad()
                t_transfer_start = time.perf_counter()
                gpu_batch = self._prepare_batch_for_gpu(cpu_batch)
                t_transfer_end = time.perf_counter()
                timing_stats['data_transfer'] += t_transfer_end - t_transfer_start
                timing_stats.setdefault('data_transfer_sync_time', 0.0)
                timing_stats['data_transfer_sync_time'] += getattr(self, '_last_data_transfer_sync_time', 0.0)
                t_forward_start = time.perf_counter()
                out = model(gpu_batch.x, gpu_batch.edge_index)
                batch_size_actual = gpu_batch.batch_size
                target_out = out[:batch_size_actual]
                target_y = gpu_batch.y[:batch_size_actual]
                loss = F.cross_entropy(target_out, target_y)
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
                epoch_loss += loss.item()
                epoch_steps += 1
                total_steps += 1
                if step % 10 == 0:
                    self.logger.logger.debug(f'Worker {self.worker_id} Epoch {epoch + 1}/{epochs}, Step {step + 1}/{max_steps}, Loss: {loss.item():.4f}, Batch size: {batch_size_actual}')
                del train_loader, loader_iterator, cpu_batch, gpu_batch
                del target_out, target_y, out, loss
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            avg_epoch_loss = epoch_loss / epoch_steps if epoch_steps > 0 else 0.0
            epoch_losses.append(avg_epoch_loss)
            total_loss += epoch_loss
            self.logger.logger.info(f'Worker {self.worker_id} 完成 Epoch {epoch + 1}/{epochs}, 平均损失: {avg_epoch_loss:.4f}, 步数: {epoch_steps}')
        avg_loss = total_loss / total_steps if total_steps > 0 else 0.0
        train_time = time.time() - start_time
        total_timing = sum(timing_stats.values())
        if total_timing > 0:
            compute_time = timing_stats['forward'] + timing_stats['backward']
            self.logger.logger.info(f"[性能分析-ParallelV2] Worker {self.worker_id} - 总时间={total_timing:.2f}s, 模型迁移={timing_stats['model_migration']:.4f}s ({timing_stats['model_migration'] / total_timing * 100:.1f}%), 采样={timing_stats['sampling']:.2f}s ({timing_stats['sampling'] / total_timing * 100:.1f}%), 数据传输={timing_stats['data_transfer']:.2f}s ({timing_stats['data_transfer'] / total_timing * 100:.1f}%), 计算时间={compute_time:.2f}s ({compute_time / total_timing * 100:.1f}%), [前向={timing_stats['forward']:.2f}s, 反向={timing_stats['backward']:.2f}s], 参数更新={timing_stats['optimizer_step']:.2f}s ({timing_stats['optimizer_step'] / total_timing * 100:.1f}%)")
        train_stats = {'epochs_completed': epochs, 'steps_completed': total_steps, 'train_time': train_time, 'epoch_losses': epoch_losses, 'avg_loss': avg_loss, 'model_parameters': sum((p.numel() for p in model.parameters())), 'arch_spec': arch_spec, 'batch_size_used': effective_batch_size, 'training_mode': 'cpu_data_gpu_compute', 'timing_stats': timing_stats}
        return (model.state_dict(), avg_loss, train_stats)

    def evaluate_architecture(self, arch_spec: Dict[str, Any], model_state: Dict) -> Dict[str, Any]:
        start_time = time.time()
        input_dim = self.full_data.num_node_features
        output_dim = self.config.get('num_classes', 7)
        model = ModelBuilder.build_model(arch_spec, input_dim, output_dim, self.config).to(self.device)
        model.load_state_dict(model_state)
        model.eval()
        eval_batch_size = self.config.get('eval_batch_size', 2048)
        results = {}
        with torch.no_grad():
            if not hasattr(self.full_data, 'val_mask'):
                self.logger.logger.warning(f'Worker {self.worker_id} 数据缺少验证集mask')
                return {'worker_id': self.worker_id, 'eval_time': 0.0, 'val_accuracy': 0.0, 'evaluation_successful': False}
            val_mask = self.full_data.val_mask
            if not val_mask.any():
                self.logger.logger.warning(f'Worker {self.worker_id} 验证集为空')
                return {'worker_id': self.worker_id, 'eval_time': 0.0, 'val_accuracy': 0.0, 'evaluation_successful': False}
            val_indices = torch.where(val_mask)[0]
            val_loader = NeighborLoader(self.cpu_data, num_neighbors=self.num_neighbors, batch_size=eval_batch_size, input_nodes=val_indices, shuffle=False, num_workers=0, pin_memory=True if self.device.type == 'cuda' else False)
            all_predictions = []
            all_targets = []
            for batch in val_loader:
                batch = batch.to(self.device)
                out = model(batch.x, batch.edge_index)
                batch_size_actual = batch.batch_size
                predictions = torch.argmax(out[:batch_size_actual], dim=1)
                all_predictions.append(predictions.cpu())
                all_targets.append(batch.y[:batch_size_actual].cpu())
                del batch, out, predictions
            del val_loader
            if all_predictions:
                all_predictions = torch.cat(all_predictions)
                all_targets = torch.cat(all_targets)
                correct = (all_predictions == all_targets).sum().item()
                total = len(all_targets)
                accuracy = correct / total if total > 0 else 0.0
                results['val_accuracy'] = accuracy
                results['val_correct'] = correct
                results['val_total'] = total
        eval_time = time.time() - start_time
        results.update({'worker_id': self.worker_id, 'eval_time': eval_time, 'total_nodes': self.full_data.num_nodes, 'arch_spec': arch_spec, 'evaluation_successful': True, 'evaluation_mode': 'cpu_data_gpu_compute_val_only'})
        return results

class ArchParallelBackendV2:

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = GraphNASLogger(name='arch_parallel_backend_v2')
        self.workers = []
        self.full_data = None
        self.worker_use_gpu = []
        self._setup_workers_count()

    def _setup_workers_count(self):
        config_num_workers = self.config.get('num_workers', None)
        if config_num_workers is not None:
            self.num_workers = config_num_workers
            self.logger.logger.info(f'使用配置中指定的worker数量: {self.num_workers} (V2模式)')
            if torch.cuda.is_available():
                num_gpus = torch.cuda.device_count()
                if self.num_workers <= num_gpus:
                    self.worker_use_gpu = [True] * self.num_workers
                    self.logger.logger.info(f'所有 {self.num_workers} 个Worker将使用GPU，数据常驻CPU')
                else:
                    self.worker_use_gpu = [True] * num_gpus + [False] * (self.num_workers - num_gpus)
                    self.logger.logger.info(f'前 {num_gpus} 个Worker使用GPU，后 {self.num_workers - num_gpus} 个Worker使用CPU，数据常驻CPU')
            else:
                self.worker_use_gpu = [False] * self.num_workers
                self.logger.logger.warning(f'没有可用的GPU，所有 {self.num_workers} 个Worker将使用CPU (V2模式)')
        elif torch.cuda.is_available():
            num_gpus = torch.cuda.device_count()
            self.num_workers = num_gpus
            self.worker_use_gpu = [True] * num_gpus
            self.logger.logger.info(f'检测到 {num_gpus} 个GPU，自动创建 {self.num_workers} 个GPU Worker (V2模式)')
        else:
            self.num_workers = 1
            self.worker_use_gpu = [False]
            self.logger.logger.warning(f'没有可用的GPU，创建 {self.num_workers} 个CPU Worker (V2模式)')
        for i in range(self.num_workers):
            gpu_info = f'GPU (宿主机GPU {i})，数据常驻CPU' if self.worker_use_gpu[i] else 'CPU'
            self.logger.logger.info(f'Worker {i} 将使用 {gpu_info}')

    def get_num_workers(self):
        return self.num_workers

    def setup_distributed_training(self, full_data: Data):
        self.full_data = full_data
        self.workers = []
        self.logger.logger.info(f'初始化 {self.num_workers} 个ArchParallelWorkerV2，数据包含 {full_data.num_nodes} 个节点，常驻CPU内存')
        for i in range(self.num_workers):
            use_gpu = self.worker_use_gpu[i]
            if use_gpu:
                worker = ArchParallelWorkerV2.options(num_gpus=1, num_cpus=1).remote(worker_id=i, full_data=full_data, use_gpu=True, config=self.config)
                self.logger.logger.info(f'创建GPU Worker {i} (V2: CPU数据+GPU计算)')
            else:
                worker = ArchParallelWorkerV2.options(num_gpus=0, num_cpus=1).remote(worker_id=i, full_data=full_data, use_gpu=False, config=self.config)
                self.logger.logger.info(f'创建CPU Worker {i} (V2: 纯CPU计算)')
            self.workers.append(worker)
        self.logger.logger.info('ArchParallelBackendV2 初始化完成')

    def train_architectures_parallel(self, arch_specs: List[Dict[str, Any]], model_states: Optional[List[Dict]]=None, epochs: int=1, batch_size: Optional[int]=None, steps_per_epoch: Optional[int]=None) -> Tuple[List[Tuple[Dict, float, Dict[str, Any]]], Dict[str, float]]:
        if not self.workers:
            raise RuntimeError('请先调用 setup_distributed_training() 初始化环境')
        overall_start = time.perf_counter()
        num_archs = len(arch_specs)
        num_workers = len(self.workers)
        if num_archs > num_workers:
            self.logger.logger.warning(f'架构数量({num_archs})大于Worker数量({num_workers})，将分批次训练 (V2模式)')
        if model_states is None:
            model_states = [None] * num_archs
        self.logger.logger.info(f'开始并行训练 {num_archs} 个架构，使用 {num_workers} 个Worker (V2模式)')
        all_results = []
        total_dispatch_time = 0.0
        total_pure_train_time = 0.0
        total_ray_communication = 0.0
        total_model_to_device_sync = 0.0
        total_load_state_dict_sync = 0.0
        total_batch_to_device_sync = 0.0
        for batch_start in range(0, num_archs, num_workers):
            batch_end = min(batch_start + num_workers, num_archs)
            batch_arch_specs = arch_specs[batch_start:batch_end]
            batch_model_states = model_states[batch_start:batch_end]
            self.logger.logger.info(f'训练批次 {batch_start // num_workers + 1}: 架构 {batch_start}-{batch_end - 1} (V2模式)')
            dispatch_start = time.perf_counter()
            training_futures = []
            for i, (arch_spec, model_state) in enumerate(zip(batch_arch_specs, batch_model_states)):
                worker_idx = i
                future = self.workers[worker_idx].train_architecture.remote(arch_spec, model_state, epochs, batch_size, steps_per_epoch)
                training_futures.append(future)
            dispatch_time = time.perf_counter() - dispatch_start
            total_dispatch_time += dispatch_time
            ray_get_start = time.perf_counter()
            batch_results = ray.get(training_futures)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            ray_get_time = time.perf_counter() - ray_get_start
            batch_train_times = [stats.get('train_time', 0.0) for _, _, stats in batch_results]
            max_train_time = max(batch_train_times) if batch_train_times else 0.0
            total_pure_train_time += max_train_time
            ray_comm_time = max(0.0, ray_get_time - max_train_time)
            total_ray_communication += ray_comm_time
            for i, (model_state, loss, stats) in enumerate(batch_results):
                arch_idx = batch_start + i
                self.logger.logger.info(f"架构 {arch_idx} 训练完成 (V2): 损失={loss:.4f}, 耗时={stats.get('train_time', 0):.2f}s, 参数量={stats.get('model_parameters', 0)}, 批量大小={stats.get('batch_size_used', 0)}")
                total_model_to_device_sync += stats.get('model_to_device_sync', 0.0)
                total_load_state_dict_sync += stats.get('load_state_dict_sync', 0.0)
                total_batch_to_device_sync += stats.get('batch_to_device_sync', 0.0)
            all_results.extend(batch_results)
        wall_clock_total = time.perf_counter() - overall_start
        timing_info = {'wall_clock_total': wall_clock_total, 'dispatch_overhead': total_dispatch_time, 'pure_train_time': total_pure_train_time, 'ray_communication': total_ray_communication, 'model_to_device_sync': total_model_to_device_sync, 'load_state_dict_sync': total_load_state_dict_sync, 'batch_to_device_sync': total_batch_to_device_sync}
        self.logger.logger.debug(f'[时间统计-ParallelV2] 墙上时钟总时间: {wall_clock_total:.2f}s, 调度: {total_dispatch_time:.4f}s ({total_dispatch_time / wall_clock_total * 100:.1f}%), 纯训练: {total_pure_train_time:.2f}s ({total_pure_train_time / wall_clock_total * 100:.1f}%), Ray通信: {total_ray_communication:.4f}s ({total_ray_communication / wall_clock_total * 100:.1f}%)')
        return (all_results, timing_info)

    def evaluate_architectures_parallel(self, arch_specs: List[Dict[str, Any]], model_states: List[Dict]) -> List[Dict[str, Any]]:
        if not self.workers:
            raise RuntimeError('请先调用 setup_distributed_training() 初始化环境')
        num_archs = len(arch_specs)
        num_workers = len(self.workers)
        if len(model_states) != num_archs:
            raise ValueError('模型状态数量必须等于架构数量')
        if num_archs > num_workers:
            self.logger.logger.warning(f'架构数量({num_archs})大于Worker数量({num_workers})，将分批次评估 (V2模式)')
        self.logger.logger.info(f'开始并行评估 {num_archs} 个架构，使用 {num_workers} 个Worker (V2模式)')
        all_results = []
        for batch_start in range(0, num_archs, num_workers):
            batch_end = min(batch_start + num_workers, num_archs)
            batch_arch_specs = arch_specs[batch_start:batch_end]
            batch_model_states = model_states[batch_start:batch_end]
            self.logger.logger.info(f'评估批次 {batch_start // num_workers + 1}: 架构 {batch_start}-{batch_end - 1} (V2模式)')
            evaluation_futures = []
            for i, (arch_spec, model_state) in enumerate(zip(batch_arch_specs, batch_model_states)):
                worker_idx = i
                future = self.workers[worker_idx].evaluate_architecture.remote(arch_spec, model_state)
                evaluation_futures.append(future)
            batch_results = ray.get(evaluation_futures)
            for i, result in enumerate(batch_results):
                arch_idx = batch_start + i
                self.logger.logger.info(f"架构 {arch_idx} 评估完成 (V2): 验证准确率={result.get('val_accuracy', 0):.4f}, 耗时={result.get('eval_time', 0):.2f}s")
            all_results.extend(batch_results)
        return all_results

    def train_epoch(self, arch_spec: Dict[str, Any], model_state: Optional[Dict]=None, epochs: int=1, batch_size: Optional[int]=None, steps_per_epoch: Optional[int]=None) -> Tuple[Dict, float]:
        if not self.workers:
            raise RuntimeError('请先调用 setup_distributed_training() 初始化环境')
        training_future = self.workers[0].train_architecture.remote(arch_spec, model_state, epochs, batch_size, steps_per_epoch)
        model_state, loss, _ = ray.get(training_future)
        return (model_state, loss)

    def evaluate_model(self, arch_spec: Dict[str, Any], model_state: Dict) -> Dict[str, Any]:
        if not self.workers:
            raise RuntimeError('请先调用 setup_distributed_training() 初始化环境')
        evaluation_future = self.workers[0].evaluate_architecture.remote(arch_spec, model_state)
        result = ray.get(evaluation_future)
        return {'global_accuracy': result.get('val_accuracy', 0.0), 'val_accuracy': result.get('val_accuracy', 0.0), 'total_unique_nodes': result.get('total_nodes', 0), 'correct_predictions': result.get('val_correct', 0), 'worker_stats': [result], 'evaluation_successful': result.get('evaluation_successful', False)}

    def cleanup(self):
        self.logger.logger.info('正在清理ArchParallelBackendV2资源...')
        self.workers = []
        self.full_data = None