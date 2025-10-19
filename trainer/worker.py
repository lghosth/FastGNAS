import math
import torch
import torch.nn.functional as F
from typing import Dict, Any, Tuple, Optional, List
import time
import ray
from trainer.builder import ModelBuilder
from batching.hierarchical_cache_manager import HierarchicalCacheManager
from utils.logger import GraphNASLogger
from utils.utils import parse_fan_out, normalize_fan_out_for_layers
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader
from core.load_estimator import MloadDataCollector

@ray.remote(num_gpus=1, num_cpus=4)
class TrainerWorker:

    def __init__(self, worker_id: int, gpu_id: int, data_partition: Data, config: Optional[Dict[str, Any]]=None, collector_ref: Optional[Any]=None):
        self.worker_id = worker_id
        self.gpu_id = gpu_id
        self.logger = GraphNASLogger(name=f'worker_{worker_id}', track_gpu=True)
        import os
        self.logger.logger.info(f"[Worker {worker_id}] CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES')}")
        self.logger.logger.info(f'[Worker {worker_id}] torch.cuda.is_available(): {torch.cuda.is_available()}')
        self.logger.logger.info(f"[Worker {worker_id}] ray.get_gpu_ids(): {(ray.get_gpu_ids() if hasattr(ray, 'get_gpu_ids') else 'N/A')}")
        if torch.cuda.is_available():
            self.device = torch.device('cuda:0')
            torch.cuda.set_device(self.device)
        else:
            self.device = torch.device('cpu')
        global_node_ids = getattr(data_partition, 'global_node_ids', None)
        self.cpu_subgraph_for_sampling = Data(edge_index=data_partition.edge_index.clone().cpu() if data_partition.edge_index is not None else None, num_nodes=data_partition.num_nodes, train_mask=data_partition.train_mask.clone().cpu() if hasattr(data_partition, 'train_mask') and data_partition.train_mask is not None else None)
        if global_node_ids is not None:
            self.cpu_subgraph_for_sampling.global_node_ids = global_node_ids.cpu() if hasattr(global_node_ids, 'to') else global_node_ids
        self.data_partition = data_partition.to(self.device)
        if global_node_ids is not None:
            self.data_partition.global_node_ids = global_node_ids.to(self.device) if hasattr(global_node_ids, 'to') else global_node_ids
        self.base_config: Dict[str, Any] = dict(config) if isinstance(config, dict) else {}
        batch_size_hint = int(self.base_config.get('batch_size', 5) or 5)
        if batch_size_hint <= 0:
            batch_size_hint = 5
        train_mask = getattr(self.data_partition, 'train_mask', None)
        if isinstance(train_mask, torch.Tensor):
            train_nodes = int(train_mask.sum().item())
        else:
            train_nodes = int(getattr(self.data_partition, 'num_nodes', 0) or 0)
        train_nodes = max(train_nodes, 1)
        if self.data_partition.edge_index is not None and self.data_partition.num_nodes is not None and (self.data_partition.num_nodes > 0):
            self.avg_degree = self.data_partition.num_edges / self.data_partition.num_nodes
        else:
            self.avg_degree = 1.0
        fan_out_raw = self.base_config.get('fan_out', -1)
        self.base_fan_out = parse_fan_out(fan_out_raw)
        normalized_fan_outs = [fan if fan > 0 else self.avg_degree for fan in self.base_fan_out]
        self.avg_fan_out = float(sum(normalized_fan_outs) / len(normalized_fan_outs))
        self.fan_out: List[int] = self.base_fan_out
        estimated_pool = math.ceil(train_nodes / batch_size_hint)
        self.logger.logger.info(f'Worker {worker_id} on GPU {gpu_id} initialized, train_nodes={train_nodes}, train_nodes / batch_size_hint = {estimated_pool}')
        estimated_pool = estimated_pool * float(self.avg_degree / self.avg_fan_out)
        self.logger.logger.info(f'Worker {worker_id} on GPU {gpu_id} initialized, train_nodes={train_nodes}, avg_fan_out={self.avg_fan_out}, avg_degree={self.avg_degree}')
        max_pool_size = int(math.ceil(max(1, min(estimated_pool, 5000))))
        use_fixed_loader = self.base_config.get('use_fixed_loader', False)
        if use_fixed_loader:
            fixed_batch_size = batch_size_hint
            fixed_fan_out = self.base_fan_out
            fixed_num_layers = self.base_config.get('gnn_layers', 2)
            self.logger.logger.info(f'Worker {worker_id} 启用固定Loader优化: bs={fixed_batch_size}, layers={fixed_num_layers}, fan_out={fixed_fan_out}')
            self.batch_manager = HierarchicalCacheManager(subgraph=self.cpu_subgraph_for_sampling, gpu_subgraph=self.data_partition, device=self.device, use_fixed_loader=True, fixed_batch_size=fixed_batch_size, fixed_fan_out=fixed_fan_out, fixed_num_layers=fixed_num_layers)
        else:
            self.batch_manager = HierarchicalCacheManager(subgraph=self.cpu_subgraph_for_sampling, gpu_subgraph=self.data_partition, device=self.device, max_cache_pools=1, max_pool_size=max_pool_size)
        self.logger.logger.info(f'Worker {worker_id} on GPU {gpu_id} initialized, train_nodes={train_nodes}, batch_size={batch_size_hint}, max_pool_size={max_pool_size}')
        self.model = None
        self.current_arch_spec = None
        data_collection_config = self.base_config.get('data_collection', {}) if self.base_config else {}
        self.enable_mload_collector = data_collection_config.get('enable_mload_collector', False)
        if self.enable_mload_collector:
            collector_type = data_collection_config.get('collector_type', 'auto')
            priority_order = data_collection_config.get('priority_order', ['partition', 'dataset'])
            device_type = data_collection_config.get('device_type', 'p100')
            dataset_dir = data_collection_config.get('dataset_dir', 'dataset/Mload_dataset')
            self.data_collector = MloadDataCollector(dataset_dir=dataset_dir, device_type=device_type, collector_type=collector_type, priority_order=priority_order, data_collection_cfg=data_collection_config)
            file_prefix = data_collection_config.get('file_prefix', 'fastgnas')
            time_window_mode = data_collection_config.get('time_window_mode', 'batch')
            self.time_window_mode = time_window_mode
            self.output_filename = f'{file_prefix}-{time_window_mode}.yaml'
            self.collector_ref = collector_ref
        else:
            self.data_collector = None
            self.time_window_mode = None
            self.output_filename = None
            self.collector_ref = None

    def train_model_on_partition(self, arch_spec: Dict[str, Any], state_dict: Dict, config: Dict[str, Any]) -> Tuple[float, Dict, int, int]:
        model_migration_time = 0.0
        if self.model is None or str(arch_spec) != str(self.current_arch_spec):
            if self.model is not None:
                try:
                    self.model.cpu()
                    del self.model
                    self.model = None
                    if torch.cuda.is_available() and self.device.type == 'cuda':
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()
                except Exception as e:
                    self.logger.logger.warning(f'清理旧模型失败: {e}')
                    pass
            t_build_start = time.perf_counter()
            num_classes = self.base_config.get('num_classes', None)
            if isinstance(num_classes, int) and num_classes > 0:
                output_dim = int(num_classes)
            else:
                raise TypeError('worker 无法确定输出维度 to determine output_dim')
            self.model = ModelBuilder.build_model(arch_spec, self.data_partition.num_node_features, output_dim, config).to(self.device)
            self.current_arch_spec = arch_spec
            t_build_end = time.perf_counter()
            model_build_time = t_build_end - t_build_start
            model_migration_time += model_build_time
            self.logger.logger.debug(f'[Worker {self.worker_id}] 模型构建耗时: {model_build_time * 1000:.2f}ms')
        if state_dict:
            t_load_start = time.perf_counter()
            self.model.load_state_dict(state_dict)
            t_load_end = time.perf_counter()
            state_load_time = t_load_end - t_load_start
            model_migration_time += state_load_time
            self.logger.logger.info(f'[Worker {self.worker_id}] ✅ 加载了保存的模型状态，耗时: {state_load_time * 1000:.2f}ms')
        else:
            self.logger.logger.warning(f'[Worker {self.worker_id}] ⚠️  state_dict为空，模型从随机初始化开始训练')
        t_opt_start = time.perf_counter()
        weight_decay = config.get('weight_decay', self.base_config.get('weight_decay', 0.0))
        optimizer = torch.optim.Adam(self.model.parameters(), lr=config.get('lr', 0.01), weight_decay=weight_decay)
        t_opt_end = time.perf_counter()
        opt_init_time = t_opt_end - t_opt_start
        model_migration_time += opt_init_time
        if model_migration_time > 0:
            self.logger.logger.debug(f'[Worker {self.worker_id}] 模型迁移总耗时: {model_migration_time * 1000:.2f}ms (构建+加载+优化器)')
        epochs = config.get('epochs_per_rotation', self.base_config.get('epochs_per_rotation', 5))
        requested_bs = config.get('batch_size', self.base_config.get('batch_size', 5))
        if 'fan_out' in config and isinstance(config['fan_out'], list):
            fan_out_list = config['fan_out']
        else:
            fan_out_raw = config.get('fan_out', None)
            if fan_out_raw is not None:
                fan_out_list = parse_fan_out(fan_out_raw)
            else:
                fan_out_list = self.base_fan_out
        steps_per_epoch = config.get('steps_per_epoch', self.base_config.get('steps_per_epoch', 1))
        requested_layers = self._get_num_layers_from_arch(arch_spec)
        self.current_fan_out = normalize_fan_out_for_layers(fan_out_list, requested_layers)
        config['fan_out'] = self.current_fan_out
        batch_times = []
        if self.time_window_mode == 'batch':
            batch_times = []
        timing_stats = {'model_migration': model_migration_time, 'batch_preparation': 0.0, 'forward': 0.0, 'backward': 0.0, 'optimizer_step': 0.0, 'memory_cleanup': 0.0}
        early_stop_patience = config.get('early_stop_patience')
        best_epoch_loss = float('inf')
        epochs_no_improve = 0
        t_start = time.perf_counter()
        for epoch in range(1, epochs + 1):
            self.model.train()
            epoch_loss_total = 0.0
            effective_steps = 0
            for step in range(steps_per_epoch):
                if self.time_window_mode == 'batch':
                    batch_start = time.perf_counter()
                t0 = time.perf_counter()
                batch = self.batch_manager.get_batch(requested_bs=requested_bs, requested_layers=requested_layers, fan_out=self.current_fan_out)
                t1 = time.perf_counter()
                timing_stats['batch_preparation'] += t1 - t0
                if batch is None:
                    self.logger.logger.warning(f'警告: 在 epoch {epoch}, step {step + 1} 未能获取到batch')
                    continue
                t_forward_start = time.perf_counter()
                optimizer.zero_grad()
                self.logger.logger.debug(f'[Worker {self.worker_id}] 开始前向传播，在第 {epoch} 轮，第 {step + 1} 步Before forward: {torch.cuda.memory_allocated(0) / 1024 ** 3:.2f} GB')
                out = self.model(batch.x, batch.edge_index)
                if isinstance(out, torch.Tensor) and isinstance(batch.y, torch.Tensor):
                    logits = out[:batch.batch_size]
                    targets = self._prepare_labels(batch.y[:batch.batch_size])
                    loss = F.cross_entropy(logits, targets)
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
                    self.logger.logger.debug(f'[Worker {self.worker_id}] 完成前向传播，在第 {epoch} 轮，第 {step + 1} 步After forward: {torch.cuda.memory_allocated(0) / 1024 ** 3:.2f} GB')
                    self.logger.logger.debug(f'Peak memory during batch: {torch.cuda.max_memory_allocated(0) / 1024 ** 3:.2f} GB')
                    torch.cuda.reset_peak_memory_stats(0)
                    epoch_loss_total += loss.item()
                    effective_steps += 1
                else:
                    self.logger.logger.warning(f'警告: 在 epoch {epoch}, step {step + 1} 的输出或标签不是张量')
                    raise ValueError(f'警告: 在 epoch {epoch}, step {step + 1} 的输出或标签不是张量')
                if self.time_window_mode == 'batch':
                    batch_time = time.perf_counter() - batch_start
                    batch_times.append(batch_time)
                t_cleanup_start = time.perf_counter()
                del batch, out, logits, targets, loss
                t_cleanup_end = time.perf_counter()
                timing_stats['memory_cleanup'] += t_cleanup_end - t_cleanup_start
            if effective_steps > 0:
                avg_epoch_loss = epoch_loss_total / effective_steps
                if avg_epoch_loss + 1e-08 < best_epoch_loss:
                    best_epoch_loss = avg_epoch_loss
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1
                if early_stop_patience is not None and epochs_no_improve >= int(early_stop_patience):
                    self.logger.logger.info(f'[Worker {self.worker_id}] 触发早停: 连续 {epochs_no_improve} 轮无loss改进 (最佳loss={best_epoch_loss:.4f})')
                    break
            cleanup_mode = config.get('memory_cleanup_mode', 'minimal')
            if cleanup_mode == 'normal' and self.device.type == 'cuda':
                t_cleanup_start = time.perf_counter()
                torch.cuda.empty_cache()
                timing_stats['memory_cleanup'] += time.perf_counter() - t_cleanup_start
        total_time = sum(timing_stats.values())
        if total_time > 0:
            compute_time = timing_stats['forward'] + timing_stats['backward']
            self.logger.logger.info(f"[性能分析-FastGNAS] Worker {self.worker_id} - 总时间={total_time:.2f}s, 模型迁移={timing_stats['model_migration']:.4f}s ({timing_stats['model_migration'] / total_time * 100:.1f}%), batch准备={timing_stats['batch_preparation']:.2f}s ({timing_stats['batch_preparation'] / total_time * 100:.1f}%), 计算时间={compute_time:.2f}s ({compute_time / total_time * 100:.1f}%), [前向={timing_stats['forward']:.2f}s, 反向={timing_stats['backward']:.2f}s], 参数更新={timing_stats['optimizer_step']:.2f}s ({timing_stats['optimizer_step'] / total_time * 100:.1f}%), 显存清理={timing_stats['memory_cleanup']:.4f}s ({timing_stats['memory_cleanup'] / total_time * 100:.1f}%)")
        if self.time_window_mode == 'batch' and batch_times:
            training_time = sum(batch_times) / len(batch_times)
            self.logger.logger.debug(f'[Worker {self.worker_id}] Batch模式: 总计{len(batch_times)}个batch, 平均时间={training_time:.4f}s, 总时间={sum(batch_times):.4f}s')
        else:
            training_time = time.perf_counter() - t_start
        val_acc, _, correct_count, incorrect_count = self._evaluate(self.model)
        updated_state_dict = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
        t_cleanup_start = time.perf_counter()
        del optimizer
        cleanup_mode = config.get('memory_cleanup_mode', 'minimal')
        if self.device.type == 'cuda' and cleanup_mode != 'none':
            torch.cuda.empty_cache()
        t_cleanup_end = time.perf_counter()
        timing_stats['memory_cleanup'] += t_cleanup_end - t_cleanup_start
        if self.enable_mload_collector and self.data_collector is not None:
            try:
                graph_info = self.data_collector.extract_graph_info(self.data_partition)
                sample = self.data_collector.add_training_record(arch_spec=arch_spec, graph_info=graph_info, training_time=training_time, config=self.base_config)
                if getattr(self, 'collector_ref', None) is not None:
                    try:
                        if hasattr(self.collector_ref, 'add_record'):
                            self.collector_ref.add_record.remote(sample)
                        else:
                            self.data_collector.save_dataset(self.output_filename)
                    except Exception as _:
                        self.data_collector.save_dataset(self.output_filename)
                else:
                    self.data_collector.save_dataset(self.output_filename)
                self.logger.logger.info(f"[Worker {self.worker_id}] 已记录训练样本: time={training_time:.4f}s, nodes={graph_info.get('vertices')}, edges={graph_info.get('edges')}")
            except Exception as e:
                self.logger.logger.warning(f'数据收集失败: {e}')
        try:
            timing_stats['wall_clock_train_time'] = float(training_time)
        except Exception:
            pass
        return (val_acc, updated_state_dict, correct_count, incorrect_count, timing_stats)

    def train_bucket_models(self, bucket_architectures: List[Dict[str, Any]], config: Dict[str, Any], state_dicts: Optional[Dict[str, Dict]]=None) -> List[Dict[str, Any]]:
        results = []
        for arch in bucket_architectures:
            arch_id = arch['arch_id']
            arch_spec = arch['arch_spec']
            model_state = None
            if state_dicts and arch_id in state_dicts:
                model_state = state_dicts[arch_id]
            try:
                val_acc, updated_state, correct_count, incorrect_count = self.train_model_on_partition(arch_spec=arch_spec, state_dict=model_state, config=config)
                result = {'arch_id': arch_id, 'success': True, 'val_acc': val_acc, 'updated_state': updated_state, 'correct_count': correct_count, 'incorrect_count': incorrect_count, 'load': arch.get('load', 0.0)}
            except Exception as e:
                result = {'arch_id': arch_id, 'success': False, 'error': str(e), 'load': arch.get('load', 0.0)}
                print(f"[Worker {self.worker_id}] 模型 '{arch_id}' 训练失败: {e}")
            results.append(result)
        return results

    def _get_num_layers_from_arch(self, arch_spec: Dict[str, Any]) -> int:
        if 'layers' in arch_spec:
            return len(arch_spec['layers'])
        elif 'cell_spec' in arch_spec:
            return arch_spec.get('num_cells_to_stack', 2)
        else:
            self.logger.logger.warning(f'警告: 未知的架构格式 {arch_spec.keys()}，使用默认层数2')
            return 2

    def _prepare_labels(self, labels: torch.Tensor) -> torch.Tensor:
        if labels.dim() == 0:
            return labels.view(1).long()
        if labels.dim() == 1:
            return labels.long()
        if labels.size(-1) == 1:
            return labels.view(-1).long()
        return labels.argmax(dim=-1).long()

    @torch.no_grad()
    def _evaluate(self, model) -> Tuple[float, float, int, int]:
        if not isinstance(self.data_partition.val_mask, torch.Tensor) or not isinstance(self.data_partition.y, torch.Tensor):
            return (0.0, 0.0, 0, 0)
        val_mask: torch.Tensor = self.data_partition.val_mask
        val_mask_sum = val_mask.sum().item()
        self.logger.logger.debug(f'[Worker {self.worker_id}] 评估: val_mask_sum={val_mask_sum}')
        if val_mask_sum == 0:
            self.logger.logger.warning(f'[Worker {self.worker_id}] 警告: 验证集为空，返回准确率0.0')
            return (0.0, 0.0, 0, 0)
        num_layers = len(getattr(model, 'layers', []))
        if num_layers <= 0:
            raise ValueError('模型不包含任何 GNN 层，无法执行评估')
        is_large = self._is_large_graph()
        self.logger.logger.debug(f"[Worker {self.worker_id}] 推理模式: {('层式采样' if is_large else '全图')}, 层数={num_layers}")
        if is_large:
            fan_out_list = self._get_fan_out_list(num_layers)
            eval_batch_size = self._get_eval_batch_size()
            val_node_ids = torch.where(val_mask)[0].cpu()
            self.logger.logger.debug(f'[Worker {self.worker_id}] 层式推理: fan_out={fan_out_list}, batch_size={eval_batch_size}, val_nodes={len(val_node_ids)}')
            logits = self._layerwise_inference(model=model, fan_out_list=fan_out_list, batch_size=eval_batch_size, target_node_ids=val_node_ids)
        else:
            logits = model(self.data_partition.x, self.data_partition.edge_index)
        self.logger.logger.debug(f'[Worker {self.worker_id}] Logits: shape={logits.shape}, min={logits.min().item():.4f}, max={logits.max().item():.4f}')
        pred = logits.argmax(dim=1)
        val_pred = pred[val_mask]
        val_true = self._prepare_labels(self.data_partition.y[val_mask])
        self.logger.logger.debug(f'[Worker {self.worker_id}] 预测: val_pred shape={val_pred.shape}, unique_values={val_pred.unique().tolist()}')
        self.logger.logger.debug(f'[Worker {self.worker_id}] 真值: val_true shape={val_true.shape}, unique_values={val_true.unique().tolist()}')
        val_correct_mask = val_pred == val_true
        val_correct_count = val_correct_mask.sum().item()
        val_incorrect_count = val_mask_sum - val_correct_count
        val_acc = val_correct_count / val_mask_sum
        self.logger.logger.info(f'[Worker {self.worker_id}] 评估完成: 准确率={val_acc:.4f} ({val_correct_count}/{val_mask_sum})')
        return (val_acc, 0.0, val_correct_count, val_incorrect_count)

    @torch.no_grad()
    def _layerwise_inference(self, model, fan_out_list: List[int], batch_size: int, target_node_ids: Optional[torch.Tensor]=None) -> torch.Tensor:
        model.eval()
        data = self.data_partition
        num_nodes = getattr(data, 'num_nodes', None)
        if num_nodes is None:
            raise ValueError('数据分区缺少 num_nodes 信息，无法执行推理')
        x_all = data.x.to(self.device)
        edge_index = data.edge_index
        eval_data = Data(x=x_all, edge_index=edge_index, num_nodes=num_nodes)
        if target_node_ids is None:
            target_node_ids = torch.arange(num_nodes, dtype=torch.long)
            self.logger.logger.info(f'[Worker {self.worker_id}] 使用所有节点作为目标节点进行多GPU全图推理')
        else:
            target_node_ids = target_node_ids.to(torch.long)
        current_features = x_all
        for layer_idx, (layer, act) in enumerate(zip(model.layers, model.activations)):
            fan_out_value = fan_out_list[layer_idx] if layer_idx < len(fan_out_list) else fan_out_list[-1]
            num_neighbors = [-1] if fan_out_value == -1 else [int(fan_out_value)]
            loader = NeighborLoader(eval_data, num_neighbors=num_neighbors, input_nodes=target_node_ids, batch_size=batch_size, shuffle=False)
            next_features: Optional[torch.Tensor] = None
            for batch in loader:
                batch = batch.to(self.device)
                n_id = batch.n_id
                h = layer(current_features[n_id], batch.edge_index)
                h = act(h)
                if next_features is None:
                    next_features = current_features.new_zeros((num_nodes, h.size(-1)))
                next_features[n_id] = h
            if next_features is None:
                raise ValueError('推理阶段未生成任何特征，可能 batch_size 设置过小')
            current_features = next_features
            eval_data = Data(x=current_features, edge_index=edge_index, num_nodes=num_nodes)
            target_node_ids = target_node_ids
        return current_features

    def _get_fan_out_list(self, num_layers: int) -> List[int]:
        if num_layers <= 0:
            raise ValueError('模型不包含任何 GNN 层')
        fan_out_raw = self.base_config.get('fan_out', [-1])
        fan_out_list = parse_fan_out(fan_out_raw)
        return normalize_fan_out_for_layers(fan_out_list, num_layers)

    def _get_eval_batch_size(self) -> int:
        eval_cfg = self.base_config.get('evaluation', {})
        eval_batch_size = int(eval_cfg.get('batch_size', self.base_config.get('batch_size', 32)))
        return max(eval_batch_size, 1)

    def _is_large_graph(self) -> bool:
        dataset_name = str(self.base_config.get('dataset', '')).lower()
        return dataset_name in {'reddit', 'ogbn-products', 'ogbn-arxiv'}

    def distributed_inference(self, arch_spec: Dict[str, Any], state_dict: Dict) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        is_large = self._is_large_graph()
        eval_device = torch.device('cpu') if is_large else self.device
        try:
            eval_model = ModelBuilder.build_model(arch_spec=arch_spec, input_dim=self.data_partition.x.size(1), output_dim=self.base_config.get('num_classes', 7), config=self.base_config).to(eval_device)
            device_name = 'CPU' if is_large else f'GPU:{self.device}'
            self.logger.logger.debug(f'Worker {self.worker_id} 在{device_name}上构建评估模型')
        except Exception as e:
            self.logger.logger.error(f'Worker {self.worker_id} 评估模型构建失败: {e}')
            raise RuntimeError('[全图评估] 模型构建失败: model_build_failed')
        if state_dict is not None and len(state_dict) > 0:
            try:
                eval_model.load_state_dict(state_dict, strict=False)
            except Exception as e:
                self.logger.logger.warning(f'Worker {self.worker_id} 加载状态失败: {e}')
        if is_large:
            eval_data_partition = Data(x=self.data_partition.x.cpu(), edge_index=self.data_partition.edge_index.cpu(), y=self.data_partition.y.cpu() if hasattr(self.data_partition, 'y') else None, num_nodes=self.data_partition.num_nodes)
            if hasattr(self.data_partition, 'eval_mask'):
                eval_data_partition.eval_mask = self.data_partition.eval_mask.cpu()
            if hasattr(self.data_partition, 'val_mask'):
                eval_data_partition.val_mask = self.data_partition.val_mask.cpu()
            if hasattr(self.data_partition, 'global_node_ids'):
                eval_data_partition.global_node_ids = self.data_partition.global_node_ids.cpu()
        else:
            eval_data_partition = self.data_partition
        eval_model.eval()
        with torch.no_grad():
            num_layers = len(getattr(eval_model, 'layers', []))
            if num_layers <= 0:
                raise ValueError('模型不包含任何 GNN 层，无法执行评估')
            if is_large:
                fan_out_list = self._get_fan_out_list(num_layers)
                eval_batch_size = self._get_eval_batch_size()
                if hasattr(eval_data_partition, 'eval_mask') and isinstance(eval_data_partition.eval_mask, torch.Tensor):
                    target_mask = eval_data_partition.eval_mask
                else:
                    target_mask = getattr(eval_data_partition, 'val_mask', None)
                target_ids = torch.where(target_mask)[0] if isinstance(target_mask, torch.Tensor) else None
                out = self._layerwise_inference_cpu(model=eval_model, data_partition=eval_data_partition, fan_out_list=fan_out_list, batch_size=eval_batch_size, target_node_ids=target_ids, device=eval_device)
            else:
                out = eval_model(eval_data_partition.x, eval_data_partition.edge_index)
            pred = out.argmax(dim=1)
            if hasattr(eval_data_partition, 'eval_mask') and isinstance(eval_data_partition.eval_mask, torch.Tensor) and hasattr(eval_data_partition, 'y'):
                eval_mask = eval_data_partition.eval_mask
                eval_mask_sum = eval_mask.sum().item()
                if eval_mask_sum > 0:
                    val_predictions = pred[eval_mask].cpu()
                    val_true_labels = self._prepare_labels(eval_data_partition.y[eval_mask]).cpu()
                    if hasattr(eval_data_partition, 'global_node_ids'):
                        val_global_ids = eval_data_partition.global_node_ids[eval_mask].cpu()
                        self.logger.logger.debug(f"Worker {self.worker_id} {('CPU' if is_large else 'GPU')}推理完成（{eval_mask_sum}个唯一验证节点）")
                    else:
                        val_local_indices = torch.where(eval_mask)[0].cpu()
                        val_global_ids = val_local_indices + self.worker_id * 10000
                        self.logger.logger.warning(f'Worker {self.worker_id} 使用 eval_mask 但缺少全局ID映射')
                    return (val_predictions, val_global_ids, val_true_labels)
            if hasattr(eval_data_partition, 'val_mask') and isinstance(eval_data_partition.val_mask, torch.Tensor) and hasattr(eval_data_partition, 'y'):
                val_mask = eval_data_partition.val_mask
                val_mask_sum = val_mask.sum().item()
                if val_mask_sum > 0:
                    self.logger.logger.warning(f'Worker {self.worker_id} 降级使用 val_mask（{val_mask_sum}个节点），可能包含与其他分区重复的验证节点，需要后续去重')
                    val_predictions = pred[val_mask].cpu()
                    val_true_labels = self._prepare_labels(eval_data_partition.y[val_mask]).cpu()
                    if hasattr(eval_data_partition, 'global_node_ids'):
                        val_global_ids = eval_data_partition.global_node_ids[val_mask].cpu()
                    else:
                        val_local_indices = torch.where(val_mask)[0].cpu()
                        val_global_ids = val_local_indices + self.worker_id * 10000
                    return (val_predictions, val_global_ids, val_true_labels)
            self.logger.logger.warning(f'Worker {self.worker_id} 没有验证数据')
            return (torch.tensor([]), torch.tensor([]), torch.tensor([]))

    @torch.no_grad()
    def _layerwise_inference_cpu(self, model, data_partition: Data, fan_out_list: List[int], batch_size: int, target_node_ids: Optional[torch.Tensor]=None, device: torch.device=torch.device('cpu')) -> torch.Tensor:
        model.eval()
        num_nodes = getattr(data_partition, 'num_nodes', None)
        if num_nodes is None:
            raise ValueError('数据分区缺少 num_nodes 信息，无法执行推理')
        x_all = data_partition.x.to(device)
        edge_index = data_partition.edge_index
        if target_node_ids is None:
            target_node_ids = torch.arange(num_nodes, dtype=torch.long)
            self.logger.logger.info(f'[Worker {self.worker_id}] CPU推理：使用所有节点作为目标')
        else:
            target_node_ids = target_node_ids.to(torch.long)
        current_features = x_all
        for layer_idx, (layer, act) in enumerate(zip(model.layers, model.activations)):
            fan_out_value = fan_out_list[layer_idx] if layer_idx < len(fan_out_list) else fan_out_list[-1]
            num_neighbors = [-1] if fan_out_value == -1 else [int(fan_out_value)]
            eval_data = Data(x=current_features, edge_index=edge_index, num_nodes=num_nodes)
            loader = NeighborLoader(eval_data, num_neighbors=num_neighbors, input_nodes=target_node_ids, batch_size=batch_size, shuffle=False)
            next_features: Optional[torch.Tensor] = None
            try:
                for batch in loader:
                    batch = batch.to(device)
                    n_id = batch.n_id
                    h = layer(current_features[n_id], batch.edge_index)
                    h = act(h)
                    if next_features is None:
                        next_features = current_features.new_zeros((num_nodes, h.size(-1)))
                    next_features[n_id] = h
                    del batch, n_id, h
            finally:
                del loader, eval_data
            if next_features is None:
                raise ValueError('推理阶段未生成任何特征，可能 batch_size 设置过小')
            if layer_idx > 0:
                del current_features
            current_features = next_features
        return current_features