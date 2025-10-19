import os
import sys
import yaml
import ray
import math
import time
import torch
import argparse
import numpy as np
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(project_root)
from baseline.components.backends.parallel_backends_v2 import ArchParallelBackendV2
from baseline.components.searchers.shapley_searcher import ShapleySearcher
from data.data_utils import get_dataset
from utils.logger import GraphNASLogger
from utils.utils import load_dataset_config
from utils.arch_converter import convert_architecture, convert_architecture_to_dgnap

def load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config

def merge_dataset_config(config: Dict[str, Any], dataset_name: str) -> Dict[str, Any]:
    dataset_config = load_dataset_config(dataset_name)
    if 'parallel_backends' not in config:
        config['parallel_backends'] = {}
    if 'training' not in config:
        config['training'] = {}
    parallel_cfg = config['parallel_backends']
    parallel_cfg.update({'dataset': dataset_name, 'num_classes': dataset_config.get('num_classes', 7)})
    config['num_classes'] = dataset_config.get('num_classes', 7)
    training_cfg = config['training']
    training_cfg.update({'batch_size': dataset_config.get('train_batch_size', 1024), 'fan_out': dataset_config.get('fan_out', [-1, -1]), 'steps_per_epoch': dataset_config.get('steps_per_epoch', 5), 'epochs_per_round': dataset_config.get('epochs_per_round', 5), 'num_rounds': dataset_config.get('num_rounds', 2)})
    return config

def set_random_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def create_log_directory(log_dir: str) -> str:
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    full_log_dir = os.path.join(log_dir, f'run_{timestamp}')
    os.makedirs(full_log_dir, exist_ok=True)
    return full_log_dir

def ensure_ray_initialized(parallel_cfg: Dict[str, Any], logger: GraphNASLogger) -> None:
    if ray.is_initialized():
        return
    init_kwargs: Dict[str, Any] = {}
    ray_config = parallel_cfg.get('ray', {})
    if isinstance(ray_config, dict):
        init_kwargs.update(ray_config.get('init_kwargs', {}))
        address = ray_config.get('address')
        if address:
            init_kwargs['address'] = address
    workers_config = parallel_cfg.get('workers')
    if workers_config and isinstance(workers_config, (list, tuple)):
        available_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
        valid_gpu_ids = []
        for idx, entry in enumerate(workers_config):
            try:
                gpu_id = int(entry)
                if 0 <= gpu_id < available_gpus:
                    valid_gpu_ids.append(gpu_id)
                else:
                    logger.logger.warning(f'workers[{idx}] = {gpu_id} 超出GPU范围 [0, {available_gpus - 1}]，已忽略')
            except (TypeError, ValueError):
                logger.logger.warning(f'workers[{idx}] = {entry} 无法解析为GPU ID，已忽略')
        if valid_gpu_ids:
            gpu_str = ','.join(map(str, valid_gpu_ids))
            os.environ['CUDA_VISIBLE_DEVICES'] = gpu_str
            logger.logger.info(f'设置 CUDA_VISIBLE_DEVICES={gpu_str}，使用指定GPU: {valid_gpu_ids}')
            if 'num_gpus' not in init_kwargs:
                init_kwargs['num_gpus'] = len(valid_gpu_ids)
    logger.logger.info(f"初始化Ray集群，参数: {(init_kwargs if init_kwargs else '默认配置')}")
    ray.init(**init_kwargs)

def load_full_graph(dataset_name: str, dataset_root: str, logger: GraphNASLogger) -> Dict[str, Any]:
    logger.logger.info(f'加载数据集: {dataset_name} (root={dataset_root})')
    dataset, data = get_dataset(name=dataset_name, root=dataset_root)
    logger.logger.info('数据集统计 => 节点: %d, 边: %d, 特征维度: %d, 类别: %d', data.num_nodes, data.num_edges, data.num_node_features, dataset.num_classes)
    logger.logger.info('mask比例 => train: %d, val: %d, test: %d', int(data.train_mask.sum().item()), int(data.val_mask.sum().item()) if hasattr(data, 'val_mask') else 0, int(data.test_mask.sum().item()) if hasattr(data, 'test_mask') else 0)
    return {'dataset': dataset, 'data': data}

def evaluate_architecture_batch(arch_specs: List[Dict[str, Any]], backend: ArchParallelBackendV2, config: Dict[str, Any], logger: GraphNASLogger, start_arch_index: int=1, stage_name: str='测试', evaluation_round: int=1) -> Tuple[List[Dict[str, Any]], List[float], Dict[str, float]]:
    results = []
    rewards = []
    timing_summary = {'wall_clock_total': 0.0, 'dispatch_overhead': 0.0, 'pure_train_time': 0.0, 'ray_communication': 0.0}
    logger.logger.info(f"\n{'=' * 50}")
    logger.logger.info(f'开始批量并行评估 {stage_name} 阶段的 {len(arch_specs)} 个架构')
    logger.logger.info(f'使用架构并行策略：多个架构同时训练')
    logger.logger.info(f"{'=' * 50}")
    training_cfg = config.get('training', {})
    epochs_per_round = training_cfg.get('epochs_per_round', 5)
    num_rounds = training_cfg.get('num_rounds', 2)
    batch_size = training_cfg.get('batch_size', 1024)
    steps_per_epoch = training_cfg.get('steps_per_epoch', 5)
    layer_wise_specs = []
    arch_ids = []
    for i, arch_spec in enumerate(arch_specs):
        arch_id = f'shap_arch_{start_arch_index + i:05d}'
        arch_ids.append(arch_id)
        try:
            layer_wise_spec = convert_architecture(arch_spec)
            layer_wise_specs.append(layer_wise_spec)
            logger.logger.info(f'架构 {arch_id} 格式转换成功')
        except Exception as e:
            logger.logger.error(f'架构 {arch_id} 格式转换失败: {e}')
            import traceback
            logger.logger.error(f'详细错误信息:\n{traceback.format_exc()}')
            results.append({'architecture_name': arch_id, 'best_accuracy': 0.0, 'best_round': 0, 'total_training_time': 0.0, 'converged': False, 'training_history': [], 'timing_stats': {}, 'architecture_spec': arch_spec, 'layer_wise_spec': {}, 'error': str(e)})
            rewards.append(0.0)
    if not layer_wise_specs:
        logger.logger.warning('没有有效的架构可以训练')
        return (results, rewards, timing_summary)
    model_states = [None] * len(layer_wise_specs)
    all_training_history = [[] for _ in range(len(layer_wise_specs))]
    total_training_times = [0.0] * len(layer_wise_specs)
    for round_idx in range(num_rounds):
        logger.logger.info(f"\n{'=' * 60}")
        logger.logger.info(f'第 {round_idx + 1}/{num_rounds} 轮并行训练')
        logger.logger.info(f'训练 {len(layer_wise_specs)} 个架构')
        logger.logger.info(f"{'=' * 60}")
        round_start_time = time.perf_counter()
        try:
            parallel_results, timing_info = backend.train_architectures_parallel(arch_specs=layer_wise_specs, model_states=model_states, epochs=epochs_per_round, batch_size=batch_size, steps_per_epoch=steps_per_epoch)
            round_duration = time.perf_counter() - round_start_time
            logger.logger.info(f'并行训练完成，耗时: {round_duration:.2f}s')
            for key in timing_summary:
                timing_summary[key] += timing_info.get(key, 0.0)
            for i, (model_state, loss, timing_stats) in enumerate(parallel_results):
                model_states[i] = model_state
                train_time = timing_stats.get('train_time', round_duration / len(layer_wise_specs))
                total_training_times[i] += train_time
                all_training_history[i].append({'round': round_idx + 1, 'loss': loss, 'duration': train_time, 'accumulated_time': total_training_times[i]})
        except Exception as e:
            logger.logger.error(f'第 {round_idx + 1} 轮并行训练失败: {e}')
            import traceback
            logger.logger.error(f'详细错误信息:\n{traceback.format_exc()}')
            for i, arch_id in enumerate(arch_ids):
                if i >= len(results):
                    results.append({'architecture_name': arch_id, 'best_accuracy': 0.0, 'best_round': 0, 'total_training_time': total_training_times[i] if i < len(total_training_times) else 0.0, 'converged': False, 'training_history': all_training_history[i] if i < len(all_training_history) else [], 'timing_stats': {}, 'architecture_spec': arch_specs[i], 'layer_wise_spec': layer_wise_specs[i] if i < len(layer_wise_specs) else {}, 'error': str(e)})
                    rewards.append(0.0)
            return (results, rewards, timing_summary)
    logger.logger.info(f"\n{'=' * 60}")
    logger.logger.info(f'评估所有 {len(layer_wise_specs)} 个训练完成的架构')
    logger.logger.info(f"{'=' * 60}")
    for i, (arch_id, layer_wise_spec, model_state) in enumerate(zip(arch_ids, layer_wise_specs, model_states)):
        try:
            eval_start_time = time.perf_counter()
            eval_result = backend.evaluate_model(layer_wise_spec, model_state)
            eval_duration = time.perf_counter() - eval_start_time
            if eval_result and eval_result.get('evaluation_successful', False):
                accuracy = eval_result.get('global_accuracy', 0.0)
                converged = True
            else:
                accuracy = 0.0
                converged = False
            logger.logger.info(f'架构 {arch_id}: 准确率 = {accuracy:.4f}')
            results.append({'architecture_name': arch_id, 'best_accuracy': accuracy, 'best_round': evaluation_round, 'total_training_time': total_training_times[i], 'converged': converged, 'training_history': all_training_history[i], 'timing_stats': {}, 'architecture_spec': arch_specs[i], 'layer_wise_spec': layer_wise_spec})
            rewards.append(accuracy)
        except Exception as e:
            logger.logger.error(f'架构 {arch_id} 评估失败: {e}')
            import traceback
            logger.logger.error(f'详细错误信息:\n{traceback.format_exc()}')
            results.append({'architecture_name': arch_id, 'best_accuracy': 0.0, 'best_round': 0, 'total_training_time': total_training_times[i], 'converged': False, 'training_history': all_training_history[i], 'timing_stats': {}, 'architecture_spec': arch_specs[i], 'layer_wise_spec': layer_wise_spec, 'error': str(e)})
            rewards.append(0.0)
    return (results, rewards, timing_summary)

def main() -> int:
    parser = argparse.ArgumentParser(description='Ad_OD-Shapley Experiment (Architecture-Parallel + Shapley Search)')
    parser.add_argument('--config', type=str, default='experiment/Ad_OD-shapley/config.yaml', help='配置文件路径')
    parser.add_argument('--dataset', type=str, help='数据集名称（从 dataset.yaml 读取配置）', required=True)
    parser.add_argument('--workers', type=int, default=None, help='覆盖配置中的worker数量')
    parser.add_argument('--rounds-archs', type=str, default=None, help='覆盖Shapley采样配置，格式：初始采样数-剪枝后采样数，例如：10-5')
    args = parser.parse_args()
    print(f'加载配置文件: {args.config}')
    config = load_config(args.config)
    if args.dataset:
        print(f'加载数据集配置: {args.dataset}')
        config = merge_dataset_config(config, args.dataset)
    else:
        raise ValueError('未指定数据集名称')
    parallel_cfg = config.get('parallel_backends', {})
    if args.workers:
        parallel_cfg['num_workers'] = args.workers
    if args.rounds_archs:
        try:
            initial_str, refined_str = args.rounds_archs.split('-')
            initial_samples = int(initial_str)
            refined_samples = int(refined_str)
            if 'architecture' not in config:
                config['architecture'] = {}
            if 'sampler' not in config['architecture']:
                config['architecture']['sampler'] = {}
            config['architecture']['sampler']['initial_samples'] = initial_samples
            config['architecture']['sampler']['refined_samples'] = refined_samples
            print(f'应用--rounds-archs参数: 初始采样数={initial_samples}, 剪枝后采样数={refined_samples}')
        except ValueError as e:
            raise ValueError(f"--rounds-archs参数格式错误，应为'初始采样数-剪枝后采样数'，例如'10-5': {e}")
    exp_cfg = config.get('experiment', {})
    use_random_seed = bool(exp_cfg.get('use_random_seed', False))
    if use_random_seed:
        seed = exp_cfg.get('random_seed', 42)
        set_random_seed(seed)
        print(f'设置随机种子: {seed}')
    else:
        print('未启用随机种子(use_random_seed=false)')
    log_config = config.get('logging', {})
    if log_config.get('save_logs', True):
        log_dir = create_log_directory(log_config.get('log_dir', 'experiment/Ad_OD-shapley/log'))
        print(f'日志目录: {log_dir}')
    else:
        log_dir = None
    logger = GraphNASLogger(name=f'ad_od_shapley_experiment_{args.dataset}', log_dir=log_dir, log_level=log_config.get('level', 'INFO'))
    try:
        logger.logger.info('=' * 80)
        logger.logger.info(f'Ad_OD-Shapley 实验开始, 数据集为{args.dataset}')
        logger.logger.info('=' * 80)
        logger.logger.info(f'数据集: {args.dataset}')
        logger.logger.info(f"Worker数量: {parallel_cfg.get('num_workers', 'auto')}")
        logger.logger.info('\n初始化Ray集群...')
        ensure_ray_initialized(parallel_cfg, logger)
        logger.logger.info('\n加载数据集...')
        dataset_info = load_full_graph(dataset_name=args.dataset, dataset_root=parallel_cfg.get('dataset_root', 'dataset'), logger=logger)
        logger.logger.info('\n初始化架构并行后端...')
        backend = ArchParallelBackendV2(config)
        logger.logger.info('设置分布式训练环境...')
        backend.setup_distributed_training(dataset_info['data'])
        arch_cfg = config.get('architecture', {})
        sampler_cfg = arch_cfg.get('sampler', {})
        shapley_cfg = sampler_cfg.get('shapley', {})
        initial_samples = sampler_cfg.get('initial_samples', 10)
        logger.logger.info(f'\n初始化Shapley搜索器...')
        searcher = ShapleySearcher(config=shapley_cfg)
        logger.logger.info(f"\n{'=' * 80}")
        logger.logger.info('阶段1: 初始架构采样、训练和评估')
        logger.logger.info(f"{'=' * 80}")
        phase1_search_time = 0.0
        phase1_train_time_wallclock = 0.0
        phase1_train_time_accumulated = 0.0
        phase1_eval_time = 0.0
        logger.logger.info(f'\n采样 {initial_samples} 个初始架构...')
        search_start = time.perf_counter()
        initial_architectures = searcher.sample_architectures(initial_samples, stage='initial')
        search_time = time.perf_counter() - search_start
        phase1_search_time += search_time
        logger.logger.info(f'架构采样完成，耗时: {search_time:.4f}s')
        initial_arch_specs = [arch['arch_spec'] for arch in initial_architectures]
        phase1_start = time.perf_counter()
        initial_results, initial_rewards, timing_info = evaluate_architecture_batch(initial_arch_specs, backend, config, logger, start_arch_index=1, stage_name='初始', evaluation_round=1)
        phase1_duration = time.perf_counter() - phase1_start
        if timing_info:
            wall_clock_total = timing_info.get('wall_clock_total', 0.0)
            phase1_train_time_wallclock += wall_clock_total
            phase1_eval_time += max(0, phase1_duration - wall_clock_total)
            logger.logger.info(f'阶段1时间分解: Backend总时间={wall_clock_total:.2f}s, 评估时间={phase1_eval_time:.2f}s')
        for result in initial_results:
            training_history = result.get('training_history', [])
            if training_history:
                last_round = training_history[-1]
                pure_train_time = last_round.get('accumulated_time', 0.0)
                phase1_train_time_accumulated += pure_train_time
        logger.logger.info(f'\n注册初始评估结果到Shapley搜索器...')
        for i, (arch_spec, reward) in enumerate(zip(initial_arch_specs, initial_rewards)):
            arch_id = f'shap_arch_{i + 1:05d}'
            logger.logger.info(f'架构 {arch_id} 格式: {arch_spec}')
            required_components = ['gnnConv1', 'gnnConv2', 'activation1', 'activation2', 'normalize1', 'normalize2', 'aggregation1', 'aggregation2', 'hidden_channels', 'dropout', 'lr', 'weight_decay', 'optimizer', 'criterion', 'pooling']
            missing_components = [comp for comp in required_components if comp not in arch_spec]
            if missing_components:
                logger.logger.warning(f'架构 {arch_id} 缺少组件: {missing_components}')
            standardized_arch_spec = arch_spec.copy()
            if 'activation1' in standardized_arch_spec:
                if standardized_arch_spec['activation1'] == 'relu':
                    standardized_arch_spec['activation1'] = 'Relu'
            if 'activation2' in standardized_arch_spec:
                if standardized_arch_spec['activation2'] == 'relu':
                    standardized_arch_spec['activation2'] = 'Relu'
            if 'gnnConv1' in standardized_arch_spec:
                if standardized_arch_spec['gnnConv1'] == 'GCNConv':
                    standardized_arch_spec['gnnConv1'] = 'GCNConv'
                elif standardized_arch_spec['gnnConv1'] == 'GATConv':
                    standardized_arch_spec['gnnConv1'] = 'GATConv'
            if 'gnnConv2' in standardized_arch_spec:
                if standardized_arch_spec['gnnConv2'] == 'GCNConv':
                    standardized_arch_spec['gnnConv2'] = 'GCNConv'
                elif standardized_arch_spec['gnnConv2'] == 'GATConv':
                    standardized_arch_spec['gnnConv2'] = 'GATConv'
            logger.logger.info(f'标准化后的架构 {arch_id}: {standardized_arch_spec}')
            try:
                searcher.register_evaluation(arch_id, standardized_arch_spec, reward)
                logger.logger.info(f'成功注册架构 {arch_id}')
            except Exception as e:
                logger.logger.error(f'注册架构 {arch_id} 失败: {e}')
                logger.logger.error(f'原始架构格式: {arch_spec}')
                logger.logger.error(f'标准化架构格式: {standardized_arch_spec}')
                raise
        logger.logger.info(f"\n{'=' * 80}")
        logger.logger.info('跳过 阶段2: 预测器训练、剪枝和精化采样（统一为仅阶段1）')
        logger.logger.info(f"{'=' * 80}")
        phase2_predictor_time = 0.0
        phase2_pruning_time = 0.0
        phase2_search_time = 0.0
        phase2_train_time_wallclock = 0.0
        phase2_train_time_accumulated = 0.0
        phase2_eval_time = 0.0
        pruning_history = []
        refined_architectures = []
        refined_arch_specs = []
        phase2_duration = 0.0
        refined_results = []
        refined_rewards = []
        for result in refined_results:
            training_history = result.get('training_history', [])
            pure_train_time = 0.0
            eval_time = 0.0
            if training_history:
                last_round = training_history[-1]
                pure_train_time = last_round.get('accumulated_time', 0.0)
                phase2_train_time_accumulated += pure_train_time
                for round_data in training_history:
                    eval_time += round_data.get('eval_duration', 0.0)
                phase2_eval_time += eval_time
        phase2_train_time_wallclock = phase2_train_time_accumulated
        all_results = initial_results
        all_rewards = initial_rewards
        global_timeline = []
        global_best_accuracy = 0.0
        cumulative_time = 0.0
        for i, result in enumerate(all_results):
            cumulative_time += result.get('total_training_time', 0.0)
            if result.get('best_accuracy', 0.0) > global_best_accuracy:
                global_best_accuracy = result.get('best_accuracy', 0.0)
            global_timeline.append({'cumulative_time': cumulative_time, 'global_best_accuracy': global_best_accuracy, 'sampled_architectures': i + 1, 'architecture_name': result.get('architecture_name', f'arch_{i + 1}'), 'accuracy': result.get('best_accuracy', 0.0)})
        if log_dir:
            try:
                normalized_timeline = []
                for entry in global_timeline:
                    new_entry = entry.copy()
                    if 'sampled_architectures' in new_entry and 'sampled_arch_count' not in new_entry:
                        new_entry['sampled_arch_count'] = new_entry.pop('sampled_architectures')
                    if 'architecture_name' in new_entry and 'current_arch_name' not in new_entry:
                        new_entry['current_arch_name'] = new_entry.pop('architecture_name')
                    if 'accuracy' in new_entry and 'current_accuracy' not in new_entry:
                        new_entry['current_accuracy'] = new_entry.pop('accuracy')
                    normalized_timeline.append(new_entry)
                from utils.export_timeline import normalize_and_save_timeline
                paths = normalize_and_save_timeline(normalized_timeline, log_dir)
                logger.logger.info(f"搜索时间线已保存到: {paths.get('csv')} {paths.get('xlsx')}")
            except Exception as e:
                logger.logger.error(f'导出搜索时间线失败: {e}')
        aggregated_timing_total = {'model_migration': 0.0, 'sampling': 0.0, 'data_transfer': 0.0, 'forward': 0.0, 'backward': 0.0, 'optimizer_step': 0.0, 'sync_time': 0.0, 'sync_transfer_time': 0.0, 'sync_compute_time': 0.0}
        aggregated_timing_avg = {'model_migration': 0.0, 'sampling': 0.0, 'data_transfer': 0.0, 'forward': 0.0, 'backward': 0.0, 'optimizer_step': 0.0, 'sync_time': 0.0, 'sync_transfer_time': 0.0, 'sync_compute_time': 0.0}
        for result in all_results:
            timing_stats = result.get('timing_stats', {})
            for key in aggregated_timing_total:
                aggregated_timing_total[key] += timing_stats.get(key, 0.0)
        num_architectures = len(all_results)
        if num_architectures > 0:
            for key in aggregated_timing_avg:
                aggregated_timing_avg[key] = aggregated_timing_total[key] / num_architectures
        logger.logger.info(f'\n⏱️  CPU时间统计 (所有架构累计):')
        total_cpu_time = sum(aggregated_timing_total.values())
        if total_cpu_time > 0:
            logger.logger.info(f'  总CPU时间: {total_cpu_time:.2f}s')
            logger.logger.info(f"  模型迁移: {aggregated_timing_total['model_migration']:.2f}s ({aggregated_timing_total['model_migration'] / total_cpu_time * 100:.1f}%)")
            logger.logger.info(f"  数据采样: {aggregated_timing_total['sampling']:.2f}s ({aggregated_timing_total['sampling'] / total_cpu_time * 100:.1f}%)")
            logger.logger.info(f"  数据传输: {aggregated_timing_total['data_transfer']:.2f}s ({aggregated_timing_total['data_transfer'] / total_cpu_time * 100:.1f}%)")
            compute_time_cpu = aggregated_timing_total['forward'] + aggregated_timing_total['backward']
            logger.logger.info(f'  计算时间: {compute_time_cpu:.2f}s ({compute_time_cpu / total_cpu_time * 100:.1f}%)')
            logger.logger.info(f"    ├─ 前向传播: {aggregated_timing_total['forward']:.2f}s")
            logger.logger.info(f"    └─ 反向传播: {aggregated_timing_total['backward']:.2f}s")
            logger.logger.info(f"  参数更新: {aggregated_timing_total['optimizer_step']:.2f}s ({aggregated_timing_total['optimizer_step'] / total_cpu_time * 100:.1f}%)")
            logger.logger.info(f"  同步开销: {aggregated_timing_total['sync_time']:.2f}s ({aggregated_timing_total['sync_time'] / total_cpu_time * 100:.1f}%)")
            logger.logger.info(f"    ├─ 参数传输: {aggregated_timing_total['sync_transfer_time']:.2f}s")
            logger.logger.info(f"    └─ 参数聚合: {aggregated_timing_total['sync_compute_time']:.2f}s")
        logger.logger.info(f'\n⏱️  平均worker时间统计:')
        total_avg_time = sum(aggregated_timing_avg.values())
        if total_avg_time > 0:
            logger.logger.info(f'  平均总时间: {total_avg_time:.2f}s')
            logger.logger.info(f"  模型迁移: {aggregated_timing_avg['model_migration']:.2f}s ({aggregated_timing_avg['model_migration'] / total_avg_time * 100:.1f}%)")
            logger.logger.info(f"  数据采样: {aggregated_timing_avg['sampling']:.2f}s ({aggregated_timing_avg['sampling'] / total_avg_time * 100:.1f}%)")
            logger.logger.info(f"  数据传输: {aggregated_timing_avg['data_transfer']:.2f}s ({aggregated_timing_avg['data_transfer'] / total_avg_time * 100:.1f}%)")
            compute_time_avg = aggregated_timing_avg['forward'] + aggregated_timing_avg['backward']
            logger.logger.info(f'  计算时间: {compute_time_avg:.2f}s ({compute_time_avg / total_avg_time * 100:.1f}%)')
            logger.logger.info(f"    ├─ 前向传播: {aggregated_timing_avg['forward']:.2f}s")
            logger.logger.info(f"    └─ 反向传播: {aggregated_timing_avg['backward']:.2f}s")
            logger.logger.info(f"  参数更新: {aggregated_timing_avg['optimizer_step']:.2f}s ({aggregated_timing_avg['optimizer_step'] / total_avg_time * 100:.1f}%)")
            logger.logger.info(f"  同步开销: {aggregated_timing_avg['sync_time']:.2f}s ({aggregated_timing_avg['sync_time'] / total_avg_time * 100:.1f}%)")
            logger.logger.info(f"    ├─ 参数传输: {aggregated_timing_avg['sync_transfer_time']:.2f}s")
            logger.logger.info(f"    └─ 参数聚合: {aggregated_timing_avg['sync_compute_time']:.2f}s")
        logger.logger.info(f'\n📊 阶段1详细时间分解:')
        if phase1_duration > 0:
            logger.logger.info(f'  ├─ 架构采样时间: {phase1_search_time:.2f}s ({phase1_search_time / phase1_duration * 100:.1f}%)')
            logger.logger.info(f'  ├─ 架构训练时间（墙上时钟）: {phase1_train_time_wallclock:.2f}s ({phase1_train_time_wallclock / phase1_duration * 100:.1f}%)')
            logger.logger.info(f'  ├─ 架构训练时间（串行累加）: {phase1_train_time_accumulated:.2f}s (保留旧方案统计)')
            logger.logger.info(f'  └─ 模型评估时间: {phase1_eval_time:.2f}s ({phase1_eval_time / phase1_duration * 100:.1f}%)')
        else:
            logger.logger.info(f'  阶段1未执行（跳过）')
        total_search_time = phase1_search_time
        total_train_time_wallclock = phase1_train_time_wallclock
        total_train_time_accumulated = phase1_train_time_accumulated
        total_eval_time = phase1_eval_time
        total_predictor_time = 0.0
        total_pruning_time = 0.0
        total_duration = phase1_duration
        accounted_time = total_search_time + total_train_time_wallclock + total_eval_time + total_predictor_time + total_pruning_time
        if accounted_time > total_duration:
            accounted_time = total_duration
            unaccounted_time = 0.0
        else:
            unaccounted_time = total_duration - accounted_time
        logger.logger.info(f'\n💡 时间验证:')
        logger.logger.info(f'  已统计时间: {accounted_time:.2f}s')
        logger.logger.info(f'  未统计时间: {unaccounted_time:.2f}s ({abs(unaccounted_time) / total_duration * 100:.1f}%)')
        if abs(unaccounted_time) > 1.0:
            logger.logger.info(f'  ⚠️  注意：未统计时间可能包含初始化、数据加载等开销')
        logger.logger.info(f'\n📈 总体时间分解:')
        logger.logger.info(f'  ├─ 架构采样时间: {total_search_time:.2f}s ({total_search_time / total_duration * 100:.1f}%)')
        logger.logger.info(f'  ├─ 架构训练时间（墙上时钟）: {total_train_time_wallclock:.2f}s ({total_train_time_wallclock / total_duration * 100:.1f}%)')
        logger.logger.info(f'  │      （可与Ad-rl、Ad_OD-rl公平对比）')
        logger.logger.info(f'  ├─ 架构训练时间（串行累加）: {total_train_time_accumulated:.2f}s (保留旧方案统计)')
        logger.logger.info(f'  ├─ 模型评估时间: {total_eval_time:.2f}s ({total_eval_time / total_duration * 100:.1f}%)')
        logger.logger.info('\n' + '=' * 80)
        logger.logger.info('实验结果汇总')
        logger.logger.info('=' * 80)
        if all_results:
            all_results.sort(key=lambda item: item['best_accuracy'], reverse=True)
            logger.logger.info(f'共测试了 {len(all_results)} 个架构:')
            for idx, result in enumerate(all_results, start=1):
                logger.logger.info(f"{idx}. {result['architecture_name']}: Best accuracy {result['best_accuracy']:.4f} (轮次 {result['best_round']}, time {result['total_training_time']:.2f}s)")
            best_result = all_results[0]
            logger.logger.info(f"\n最佳架构: {best_result['architecture_name']}")
            logger.logger.info(f"最佳准确率: {best_result['best_accuracy']:.4f}")
            logger.logger.info(f"训练时间: {best_result['total_training_time']:.2f}s")
            logger.logger.info(f"收敛情况: {('已收敛' if best_result['converged'] else '未收敛')}")
            if log_dir:
                import json
                results_file = os.path.join(log_dir, 'experiment_results.json')
                with open(results_file, 'w', encoding='utf-8') as f:
                    json.dump({'config': config, 'results': all_results, 'timeline': global_timeline, 'pruning_history': pruning_history, 'timing_stats': {'total_cpu_time': aggregated_timing_total, 'average_worker_time': aggregated_timing_avg}}, f, indent=2, ensure_ascii=False)
                logger.logger.info(f'实验结果已保存到: {results_file}')
        else:
            logger.logger.warning('没有成功的实验结果')
        total_training_duration = phase1_duration
        logger.logger.info('\n' + '=' * 80)
        logger.logger.info('最终时间统计')
        logger.logger.info('=' * 80)
        logger.logger.info(f'总墙上时钟时间: {total_training_duration:.2f}s')
        num_total_archs = len(all_results)
        num_successful_archs = sum((1 for r in all_results if r.get('converged', False)))
        num_failed_archs = num_total_archs - num_successful_archs
        if total_search_time > 0 or total_train_time_accumulated > 0 or total_eval_time > 0:
            logger.logger.info(f'\n时间分解:')
            logger.logger.info(f'  |-- 架构采样时间: {total_search_time:.2f}s ({total_search_time / total_training_duration * 100:.1f}%)')
            logger.logger.info(f'  |-- 架构训练时间（墙上时钟）: {total_train_time_wallclock:.2f}s ({total_train_time_wallclock / total_training_duration * 100:.1f}%)')
            logger.logger.info(f'  |-- 架构训练时间（串行累加）: {total_train_time_accumulated:.2f}s (保留旧方案统计)')
            logger.logger.info(f'  |-- 模型评估时间: {total_eval_time:.2f}s ({total_eval_time / total_training_duration * 100:.1f}%)')
            logger.logger.info(f'\n架构统计:')
            logger.logger.info(f'  总评估架构数: {num_total_archs}')
            logger.logger.info(f'  成功架构数: {num_successful_archs}')
            logger.logger.info(f'  失败架构数: {num_failed_archs}')
            if num_total_archs > 0:
                avg_train_time_per_arch = total_train_time_wallclock / num_total_archs
                avg_eval_time_per_arch = total_eval_time / num_total_archs
                logger.logger.info(f'\n平均每架构时间:')
                logger.logger.info(f'  平均训练时间（墙上时钟）: {avg_train_time_per_arch:.5f}s')
                logger.logger.info(f'  平均评估时间: {avg_eval_time_per_arch:.5f}s')
                logger.logger.info(f'  平均总时间: {avg_train_time_per_arch + avg_eval_time_per_arch:.5f}s')
            accounted_time = total_search_time + total_train_time_wallclock + total_eval_time + total_predictor_time + total_pruning_time
            if accounted_time > total_training_duration:
                accounted_time = total_training_duration
                unaccounted_time = 0.0
            else:
                unaccounted_time = total_training_duration - accounted_time
            logger.logger.info(f'\n时间验证:')
            logger.logger.info(f'  已统计时间: {accounted_time:.2f}s')
            logger.logger.info(f'  未统计时间: {unaccounted_time:.2f}s ({abs(unaccounted_time) / total_training_duration * 100:.1f}%)')
            if abs(unaccounted_time) > 1.0:
                logger.logger.info(f'  注意：未统计时间可能包含初始化、数据加载等开销')
        logger.logger.info('\n' + '=' * 80)
        logger.logger.info('📊 实验汇总（统一格式）')
        logger.logger.info('=' * 80)
        logger.logger.info(f'实验-训练数据集: {args.dataset}')
        logger.logger.info(f'实验-纯训练时间: {total_train_time_wallclock:.6f}s (不含评估)')
        logger.logger.info(f'实验-训练架构数: {num_total_archs}')
        if all_results and num_total_archs > 0:
            avg_train_time = total_train_time_wallclock / num_total_archs
            logger.logger.info(f'实验-平均架构训练时间: {avg_train_time:.6f}s')
            best_result = max(all_results, key=lambda x: x['best_accuracy'])
            logger.logger.info(f"实验-最优架构精度: {best_result['best_accuracy']:.6f}")
            logger.logger.info(f"实验-最优架构编号: {best_result['architecture_name']}")
            logger.logger.info(f"实验-最优架构出现轮次: {best_result['best_round']}")
    except Exception as exc:
        logger.logger.error(f'实验执行失败: {exc}')
        import traceback
        traceback.print_exc()
        return 1
    finally:
        backend.cleanup()
        if ray.is_initialized():
            ray.shutdown()
    logger.logger.info('\n' + '=' * 80)
    logger.logger.info(f'Ad_OD-Shapley 实验完成，数据集为 {args.dataset}')
    logger.logger.info('=' * 80)
    return 0
if __name__ == '__main__':
    raise SystemExit(main())