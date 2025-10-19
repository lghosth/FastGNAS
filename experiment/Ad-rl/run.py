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
from baseline.components.backends.parallel_backends import ArchParallelBackend
from baseline.components.searchers.rl_searcher import RLBalancedSearcher
from data.data_utils import get_dataset
from utils.logger import GraphNASLogger
from utils.utils import load_dataset_config

def load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config

def merge_dataset_config(config: Dict[str, Any], dataset_name: str) -> Dict[str, Any]:
    dataset_config_path = os.path.join(project_root, 'experiment/dataset.yaml')
    dataset_cfg = load_dataset_config(dataset_name, dataset_config_path)
    parallel_backends = config.get('parallel_backends', {})
    parallel_backends['dataset'] = dataset_cfg.get('name', dataset_name)
    if 'training' not in config:
        config['training'] = {}
    training_cfg = config['training']
    if 'train_batch_size' in dataset_cfg:
        training_cfg['batch_size'] = dataset_cfg['train_batch_size']
    if 'eval_batch_size' in dataset_cfg:
        training_cfg['eval_batch_size'] = dataset_cfg['eval_batch_size']
    if 'fan_out' in dataset_cfg:
        training_cfg['fan_out'] = dataset_cfg['fan_out']
    if 'steps_per_epoch' in dataset_cfg:
        training_cfg['steps_per_epoch'] = dataset_cfg['steps_per_epoch']
    if 'epochs_per_round' in dataset_cfg:
        training_cfg['epochs_per_round'] = dataset_cfg['epochs_per_round']
    if 'num_rounds' in dataset_cfg:
        training_cfg['num_rounds'] = dataset_cfg['num_rounds']
    if 'num_classes' in dataset_cfg:
        training_cfg['num_classes'] = dataset_cfg['num_classes']
        parallel_backends['num_classes'] = dataset_cfg['num_classes']
    return config

def set_random_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
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
                    logger.logger.warning(f'workers[{idx}] = {gpu_id} exceeds GPU range [0, {available_gpus - 1}], ignored')
            except (TypeError, ValueError):
                logger.logger.warning(f'workers[{idx}] = {entry} cannot be parsed as GPU ID, ignored')
        if valid_gpu_ids:
            gpu_str = ','.join(map(str, valid_gpu_ids))
            os.environ['CUDA_VISIBLE_DEVICES'] = gpu_str
            logger.logger.info(f'Set CUDA_VISIBLE_DEVICES={gpu_str}, using specified GPUs: {valid_gpu_ids}')
            if 'num_gpus' not in init_kwargs:
                init_kwargs['num_gpus'] = len(valid_gpu_ids)
    logger.logger.info(f"Initializing Ray cluster, parameters: {(init_kwargs if init_kwargs else 'default config')}")
    ray.init(**init_kwargs)

def load_graph_data(dataset_name: str, dataset_root: str, logger: GraphNASLogger):
    logger.logger.info(f'加载数据集: {dataset_name} (root={dataset_root})')
    dataset, data = get_dataset(name=dataset_name, root=dataset_root)
    logger.logger.info(f'数据集加载完成 - 节点数: {data.num_nodes}, 边数: {data.num_edges}, 特征维度: {data.num_node_features}, 类别数: {dataset.num_classes}')
    return data

def sample_architectures_for_experiment(config: Dict[str, Any], logger: GraphNASLogger, searcher: Optional[RLBalancedSearcher]=None, num_samples: Optional[int]=None) -> Tuple[List[Dict[str, Any]], Optional[RLBalancedSearcher]]:
    arch_cfg = config.get('architecture', {})
    sampler_cfg = arch_cfg.get('sampler', {})
    total_samples = sampler_cfg.get('total_samples')
    controller_cfg = dict(sampler_cfg.get('controller', {}))
    balance_cfg = sampler_cfg.get('balance') if sampler_cfg.get('use_balancer') else None
    sampled: List[Dict[str, Any]] = []
    if total_samples:
        if searcher is None:
            seed = config.get('experiment', {}).get('random_seed')
            if seed is not None:
                controller_cfg.setdefault('random_seed', seed)
            logger.logger.info(f'Initializing RLBalancedSearcher, planning to sample {total_samples} architectures')
            searcher = RLBalancedSearcher(controller_cfg=controller_cfg, balance_cfg=balance_cfg)
        sample_count = num_samples or total_samples
        if balance_cfg and searcher.has_balancer():
            searcher.balance_controller.batch_size = sample_count
            raw_samples = searcher.generate_balanced_batch()
        else:
            raw_samples = searcher.sample_architectures(sample_count)
        raw_samples = raw_samples[:sample_count]
        for idx, sample in enumerate(raw_samples):
            arch_name = sample.get('arch_id') or f'rl_arch_{idx + 1:03d}'
            arch_spec = sample.get('arch_spec', sample)
            sampled.append({'name': arch_name, 'arch_spec': arch_spec})
            logger.logger.info(f'  -> Sampled architecture: {arch_name}')
        return (sampled, searcher)
    static_archs = arch_cfg.get('test_architectures', [])
    for idx, arch_entry in enumerate(static_archs):
        arch_name = arch_entry.get('name', f'static_arch_{idx + 1:03d}')
        arch_spec = {k: v for k, v in arch_entry.items() if k != 'name'}
        sampled.append({'name': arch_name, 'arch_spec': arch_spec})
    if not sampled and 'test_arch' in arch_cfg:
        sampled.append({'name': 'default_arch', 'arch_spec': arch_cfg['test_arch']})
    return (sampled, searcher)

def evaluate_architecture_batch(architectures: List[Dict[str, Any]], backend: ArchParallelBackend, config: Dict[str, Any], logger: GraphNASLogger, start_arch_index: int=1, evaluation_round: int=1) -> Tuple[List[Dict[str, Any]], Dict[str, float], Dict[str, float]]:
    batch_results: List[Dict[str, Any]] = []
    rewards: Dict[str, float] = {}
    if not architectures:
        logger.logger.warning('Architecture batch is empty, skipping evaluation')
        return (batch_results, rewards, {})
    arch_specs = [arch_info['arch_spec'] for arch_info in architectures]
    arch_names = [arch_info['name'] for arch_info in architectures]
    training_cfg = config['training']
    epochs_per_round = training_cfg['epochs_per_round']
    num_rounds = training_cfg['num_rounds']
    batch_size = training_cfg.get('batch_size')
    steps_per_epoch = training_cfg.get('steps_per_epoch')
    logger.logger.info(f'Starting parallel training of {len(arch_specs)} architectures, {num_rounds} rounds per arch, {epochs_per_round} epochs per round')
    model_states = [None] * len(arch_specs)
    best_accuracies = [0.0] * len(arch_specs)
    best_rounds = [evaluation_round] * len(arch_specs)
    training_histories = [[] for _ in range(len(arch_specs))]
    total_training_times = [0.0] * len(arch_specs)
    converged_flags = [True] * len(arch_specs)
    accumulated_timing_stats = [{'model_migration': 0.0, 'sampling': 0.0, 'data_transfer': 0.0, 'forward': 0.0, 'backward': 0.0, 'optimizer_step': 0.0} for _ in range(len(arch_specs))]
    timing_summary = {'wall_clock_total': 0.0, 'dispatch_overhead': 0.0, 'pure_train_time': 0.0, 'ray_communication': 0.0}
    try:
        for round_idx in range(1, num_rounds + 1):
            logger.logger.info(f'\n--- Parallel Batch 第 {round_idx}/{num_rounds} Training ---')
            round_start = time.time()
            parallel_results, timing_info = backend.train_architectures_parallel(arch_specs=arch_specs, model_states=model_states, epochs=epochs_per_round, batch_size=batch_size, steps_per_epoch=steps_per_epoch)
            for key in timing_summary:
                timing_summary[key] += timing_info.get(key, 0.0)
            for i, (new_state, avg_loss, train_stats) in enumerate(parallel_results):
                model_states[i] = new_state
                total_training_times[i] += train_stats.get('train_time', 0.0)
                timing_stats = train_stats.get('timing_stats', {})
                for key in accumulated_timing_stats[i]:
                    accumulated_timing_stats[i][key] += timing_stats.get(key, 0.0)
            evaluation_results = backend.evaluate_architectures_parallel(arch_specs, model_states)
            round_time = time.time() - round_start
            for i, eval_result in enumerate(evaluation_results):
                _, avg_loss, train_stats = parallel_results[i]
                accuracy = eval_result.get('val_accuracy', 0.0)
                evaluation_successful = eval_result.get('evaluation_successful', True)
                training_histories[i].append({'round': round_idx, 'loss': avg_loss, 'accuracy': accuracy, 'evaluation': eval_result, 'time': train_stats.get('train_time', 0.0)})
                if not evaluation_successful:
                    logger.logger.warning(f'Architecture {arch_names[i]} round {round_idx} evaluation unsuccessful; details: {eval_result}')
                logger.logger.info(f'Architecture {arch_names[i]} round {round_idx} - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}')
                if round_idx == 1 or accuracy > best_accuracies[i]:
                    best_accuracies[i] = accuracy
                    logger.logger.info(f'Found better result for architecture {arch_names[i]}! 最佳准确率: {best_accuracies[i]:.4f}')
            logger.logger.info(f'批次轮次 {round_idx} completed, time: {round_time:.2f}s')
    except Exception as exc:
        logger.logger.error(f'Error in parallel training batch: {exc}')
        import traceback
        traceback.print_exc()
        converged_flags = [False] * len(arch_specs)
    for i in range(len(arch_specs)):
        result = {'architecture_name': arch_names[i], 'architecture_spec': arch_specs[i], 'architecture_index': start_arch_index + i, 'training_history': training_histories[i], 'best_accuracy': best_accuracies[i], 'best_round': best_rounds[i], 'total_training_time': total_training_times[i], 'converged': converged_flags[i], 'timing_stats': accumulated_timing_stats[i]}
        batch_results.append(result)
        rewards[arch_names[i]] = best_accuracies[i]
        logger.logger.info(f'\nArchitecture {arch_names[i]} training completed! 最佳准确率: {best_accuracies[i]:.4f} (round {best_rounds[i]}), total training time: {total_training_times[i]:.2f}s')
    return (batch_results, rewards, timing_summary)

def main() -> int:
    parser = argparse.ArgumentParser(description='Ad-RL Experiment (Architecture-Parallel - parallel backend)')
    parser.add_argument('--config', type=str, default='experiment/Ad-rl/config.yaml', help='配置文件路径')
    parser.add_argument('--dataset', type=str, help='数据集名称（从 dataset.yaml 读取配置）', required=True)
    parser.add_argument('--workers', type=int, default=None, help='覆盖配置中的worker数量')
    parser.add_argument('--rounds-archs', type=str, default=None, help='覆盖RL采样配置：reward_times-总架构数，例如：3-10')
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
            rounds_str, archs_str = args.rounds_archs.split('-')
            rounds = int(rounds_str)
            archs = int(archs_str)
            if 'architecture' not in config:
                config['architecture'] = {}
            if 'sampler' not in config['architecture']:
                config['architecture']['sampler'] = {}
            config['architecture']['sampler']['reward_times'] = rounds
            config['architecture']['sampler']['total_samples'] = archs
            print(f'应用--rounds-archs参数: reward_times={rounds}, total_samples={archs}')
        except ValueError as e:
            raise ValueError(f"--rounds-archs参数格式错误，应为'reward_times-总架构数'，例如'3-10': {e}")
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
        log_dir = create_log_directory(log_config.get('log_dir', 'experiment/Ad-rl/log'))
        print(f'日志目录: {log_dir}')
    else:
        log_dir = None
    logger = GraphNASLogger(name=f'ad_experiment_{args.dataset}', log_dir=log_dir, log_level=log_config.get('level', 'INFO'))
    logger.logger.info('=' * 80)
    logger.logger.info(f'Ad-RL Experiment Start (Architecture-Parallel - parallel), Dataset: {args.dataset}')
    logger.logger.info('=' * 80)
    logger.logger.info(f"Dataset: {parallel_cfg['dataset']}")
    logger.logger.info(f"Worker数量: {parallel_cfg['num_workers']}")
    logger.logger.info(f"Backend: {parallel_cfg.get('backend', 'parallel')}")
    logger.logger.info(f"学习率: {parallel_cfg.get('learning_rate', 0.01)}")
    logger.logger.info(f"训练配置: {config['training']}")
    ensure_ray_initialized(parallel_cfg, logger)
    try:
        logger.logger.info('\n加载图数据...')
        dataset_root = parallel_cfg.get('dataset_root', 'dataset')
        full_data = load_graph_data(args.dataset, dataset_root, logger)
        logger.logger.info('\n初始化架构并行后端 (parallel)...')
        full_config = {**parallel_cfg, 'training': config.get('training', {})}
        backend = ArchParallelBackend(full_config)
        backend.setup_distributed_training(full_data)
        logger.logger.info(f'后端初始化完成 => Worker数量: {backend.get_num_workers()}')
        training_phase_start = time.perf_counter()
        training_duration: Optional[float] = None
        sampler_cfg = config.get('architecture', {}).get('sampler', {})
        total_samples = sampler_cfg.get('total_samples')
        reward_times = sampler_cfg.get('reward_times', 0)
        all_results: List[Dict[str, Any]] = []
        total_search_time = 0.0
        total_train_time = 0.0
        total_scheduling_overhead = 0.0
        total_eval_time = 0.0
        search_eval_start = time.perf_counter()
        if total_samples:
            effective_reward_times = reward_times if reward_times and reward_times > 0 else 1
            if reward_times != effective_reward_times:
                logger.logger.warning('reward_times <= 0，默认视为1次更新')
            batch_size_estimate = max(1, math.ceil(total_samples / effective_reward_times))
            logger.logger.info(f'总计需要评估 {total_samples} 个架构，将分 {math.ceil(total_samples / batch_size_estimate)} 批，每批约 {batch_size_estimate} 个')
            processed_total = 0
            batch_index = 0
            searcher: Optional[RLBalancedSearcher] = None
            while processed_total < total_samples:
                batch_index += 1
                remaining = total_samples - processed_total
                target_batch = min(batch_size_estimate, remaining)
                search_start = time.perf_counter()
                batch_archs, searcher = sample_architectures_for_experiment(config, logger, searcher=searcher, num_samples=target_batch)
                search_time = time.perf_counter() - search_start
                total_search_time += search_time
                if not batch_archs:
                    logger.logger.warning('采样批次为空，提前结束评估')
                    break
                if len(batch_archs) > target_batch:
                    batch_archs = batch_archs[:target_batch]
                logger.logger.info(f'开始评估第 {batch_index} 批架构，共 {len(batch_archs)}')
                train_eval_start = time.perf_counter()
                batch_results, batch_rewards, timing_info = evaluate_architecture_batch(batch_archs, backend, config, logger, start_arch_index=processed_total + 1, evaluation_round=batch_index)
                train_eval_time = time.perf_counter() - train_eval_start
                if timing_info:
                    wall_clock_total = timing_info.get('wall_clock_total', 0.0)
                    dispatch_overhead = timing_info.get('dispatch_overhead', 0.0)
                    pure_train = timing_info.get('pure_train_time', 0.0)
                    ray_comm = timing_info.get('ray_communication', 0.0)
                    total_train_time += wall_clock_total
                    total_scheduling_overhead += dispatch_overhead
                    batch_eval_time = max(0, train_eval_time - wall_clock_total)
                    total_eval_time += batch_eval_time
                    logger.logger.debug(f'批次 {batch_index} 时间分解: Backend总时间={wall_clock_total:.2f}s (调度={dispatch_overhead:.4f}s, 纯训练={pure_train:.2f}s, Ray通信={ray_comm:.4f}s), 评估={batch_eval_time:.2f}s')
                elif batch_results:
                    max_train_time = max([r.get('total_training_time', 0.0) for r in batch_results])
                    total_train_time += max_train_time
                    batch_eval_time = max(0, train_eval_time - max_train_time)
                    total_eval_time += batch_eval_time
                all_results.extend(batch_results)
                processed_total += len(batch_archs)
                if batch_rewards and effective_reward_times:
                    logger.logger.info(f'第 {batch_index} 批评估完成，回传 {len(batch_rewards)} 个奖励')
                    searcher.update_rewards(batch_rewards, default_reward=0.0)
            if processed_total < total_samples:
                logger.logger.warning(f'计划评估 {total_samples} 个架构，但实际完成 {processed_total}')
        else:
            test_archs, _ = sample_architectures_for_experiment(config, logger)
            batch_results, _, _ = evaluate_architecture_batch(test_archs, backend, config, logger, evaluation_round=1)
            all_results.extend(batch_results)
        search_eval_duration = time.perf_counter() - search_eval_start
        logger.logger.info(f'\n' + '=' * 80)
        logger.logger.info('⏱️  时间统计汇总')
        logger.logger.info('=' * 80)
        logger.logger.info(f'总耗时: {search_eval_duration:.2f}s')
        logger.logger.info(f'  ├─ 架构采样时间: {total_search_time:.2f}s ({total_search_time / search_eval_duration * 100:.1f}%)')
        logger.logger.info(f'  ├─ 架构训练时间: {total_train_time:.2f}s ({total_train_time / search_eval_duration * 100:.1f}%)')
        if total_scheduling_overhead > 0.0:
            logger.logger.info(f'  │   ├─ 其中调度开销: {total_scheduling_overhead:.2f}s ({total_scheduling_overhead / total_train_time * 100:.1f}%)')
            pure_train_only = total_train_time - total_scheduling_overhead
            logger.logger.info(f'  │   └─ 纯训练时间: {pure_train_only:.2f}s ({pure_train_only / total_train_time * 100:.1f}%)')
        logger.logger.info(f'  └─ 架构评估时间: {total_eval_time:.2f}s ({total_eval_time / search_eval_duration * 100:.1f}%)')
        if all_results:
            aggregated_timing = {'model_migration': 0.0, 'sampling': 0.0, 'data_transfer': 0.0, 'forward': 0.0, 'backward': 0.0, 'optimizer_step': 0.0}
            num_workers = config.get('parallel_backends', {}).get('num_workers', 4)
            wall_clock_timing = {'model_migration': 0.0, 'sampling': 0.0, 'data_transfer': 0.0, 'forward': 0.0, 'backward': 0.0, 'optimizer_step': 0.0}
            for batch_start in range(0, len(all_results), num_workers):
                batch_end = min(batch_start + num_workers, len(all_results))
                batch_results = all_results[batch_start:batch_end]
                for key in wall_clock_timing:
                    batch_max = max((r.get('timing_stats', {}).get(key, 0.0) for r in batch_results), default=0.0)
                    wall_clock_timing[key] += batch_max
            for result in all_results:
                timing_stats = result.get('timing_stats', {})
                for key in aggregated_timing:
                    aggregated_timing[key] += timing_stats.get(key, 0.0)
            total_wall_clock = sum(wall_clock_timing.values())
            if total_wall_clock > 0:
                logger.logger.info(f'\n⏱️  训练细粒度时间分解 (墙上时钟时间 - 仅训练阶段):')
                logger.logger.info(f'  纯训练时间: {total_wall_clock:.2f}s (架构并行批次，各批次取max后累加)')
                logger.logger.info(f"  ├─ 模型迁移: {wall_clock_timing['model_migration']:.2f}s ({wall_clock_timing['model_migration'] / total_wall_clock * 100:.1f}%)")
                logger.logger.info(f"  ├─ 数据采样: {wall_clock_timing['sampling']:.2f}s ({wall_clock_timing['sampling'] / total_wall_clock * 100:.1f}%)")
                logger.logger.info(f"  ├─ 数据传输: {wall_clock_timing['data_transfer']:.2f}s ({wall_clock_timing['data_transfer'] / total_wall_clock * 100:.1f}%)")
                compute_time = wall_clock_timing['forward'] + wall_clock_timing['backward']
                logger.logger.info(f'  ├─ 计算时间: {compute_time:.2f}s ({compute_time / total_wall_clock * 100:.1f}%)')
                logger.logger.info(f"  │  ├─ 前向传播: {wall_clock_timing['forward']:.2f}s")
                logger.logger.info(f"  │  └─ 反向传播: {wall_clock_timing['backward']:.2f}s")
                logger.logger.info(f"  └─ 参数更新: {wall_clock_timing['optimizer_step']:.2f}s ({wall_clock_timing['optimizer_step'] / total_wall_clock * 100:.1f}%)")
                eval_overhead = max(0, total_train_time - total_wall_clock)
                if eval_overhead > 0.01:
                    logger.logger.info(f'\n  评估开销: {eval_overhead:.2f}s (架构训练时间 - 纯训练时间)')
            total_cpu_time = sum(aggregated_timing.values())
            if total_cpu_time > 0:
                logger.logger.info(f'\n⏱️  训练细粒度时间分解 (所有架构CPU时间累计):')
                logger.logger.info(f'  总CPU时间: {total_cpu_time:.2f}s')
                logger.logger.info(f"  ├─ 模型迁移: {aggregated_timing['model_migration']:.2f}s ({aggregated_timing['model_migration'] / total_cpu_time * 100:.1f}%)")
                logger.logger.info(f"  ├─ 数据采样: {aggregated_timing['sampling']:.2f}s ({aggregated_timing['sampling'] / total_cpu_time * 100:.1f}%)")
                logger.logger.info(f"  ├─ 数据传输: {aggregated_timing['data_transfer']:.2f}s ({aggregated_timing['data_transfer'] / total_cpu_time * 100:.1f}%)")
                compute_time_cpu = aggregated_timing['forward'] + aggregated_timing['backward']
                logger.logger.info(f'  ├─ 计算时间: {compute_time_cpu:.2f}s ({compute_time_cpu / total_cpu_time * 100:.1f}%)')
                logger.logger.info(f"  │  ├─ 前向传播: {aggregated_timing['forward']:.2f}s")
                logger.logger.info(f"  │  └─ 反向传播: {aggregated_timing['backward']:.2f}s")
                logger.logger.info(f"  └─ 参数更新: {aggregated_timing['optimizer_step']:.2f}s ({aggregated_timing['optimizer_step'] / total_cpu_time * 100:.1f}%)")
            if total_cpu_time > 0 and total_wall_clock > 0:
                parallelism_ratio = total_cpu_time / total_wall_clock
                theoretical_max = num_workers * (len(all_results) // num_workers + (1 if len(all_results) % num_workers else 0))
                efficiency = parallelism_ratio / theoretical_max * 100 if theoretical_max > 0 else 0
                logger.logger.info(f'\n⚡ 架构并行效率分析:')
                logger.logger.info(f'  Worker数量: {num_workers}')
                logger.logger.info(f'  架构总数: {len(all_results)}')
                logger.logger.info(f'  并行批次数: {(len(all_results) + num_workers - 1) // num_workers}')
                logger.logger.info(f'  墙上时钟时间: {total_wall_clock:.2f}s')
                logger.logger.info(f'  CPU累计时间: {total_cpu_time:.2f}s')
                logger.logger.info(f'  并行倍率: {parallelism_ratio:.2f}x')
                logger.logger.info(f'  理论最大倍率: {theoretical_max}x')
                logger.logger.info(f'  并行效率: {efficiency:.1f}%')
                if efficiency < 50:
                    logger.logger.info(f'  ⚠️  并行效率较低，可能存在：')
                    logger.logger.info(f'      - 架构间负载不均衡')
                    logger.logger.info(f'      - 批次内等待时间过长')
                elif efficiency < 80:
                    logger.logger.info(f'  ✅ 并行效率正常')
                else:
                    logger.logger.info(f'  🎉 并行效率优秀')
        logger.logger.info('\n' + '=' * 80)
        logger.logger.info('实验结果汇总')
        logger.logger.info('=' * 80)
        if all_results:
            all_results.sort(key=lambda item: item['best_accuracy'], reverse=True)
            logger.logger.info(f'共测试了 {len(all_results)} architectures:')
            for idx, result in enumerate(all_results, start=1):
                logger.logger.info(f"{idx}. {result['architecture_name']}: Best accuracy {result['best_accuracy']:.4f} (round {result['best_round']}, time {result['total_training_time']:.2f}s)")
            best_result = all_results[0]
            logger.logger.info(f"\n最佳架构: {best_result['architecture_name']}")
            logger.logger.info(f"最佳准确率: {best_result['best_accuracy']:.4f}")
            logger.logger.info(f"训练时间: {best_result['total_training_time']:.2f}s")
            logger.logger.info(f"收敛情况: {('已收敛' if best_result['converged'] else '未收敛')}")
            if log_dir:
                import csv
                import json
                global_timeline = []
                global_best_accuracy = 0.0
                best_arch_index = 0
                cumulative_time = 0.0
                all_results_sorted_by_index = sorted(all_results, key=lambda x: x.get('architecture_index', 0))
                for result in all_results_sorted_by_index:
                    arch_index = result.get('architecture_index', 0)
                    arch_name = result['architecture_name']
                    training_history = result.get('training_history', [])
                    for round_data in training_history:
                        current_accuracy = round_data.get('accuracy', 0.0)
                        round_time = round_data.get('time', 0.0)
                        cumulative_time += round_time
                        if current_accuracy > global_best_accuracy:
                            global_best_accuracy = current_accuracy
                            best_arch_index = arch_index
                        global_timeline.append({'cumulative_time': cumulative_time, 'global_best_accuracy': global_best_accuracy, 'sampled_arch_count': arch_index, 'best_arch_index': best_arch_index, 'current_arch_name': arch_name, 'current_round': round_data.get('round', 0), 'current_accuracy': current_accuracy})
                try:
                    from utils.export_timeline import normalize_and_save_timeline
                    paths = normalize_and_save_timeline(global_timeline, log_dir)
                    logger.logger.info(f"\n搜索时间线已保存到: {paths.get('csv')} {paths.get('xlsx')}")
                    logger.logger.info(f'   - 共 {len(global_timeline)} 个数据点')
                    logger.logger.info(f'   - 最终最优准确率: {global_best_accuracy:.4f}')
                    logger.logger.info(f'   - 最优架构出现在第{best_arch_index}')
                except Exception as e:
                    logger.logger.error(f'导出搜索时间线失败: {e}')
                results_file = os.path.join(log_dir, 'experiment_results.json')
                with open(results_file, 'w', encoding='utf-8') as result_file:
                    json.dump(all_results, result_file, indent=2, default=str)
                logger.logger.info(f'实验结果已保存到: {results_file}')
        else:
            logger.logger.warning('没有成功的实验结果')
        training_duration = time.perf_counter() - training_phase_start
    except Exception as exc:
        logger.logger.error(f'实验执行失败: {exc}')
        import traceback
        traceback.print_exc()
        return 1
    finally:
        if ray.is_initialized():
            ray.shutdown()
    if training_duration is None:
        training_duration = time.perf_counter() - training_phase_start
    logger.logger.info(f'并行训练总耗时(不含初始化): {training_duration:.2f}s')
    logger.logger.info('\n' + '=' * 80)
    logger.logger.info('📊 实验汇总（统一格式）')
    logger.logger.info('=' * 80)
    logger.logger.info(f'实验-训练数据集: {args.dataset}')
    logger.logger.info(f'实验-纯训练时间: {total_train_time:.6f}s (不含评估)')
    logger.logger.info(f'实验-训练架构数: {len(all_results)}')
    if all_results and len(all_results) > 0:
        avg_train_time = total_train_time / len(all_results)
        logger.logger.info(f'实验-平均架构训练时间: {avg_train_time:.6f}s')
        best_result = max(all_results, key=lambda x: x['best_accuracy'])
        logger.logger.info(f"实验-最优架构精度: {best_result['best_accuracy']:.6f}")
        logger.logger.info(f"实验-最优架构编号: {best_result['architecture_name']}")
        logger.logger.info(f"实验-最优架构出现轮次: {best_result['best_round']}")
    logger.logger.info('\n' + '=' * 80)
    logger.logger.info(f'Ad-RL 实验完成，数据集为 {args.dataset}')
    logger.logger.info('=' * 80)
    return 0
if __name__ == '__main__':
    raise SystemExit(main())