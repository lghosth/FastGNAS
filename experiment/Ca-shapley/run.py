import os
import sys
import yaml
import torch
import argparse
import numpy as np
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
import time
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(project_root)
from baseline.components.backends.ca_backends import CABackends
from baseline.components.searchers.shapley_searcher import ShapleySearcher
from utils.logger import GraphNASLogger
from utils.utils import load_dataset_config
from utils.arch_converter import convert_architecture, convert_architecture_to_dgnap

def load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config

def merge_dataset_config(config: Dict[str, Any], dataset_name: str) -> Dict[str, Any]:
    dataset_config_path = os.path.join(project_root, 'experiment/dataset.yaml')
    dataset_cfg = load_dataset_config(dataset_name, dataset_config_path)
    if 'ca_backends' not in config:
        config['ca_backends'] = {}
    ca_backends = config['ca_backends']
    ca_backends['dataset'] = dataset_cfg.get('name', dataset_name)
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
        ca_backends['num_classes'] = dataset_cfg['num_classes']
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

def run_single_architecture_experiment(ca_backends: CABackends, arch_spec: Dict[str, Any], arch_name: str, config: Dict[str, Any], logger: GraphNASLogger, arch_index: int=0, evaluation_round: int=1) -> Dict[str, Any]:
    original_arch_spec = arch_spec.copy()
    arch_spec = convert_architecture(arch_spec)
    logger.logger.info(f'=' * 50)
    logger.logger.info(f'开始测试架构: {arch_name}')
    logger.logger.info(f'原始DGNAP格式: {original_arch_spec}')
    logger.logger.info(f'转换后Layer-wise格式: {arch_spec}')
    training_config = config['training']
    epochs_per_round = training_config.get('epochs_per_round', 5)
    num_rounds = training_config.get('num_rounds', 2)
    batch_size = training_config.get('batch_size')
    steps_per_epoch = training_config.get('steps_per_epoch')
    results = {'architecture_name': arch_name, 'architecture_spec': arch_spec, 'original_arch_spec': original_arch_spec, 'architecture_index': arch_index, 'training_history': [], 'best_accuracy': 0.0, 'best_round': evaluation_round, 'total_training_time': 0.0, 'converged': True, 'timing_stats': {'model_build': 0.0, 'model_migration': 0.0, 'sampling': 0.0, 'data_transfer': 0.0, 'forward': 0.0, 'backward': 0.0, 'optimizer_step': 0.0, 'sync_time': 0.0, 'sync_transfer_time': 0.0, 'sync_compute_time': 0.0}}
    model_state = None
    best_accuracy = 0.0
    start_time = time.perf_counter()
    accumulated_time = 0.0
    for round_num in range(1, num_rounds + 1):
        logger.logger.info(f'\n--- 第 {round_num}/{num_rounds} 轮训练 ---')
        try:
            train_start_time = time.perf_counter()
            model_state, avg_loss, timing_stats = ca_backends.train_epoch(arch_spec=arch_spec, model_state=model_state, epochs=epochs_per_round, batch_size=batch_size, steps_per_epoch=steps_per_epoch)
            train_time = time.perf_counter() - train_start_time
            accumulated_time += train_time
            for key in results['timing_stats']:
                if key in timing_stats:
                    results['timing_stats'][key] += timing_stats[key]
            eval_start_time = time.perf_counter()
            evaluation_result = ca_backends.evaluate_model(arch_spec, model_state)
            accuracy = evaluation_result.get('global_accuracy', 0.0)
            evaluation_successful = evaluation_result.get('evaluation_successful', True)
            eval_time = time.perf_counter() - eval_start_time
            if round_num == 1 or accuracy > best_accuracy:
                best_accuracy = accuracy
                results['best_accuracy'] = best_accuracy
                logger.logger.info(f'发现更好结果! 最佳准确率: {best_accuracy:.4f}')
            round_result = {'round': round_num, 'loss': avg_loss, 'accuracy': accuracy, 'evaluation': evaluation_result, 'train_time': train_time, 'eval_time': eval_time, 'accumulated_time': accumulated_time, 'best_accuracy_so_far': best_accuracy}
            results['training_history'].append(round_result)
            if not evaluation_successful:
                logger.logger.warning(f'轮次 {round_num} 评估未成功，使用准确率 0.0')
            logger.logger.info(f'轮次 {round_num} 完成 - 损失: {avg_loss:.4f}, 准确率: {accuracy:.4f}, 训练耗时: {train_time:.2f}s, 累计: {accumulated_time:.2f}s')
        except Exception as e:
            logger.logger.error(f'第 {round_num} 轮训练出错: {e}')
            import traceback
            traceback.print_exc()
            break
    total_time = time.perf_counter() - start_time
    results['total_training_time'] = total_time
    logger.logger.info(f'\n架构 {arch_name} 训练完成!')
    logger.logger.info(f"最佳准确率: {results['best_accuracy']:.4f} (第 {results['best_round']} 轮)")
    logger.logger.info(f'总训练时间: {total_time:.2f} 秒')
    timing_stats = results['timing_stats']
    total_timing = sum(timing_stats.values())
    if total_timing > 0:
        logger.logger.info(f'\n⏱️  详细时间分析:')
        logger.logger.info(f'  总计时: {total_timing:.2f}s')
        logger.logger.info(f"  模型迁移: {timing_stats['model_migration']:.4f}s ({timing_stats['model_migration'] / total_timing * 100:.1f}%)")
        logger.logger.info(f"  数据采样: {timing_stats['sampling']:.2f}s ({timing_stats['sampling'] / total_timing * 100:.1f}%)")
        logger.logger.info(f"  数据传输: {timing_stats['data_transfer']:.2f}s ({timing_stats['data_transfer'] / total_timing * 100:.1f}%)")
        compute_time = timing_stats['forward'] + timing_stats['backward']
        logger.logger.info(f'  计算时间: {compute_time:.2f}s ({compute_time / total_timing * 100:.1f}%)')
        logger.logger.info(f"    ├─ 前向传播: {timing_stats['forward']:.2f}s")
        logger.logger.info(f"    └─ 反向传播: {timing_stats['backward']:.2f}s")
        logger.logger.info(f"  参数更新: {timing_stats['optimizer_step']:.2f}s ({timing_stats['optimizer_step'] / total_timing * 100:.1f}%)")
        logger.logger.info(f"  同步开销: {timing_stats['sync_time']:.2f}s ({timing_stats['sync_time'] / total_timing * 100:.1f}%)")
        logger.logger.info(f"    ├─ 参数传输: {timing_stats['sync_transfer_time']:.2f}s")
        logger.logger.info(f"    └─ 参数聚合: {timing_stats['sync_compute_time']:.2f}s")
    return results

def evaluate_architecture_batch(architectures: List[Dict[str, Any]], ca_backends: CABackends, config: Dict[str, Any], logger: GraphNASLogger, start_arch_index: int=1, stage_name: str='initial', evaluation_round: int=1) -> Tuple[List[Dict[str, Any]], Dict[str, float]]:
    batch_results: List[Dict[str, Any]] = []
    rewards: Dict[str, float] = {}
    logger.logger.info(f"\n{'=' * 80}")
    logger.logger.info(f'开始评估 {stage_name} 阶段的 {len(architectures)} 个架构')
    logger.logger.info(f"{'=' * 80}")
    for idx, arch_info in enumerate(architectures):
        arch_name = arch_info['arch_id']
        arch_spec = arch_info['arch_spec']
        current_arch_index = start_arch_index + idx
        try:
            result = run_single_architecture_experiment(ca_backends=ca_backends, arch_spec=arch_spec, arch_name=arch_name, config=config, logger=logger, arch_index=current_arch_index, evaluation_round=evaluation_round)
            batch_results.append(result)
            rewards[arch_name] = result['best_accuracy']
        except Exception as e:
            logger.logger.error(f'架构 {arch_name} 测试失败: {e}')
            import traceback
            traceback.print_exc()
            rewards[arch_name] = 0.0
    return (batch_results, rewards)

def main():
    parser = argparse.ArgumentParser(description='CA-Shapley 实验')
    parser.add_argument('--config', type=str, default='experiment/Ca-shapley/config.yaml', help='配置文件路径')
    parser.add_argument('--dataset', type=str, help='数据集名称', required=True)
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
    if args.workers:
        config['ca_backends']['num_workers'] = args.workers
    if args.rounds_archs:
        try:
            initial_str, refined_str = args.rounds_archs.split('-')
            initial_samples = int(initial_str)
            refined_samples = int(refined_str)
            if 'ca_backends' not in config:
                config['ca_backends'] = {}
            if 'sampler' not in config['ca_backends']:
                config['ca_backends']['sampler'] = {}
            config['ca_backends']['sampler']['initial_samples'] = initial_samples
            config['ca_backends']['sampler']['refined_samples'] = refined_samples
            print(f'应用--rounds-archs参数: 初始采样数={initial_samples}, 剪枝后采样数={refined_samples}')
        except ValueError as e:
            raise ValueError(f"--rounds-archs参数格式错误，应为'初始采样数-剪枝后采样数'，例如'10-5': {e}")
    ca_config = config.get('ca_backends', {})
    if ca_config.get('device', 'cuda') == 'cuda':
        worker_list = ca_config.get('workers', [])
        if worker_list:
            visible_devices = ','.join(map(str, worker_list))
            os.environ['CUDA_VISIBLE_DEVICES'] = visible_devices
            print(f'设置 CUDA_VISIBLE_DEVICES={visible_devices}')
    exp_cfg = config.get('ca_backends', {}).get('experiment', {})
    use_random_seed = bool(exp_cfg.get('use_random_seed', False))
    if use_random_seed:
        seed = exp_cfg.get('random_seed', 42)
        set_random_seed(seed)
        print(f'设置随机种子: {seed}')
    else:
        print('未启用随机种子(use_random_seed=false)')
    log_config = config.get('ca_backends', {}).get('logging', {})
    if log_config.get('save_logs', True):
        log_dir = create_log_directory(log_config.get('log_dir', 'experiment/Ca-shapley/log'))
        print(f'日志目录: {log_dir}')
    else:
        log_dir = None
    logger = GraphNASLogger(name=f'ca_shapley_experiment_{args.dataset}', log_dir=log_dir, log_level=log_config.get('level', 'INFO'))
    logger.logger.info('=' * 80)
    logger.logger.info(f'CA-Shapley 实验开始, 数据集为{args.dataset}')
    logger.logger.info('=' * 80)
    logger.logger.info(f"数据集: {ca_config['dataset']}")
    logger.logger.info(f"Worker数量: {ca_config['num_workers']}")
    try:
        logger.logger.info('\n初始化数据并行后端...')
        full_config = {**ca_config, 'training': config.get('training', {})}
        ca_backends = CABackends(full_config)
        ca_backends.setup_distributed_training()
        partition_info = ca_backends.get_partition_info()
        logger.logger.info('\n数据分区信息:')
        logger.logger.info(f"分区数量: {partition_info['num_partitions']}")
        logger.logger.info(f"总节点数: {partition_info['total_nodes']}")
        logger.logger.info('\n' + '=' * 80)
        logger.logger.info('阶段1: 初始架构采样、训练和评估')
        logger.logger.info('=' * 80)
        sampler_config = ca_config.get('sampler', {})
        shapley_config = sampler_config.get('shapley', {})
        initial_samples = sampler_config.get('initial_samples', 50)
        phase1_search_time = 0.0
        phase1_train_time_wallclock = 0.0
        phase1_train_time_accumulated = 0.0
        phase1_eval_time = 0.0
        logger.logger.info(f'\n初始化Shapley搜索器...')
        searcher = ShapleySearcher(config=shapley_config)
        logger.logger.info(f'\n采样 {initial_samples} 个初始架构...')
        search_start = time.perf_counter()
        initial_architectures = searcher.sample_architectures(initial_samples, stage='initial')
        search_time = time.perf_counter() - search_start
        phase1_search_time += search_time
        phase1_start = time.perf_counter()
        initial_results, initial_rewards = evaluate_architecture_batch(initial_architectures, ca_backends, config, logger, start_arch_index=1, stage_name='初始', evaluation_round=1)
        phase1_duration = time.perf_counter() - phase1_start
        for result in initial_results:
            training_history = result.get('training_history', [])
            pure_train_time = 0.0
            if training_history:
                last_round = training_history[-1]
                pure_train_time = last_round.get('accumulated_time', 0.0)
                phase1_train_time_accumulated += pure_train_time
            total_time = result.get('total_training_time', 0.0)
            result_eval_time = total_time - pure_train_time
            phase1_eval_time += result_eval_time
        phase1_train_time_wallclock = phase1_train_time_accumulated
        logger.logger.info(f'\n注册 {len(initial_results)} 个评估结果到Shapley搜索器...')
        for result in initial_results:
            dgnap_arch_spec = convert_architecture_to_dgnap(result['architecture_spec'])
            logger.logger.info(f"注册架构 {result['architecture_name']}: layer-wise -> DGNAP")
            logger.logger.info(f"  Layer-wise: {result['architecture_spec']}")
            logger.logger.info(f'  DGNAP: {dgnap_arch_spec}')
            searcher.register_evaluation(arch_id=result['architecture_name'], arch_spec=dgnap_arch_spec, reward=result['best_accuracy'], metadata={'total_time': result['total_training_time']})
        logger.logger.info(f'\n阶段1完成! 耗时: {phase1_duration:.2f}s')
        best_initial = max(initial_results, key=lambda x: x['best_accuracy'])
        logger.logger.info(f"最佳准确率: {best_initial['best_accuracy']:.4f}")
        phase2_predictor_time = 0.0
        phase2_pruning_time = 0.0
        phase2_search_time = 0.0
        phase2_train_time_wallclock = 0.0
        phase2_train_time_accumulated = 0.0
        phase2_eval_time = 0.0
        pruned_space = {}
        pruning_history = []
        refined_architectures = []
        phase2_duration = 0.0
        refined_results = []
        refined_rewards = []
        for result in refined_results:
            training_history = result.get('training_history', [])
            pure_train_time = 0.0
            if training_history:
                last_round = training_history[-1]
                pure_train_time = last_round.get('accumulated_time', 0.0)
                phase2_train_time_accumulated += pure_train_time
            total_time = result.get('total_training_time', 0.0)
            result_eval_time = total_time - pure_train_time
            phase2_eval_time += result_eval_time
        phase2_train_time_wallclock = phase2_train_time_accumulated
        all_results = initial_results
        all_results.sort(key=lambda x: x['best_accuracy'], reverse=True)
        logger.logger.info('\n' + '=' * 80)
        logger.logger.info('实验结果汇总')
        logger.logger.info('=' * 80)
        logger.logger.info(f'\n共测试了 {len(all_results)} 个架构')
        best_result = all_results[0]
        logger.logger.info(f"\n最佳架构: {best_result['architecture_name']}")
        logger.logger.info(f"最佳准确率: {best_result['best_accuracy']:.4f}")
        logger.logger.info(f"训练时间: {best_result['total_training_time']:.2f} 秒")
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
                round_time = round_data.get('accumulated_time', 0.0)
                current_accuracy = round_data.get('accuracy', 0.0)
                cumulative_time += round_data.get('train_time', 0.0)
                if current_accuracy > global_best_accuracy:
                    global_best_accuracy = current_accuracy
                    best_arch_index = arch_index
                global_timeline.append({'cumulative_time': cumulative_time, 'global_best_accuracy': global_best_accuracy, 'sampled_arch_count': arch_index, 'best_arch_index': best_arch_index, 'current_arch_name': arch_name, 'current_round': round_data.get('round', 0), 'current_accuracy': current_accuracy})
        if log_dir and global_timeline:
            try:
                from utils.export_timeline import normalize_and_save_timeline
                paths = normalize_and_save_timeline(global_timeline, log_dir)
                logger.logger.info(f"\n📊 搜索时间线已保存到: {paths.get('csv')} {paths.get('xlsx')}")
                logger.logger.info(f'   - 共 {len(global_timeline)} 个数据点')
                logger.logger.info(f'   - 最终最优准确率: {global_best_accuracy:.4f}')
                logger.logger.info(f'   - 最优架构出现在第 {best_arch_index} 个采样架构')
            except Exception as e:
                logger.logger.error(f'导出搜索时间线失败: {e}')
        aggregated_timing_total = {'model_build': 0.0, 'model_migration': 0.0, 'sampling': 0.0, 'data_transfer': 0.0, 'forward': 0.0, 'backward': 0.0, 'optimizer_step': 0.0, 'sync_time': 0.0, 'sync_transfer_time': 0.0, 'sync_compute_time': 0.0}
        aggregated_timing_avg = {'model_build': 0.0, 'model_migration': 0.0, 'sampling': 0.0, 'data_transfer': 0.0, 'forward': 0.0, 'backward': 0.0, 'optimizer_step': 0.0, 'sync_time': 0.0, 'sync_transfer_time': 0.0, 'sync_compute_time': 0.0}
        num_workers = ca_config.get('num_workers', 1)
        for result in all_results:
            timing_stats = result.get('timing_stats', {})
            for key in aggregated_timing_total:
                aggregated_timing_total[key] += timing_stats.get(key, 0.0)
                aggregated_timing_avg[key] += timing_stats.get(key, 0.0) / num_workers
        total_timing_cpu = sum(aggregated_timing_total.values())
        if total_timing_cpu > 0:
            logger.logger.info(f'\n⏱️  CPU时间累计 (所有架构 × 所有Worker之和):')
            logger.logger.info(f'  总计时: {total_timing_cpu:.2f}s')
            logger.logger.info(f"  模型构建: {aggregated_timing_total['model_build']:.2f}s ({aggregated_timing_total['model_build'] / total_timing_cpu * 100:.1f}%)")
            logger.logger.info(f"  模型迁移: {aggregated_timing_total['model_migration']:.2f}s ({aggregated_timing_total['model_migration'] / total_timing_cpu * 100:.1f}%)")
            logger.logger.info(f"  数据采样: {aggregated_timing_total['sampling']:.2f}s ({aggregated_timing_total['sampling'] / total_timing_cpu * 100:.1f}%)")
            logger.logger.info(f"  数据传输: {aggregated_timing_total['data_transfer']:.2f}s ({aggregated_timing_total['data_transfer'] / total_timing_cpu * 100:.1f}%)")
            compute_time = aggregated_timing_total['forward'] + aggregated_timing_total['backward']
            logger.logger.info(f'  计算时间: {compute_time:.2f}s ({compute_time / total_timing_cpu * 100:.1f}%)')
            logger.logger.info(f"    ├─ 前向传播: {aggregated_timing_total['forward']:.2f}s")
            logger.logger.info(f"    └─ 反向传播: {aggregated_timing_total['backward']:.2f}s")
            logger.logger.info(f"  参数更新: {aggregated_timing_total['optimizer_step']:.2f}s ({aggregated_timing_total['optimizer_step'] / total_timing_cpu * 100:.1f}%)")
            logger.logger.info(f"  同步开销: {aggregated_timing_total['sync_time']:.2f}s ({aggregated_timing_total['sync_time'] / total_timing_cpu * 100:.1f}%)")
            logger.logger.info(f"    ├─ 参数传输: {aggregated_timing_total['sync_transfer_time']:.2f}s")
            logger.logger.info(f"    └─ 参数聚合: {aggregated_timing_total['sync_compute_time']:.2f}s")
        total_timing_avg = sum(aggregated_timing_avg.values())
        if total_timing_avg > 0:
            logger.logger.info(f'\n⏱️  平均Worker时间 (接近实际墙上时钟耗时):')
            logger.logger.info(f'  总计时: {total_timing_avg:.2f}s')
            logger.logger.info(f"  模型构建: {aggregated_timing_avg['model_build']:.2f}s ({aggregated_timing_avg['model_build'] / total_timing_avg * 100:.1f}%)")
            logger.logger.info(f"  模型迁移: {aggregated_timing_avg['model_migration']:.2f}s ({aggregated_timing_avg['model_migration'] / total_timing_avg * 100:.1f}%)")
            logger.logger.info(f"  数据采样: {aggregated_timing_avg['sampling']:.2f}s ({aggregated_timing_avg['sampling'] / total_timing_avg * 100:.1f}%)")
            logger.logger.info(f"  数据传输: {aggregated_timing_avg['data_transfer']:.2f}s ({aggregated_timing_avg['data_transfer'] / total_timing_avg * 100:.1f}%)")
            compute_time_avg = aggregated_timing_avg['forward'] + aggregated_timing_avg['backward']
            logger.logger.info(f'  计算时间: {compute_time_avg:.2f}s ({compute_time_avg / total_timing_avg * 100:.1f}%)')
            logger.logger.info(f"    ├─ 前向传播: {aggregated_timing_avg['forward']:.2f}s")
            logger.logger.info(f"    └─ 反向传播: {aggregated_timing_avg['backward']:.2f}s")
            logger.logger.info(f"  参数更新: {aggregated_timing_avg['optimizer_step']:.2f}s ({aggregated_timing_avg['optimizer_step'] / total_timing_avg * 100:.1f}%)")
            logger.logger.info(f"  同步开销: {aggregated_timing_avg['sync_time']:.2f}s ({aggregated_timing_avg['sync_time'] / total_timing_avg * 100:.1f}%)")
            logger.logger.info(f"    ├─ 参数传输: {aggregated_timing_avg['sync_transfer_time']:.2f}s")
            logger.logger.info(f"    └─ 参数聚合: {aggregated_timing_avg['sync_compute_time']:.2f}s")
        if log_dir:
            import json
            import csv
            results_file = os.path.join(log_dir, 'experiment_results.json')
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump(all_results, f, indent=2, default=str)
            logger.logger.info(f'\n实验结果已保存到: {results_file}')
            pruning_file = os.path.join(log_dir, 'pruning_history.json')
            with open(pruning_file, 'w', encoding='utf-8') as f:
                json.dump(pruning_history, f, indent=2, default=str)
            logger.logger.info(f'剪枝历史已保存到: {pruning_file}')
        total_duration = phase1_duration + phase2_duration
        logger.logger.info('\n' + '=' * 80)
        logger.logger.info('⏱️  总体时间统计')
        logger.logger.info('=' * 80)
        logger.logger.info(f'总耗时: {total_duration:.2f}s')
        logger.logger.info(f'  阶段1(初始训练评估): {phase1_duration:.2f}s ({phase1_duration / total_duration * 100:.1f}%)')
        logger.logger.info(f'  阶段2(剪枝+采样训练): {phase2_duration:.2f}s ({phase2_duration / total_duration * 100:.1f}%)')
        logger.logger.info(f'\n📊 阶段1详细时间分解:')
        if phase1_duration > 0:
            logger.logger.info(f'  ├─ 架构采样时间: {phase1_search_time:.2f}s ({phase1_search_time / phase1_duration * 100:.1f}%)')
            logger.logger.info(f'  ├─ 架构训练时间（墙上时钟）: {phase1_train_time_wallclock:.2f}s ({phase1_train_time_wallclock / phase1_duration * 100:.1f}%) ⭐可与Ad-rl对比')
            logger.logger.info(f'  ├─ 架构训练时间（串行累加）: {phase1_train_time_accumulated:.2f}s ({phase1_train_time_accumulated / phase1_duration * 100:.1f}%) [旧方案保留]')
            logger.logger.info(f'  └─ 模型评估时间: {phase1_eval_time:.2f}s ({phase1_eval_time / phase1_duration * 100:.1f}%)')
        else:
            logger.logger.info(f'  阶段1未执行（跳过）')
        logger.logger.info(f'\n📊 阶段2详细时间分解:')
        if phase2_duration > 0:
            logger.logger.info(f'  ├─ 预测器训练时间: {phase2_predictor_time:.2f}s ({phase2_predictor_time / phase2_duration * 100:.1f}%)')
            logger.logger.info(f'  ├─ 搜索空间剪枝时间: {phase2_pruning_time:.2f}s ({phase2_pruning_time / phase2_duration * 100:.1f}%)')
            logger.logger.info(f'  ├─ 架构采样时间: {phase2_search_time:.2f}s ({phase2_search_time / phase2_duration * 100:.1f}%)')
            logger.logger.info(f'  ├─ 架构训练时间（墙上时钟）: {phase2_train_time_wallclock:.2f}s ({phase2_train_time_wallclock / phase2_duration * 100:.1f}%) ⭐可与Ad-rl对比')
            logger.logger.info(f'  ├─ 架构训练时间（串行累加）: {phase2_train_time_accumulated:.2f}s ({phase2_train_time_accumulated / phase2_duration * 100:.1f}%) [旧方案保留]')
            logger.logger.info(f'  └─ 模型评估时间: {phase2_eval_time:.2f}s ({phase2_eval_time / phase2_duration * 100:.1f}%)')
        else:
            logger.logger.info(f'  阶段2未执行（跳过）')
        total_search_time = phase1_search_time + phase2_search_time
        total_train_time_wallclock = phase1_train_time_wallclock + phase2_train_time_wallclock
        total_train_time_accumulated = phase1_train_time_accumulated + phase2_train_time_accumulated
        total_eval_time = phase1_eval_time + phase2_eval_time
        total_predictor_time = phase2_predictor_time
        total_pruning_time = phase2_pruning_time
        accounted_time = total_search_time + total_train_time_wallclock + total_eval_time + total_predictor_time + total_pruning_time
        unaccounted_time = total_duration - accounted_time
        logger.logger.info(f'\n💡 时间验证:')
        logger.logger.info(f'  已统计时间: {accounted_time:.2f}s')
        logger.logger.info(f'  未统计时间: {unaccounted_time:.2f}s ({abs(unaccounted_time) / total_duration * 100:.1f}%)')
        if abs(unaccounted_time) > 1.0:
            logger.logger.info(f'  ⚠️  注意：未统计时间可能包含初始化、数据加载等开销')
        logger.logger.info(f'\n📈 总体时间分解:')
        logger.logger.info(f'  ├─ 架构采样时间: {total_search_time:.2f}s ({total_search_time / total_duration * 100:.1f}%)')
        logger.logger.info(f'  ├─ 架构训练时间（墙上时钟）: {total_train_time_wallclock:.2f}s ({total_train_time_wallclock / total_duration * 100:.1f}%) ⭐可与Ad-rl对比')
        logger.logger.info(f'  ├─ 架构训练时间（串行累加）: {total_train_time_accumulated:.2f}s ({total_train_time_accumulated / total_duration * 100:.1f}%) [旧方案保留]')
        logger.logger.info(f'  ├─ 模型评估时间: {total_eval_time:.2f}s ({total_eval_time / total_duration * 100:.1f}%)')
        logger.logger.info(f'  ├─ 预测器训练时间: {total_predictor_time:.2f}s ({total_predictor_time / total_duration * 100:.1f}%)')
        logger.logger.info(f'  └─ 搜索空间剪枝时间: {total_pruning_time:.2f}s ({total_pruning_time / total_duration * 100:.1f}%)')
        logger.logger.info('\n' + '=' * 80)
        logger.logger.info('📊 实验汇总（统一格式）')
        logger.logger.info('=' * 80)
        logger.logger.info(f'实验-训练数据集: {args.dataset}')
        logger.logger.info(f'实验-纯训练时间: {total_train_time_wallclock:.6f}s (不含评估)')
        logger.logger.info(f'实验-训练架构数: {len(all_results)}')
        if all_results and len(all_results) > 0:
            avg_train_time = total_train_time_wallclock / len(all_results)
            logger.logger.info(f'实验-平均架构训练时间: {avg_train_time:.6f}s')
            best_result = max(all_results, key=lambda x: x['best_accuracy'])
            logger.logger.info(f"实验-最优架构精度: {best_result['best_accuracy']:.6f}")
            logger.logger.info(f"实验-最优架构编号: {best_result['architecture_name']}")
            logger.logger.info(f"实验-最优架构出现轮次: {best_result['best_round']}")
    except Exception as e:
        logger.logger.error(f'实验执行失败: {e}')
        import traceback
        traceback.print_exc()
        return 1
    logger.logger.info('\n' + '=' * 80)
    logger.logger.info(f'CA-Shapley 实验完成，数据集为{args.dataset}')
    logger.logger.info('=' * 80)
    return 0
if __name__ == '__main__':
    exit(main())