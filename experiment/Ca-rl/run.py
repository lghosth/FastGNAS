import os
import sys
import yaml
import torch
import argparse
import numpy as np
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
import time
import math
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(project_root)
from baseline.components.backends.ca_backends import CABackends
from baseline.components.searchers.rl_searcher import RLBalancedSearcher
from utils.logger import GraphNASLogger
from utils.utils import load_dataset_config

def load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config

def merge_dataset_config(config: Dict[str, Any], dataset_name: str) -> Dict[str, Any]:
    dataset_config_path = os.path.join(project_root, 'experiment/dataset.yaml')
    dataset_cfg = load_dataset_config(dataset_name, dataset_config_path)
    ca_backends = config.get('ca_backends', {})
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
    logger.logger.info(f'=' * 50)
    logger.logger.info(f'开始测试架构: {arch_name}')
    logger.logger.info(f'架构规格: {arch_spec}')
    training_config = config['training']
    epochs_per_round = training_config.get('epochs_per_round', 5)
    num_rounds = training_config.get('num_rounds', 2)
    batch_size = training_config.get('batch_size')
    steps_per_epoch = training_config.get('steps_per_epoch')
    results = {'architecture_name': arch_name, 'architecture_spec': arch_spec, 'architecture_index': arch_index, 'training_history': [], 'best_accuracy': 0.0, 'best_round': evaluation_round, 'total_training_time': 0.0, 'converged': True, 'timing_stats': {'model_migration': 0.0, 'sampling': 0.0, 'data_transfer': 0.0, 'forward': 0.0, 'backward': 0.0, 'optimizer_step': 0.0, 'sync_time': 0.0}}
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
                logger.logger.info(f'🎉 发现更好结果! 最佳准确率: {best_accuracy:.4f}')
            round_result = {'round': round_num, 'loss': avg_loss, 'accuracy': accuracy, 'evaluation': evaluation_result, 'train_time': train_time, 'eval_time': eval_time, 'accumulated_time': accumulated_time, 'best_accuracy_so_far': best_accuracy}
            results['training_history'].append(round_result)
            if not evaluation_successful:
                logger.logger.warning(f'轮次 {round_num} 评估未成功，使用准确率 0.0; 详情: {evaluation_result}')
            logger.logger.info(f'轮次 {round_num} 完成 - 损失: {avg_loss:.4f}, 准确率: {accuracy:.4f}, 训练耗时: {train_time:.2f}s, 评估耗时: {eval_time:.2f}s, 累计时间: {accumulated_time:.2f}s, 历史最优: {best_accuracy:.4f}')
        except Exception as e:
            logger.logger.error(f'第 {round_num} 轮训练出错: {e}')
            import traceback
            traceback.print_exc()
            break
    total_time = time.perf_counter() - start_time
    results['total_training_time'] = total_time
    logger.logger.info(f'\n🏁 架构 {arch_name} 训练完成!')
    logger.logger.info(f"最佳准确率: {results['best_accuracy']:.4f} (第 {results['best_round']} 轮)")
    logger.logger.info(f'总训练时间: {total_time:.2f} 秒')
    logger.logger.info(f"是否收敛: {('是' if results['converged'] else '否')}")
    timing_stats = results['timing_stats']
    cpu_total = sum(timing_stats.values())
    num_workers = getattr(ca_backends, 'num_workers', 1) or 1
    cpu_avg = cpu_total / num_workers if num_workers > 0 else cpu_total
    if cpu_total > 0:
        logger.logger.info(f'\n⏱️  详细时间分析:')
        logger.logger.info(f'  CPU累计总计时 (所有Worker之和): {cpu_total:.2f}s')
        logger.logger.info(f'  平均每Worker时间 (近似墙上时钟): {cpu_avg:.2f}s (Workers={num_workers})')
        logger.logger.info(f'  墙上时钟总训练时间: {total_time:.2f}s')
        logger.logger.info(f'\n  训练过程分解 (基于CPU累计):')
        logger.logger.info(f"    ├─ 模型迁移: {timing_stats['model_migration']:.4f}s ({timing_stats['model_migration'] / cpu_total * 100:.1f}% of CPU累计, {timing_stats['model_migration'] / num_workers:.4f}s/worker)")
        logger.logger.info(f"    ├─ 数据采样: {timing_stats['sampling']:.2f}s ({timing_stats['sampling'] / cpu_total * 100:.1f}% of CPU累计, {timing_stats['sampling'] / num_workers:.2f}s/worker)")
        logger.logger.info(f"    ├─ 数据传输: {timing_stats['data_transfer']:.2f}s ({timing_stats['data_transfer'] / cpu_total * 100:.1f}% of CPU累计, {timing_stats['data_transfer'] / num_workers:.2f}s/worker)")
        compute_time = timing_stats['forward'] + timing_stats['backward']
        logger.logger.info(f'    ├─ 计算时间: {compute_time:.2f}s ({compute_time / cpu_total * 100:.1f}% of CPU累计, {compute_time / num_workers:.2f}s/worker)')
        logger.logger.info(f"    │   ├─ 前向传播: {timing_stats['forward']:.2f}s")
        logger.logger.info(f"    │   └─ 反向传播: {timing_stats['backward']:.2f}s")
        logger.logger.info(f"    ├─ 参数更新: {timing_stats['optimizer_step']:.2f}s ({timing_stats['optimizer_step'] / cpu_total * 100:.1f}% of CPU累计, {timing_stats['optimizer_step'] / num_workers:.2f}s/worker)")
        logger.logger.info(f"    └─ 同步开销: {timing_stats['sync_time']:.2f}s ({timing_stats['sync_time'] / cpu_total * 100:.1f}% of CPU累计, {timing_stats['sync_time'] / num_workers:.2f}s/worker)")
        logger.logger.info(f'        (Ray通信 + 参数聚合，详见调试日志)')
    return results

def sample_architectures_for_experiment(config: Dict[str, Any], logger: GraphNASLogger, searcher: Optional[RLBalancedSearcher]=None, num_samples: Optional[int]=None) -> Tuple[List[Dict[str, Any]], Optional[RLBalancedSearcher]]:
    ca_backends_config = config.get('ca_backends', {})
    sampler_config = ca_backends_config.get('sampler', {})
    total_samples = sampler_config.get('total_samples')
    controller_cfg = dict(sampler_config.get('controller', {}))
    balance_cfg = sampler_config.get('balance') if sampler_config.get('use_balancer') else None
    sampled_architectures: List[Dict[str, Any]] = []
    if total_samples:
        if searcher is None:
            seed = config.get('ca_backends', {}).get('experiment', {}).get('random_seed')
            if seed is not None:
                controller_cfg.setdefault('random_seed', seed)
            logger.logger.info(f'初始化 RLBalancedSearcher，计划采样 {total_samples} 个架构')
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
            sampled_architectures.append({'name': arch_name, 'arch_spec': arch_spec})
            logger.logger.info(f'  -> 采样架构: {arch_name}')
        return (sampled_architectures, searcher)
    logger.logger.warning('未配置 total_samples，无法进行 RL 采样')
    return (sampled_architectures, searcher)

def evaluate_architecture_batch(architectures: List[Dict[str, Any]], ca_backends: CABackends, config: Dict[str, Any], logger: GraphNASLogger, start_arch_index: int=1, evaluation_round: int=1) -> Tuple[List[Dict[str, Any]], Dict[str, float]]:
    batch_results: List[Dict[str, Any]] = []
    rewards: Dict[str, float] = {}
    for idx, arch_info in enumerate(architectures):
        arch_name = arch_info['name']
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
    parser = argparse.ArgumentParser(description='CA-RL 实验')
    parser.add_argument('--config', type=str, default='experiment/Ca-rl/config.yaml', help='配置文件路径')
    parser.add_argument('--dataset', type=str, help='数据集名称（从 dataset.yaml 读取配置）', required=True)
    parser.add_argument('--workers', type=int, default=None, help='覆盖配置中的worker数量')
    parser.add_argument('--rounds-archs', type=str, default=None, help='覆盖RL训练配置，格式：轮次-架构数，例如：3-10')
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
        try:
            config['ca_backends']['workers'] = list(range(int(args.workers)))
        except Exception:
            pass
    if args.rounds_archs:
        try:
            rounds_str, archs_str = args.rounds_archs.split('-')
            rounds = int(rounds_str)
            archs = int(archs_str)
            if 'ca_backends' not in config:
                config['ca_backends'] = {}
            if 'sampler' not in config['ca_backends']:
                config['ca_backends']['sampler'] = {}
            config['ca_backends']['sampler']['total_samples'] = archs
            config['ca_backends']['sampler']['reward_times'] = rounds
            print(f'应用--rounds-archs参数: 轮次={rounds}, 架构数={archs}')
        except ValueError as e:
            raise ValueError(f"--rounds-archs参数格式错误，应为'轮次-架构数'，例如'3-10': {e}")
    ca_config = config.get('ca_backends', {})
    if ca_config.get('device', 'cuda') == 'cuda':
        worker_list = ca_config.get('workers', [])
        if worker_list:
            visible_devices = ','.join(map(str, worker_list))
            os.environ['CUDA_VISIBLE_DEVICES'] = visible_devices
            print(f'设置 CUDA_VISIBLE_DEVICES={visible_devices}')
        else:
            os.environ.pop('CUDA_VISIBLE_DEVICES', None)
            print('未指定 workers，使用系统默认可见GPU')
    seed_cfg = config.get('ca_backends', {}).get('experiment', {})
    use_random_seed = bool(seed_cfg.get('use_random_seed', False))
    if use_random_seed:
        seed = seed_cfg.get('random_seed', 42)
        set_random_seed(seed)
        print(f'设置随机种子: {seed}')
    else:
        print('未启用随机种子(use_random_seed=false)')
    log_config = config.get('ca_backends', {}).get('logging', {})
    if log_config.get('save_logs', True):
        log_dir = create_log_directory(log_config.get('log_dir', 'experiment/Ca-rl/log'))
        print(f'日志目录: {log_dir}')
    else:
        log_dir = None
    logger = GraphNASLogger(name=f'ca_rl_experiment_{args.dataset}', log_dir=log_dir, log_level=log_config.get('level', 'INFO'))
    logger.logger.info('=' * 80)
    logger.logger.info(f'CA-RL 实验开始, 数据集为{args.dataset}')
    logger.logger.info('=' * 80)
    logger.logger.info(f"数据集: {ca_config['dataset']}")
    logger.logger.info(f"Worker数量: {ca_config['num_workers']}")
    logger.logger.info(f"设备类型: {ca_config.get('device', 'cuda')}")
    logger.logger.info(f"学习率: {ca_config.get('learning_rate', 0.01)}")
    logger.logger.info(f"训练配置: {config['training']}")
    try:
        logger.logger.info('\n初始化数据并行后端...')
        full_config = {**ca_config, 'training': config.get('training', {})}
        ca_backends = CABackends(full_config)
        ca_backends.setup_distributed_training()
        partition_info = ca_backends.get_partition_info()
        logger.logger.info('\n📊 数据分区信息:')
        logger.logger.info(f"分区数量: {partition_info['num_partitions']}")
        logger.logger.info(f"总节点数: {partition_info['total_nodes']}")
        logger.logger.info(f"总边数: {partition_info['total_edges']}")
        for partition in partition_info['partitions']:
            logger.logger.info(f"  分区 {partition['id']}: {partition['nodes']} 节点, 训练={partition['train_nodes']}, 验证={partition['val_nodes']}")
        training_phase_start = time.perf_counter()
        search_eval_start = time.perf_counter()
        search_eval_duration = 0.0
        sampler_config = ca_config.get('sampler', {})
        total_samples = sampler_config.get('total_samples')
        reward_times = sampler_config.get('reward_times', 0)
        all_results: List[Dict[str, Any]] = []
        total_search_time = 0.0
        total_train_time_wallclock = 0.0
        total_train_time_accumulated = 0.0
        total_eval_time = 0.0
        if total_samples:
            if reward_times is None or reward_times <= 0:
                logger.logger.warning('reward_times <= 0，默认视为1次更新')
                effective_reward_times = 1
            else:
                effective_reward_times = reward_times
            batch_size = max(1, math.ceil(total_samples / effective_reward_times))
            logger.logger.info(f'总计需要评估 {total_samples} 个架构，将分 {math.ceil(total_samples / batch_size)} 批，每批约 {batch_size} 个')
            processed_total = 0
            batch_index = 0
            searcher: Optional[RLBalancedSearcher] = None
            while processed_total < total_samples:
                batch_index += 1
                remaining = total_samples - processed_total
                target_batch_size = min(batch_size, remaining)
                search_start = time.perf_counter()
                batch_architectures, searcher = sample_architectures_for_experiment(config, logger, searcher=searcher, num_samples=target_batch_size)
                search_time = time.perf_counter() - search_start
                total_search_time += search_time
                if not batch_architectures:
                    logger.logger.warning('采样批次为空，提前结束评估')
                    break
                if len(batch_architectures) > target_batch_size:
                    batch_architectures = batch_architectures[:target_batch_size]
                logger.logger.info(f'开始评估第 {batch_index} 批架构，共 {len(batch_architectures)} 个')
                train_eval_start = time.perf_counter()
                batch_results, batch_rewards = evaluate_architecture_batch(batch_architectures, ca_backends, config, logger, start_arch_index=processed_total + 1, evaluation_round=batch_index)
                batch_wallclock_time = time.perf_counter() - train_eval_start
                batch_accumulated_train_time = 0.0
                batch_accumulated_eval_time = 0.0
                for result in batch_results:
                    training_history = result.get('training_history', [])
                    pure_train_time = 0.0
                    if training_history:
                        last_round = training_history[-1]
                        pure_train_time = last_round.get('accumulated_time', 0.0)
                        batch_accumulated_train_time += pure_train_time
                    total_time = result.get('total_training_time', 0.0)
                    result_eval_time = total_time - pure_train_time
                    batch_accumulated_eval_time += result_eval_time
                total_train_time_accumulated += batch_accumulated_train_time
                total_eval_time += batch_accumulated_eval_time
                batch_train_wallclock = batch_accumulated_train_time
                total_train_time_wallclock += batch_train_wallclock
                all_results.extend(batch_results)
                processed_total += len(batch_architectures)
                if batch_rewards and effective_reward_times:
                    logger.logger.info(f'第 {batch_index} 批评估完成，回传 {len(batch_rewards)} 个奖励')
                    searcher.update_rewards(batch_rewards, default_reward=0.0)
            if processed_total < total_samples:
                logger.logger.warning(f'计划评估 {total_samples} 个架构，但实际完成 {processed_total} 个')
        search_eval_duration = time.perf_counter() - search_eval_start
        logger.logger.info(f'\n' + '=' * 80)
        logger.logger.info('⏱️  时间统计汇总')
        logger.logger.info('=' * 80)
        logger.logger.info(f'总耗时: {search_eval_duration:.2f}s')
        logger.logger.info(f'  ├─ 架构采样时间: {total_search_time:.2f}s ({total_search_time / search_eval_duration * 100:.1f}%)')
        logger.logger.info(f'  ├─ 架构训练时间（墙上时钟）: {total_train_time_wallclock:.2f}s ({total_train_time_wallclock / search_eval_duration * 100:.1f}%) ⭐可与Ad-rl对比')
        logger.logger.info(f'  ├─ 架构训练时间（串行累加）: {total_train_time_accumulated:.2f}s ({total_train_time_accumulated / search_eval_duration * 100:.1f}%) [旧方案保留]')
        logger.logger.info(f'  └─ 架构评估时间: {total_eval_time:.2f}s ({total_eval_time / search_eval_duration * 100:.1f}%)')
        logger.logger.info('\n' + '=' * 80)
        logger.logger.info('实验结果汇总')
        logger.logger.info('=' * 80)
        if all_results:
            all_results.sort(key=lambda x: x['best_accuracy'], reverse=True)
            logger.logger.info(f'共测试了 {len(all_results)} 个架构')
            best_result = all_results[0]
            logger.logger.info(f"\n🏆 最佳架构: {best_result['architecture_name']}")
            logger.logger.info(f"🎯 最佳准确率: {best_result['best_accuracy']:.4f}")
            logger.logger.info(f"⚡ 训练时间: {best_result['total_training_time']:.2f} 秒")
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
            aggregated_timing_total = {'model_migration': 0.0, 'sampling': 0.0, 'data_transfer': 0.0, 'forward': 0.0, 'backward': 0.0, 'optimizer_step': 0.0, 'sync_time': 0.0}
            aggregated_timing_avg = {'model_migration': 0.0, 'sampling': 0.0, 'data_transfer': 0.0, 'forward': 0.0, 'backward': 0.0, 'optimizer_step': 0.0, 'sync_time': 0.0}
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
                logger.logger.info(f"  模型迁移: {aggregated_timing_total['model_migration']:.2f}s ({aggregated_timing_total['model_migration'] / total_timing_cpu * 100:.1f}%)")
                logger.logger.info(f"  数据采样: {aggregated_timing_total['sampling']:.2f}s ({aggregated_timing_total['sampling'] / total_timing_cpu * 100:.1f}%)")
                logger.logger.info(f"  数据传输: {aggregated_timing_total['data_transfer']:.2f}s ({aggregated_timing_total['data_transfer'] / total_timing_cpu * 100:.1f}%)")
                compute_time = aggregated_timing_total['forward'] + aggregated_timing_total['backward']
                logger.logger.info(f'  计算时间: {compute_time:.2f}s ({compute_time / total_timing_cpu * 100:.1f}%)')
                logger.logger.info(f"    ├─ 前向传播: {aggregated_timing_total['forward']:.2f}s")
                logger.logger.info(f"    └─ 反向传播: {aggregated_timing_total['backward']:.2f}s")
                logger.logger.info(f"  参数更新: {aggregated_timing_total['optimizer_step']:.2f}s ({aggregated_timing_total['optimizer_step'] / total_timing_cpu * 100:.1f}%)")
                logger.logger.info(f"  同步开销: {aggregated_timing_total['sync_time']:.2f}s ({aggregated_timing_total['sync_time'] / total_timing_cpu * 100:.1f}%)")
                logger.logger.info(f'      (Ray通信 + 参数聚合，详见调试日志)')
            total_timing_avg = sum(aggregated_timing_avg.values())
            if total_timing_avg > 0:
                logger.logger.info(f'\n⏱️  平均Worker时间 (接近实际墙上时钟耗时):')
                logger.logger.info(f'  总计时: {total_timing_avg:.2f}s')
                logger.logger.info(f"  模型迁移: {aggregated_timing_avg['model_migration']:.2f}s ({aggregated_timing_avg['model_migration'] / total_timing_avg * 100:.1f}%)")
                logger.logger.info(f"  数据采样: {aggregated_timing_avg['sampling']:.2f}s ({aggregated_timing_avg['sampling'] / total_timing_avg * 100:.1f}%)")
                logger.logger.info(f"  数据传输: {aggregated_timing_avg['data_transfer']:.2f}s ({aggregated_timing_avg['data_transfer'] / total_timing_avg * 100:.1f}%)")
                compute_time_avg = aggregated_timing_avg['forward'] + aggregated_timing_avg['backward']
                logger.logger.info(f'  计算时间: {compute_time_avg:.2f}s ({compute_time_avg / total_timing_avg * 100:.1f}%)')
                logger.logger.info(f"    ├─ 前向传播: {aggregated_timing_avg['forward']:.2f}s")
                logger.logger.info(f"    └─ 反向传播: {aggregated_timing_avg['backward']:.2f}s")
                logger.logger.info(f"  参数更新: {aggregated_timing_avg['optimizer_step']:.2f}s ({aggregated_timing_avg['optimizer_step'] / total_timing_avg * 100:.1f}%)")
                logger.logger.info(f"  同步开销: {aggregated_timing_avg['sync_time']:.2f}s ({aggregated_timing_avg['sync_time'] / total_timing_avg * 100:.1f}%)")
                logger.logger.info(f'      (Ray通信 + 参数聚合，详见调试日志)')
            if log_dir:
                import json
                results_file = os.path.join(log_dir, 'experiment_results.json')
                with open(results_file, 'w') as f:
                    json.dump(all_results, f, indent=2, default=str)
                logger.logger.info(f'📄 实验结果已保存到: {results_file}')
        else:
            logger.logger.warning('⚠️ 没有成功的实验结果')
        total_training_duration = time.perf_counter() - training_phase_start
        logger.logger.info('\n' + '=' * 80)
        logger.logger.info('⏱️  最终时间统计')
        logger.logger.info('=' * 80)
        logger.logger.info(f'总耗时（墙上时钟）: {total_training_duration:.2f}s')
        if search_eval_duration > 0:
            search_percent = search_eval_duration / total_training_duration * 100
            logger.logger.info(f'  搜索评估阶段: {search_eval_duration:.2f}s ({search_percent:.1f}%)')
        if total_search_time > 0 or total_train_time_wallclock > 0 or total_eval_time > 0:
            logger.logger.info(f'\n📊 时间分解（基于墙上时钟）:')
            logger.logger.info(f'  ├─ 架构采样时间: {total_search_time:.2f}s ({total_search_time / total_training_duration * 100:.1f}%)')
            logger.logger.info(f'  ├─ 架构训练时间（墙上时钟）: {total_train_time_wallclock:.2f}s ({total_train_time_wallclock / total_training_duration * 100:.1f}%) ⭐可与Ad-rl对比')
            logger.logger.info(f'  └─ 模型评估时间: {total_eval_time:.2f}s ({total_eval_time / total_training_duration * 100:.1f}%)')
            logger.logger.info(f'\n📊 时间分解（基于串行累加，旧方案保留）:')
            logger.logger.info(f'  ├─ 架构采样时间: {total_search_time:.2f}s ({total_search_time / total_training_duration * 100:.1f}%)')
            logger.logger.info(f'  ├─ 架构训练时间（串行累加）: {total_train_time_accumulated:.2f}s ({total_train_time_accumulated / total_training_duration * 100:.1f}%)')
            logger.logger.info(f'  └─ 模型评估时间: {total_eval_time:.2f}s ({total_eval_time / total_training_duration * 100:.1f}%)')
            accounted_time_wallclock = total_search_time + total_train_time_wallclock + total_eval_time
            unaccounted_time_wallclock = total_training_duration - accounted_time_wallclock
            logger.logger.info(f'\n💡 时间验证（墙上时钟）:')
            logger.logger.info(f'  已统计时间: {accounted_time_wallclock:.2f}s')
            logger.logger.info(f'  未统计时间: {unaccounted_time_wallclock:.2f}s ({abs(unaccounted_time_wallclock) / total_training_duration * 100:.1f}%)')
            if abs(unaccounted_time_wallclock) > 1.0:
                logger.logger.info(f'  ⚠️  注意：未统计时间可能包含初始化、数据加载等开销')
            accounted_time_accumulated = total_search_time + total_train_time_accumulated + total_eval_time
            unaccounted_time_accumulated = total_training_duration - accounted_time_accumulated
            logger.logger.info(f'\n💡 时间验证（串行累加）:')
            logger.logger.info(f'  已统计时间: {accounted_time_accumulated:.2f}s')
            logger.logger.info(f'  未统计时间: {unaccounted_time_accumulated:.2f}s ({abs(unaccounted_time_accumulated) / total_training_duration * 100:.1f}%)')
            logger.logger.info(f'\n📌 时间统计说明:')
            logger.logger.info(f'  • 墙上时钟时间：实际物理耗时，与Ad-rl/Ad_OD-rl可公平对比')
            logger.logger.info(f'  • 串行累加时间：所有架构训练时间相加，旧方案统计（仅供参考）')
            logger.logger.info(f'  • Ca-rl采用串行训练，因此墙上时钟≈串行累加（差异来自架构间切换开销）')
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
    logger.logger.info(f'CA-RL 实验完成，数据集为{args.dataset}')
    logger.logger.info(f"使用 {config['ca_backends']['num_workers']} 个GPU")
    logger.logger.info('=' * 80)
    return 0
if __name__ == '__main__':
    exit(main())