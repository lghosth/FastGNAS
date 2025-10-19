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
from baseline.components.backends.co_backends import COBackends
from baseline.components.searchers.rl_searcher import RLBalancedSearcher
from utils.logger import GraphNASLogger

def load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
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
    full_log_dir = os.path.join(log_dir, f'co_run_{timestamp}')
    os.makedirs(full_log_dir, exist_ok=True)
    return full_log_dir

def run_single_architecture_experiment(co_backends: COBackends, arch_spec: Dict[str, Any], arch_name: str, config: Dict[str, Any], logger: GraphNASLogger) -> Dict[str, Any]:
    logger.logger.info(f'=' * 50)
    logger.logger.info(f'开始测试CO架构: {arch_name}')
    logger.logger.info(f'架构规格: {arch_spec}')
    training_config = config['training']
    epochs_per_round = training_config['epochs_per_round']
    num_rounds = training_config['num_rounds']
    batch_size = training_config.get('batch_size')
    steps_per_epoch = training_config.get('steps_per_epoch')
    results = {'architecture_name': arch_name, 'architecture_spec': arch_spec, 'training_history': [], 'best_accuracy': 0.0, 'best_round': 0, 'total_training_time': 0.0, 'converged': True}
    model_state = None
    best_accuracy = 0.0
    start_time = time.time()
    for round_num in range(1, num_rounds + 1):
        logger.logger.info(f'--- CO第 {round_num}/{num_rounds} 轮训练 ---')
        round_start_time = time.time()
        try:
            model_state, avg_loss = co_backends.train_epoch(arch_spec=arch_spec, model_state=model_state, epochs=epochs_per_round, batch_size=batch_size, steps_per_epoch=steps_per_epoch)
            logger.logger.info(f'--- CO第 {round_num}/{num_rounds} 轮训练--完成，开始评估 ---')
            evaluation_result = co_backends.evaluate_model(arch_spec, model_state)
            accuracy = evaluation_result.get('global_accuracy', 0.0)
            evaluation_successful = evaluation_result.get('evaluation_successful', True)
            round_time = time.time() - round_start_time
            round_result = {'round': round_num, 'loss': avg_loss, 'accuracy': accuracy, 'evaluation': evaluation_result, 'time': round_time}
            results['training_history'].append(round_result)
            if not evaluation_successful:
                logger.logger.warning(f'CO轮次 {round_num} 评估未成功，使用准确率 0.0')
            logger.logger.info(f'CO轮次 {round_num} 完成 - 损失: {avg_loss:.4f}, 准确率: {accuracy:.4f}, 耗时: {round_time:.2f}s')
            if round_num == 1 or accuracy > best_accuracy:
                best_accuracy = accuracy
                results['best_accuracy'] = best_accuracy
                results['best_round'] = round_num
                logger.logger.info(f'🎉 CO发现更好结果! 最佳准确率: {best_accuracy:.4f}')
        except Exception as e:
            logger.logger.error(f'CO第 {round_num} 轮训练出错: {e}')
            import traceback
            traceback.print_exc()
            break
    total_time = time.time() - start_time
    results['total_training_time'] = total_time
    logger.logger.info(f'\n🏁 CO架构 {arch_name} 训练完成!')
    logger.logger.info(f"最佳准确率: {results['best_accuracy']:.4f} (第 {results['best_round']} 轮)")
    logger.logger.info(f'总训练时间: {total_time:.2f} 秒')
    return results

def sample_architectures_for_experiment(config: Dict[str, Any], logger: GraphNASLogger, searcher: Optional[RLBalancedSearcher]=None, num_samples: Optional[int]=None) -> Tuple[List[Dict[str, Any]], Optional[RLBalancedSearcher]]:
    arch_config = config.get('architecture', {})
    sampler_config = arch_config.get('sampler', {})
    total_samples = sampler_config.get('total_samples')
    controller_cfg = dict(sampler_config.get('controller', {}))
    balance_cfg = sampler_config.get('balance') if sampler_config.get('use_balancer') else None
    sampled_architectures: List[Dict[str, Any]] = []
    if total_samples:
        if searcher is None:
            seed = config.get('experiment', {}).get('random_seed')
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
            arch_name = sample.get('arch_id') or f'co_rl_arch_{idx + 1:03d}'
            arch_spec = sample.get('arch_spec', sample)
            sampled_architectures.append({'name': arch_name, 'arch_spec': arch_spec})
            logger.logger.info(f'  -> CO采样架构: {arch_name}')
        return (sampled_architectures, searcher)
    static_architectures = arch_config.get('test_architectures', [])
    for idx, arch_entry in enumerate(static_architectures):
        arch_name = arch_entry.get('name', f'co_static_arch_{idx + 1:03d}')
        arch_spec = {k: v for k, v in arch_entry.items() if k != 'name'}
        sampled_architectures.append({'name': arch_name, 'arch_spec': arch_spec})
    if not sampled_architectures and 'test_arch' in arch_config:
        sampled_architectures.append({'name': 'co_default_arch', 'arch_spec': arch_config['test_arch']})
    return (sampled_architectures, searcher)

def evaluate_architecture_batch(architectures: List[Dict[str, Any]], co_backends: COBackends, config: Dict[str, Any], logger: GraphNASLogger) -> Tuple[List[Dict[str, Any]], Dict[str, float]]:
    batch_results: List[Dict[str, Any]] = []
    rewards: Dict[str, float] = {}
    for arch_info in architectures:
        arch_name = arch_info['name']
        arch_spec = arch_info['arch_spec']
        try:
            result = run_single_architecture_experiment(co_backends=co_backends, arch_spec=arch_spec, arch_name=arch_name, config=config, logger=logger)
            batch_results.append(result)
            rewards[arch_name] = result['best_accuracy']
        except Exception as e:
            logger.logger.error(f'CO架构 {arch_name} 测试失败: {e}')
            import traceback
            traceback.print_exc()
            rewards[arch_name] = 0.0
    return (batch_results, rewards)

def main():
    parser = argparse.ArgumentParser(description='CO(Communication-Only)数据并行训练基线测试')
    parser.add_argument('--config', type=str, default='baseline/gnas_co_rl/config.yaml', help='配置文件路径')
    parser.add_argument('--arch', type=str, default=None, help='指定测试的架构名称')
    parser.add_argument('--dataset', type=str, default=None, help='覆盖配置中的数据集设置')
    parser.add_argument('--workers', type=int, default=None, help='覆盖配置中的worker数量')
    args = parser.parse_args()
    print(f'加载CO配置文件: {args.config}')
    config = load_config(args.config)
    if args.dataset:
        config['co_backends']['dataset'] = args.dataset
    if args.workers:
        config['co_backends']['num_workers'] = args.workers
    seed = config.get('experiment', {}).get('random_seed', 42)
    set_random_seed(seed)
    print(f'设置随机种子: {seed}')
    log_config = config.get('logging', {})
    if log_config.get('save_logs', True):
        log_dir = create_log_directory(log_config.get('log_dir', 'logs'))
        print(f'CO日志目录: {log_dir}')
    else:
        log_dir = None
    logger = GraphNASLogger(name='co_baseline_experiment', log_dir=log_dir, log_level=log_config.get('level', 'INFO'))
    logger.logger.info('=' * 80)
    logger.logger.info('CO(Communication-Only)数据并行训练基线实验开始')
    logger.logger.info('=' * 80)
    co_config = config['co_backends']
    logger.logger.info(f"数据集: {co_config['dataset']}")
    logger.logger.info(f"Worker数量: {co_config['num_workers']}")
    logger.logger.info(f"设备类型: {co_config['device']}")
    logger.logger.info(f"学习率: {co_config['learning_rate']}")
    logger.logger.info(f"分区策略: {co_config['partition_strategy']}")
    try:
        logger.logger.info('\n初始化CO数据并行后端...')
        co_backends = COBackends(config)
        co_backends.setup_distributed_training()
        partition_info = co_backends.get_partition_info()
        logger.logger.info('\n📊 CO数据分区信息:')
        logger.logger.info(f"分区数量: {partition_info['num_partitions']}")
        logger.logger.info(f"分区策略: {partition_info['strategy']}")
        logger.logger.info(f"总节点数: {partition_info['total_nodes']}")
        logger.logger.info(f"总边数: {partition_info['total_edges']}")
        for partition in partition_info['partitions']:
            logger.logger.info(f"  分区 {partition['id']}: {partition['nodes']} 节点, 训练={partition['train_nodes']}, 验证={partition['val_nodes']}, 测试={partition['test_nodes']}")
        training_phase_start = time.perf_counter()
        sampler_config = config.get('architecture', {}).get('sampler', {})
        total_samples = sampler_config.get('total_samples')
        reward_times = sampler_config.get('reward_times', 0)
        all_results: List[Dict[str, Any]] = []
        if args.arch:
            logger.logger.info(f'测试config文件中的指定架构')
            arch_config = config['architecture']
            candidates = [arch for arch in arch_config.get('test_architectures', []) if arch.get('name') == args.arch]
            if not candidates:
                raise ValueError(f'未找到架构: {args.arch}')
            test_architectures = [{'name': cand['name'], 'arch_spec': {k: v for k, v in cand.items() if k != 'name'}} for cand in candidates]
            batch_results, _ = evaluate_architecture_batch(test_architectures, co_backends, config, logger)
            all_results.extend(batch_results)
        elif total_samples:
            logger.logger.info(f'使用搜索策略采样架构')
            if reward_times is None or reward_times <= 0:
                logger.logger.warning('reward_times <= 0，默认视为1次更新')
                effective_reward_times = 1
            else:
                effective_reward_times = reward_times
            batch_size = max(1, math.ceil(total_samples / effective_reward_times))
            logger.logger.info(f'总计需要评估 {total_samples} 个CO架构，将分 {math.ceil(total_samples / batch_size)} 批，每批约 {batch_size} 个')
            processed_total = 0
            batch_index = 0
            searcher: Optional[RLBalancedSearcher] = None
            while processed_total < total_samples:
                batch_index += 1
                remaining = total_samples - processed_total
                target_batch_size = min(batch_size, remaining)
                batch_architectures, searcher = sample_architectures_for_experiment(config, logger, searcher=searcher, num_samples=target_batch_size)
                if not batch_architectures:
                    logger.logger.warning('CO采样批次为空，提前结束评估')
                    break
                if len(batch_architectures) > target_batch_size:
                    batch_architectures = batch_architectures[:target_batch_size]
                logger.logger.info(f'开始评估第 {batch_index} 批CO架构，共 {len(batch_architectures)} 个')
                batch_results, batch_rewards = evaluate_architecture_batch(batch_architectures, co_backends, config, logger)
                all_results.extend(batch_results)
                processed_total += len(batch_architectures)
                if batch_rewards and effective_reward_times:
                    logger.logger.info(f'第 {batch_index} 批CO评估完成，回传 {len(batch_rewards)} 个奖励')
                    searcher.update_rewards(batch_rewards, default_reward=0.0)
            if processed_total < total_samples:
                logger.logger.warning(f'计划评估 {total_samples} 个CO架构，但实际完成 {processed_total} 个')
        else:
            test_architectures, _ = sample_architectures_for_experiment(config, logger)
            batch_results, _ = evaluate_architecture_batch(test_architectures, co_backends, config, logger)
            all_results.extend(batch_results)
        logger.logger.info('\n' + '=' * 80)
        logger.logger.info('CO实验结果汇总')
        logger.logger.info('=' * 80)
        if all_results:
            all_results.sort(key=lambda x: x['best_accuracy'], reverse=True)
            logger.logger.info(f'共测试了 {len(all_results)} 个CO架构:')
            for i, result in enumerate(all_results):
                logger.logger.info(f"{i + 1}. {result['architecture_name']}: 最佳准确率 {result['best_accuracy']:.4f} (第 {result['best_round']} 轮, 用时 {result['total_training_time']:.2f}s)")
            best_result = all_results[0]
            logger.logger.info(f"\n🏆 最佳CO架构: {best_result['architecture_name']}")
            logger.logger.info(f"🎯 最佳准确率: {best_result['best_accuracy']:.4f}")
            logger.logger.info(f"⚡ 训练时间: {best_result['total_training_time']:.2f} 秒")
            if log_dir:
                import json
                results_file = os.path.join(log_dir, 'co_experiment_results.json')
                with open(results_file, 'w') as f:
                    json.dump(all_results, f, indent=2, default=str)
                logger.logger.info(f'📄 CO实验结果已保存到: {results_file}')
        else:
            logger.logger.warning('⚠️ 没有成功的CO实验结果')
        total_training_duration = time.perf_counter() - training_phase_start
        logger.logger.info(f'🔥 CO训练总耗时(不含分区初始化): {total_training_duration:.2f}s')
    except Exception as e:
        logger.logger.error(f'CO实验执行失败: {e}')
        import traceback
        traceback.print_exc()
        return 1
    logger.logger.info('\n' + '=' * 80)
    logger.logger.info('CO(Communication-Only)数据并行训练基线实验完成')
    logger.logger.info('=' * 80)
    return 0
if __name__ == '__main__':
    exit(main())