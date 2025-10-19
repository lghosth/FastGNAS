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
    full_log_dir = os.path.join(log_dir, f'run_{timestamp}')
    os.makedirs(full_log_dir, exist_ok=True)
    return full_log_dir

def run_single_architecture_experiment(ca_backends: CABackends, arch_spec: Dict[str, Any], arch_name: str, config: Dict[str, Any], logger: GraphNASLogger) -> Dict[str, Any]:
    logger.logger.info(f'=' * 50)
    logger.logger.info(f'开始测试架构: {arch_name}')
    logger.logger.info(f'架构规格: {arch_spec}')
    training_config = config['training']
    epochs_per_round = training_config['epochs_per_round']
    num_rounds = training_config['num_rounds']
    batch_size = training_config.get('batch_size')
    steps_per_epoch = training_config.get('steps_per_epoch')
    results = {'architecture_name': arch_name, 'architecture_spec': arch_spec, 'training_history': [], 'best_accuracy': 0.0, 'best_round': 0, 'total_training_time': 0.0, 'converged': True}
    model_state = None
    best_accuracy = 0.0
    start_time = time.perf_counter()
    for round_num in range(1, num_rounds + 1):
        logger.logger.info(f'\n--- 第 {round_num}/{num_rounds} 轮训练 ---')
        round_start_time = time.perf_counter()
        try:
            model_state, avg_loss, timing_stats = ca_backends.train_epoch(arch_spec=arch_spec, model_state=model_state, epochs=epochs_per_round, batch_size=batch_size, steps_per_epoch=steps_per_epoch)
            evaluation_result = ca_backends.evaluate_model(arch_spec, model_state)
            accuracy = evaluation_result.get('global_accuracy', 0.0)
            evaluation_successful = evaluation_result.get('evaluation_successful', True)
            round_time = time.perf_counter() - round_start_time
            round_result = {'round': round_num, 'loss': avg_loss, 'accuracy': accuracy, 'evaluation': evaluation_result, 'time': round_time}
            results['training_history'].append(round_result)
            if not evaluation_successful:
                logger.logger.warning(f'轮次 {round_num} 评估未成功，使用准确率 0.0; 详情: {evaluation_result}')
            logger.logger.info(f'轮次 {round_num} 完成 - 损失: {avg_loss:.4f}, 准确率: {accuracy:.4f}, 耗时: {round_time:.2f}s')
            if evaluation_successful and hasattr(ca_backends, 'analyze_evaluation_result'):
                detailed_report = ca_backends.analyze_evaluation_result(evaluation_result)
                logger.logger.debug(detailed_report)
            if round_num == 1 or accuracy > best_accuracy:
                best_accuracy = accuracy
                results['best_accuracy'] = best_accuracy
                results['best_round'] = round_num
                logger.logger.info(f'🎉 发现更好结果! 最佳准确率: {best_accuracy:.4f}')
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
            arch_name = sample.get('arch_id') or f'rl_arch_{idx + 1:03d}'
            arch_spec = sample.get('arch_spec', sample)
            sampled_architectures.append({'name': arch_name, 'arch_spec': arch_spec})
            logger.logger.info(f'  -> 采样架构: {arch_name}')
        return (sampled_architectures, searcher)
    static_architectures = arch_config.get('test_architectures', [])
    for idx, arch_entry in enumerate(static_architectures):
        arch_name = arch_entry.get('name', f'static_arch_{idx + 1:03d}')
        arch_spec = {k: v for k, v in arch_entry.items() if k != 'name'}
        sampled_architectures.append({'name': arch_name, 'arch_spec': arch_spec})
    if not sampled_architectures and 'test_arch' in arch_config:
        sampled_architectures.append({'name': 'default_arch', 'arch_spec': arch_config['test_arch']})
    return (sampled_architectures, searcher)

def evaluate_architecture_batch(architectures: List[Dict[str, Any]], ca_backends: CABackends, config: Dict[str, Any], logger: GraphNASLogger) -> Tuple[List[Dict[str, Any]], Dict[str, float]]:
    batch_results: List[Dict[str, Any]] = []
    rewards: Dict[str, float] = {}
    for arch_info in architectures:
        arch_name = arch_info['name']
        arch_spec = arch_info['arch_spec']
        try:
            result = run_single_architecture_experiment(ca_backends=ca_backends, arch_spec=arch_spec, arch_name=arch_name, config=config, logger=logger)
            batch_results.append(result)
            rewards[arch_name] = result['best_accuracy']
        except Exception as e:
            logger.logger.error(f'架构 {arch_name} 测试失败: {e}')
            import traceback
            traceback.print_exc()
            rewards[arch_name] = 0.0
    return (batch_results, rewards)

def main():
    parser = argparse.ArgumentParser(description='数据并行训练基线测试')
    parser.add_argument('--config', type=str, default='baseline/gnas_ca_rl/config.yaml', help='配置文件路径')
    parser.add_argument('--arch', type=str, default=None, help='指定测试的架构名称（在config中定义）')
    parser.add_argument('--dataset', type=str, default=None, help='覆盖配置中的数据集设置')
    parser.add_argument('--workers', type=int, default=None, help='覆盖配置中的worker数量')
    args = parser.parse_args()
    print(f'加载配置文件: {args.config}')
    config = load_config(args.config)
    if args.dataset:
        config['ca_backends']['dataset'] = args.dataset
    if args.workers:
        config['ca_backends']['num_workers'] = args.workers
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
        log_dir = create_log_directory(log_config.get('log_dir', 'logs'))
        print(f'日志目录: {log_dir}')
    else:
        log_dir = None
    logger = GraphNASLogger(name='ca_baseline_experiment', log_dir=log_dir, log_level=log_config.get('level', 'INFO'))
    logger.logger.info('=' * 80)
    logger.logger.info('数据并行训练基线实验开始')
    logger.logger.info('=' * 80)
    ca_config = config['ca_backends']
    logger.logger.info(f"数据集: {ca_config['dataset']}")
    logger.logger.info(f"Worker数量: {ca_config['num_workers']}")
    logger.logger.info(f"设备类型: {ca_config['device']}")
    logger.logger.info(f"学习率: {ca_config['learning_rate']}")
    try:
        logger.logger.info('\n初始化数据并行后端...')
        ca_backends = CABackends(ca_config)
        ca_backends.setup_distributed_training()
        partition_info = ca_backends.get_partition_info()
        logger.logger.info('\n📊 数据分区信息:')
        logger.logger.info(f"分区数量: {partition_info['num_partitions']}")
        logger.logger.info(f"总节点数: {partition_info['total_nodes']}")
        logger.logger.info(f"总边数: {partition_info['total_edges']}")
        for partition in partition_info['partitions']:
            logger.logger.info(f"  分区 {partition['id']}: {partition['nodes']} 节点, 训练={partition['train_nodes']}, 验证={partition['val_nodes']}, 测试={partition['test_nodes']}")
        training_phase_start = time.perf_counter()
        search_eval_start = time.perf_counter()
        search_eval_duration = 0.0
        retrain_duration = 0.0
        sampler_config = config.get('architecture', {}).get('sampler', {})
        total_samples = sampler_config.get('total_samples')
        reward_times = sampler_config.get('reward_times', 0)
        all_results: List[Dict[str, Any]] = []
        if args.arch:
            arch_config = config['architecture']
            candidates = [arch for arch in arch_config.get('test_architectures', []) if arch.get('name') == args.arch]
            if not candidates:
                raise ValueError(f'未找到架构: {args.arch}')
            test_architectures = [{'name': cand['name'], 'arch_spec': {k: v for k, v in cand.items() if k != 'name'}} for cand in candidates]
            batch_results, _ = evaluate_architecture_batch(test_architectures, ca_backends, config, logger)
            all_results.extend(batch_results)
        elif total_samples:
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
                batch_architectures, searcher = sample_architectures_for_experiment(config, logger, searcher=searcher, num_samples=target_batch_size)
                if not batch_architectures:
                    logger.logger.warning('采样批次为空，提前结束评估')
                    break
                if len(batch_architectures) > target_batch_size:
                    batch_architectures = batch_architectures[:target_batch_size]
                logger.logger.info(f'开始评估第 {batch_index} 批架构，共 {len(batch_architectures)} 个')
                batch_results, batch_rewards = evaluate_architecture_batch(batch_architectures, ca_backends, config, logger)
                all_results.extend(batch_results)
                processed_total += len(batch_architectures)
                if batch_rewards and effective_reward_times:
                    logger.logger.info(f'第 {batch_index} 批评估完成，回传 {len(batch_rewards)} 个奖励')
                    searcher.update_rewards(batch_rewards, default_reward=0.0)
            if processed_total < total_samples:
                logger.logger.warning(f'计划评估 {total_samples} 个架构，但实际完成 {processed_total} 个')
        else:
            test_architectures, _ = sample_architectures_for_experiment(config, logger)
            batch_results, _ = evaluate_architecture_batch(test_architectures, ca_backends, config, logger)
            all_results.extend(batch_results)
        search_eval_duration = time.perf_counter() - search_eval_start
        logger.logger.info(f'\n⏱️  搜索评估阶段耗时: {search_eval_duration:.2f}s')
        if total_samples and all_results:
            retrain_start = time.perf_counter()
            logger.logger.info('\n' + '=' * 80)
            logger.logger.info('开始 Top-5 架构重训练阶段')
            logger.logger.info('=' * 80)
            all_results.sort(key=lambda x: x['best_accuracy'], reverse=True)
            top_k = 5
            top_k_architectures = all_results[:min(top_k, len(all_results))]
            logger.logger.info(f'选择Top-{len(top_k_architectures)}架构进行重训练:')
            for i, result in enumerate(top_k_architectures):
                logger.logger.info(f"  {i + 1}. {result['architecture_name']}: {result['best_accuracy']:.4f}")
            retrain_config = config.copy()
            retrain_config['training']['num_rounds'] = 200
            retrain_config['training']['epochs_per_round'] = config['training'].get('epochs_per_round', 5)
            logger.logger.info(f"重训练参数: 轮数={retrain_config['training']['num_rounds']}, 每轮epochs={retrain_config['training']['epochs_per_round']}")
            retrain_results = []
            for i, original_result in enumerate(top_k_architectures):
                arch_name = f"{original_result['architecture_name']}_retrain"
                arch_spec = original_result['architecture_spec']
                logger.logger.info(f'\n--- 重训练架构 {i + 1}/{len(top_k_architectures)}: {arch_name} ---')
                try:
                    retrain_result = run_single_architecture_experiment(ca_backends=ca_backends, arch_spec=arch_spec, arch_name=arch_name, config=retrain_config, logger=logger)
                    retrain_result['original_best_accuracy'] = original_result['best_accuracy']
                    retrain_result['accuracy_improvement'] = retrain_result['best_accuracy'] - original_result['best_accuracy']
                    retrain_result['original_arch_name'] = original_result['architecture_name']
                    retrain_results.append(retrain_result)
                    logger.logger.info(f'🔄 重训练完成: {arch_name}')
                    logger.logger.info(f"   原始准确率: {original_result['best_accuracy']:.4f}")
                    logger.logger.info(f"   重训练准确率: {retrain_result['best_accuracy']:.4f}")
                    logger.logger.info(f"   准确率提升: {retrain_result['accuracy_improvement']:.4f}")
                except Exception as e:
                    logger.logger.error(f'重训练架构 {arch_name} 失败: {e}')
                    import traceback
                    traceback.print_exc()
            all_results.extend(retrain_results)
            logger.logger.info('\n📊 Top-5重训练结果总结:')
            for result in retrain_results:
                improvement = result.get('accuracy_improvement', 0.0)
                status = '↗️ 提升' if improvement > 0 else '➡️ 持平' if improvement == 0 else '↘️ 下降'
                logger.logger.info(f"  {result['architecture_name']}: {result['original_best_accuracy']:.4f} → {result['best_accuracy']:.4f} ({improvement:+.4f}) {status}")
            retrain_duration = time.perf_counter() - retrain_start
            logger.logger.info(f'\n⏱️  重训练阶段耗时: {retrain_duration:.2f}s')
        logger.logger.info('\n' + '=' * 80)
        logger.logger.info('实验结果汇总')
        logger.logger.info('=' * 80)
        if all_results:
            all_results.sort(key=lambda x: x['best_accuracy'], reverse=True)
            original_results = [r for r in all_results if not r['architecture_name'].endswith('_retrain')]
            retrain_results = [r for r in all_results if r['architecture_name'].endswith('_retrain')]
            logger.logger.info(f'共测试了 {len(original_results)} 个架构，重训练了 {len(retrain_results)} 个架构:')
            logger.logger.info(f'\n📋 所有架构结果（按准确率排序）:')
            for i, result in enumerate(all_results):
                arch_type = '🔄' if result['architecture_name'].endswith('_retrain') else '🔍'
                logger.logger.info(f"{i + 1:2d}. {arch_type} {result['architecture_name']}: 准确率 {result['best_accuracy']:.4f} (第 {result['best_round']} 轮, 用时 {result['total_training_time']:.2f}s)")
            best_result = all_results[0]
            is_retrained = best_result['architecture_name'].endswith('_retrain')
            logger.logger.info(f"\n🏆 最佳架构: {best_result['architecture_name']}")
            logger.logger.info(f"🎯 最佳准确率: {best_result['best_accuracy']:.4f}")
            logger.logger.info(f"⚡ 训练时间: {best_result['total_training_time']:.2f} 秒")
            logger.logger.info(f"🔄 收敛情况: {('已收敛' if best_result['converged'] else '未收敛')}")
            if is_retrained:
                logger.logger.info(f"🔄 重训练类型: 来源于 {best_result.get('original_arch_name', 'Unknown')}")
                logger.logger.info(f"📈 准确率提升: {best_result.get('accuracy_improvement', 0.0):.4f}")
            if retrain_results:
                logger.logger.info(f'\n📊 重训练效果统计:')
                improved_count = sum((1 for r in retrain_results if r.get('accuracy_improvement', 0) > 0))
                avg_improvement = sum((r.get('accuracy_improvement', 0) for r in retrain_results)) / len(retrain_results)
                logger.logger.info(f'  提升架构数: {improved_count}/{len(retrain_results)}')
                logger.logger.info(f'  平均提升: {avg_improvement:.4f}')
                best_improvement = max(retrain_results, key=lambda x: x.get('accuracy_improvement', 0))
                logger.logger.info(f"  最大提升: {best_improvement['architecture_name']} ({best_improvement.get('accuracy_improvement', 0):.4f})")
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
        logger.logger.info('⏱️  时间统计')
        logger.logger.info('=' * 80)
        logger.logger.info(f'总耗时(不含分区初始化): {total_training_duration:.2f}s')
        if search_eval_duration > 0:
            search_percent = search_eval_duration / total_training_duration * 100
            logger.logger.info(f'  搜索评估阶段: {search_eval_duration:.2f}s ({search_percent:.1f}%)')
        if retrain_duration > 0:
            retrain_percent = retrain_duration / total_training_duration * 100
            logger.logger.info(f'  重训练阶段: {retrain_duration:.2f}s ({retrain_percent:.1f}%)')
        if search_eval_duration > 0 and retrain_duration > 0:
            other_duration = total_training_duration - search_eval_duration - retrain_duration
            if other_duration > 0:
                other_percent = other_duration / total_training_duration * 100
                logger.logger.info(f'  其他开销: {other_duration:.2f}s ({other_percent:.1f}%)')
    except Exception as e:
        logger.logger.error(f'实验执行失败: {e}')
        import traceback
        traceback.print_exc()
        return 1
    logger.logger.info('\n' + '=' * 80)
    logger.logger.info('数据并行训练基线实验完成')
    logger.logger.info('=' * 80)
    return 0
if __name__ == '__main__':
    exit(main())