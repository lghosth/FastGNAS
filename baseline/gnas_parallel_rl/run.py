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
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(CURRENT_DIR))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)
from baseline.components.searchers.rl_searcher import RLBalancedSearcher
from data.data_utils import get_dataset
from utils.logger import GraphNASLogger

def load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, 'r', encoding='utf-8') as file:
        return yaml.safe_load(file)

def get_backend_class(backend_name: str):
    if backend_name == 'parallel':
        from baseline.components.backends.parallel_backends import ArchParallelBackend
        print('加载backend: ArchParallelBackend')
        return ArchParallelBackend
    elif backend_name == 'parallel_v2':
        from baseline.components.backends.parallel_backends_v2 import ArchParallelBackendV2
        print('加载backend: ArchParallelBackendV2')
        return ArchParallelBackendV2
    else:
        raise ValueError(f"不支持的backend: {backend_name}，仅支持 'parallel' 或 'parallel_v2'")

def set_random_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def create_log_directory(base_dir: str) -> str:
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = os.path.join(base_dir, f'parallel_run_{timestamp}')
    os.makedirs(log_dir, exist_ok=True)
    return log_dir

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
                logger.logger.warning(f'workers[{idx}] = {entry} 无法解析为GPU编号，已忽略')
        if valid_gpu_ids:
            gpu_str = ','.join(map(str, valid_gpu_ids))
            os.environ['CUDA_VISIBLE_DEVICES'] = gpu_str
            logger.logger.info(f'设置CUDA_VISIBLE_DEVICES={gpu_str}，使用指定GPU: {valid_gpu_ids}')
            if 'num_gpus' not in init_kwargs:
                init_kwargs['num_gpus'] = len(valid_gpu_ids)
    logger.logger.info(f"初始化Ray集群, 参数: {(init_kwargs if init_kwargs else '默认配置')}")
    ray.init(**init_kwargs)

def load_full_graph(dataset_name: str, dataset_root: str, logger: GraphNASLogger) -> Dict[str, Any]:
    logger.logger.info(f'加载数据集: {dataset_name} (root={dataset_root})')
    dataset, data = get_dataset(name=dataset_name, root=dataset_root)
    logger.logger.info('数据集统计 => 节点: %d, 边: %d, 特征维度: %d, 类别: %d', data.num_nodes, data.num_edges, data.num_node_features, dataset.num_classes)
    logger.logger.info('mask比例 => train: %d, val: %d, test: %d', int(data.train_mask.sum().item()), int(data.val_mask.sum().item()) if hasattr(data, 'val_mask') else 0, int(data.test_mask.sum().item()) if hasattr(data, 'test_mask') else 0)
    return {'dataset': dataset, 'data': data}

def run_single_architecture_experiment(backend: Any, arch_spec: Dict[str, Any], arch_name: str, config: Dict[str, Any], logger: GraphNASLogger) -> Dict[str, Any]:
    logger.logger.info('=' * 50)
    logger.logger.info(f'开始测试架构: {arch_name}')
    logger.logger.info(f'架构规格: {arch_spec}')
    training_cfg = config['training']
    epochs_per_round = training_cfg['epochs_per_round']
    num_rounds = training_cfg['num_rounds']
    batch_size = training_cfg.get('batch_size')
    steps_per_epoch = training_cfg.get('steps_per_epoch')
    results: Dict[str, Any] = {'architecture_name': arch_name, 'architecture_spec': arch_spec, 'training_history': [], 'best_accuracy': 0.0, 'best_round': 0, 'total_training_time': 0.0, 'converged': True}
    best_accuracy = 0.0
    model_state: Optional[Dict[str, Any]] = None
    start_time = time.time()
    for round_idx in range(1, num_rounds + 1):
        logger.logger.info('\n--- Parallel 第 %d/%d 轮训练 ---', round_idx, num_rounds)
        round_start = time.time()
        try:
            model_state, avg_loss = backend.train_epoch(arch_spec=arch_spec, model_state=model_state, epochs=epochs_per_round, batch_size=batch_size, steps_per_epoch=steps_per_epoch)
            evaluation_result = backend.evaluate_model(arch_spec, model_state)
            accuracy = evaluation_result.get('global_accuracy', 0.0)
            evaluation_successful = evaluation_result.get('evaluation_successful', True)
            round_time = time.time() - round_start
            results['training_history'].append({'round': round_idx, 'loss': avg_loss, 'accuracy': accuracy, 'evaluation': evaluation_result, 'time': round_time})
            if not evaluation_successful:
                logger.logger.warning('Parallel 轮次 %d 评估未成功，准确率按0处理; 详情: %s', round_idx, evaluation_result)
            logger.logger.info('Parallel 轮次 %d 完成 - 损失: %.4f, 准确率: %.4f, 耗时: %.2fs', round_idx, avg_loss, accuracy, round_time)
            if round_idx == 1 or accuracy > best_accuracy:
                best_accuracy = accuracy
                results['best_accuracy'] = best_accuracy
                results['best_round'] = round_idx
                logger.logger.info('🎉 Parallel 发现更好结果! 最佳准确率: %.4f', best_accuracy)
        except Exception as exc:
            logger.logger.error('Parallel 第 %d 轮训练出错: %s', round_idx, exc)
            import traceback
            traceback.print_exc()
            results['converged'] = False
            break
    total_time = time.time() - start_time
    results['total_training_time'] = total_time
    logger.logger.info('\n🏁 Parallel 架构 %s 训练完成!', arch_name)
    logger.logger.info('最佳准确率: %.4f (第 %d 轮)', results['best_accuracy'], results['best_round'])
    logger.logger.info('总训练时间: %.2f 秒', total_time)
    logger.logger.info('是否收敛: %s', '是' if results['converged'] else '否')
    return results

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
            logger.logger.info('初始化 RLBalancedSearcher，计划采样 %d 个架构', total_samples)
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
            logger.logger.info('  -> 采样架构: %s', arch_name)
        return (sampled, searcher)
    static_archs = arch_cfg.get('test_architectures', [])
    for idx, arch_entry in enumerate(static_archs):
        arch_name = arch_entry.get('name', f'static_arch_{idx + 1:03d}')
        arch_spec = {k: v for k, v in arch_entry.items() if k != 'name'}
        sampled.append({'name': arch_name, 'arch_spec': arch_spec})
    if not sampled and 'test_arch' in arch_cfg:
        sampled.append({'name': 'default_arch', 'arch_spec': arch_cfg['test_arch']})
    return (sampled, searcher)

def evaluate_architecture_batch(architectures: List[Dict[str, Any]], backend: Any, config: Dict[str, Any], logger: GraphNASLogger) -> Tuple[List[Dict[str, Any]], Dict[str, float]]:
    batch_results: List[Dict[str, Any]] = []
    rewards: Dict[str, float] = {}
    if not architectures:
        logger.logger.warning('架构批次为空，跳过评估')
        return (batch_results, rewards)
    arch_specs = [arch_info['arch_spec'] for arch_info in architectures]
    arch_names = [arch_info['name'] for arch_info in architectures]
    training_cfg = config['training']
    epochs_per_round = training_cfg['epochs_per_round']
    num_rounds = training_cfg['num_rounds']
    batch_size = training_cfg.get('batch_size')
    steps_per_epoch = training_cfg.get('steps_per_epoch')
    logger.logger.info('开始并行训练 %d 个架构，每个架构 %d 轮，每轮 %d epochs', len(arch_specs), num_rounds, epochs_per_round)
    model_states = [None] * len(arch_specs)
    best_accuracies = [0.0] * len(arch_specs)
    best_rounds = [0] * len(arch_specs)
    training_histories = [[] for _ in range(len(arch_specs))]
    total_training_times = [0.0] * len(arch_specs)
    converged_flags = [True] * len(arch_specs)
    try:
        for round_idx in range(1, num_rounds + 1):
            logger.logger.info('\n--- Parallel Batch 第 %d/%d 轮训练 ---', round_idx, num_rounds)
            round_start = time.time()
            parallel_results = backend.train_architectures_parallel(arch_specs=arch_specs, model_states=model_states, epochs=epochs_per_round, batch_size=batch_size, steps_per_epoch=steps_per_epoch)
            for i, (new_state, avg_loss, train_stats) in enumerate(parallel_results):
                model_states[i] = new_state
                total_training_times[i] += train_stats.get('train_time', 0.0)
            evaluation_results = backend.evaluate_architectures_parallel(arch_specs, model_states)
            round_time = time.time() - round_start
            for i, eval_result in enumerate(evaluation_results):
                _, avg_loss, train_stats = parallel_results[i]
                accuracy = eval_result.get('val_accuracy', 0.0)
                evaluation_successful = eval_result.get('evaluation_successful', True)
                training_histories[i].append({'round': round_idx, 'loss': avg_loss, 'accuracy': accuracy, 'evaluation': eval_result, 'time': train_stats.get('train_time', 0.0)})
                if not evaluation_successful:
                    logger.logger.warning('架构 %s 轮次 %d 评估未成功; 详情: %s', arch_names[i], round_idx, eval_result)
                logger.logger.info('架构 %s 轮次 %d - 损失: %.4f, 准确率: %.4f', arch_names[i], round_idx, avg_loss, accuracy)
                if round_idx == 1 or accuracy > best_accuracies[i]:
                    best_accuracies[i] = accuracy
                    best_rounds[i] = round_idx
                    logger.logger.info('🎉 架构 %s 发现更好结果! 最佳准确率: %.4f', arch_names[i], best_accuracies[i])
            logger.logger.info('批次轮次 %d 完成，耗时: %.2fs', round_idx, round_time)
    except Exception as exc:
        logger.logger.error('并行训练批次出错: %s', exc)
        import traceback
        traceback.print_exc()
        converged_flags = [False] * len(arch_specs)
    for i in range(len(arch_specs)):
        result = {'architecture_name': arch_names[i], 'architecture_spec': arch_specs[i], 'training_history': training_histories[i], 'best_accuracy': best_accuracies[i], 'best_round': best_rounds[i], 'total_training_time': total_training_times[i], 'converged': converged_flags[i]}
        batch_results.append(result)
        rewards[arch_names[i]] = best_accuracies[i]
        logger.logger.info('\n🏁 架构 %s 训练完成! 最佳准确率: %.4f (第 %d 轮), 总训练时间: %.2f 秒', arch_names[i], best_accuracies[i], best_rounds[i], total_training_times[i])
    return (batch_results, rewards)

def main() -> int:
    parser = argparse.ArgumentParser(description='架构并行训练基线测试')
    parser.add_argument('--config', type=str, default='baseline/gnas_parallel_rl/config.yaml', help='配置文件路径')
    parser.add_argument('--arch', type=str, default=None, help='指定测试的架构名称（在config中定义）')
    parser.add_argument('--dataset', type=str, default=None, help='覆盖配置中的数据集设置')
    parser.add_argument('--workers', type=int, default=None, help='覆盖配置中的worker数量 (仅CPU模式生效)')
    parser.add_argument('--dataset-root', type=str, default=None, help='自定义数据集根目录')
    parser.add_argument('--backend', type=str, default=None, choices=['parallel', 'parallel_v2'], help='指定使用的backend: parallel 或 parallel_v2')
    args = parser.parse_args()
    print(f'加载配置文件: {args.config}')
    config = load_config(args.config)
    parallel_cfg = config.get('parallel_backends', {})
    if args.dataset:
        parallel_cfg['dataset'] = args.dataset
    if args.workers:
        parallel_cfg['num_workers'] = args.workers
    if args.dataset_root:
        parallel_cfg['dataset_root'] = args.dataset_root
    if args.backend:
        parallel_cfg['backend'] = args.backend
        print(f'命令行指定backend: {args.backend}')
    config['parallel_backends'] = parallel_cfg
    seed = config.get('experiment', {}).get('random_seed', 42)
    set_random_seed(seed)
    print(f'设置随机种子: {seed}')
    log_cfg = config.get('logging', {})
    if log_cfg.get('save_logs', True):
        log_dir = create_log_directory(log_cfg.get('log_dir', 'logs'))
        print(f'日志目录: {log_dir}')
    else:
        log_dir = None
    logger = GraphNASLogger(name='parallel_baseline_experiment', log_dir=log_dir, log_level=log_cfg.get('level', 'INFO'))
    logger.logger.info('=' * 80)
    logger.logger.info('架构并行训练基线实验开始')
    logger.logger.info('=' * 80)
    ensure_ray_initialized(parallel_cfg, logger)
    dataset_root = parallel_cfg.get('dataset_root', 'dataset')
    dataset_bundle = load_full_graph(parallel_cfg.get('dataset', 'Cora'), dataset_root, logger)
    full_data = dataset_bundle['data']
    parallel_cfg.setdefault('num_classes', dataset_bundle['dataset'].num_classes)
    backend_name = parallel_cfg.get('backend', 'parallel')
    logger.logger.info(f'使用backend: {backend_name}')
    try:
        BackendClass = get_backend_class(backend_name)
        full_config = {**parallel_cfg, 'training': config.get('training', {})}
        backend = BackendClass(full_config)
        backend.setup_distributed_training(full_data)
    except ValueError as e:
        logger.logger.error(str(e))
        return 1
    except Exception as e:
        logger.logger.error(f'Backend初始化失败: {e}')
        import traceback
        traceback.print_exc()
        return 1
    logger.logger.info('后端初始化完成 => Worker数量: %d', backend.get_num_workers())
    training_phase_start = time.perf_counter()
    training_duration: Optional[float] = None
    sampler_cfg = config.get('architecture', {}).get('sampler', {})
    total_samples = sampler_cfg.get('total_samples')
    reward_times = sampler_cfg.get('reward_times', 0)
    all_results: List[Dict[str, Any]] = []
    try:
        if args.arch:
            arch_cfg = config['architecture']
            candidates = [arch for arch in arch_cfg.get('test_architectures', []) if arch.get('name') == args.arch]
            if not candidates:
                raise ValueError(f'未找到架构: {args.arch}')
            test_archs = [{'name': cand['name'], 'arch_spec': {k: v for k, v in cand.items() if k != 'name'}} for cand in candidates]
            batch_results, _ = evaluate_architecture_batch(test_archs, backend, config, logger)
            all_results.extend(batch_results)
        elif total_samples:
            effective_reward_times = reward_times if reward_times and reward_times > 0 else 1
            if reward_times != effective_reward_times:
                logger.logger.warning('reward_times <= 0，默认视为1次更新')
            batch_size_estimate = max(1, math.ceil(total_samples / effective_reward_times))
            logger.logger.info('总计需要评估 %d 个架构，将分 %d 批，每批约 %d 个', total_samples, math.ceil(total_samples / batch_size_estimate), batch_size_estimate)
            processed_total = 0
            batch_index = 0
            searcher: Optional[RLBalancedSearcher] = None
            while processed_total < total_samples:
                batch_index += 1
                remaining = total_samples - processed_total
                target_batch = min(batch_size_estimate, remaining)
                batch_archs, searcher = sample_architectures_for_experiment(config, logger, searcher=searcher, num_samples=target_batch)
                if not batch_archs:
                    logger.logger.warning('采样批次为空，提前结束评估')
                    break
                if len(batch_archs) > target_batch:
                    batch_archs = batch_archs[:target_batch]
                logger.logger.info('开始评估第 %d 批架构，共 %d 个', batch_index, len(batch_archs))
                batch_results, batch_rewards = evaluate_architecture_batch(batch_archs, backend, config, logger)
                all_results.extend(batch_results)
                processed_total += len(batch_archs)
                if batch_rewards and effective_reward_times:
                    logger.logger.info('第 %d 批评估完成，回传 %d 个奖励', batch_index, len(batch_rewards))
                    searcher.update_rewards(batch_rewards, default_reward=0.0)
            if processed_total < total_samples:
                logger.logger.warning('计划评估 %d 个架构，但实际完成 %d 个', total_samples, processed_total)
        else:
            test_archs, _ = sample_architectures_for_experiment(config, logger)
            batch_results, _ = evaluate_architecture_batch(test_archs, backend, config, logger)
            all_results.extend(batch_results)
        logger.logger.info('\n' + '=' * 80)
        logger.logger.info('实验结果汇总')
        logger.logger.info('=' * 80)
        if all_results:
            all_results.sort(key=lambda item: item['best_accuracy'], reverse=True)
            logger.logger.info('共测试了 %d 个架构:', len(all_results))
            for idx, result in enumerate(all_results, start=1):
                logger.logger.info('%d. %s: 最佳准确率 %.4f (第 %d 轮, 用时 %.2fs)', idx, result['architecture_name'], result['best_accuracy'], result['best_round'], result['total_training_time'])
            best_result = all_results[0]
            logger.logger.info('\n🏆 最佳架构: %s', best_result['architecture_name'])
            logger.logger.info('🎯 最佳准确率: %.4f', best_result['best_accuracy'])
            logger.logger.info('⚡ 训练时间: %.2f 秒', best_result['total_training_time'])
            logger.logger.info('🔄 收敛情况: %s', '已收敛' if best_result['converged'] else '未收敛')
            if log_dir:
                import json
                results_file = os.path.join(log_dir, 'experiment_results.json')
                with open(results_file, 'w', encoding='utf-8') as result_file:
                    json.dump(all_results, result_file, indent=2, default=str)
                logger.logger.info('📄 实验结果已保存到: %s', results_file)
        else:
            logger.logger.warning('⚠️ 没有成功的实验结果')
        training_duration = time.perf_counter() - training_phase_start
    except Exception as exc:
        logger.logger.error('实验执行失败: %s', exc)
        import traceback
        traceback.print_exc()
        return 1
    finally:
        backend.cleanup()
        if ray.is_initialized():
            ray.shutdown()
    if training_duration is None:
        training_duration = time.perf_counter() - training_phase_start
    logger.logger.info('🔥 并行训练总耗时(不含初始化): %.2fs', training_duration)
    logger.logger.info('\n' + '=' * 80)
    logger.logger.info('架构并行训练基线实验完成')
    logger.logger.info('=' * 80)
    return 0
if __name__ == '__main__':
    raise SystemExit(main())