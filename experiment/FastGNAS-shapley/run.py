import os
import sys
import yaml
import torch
import ray
import argparse
import numpy as np
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
import time
import copy
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(project_root)
from core.dispatcher_v2 import Dispatcher
from trainer.worker import TrainerWorker
from data.partition import PartitionGraph
from utils.logger import GraphNASLogger
from core.load_estimator import MloadPredictor, DataCollectorActor
from core.balance_controller import BalanceController
from utils.utils import load_dataset_config, format_architecture_for_display
from baseline.components.searchers.shapley_searcher import ShapleySearcher
from utils.arch_converter import convert_architecture_to_dgnap

def load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config

def merge_dataset_config(config: Dict[str, Any], dataset_name: str) -> Dict[str, Any]:
    dataset_cfg = load_dataset_config(dataset_name)
    training_cfg = config.get('training', {})
    training_cfg['dataset'] = dataset_cfg.get('name', dataset_name)
    if 'train_batch_size' in dataset_cfg:
        training_cfg['batch_size'] = dataset_cfg['train_batch_size']
    if 'eval_batch_size' in dataset_cfg:
        training_cfg['evaluation'] = training_cfg.get('evaluation', {})
        training_cfg['evaluation']['batch_size'] = dataset_cfg['eval_batch_size']
    if 'fan_out' in dataset_cfg:
        training_cfg['fan_out'] = dataset_cfg['fan_out']
    if 'steps_per_epoch' in dataset_cfg:
        training_cfg['steps_per_epoch'] = dataset_cfg['steps_per_epoch']
    if 'epochs_per_round' in dataset_cfg:
        training_cfg['epochs_per_round'] = dataset_cfg['epochs_per_round']
    if 'num_rounds' in dataset_cfg:
        training_cfg['num_rounds'] = dataset_cfg['num_rounds']
    if 'partition_strategy' in dataset_cfg:
        training_cfg['train_partition_strategy'] = dataset_cfg['partition_strategy']
    if 'num_classes' in dataset_cfg:
        training_cfg['num_classes'] = dataset_cfg['num_classes']
    fastgnas_cfg = config.get('fastgnas_backends', {})
    fastgnas_cfg['dataset'] = dataset_cfg['name']
    nas_cfg = config.get('nas', {})
    if 'gnn_layers' in dataset_cfg:
        nas_cfg['gnn_layers'] = dataset_cfg['gnn_layers']
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

def predict_architecture_load(arch_spec: Dict[str, Any], load_predictor: Optional[Any], graph_info: Dict[str, Any], device_info: Dict[str, Any]=None) -> float:
    if load_predictor is None:
        return 1.0
    try:
        architecture = convert_dgnap_to_load_predictor_format(arch_spec)
        sample = {'architecture': architecture, 'graph': graph_info, 'device': device_info or {'gpu_cores': 10752, 'core_frequency': 2.1, 'fp_capability': 8.6}}
        predicted_load = load_predictor.predict(sample)
        return predicted_load
    except Exception as e:
        print(f'负载预测失败: {e}，使用默认值 1.0')
        return 1.0

def convert_dgnap_to_load_predictor_format(arch_spec: Dict[str, Any]) -> List[Dict[str, Any]]:
    layers = []
    if 'gnnConv1' in arch_spec and 'activation1' in arch_spec:
        layer1 = {'activation': arch_spec.get('activation1', 'relu').lower(), 'attention_type': 'gat' if 'gat' in arch_spec.get('gnnConv1', '').lower() else 'gcn', 'aggregation_type': 'mean', 'attention_heads': 1, 'hidden_units': 64}
        layers.append(layer1)
    if 'gnnConv2' in arch_spec and 'activation2' in arch_spec:
        layer2 = {'activation': arch_spec.get('activation2', 'relu').lower(), 'attention_type': 'gat' if 'gat' in arch_spec.get('gnnConv2', '').lower() else 'gcn', 'aggregation_type': 'mean', 'attention_heads': 1, 'hidden_units': 64}
        layers.append(layer2)
    while len(layers) < 2:
        layers.append({'activation': 'relu', 'attention_type': 'gcn', 'aggregation_type': 'mean', 'attention_heads': 1, 'hidden_units': 64})
    return layers

def evaluate_balance(architectures: List[Dict], num_workers: int, tolerance: float) -> Tuple[bool, float]:
    if not architectures:
        return (True, 0.0)
    loads = [arch['load'] for arch in architectures]
    total_load = sum(loads)
    avg_load_per_worker = total_load / num_workers
    max_deviation = 0.0
    for load in loads:
        deviation = abs(load - avg_load_per_worker)
        max_deviation = max(max_deviation, deviation)
    is_balanced = max_deviation <= avg_load_per_worker * tolerance
    load_diff = max_deviation
    return (is_balanced, load_diff)

def generate_balanced_shapley_architectures(searcher: Any, load_predictor: Any, batch_size: int, min_architectures: int, tolerance: float, num_workers: int, graph_info: Dict[str, Any], device_info: Dict[str, Any], logger: Any) -> List[Dict[str, Any]]:
    logger.logger.info(f'\n=== 开始Shapley负载均衡架构生成 ===')
    logger.logger.info(f'批量大小: {batch_size}, 最小架构数: {min_architectures}')
    logger.logger.info(f'容忍度: {tolerance}, 工作节点: {num_workers}')
    logger.logger.info(f'步骤1：使用Shapley搜索器生成 {batch_size} 个架构...')
    start_time = time.perf_counter()
    try:
        raw_architectures = searcher.sample_architectures(batch_size, stage='initial')
        logger.logger.info(f'成功生成 {len(raw_architectures)} 个架构')
    except Exception as e:
        logger.logger.error(f'Shapley架构生成失败: {e}')
        raise ValueError(f'Shapley架构生成失败: {e}')
    generation_time = time.perf_counter() - start_time
    logger.logger.info(f'架构生成耗时: {generation_time:.4f}s')
    logger.logger.info(f'步骤2：预测 {len(raw_architectures)} 个架构的负载...')
    start_time = time.perf_counter()
    architectures_with_load = []
    for i, arch in enumerate(raw_architectures):
        try:
            arch_spec = arch['arch_spec']
            layers = convert_dgnap_to_load_predictor_format(arch_spec)
            load_input = {'architecture': layers, 'graph': graph_info, 'device': device_info}
            if load_predictor:
                predicted_load = load_predictor.predict(load_input)
            else:
                import random
                predicted_load = random.uniform(0.1, 2.0)
            architectures_with_load.append({'arch_id': arch.get('arch_id', f'shap_arch_{i:05d}'), 'arch_spec': arch_spec, 'load': predicted_load})
            logger.logger.debug(f"  架构 {architectures_with_load[-1]['arch_id']}: 负载 = {predicted_load:.4f}")
        except Exception as e:
            logger.logger.warning(f'架构 {i} 负载预测失败: {e}')
            default_load = 1.0
            architectures_with_load.append({'arch_id': arch.get('arch_id', f'shap_arch_{i:05d}'), 'arch_spec': arch['arch_spec'], 'load': default_load})
    prediction_time = time.perf_counter() - start_time
    logger.logger.info(f'负载预测耗时: {prediction_time:.4f}s')
    logger.logger.info(f'步骤3：寻找负载均衡的最优架构组合...')
    start_time = time.perf_counter()
    best_combination = _find_optimal_shapley_assignment(architectures_with_load, min_architectures, num_workers, tolerance, logger)
    assignment_time = time.perf_counter() - start_time
    logger.logger.info(f'负载均衡分配耗时: {assignment_time:.4f}s')
    final_is_balanced, final_load_diff = evaluate_balance(best_combination, num_workers, tolerance)
    logger.logger.info(f'\n=== Shapley负载均衡架构生成完成 ===')
    logger.logger.info(f'最终架构数量: {len(best_combination)}')
    logger.logger.info(f"负载均衡状态: {('✅ 均衡' if final_is_balanced else '❌ 不均衡')}")
    logger.logger.info(f'负载差异: {final_load_diff:.4f}')
    logger.logger.info(f'容忍度: {tolerance:.1%}')
    loads = [arch['load'] for arch in best_combination]
    total_load = sum(loads)
    avg_load = total_load / len(loads) if loads else 0
    max_load = max(loads) if loads else 0
    min_load = min(loads) if loads else 0
    logger.logger.info(f'负载统计:')
    logger.logger.info(f'  - 总负载: {total_load:.4f}')
    logger.logger.info(f'  - 平均负载: {avg_load:.4f}')
    logger.logger.info(f'  - 最大负载: {max_load:.4f}')
    logger.logger.info(f'  - 最小负载: {min_load:.4f}')
    logger.logger.info(f'各架构负载详情:')
    for i, arch in enumerate(best_combination):
        logger.logger.info(f"  架构{i + 1} ({arch['arch_id']}): 负载 = {arch['load']:.4f}")
    return best_combination

def _find_optimal_shapley_assignment(architectures: List[Dict], min_architectures: int, num_workers: int, tolerance: float, logger: Any) -> List[Dict]:
    best_combination = None
    best_load_diff = float('inf')
    logger.logger.info(f'开始顺序评估 {len(architectures)} 个架构...')
    logger.logger.info(f'要求最少 {min_architectures} 个架构')
    if len(architectures) < min_architectures:
        logger.logger.warning(f'可用架构数量 ({len(architectures)}) 少于最小要求 ({min_architectures})')
        logger.logger.info('返回所有可用架构')
        return architectures
    for num_arch in range(min_architectures, len(architectures) + 1):
        current_combination = architectures[:num_arch]
        is_balanced, load_diff = evaluate_balance(current_combination, num_workers, tolerance)
        logger.logger.debug(f'评估前 {num_arch} 个架构:')
        logger.logger.debug(f'  - 负载差异: {load_diff:.4f}')
        logger.logger.debug(f'  - 是否均衡: {is_balanced}')
        if is_balanced:
            logger.logger.info(f'  ✅ 找到满足容忍度的分配：使用前 {num_arch} 个架构')
            logger.logger.info(f'  - 负载差异: {load_diff:.4f}')
            logger.logger.info(f'  - 容忍度: {tolerance:.1%}')
            return current_combination
        if load_diff < best_load_diff:
            best_load_diff = load_diff
            best_combination = current_combination.copy()
            logger.logger.debug(f'    📊 更新最佳组合：负载差异 {load_diff:.4f}')
    if best_combination is None:
        logger.logger.warning('未找到任何组合，返回最小数量架构')
        best_combination = architectures[:min_architectures]
    else:
        logger.logger.warning(f'未找到满足均衡条件的组合，返回负载差异最小的组合')
        logger.logger.info(f'  - 最佳组合: {len(best_combination)} 个架构')
        logger.logger.info(f'  - 负载差异: {best_load_diff:.4f}')
        logger.logger.info(f'  - 容忍度: {tolerance:.1%}')
    logger.logger.info(f'最终选择 {len(best_combination)} 个架构进行负载均衡训练')
    return best_combination

def run_single_architecture_experiment(arch_spec: Dict[str, Any], arch_id: str, dispatcher: Dispatcher, training_config: Dict[str, Any], logger: GraphNASLogger, load_predictor: Optional[Any]=None, graph_info: Optional[Dict[str, Any]]=None, stage_name: str='测试') -> Dict[str, Any]:
    logger.logger.info(f'开始测试架构: {arch_id}')
    layers = convert_dgnap_to_load_predictor_format(arch_spec)
    layer_wise_spec = {'layers': layers}
    logger.logger.info(f'原始DGNAP格式: {arch_spec}')
    logger.logger.info(f'转换后Layer-wise格式: {layer_wise_spec}')
    epochs_per_round = training_config.get('epochs_per_round', 5)
    num_rounds = training_config.get('num_rounds', 2)
    batch_size = training_config.get('batch_size', 1024)
    steps_per_epoch = training_config.get('steps_per_epoch', 5)
    best_accuracy = 0.0
    best_round = 0
    training_history = []
    total_training_time = 0.0
    aggregated_timing = {'model_migration': 0.0, 'sampling': 0.0, 'data_transfer': 0.0, 'forward': 0.0, 'backward': 0.0, 'optimizer_step': 0.0, 'sync_time': 0.0, 'sync_transfer_time': 0.0, 'sync_compute_time': 0.0}
    for round_idx in range(num_rounds):
        logger.logger.info(f'\n--- 第 {round_idx + 1}/{num_rounds} 轮训练 ---')
        round_start_time = time.perf_counter()
        predicted_load = predict_architecture_load(arch_spec=arch_spec, load_predictor=load_predictor, graph_info=graph_info or {})
        logger.logger.info(f'架构 {arch_id} 预测负载: {predicted_load:.4f}')
        submit_result = ray.get(dispatcher.submit_architectures.remote(architectures=[{'arch_id': f'{arch_id}_round_{round_idx + 1}', 'arch_spec': layer_wise_spec, 'load': predicted_load}], strategy='subtask'))
        train_out = ray.get(dispatcher.run_training.remote(num_rounds=1, epochs_per_round=epochs_per_round, enable_global_evaluation=True))
        results = ray.get(dispatcher.get_results.remote(format='top_k'))
        round_duration = time.perf_counter() - round_start_time
        total_training_time += round_duration
        if results and 'top_models' in results and (len(results['top_models']) > 0):
            model_result = results['top_models'][0]
            accuracy = model_result.get('best_accuracy', 0.0)
            loss = 0.0
            converged = True
            timing_stats = {}
        else:
            accuracy = 0.0
            loss = float('inf')
            converged = False
            timing_stats = {}
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_round = round_idx + 1
            logger.logger.info(f'发现更好结果! 最佳准确率: {best_accuracy:.4f}')
        training_history.append({'round': round_idx + 1, 'accuracy': accuracy, 'loss': loss, 'converged': converged, 'duration': round_duration, 'accumulated_time': total_training_time})
        logger.logger.info(f'轮次 {round_idx + 1} 完成 - 损失: {loss:.4f}, 准确率: {accuracy:.4f}, 训练耗时: {round_duration:.2f}s, 累计: {total_training_time:.2f}s')
        for key in aggregated_timing:
            if key in timing_stats:
                aggregated_timing[key] += timing_stats[key]
    return {'architecture_name': arch_id, 'best_accuracy': best_accuracy, 'best_round': best_round, 'total_training_time': total_training_time, 'converged': converged, 'training_history': training_history, 'timing_stats': aggregated_timing, 'architecture_spec': arch_spec, 'layer_wise_spec': layer_wise_spec}

def evaluate_architecture_batch(arch_specs: List[Dict[str, Any]], dispatcher: Dispatcher, training_config: Dict[str, Any], logger: GraphNASLogger, load_predictor: Optional[Any]=None, graph_info: Optional[Dict[str, Any]]=None, start_arch_index: int=1, stage_name: str='测试') -> Tuple[List[Dict[str, Any]], List[float]]:
    results = []
    rewards = []
    logger.logger.info(f"\n{'=' * 50}")
    logger.logger.info(f'开始批量评估 {stage_name} 阶段的 {len(arch_specs)} 个架构')
    logger.logger.info(f'使用 FastGNAS 负载均衡策略：批量提交+负载选择')
    logger.logger.info(f"{'=' * 50}")
    epochs_per_round = training_config.get('epochs_per_round', 5)
    num_rounds = training_config.get('num_rounds', 2)
    architectures_with_load = []
    arch_id_list = []
    for i, arch_spec in enumerate(arch_specs):
        arch_id = f'shap_arch_{start_arch_index + i:05d}'
        arch_id_list.append(arch_id)
        try:
            layers = convert_dgnap_to_load_predictor_format(arch_spec)
            layer_wise_spec = {'layers': layers}
            predicted_load = predict_architecture_load(arch_spec=arch_spec, load_predictor=load_predictor, graph_info=graph_info or {})
            architectures_with_load.append({'arch_id': arch_id, 'arch_spec': layer_wise_spec, 'load': predicted_load, 'original_dgnap': arch_spec})
            logger.logger.info(f'架构 {arch_id}: 格式转换成功, 预测负载 = {predicted_load:.4f}')
        except Exception as e:
            logger.logger.error(f'架构 {arch_id} 准备失败: {e}')
            results.append({'architecture_name': arch_id, 'best_accuracy': 0.0, 'best_round': 0, 'total_training_time': 0.0, 'converged': False, 'training_history': [], 'timing_stats': {}, 'architecture_spec': arch_spec, 'layer_wise_spec': {}, 'error': str(e)})
            rewards.append(0.0)
    if not architectures_with_load:
        logger.logger.warning('没有有效的架构可以训练')
        return (results, rewards, arch_id_list)
    logger.logger.info(f'\n=== 批量提交 {len(architectures_with_load)} 个架构到 Dispatcher ===')
    try:
        submit_result = ray.get(dispatcher.submit_architectures.remote(architectures=architectures_with_load, strategy='subtask'))
        logger.logger.info(f'架构提交成功')
    except Exception as e:
        logger.logger.error(f'架构提交失败: {e}')
        import traceback
        logger.logger.error(f'详细错误:\n{traceback.format_exc()}')
        for arch in architectures_with_load:
            if arch['arch_id'] not in [r['architecture_name'] for r in results]:
                results.append({'architecture_name': arch['arch_id'], 'best_accuracy': 0.0, 'best_round': 0, 'total_training_time': 0.0, 'converged': False, 'training_history': [], 'timing_stats': {}, 'architecture_spec': arch['original_dgnap'], 'layer_wise_spec': arch['arch_spec'], 'error': str(e)})
                rewards.append(0.0)
        return (results, rewards, arch_id_list)
    logger.logger.info(f'\n=== 开始训练（负载均衡模式，共 {num_rounds} 轮）===')
    training_start = time.perf_counter()
    try:
        train_out = ray.get(dispatcher.run_training.remote(num_rounds=num_rounds, epochs_per_round=epochs_per_round, enable_global_evaluation=True))
        training_duration = time.perf_counter() - training_start
        logger.logger.info(f'训练完成，总耗时: {training_duration:.2f}s')
    except Exception as e:
        logger.logger.error(f'训练失败: {e}')
        import traceback
        logger.logger.error(f'详细错误:\n{traceback.format_exc()}')
        for arch in architectures_with_load:
            if arch['arch_id'] not in [r['architecture_name'] for r in results]:
                results.append({'architecture_name': arch['arch_id'], 'best_accuracy': 0.0, 'best_round': 0, 'total_training_time': 0.0, 'converged': False, 'training_history': [], 'timing_stats': {}, 'architecture_spec': arch['original_dgnap'], 'layer_wise_spec': arch['arch_spec'], 'error': str(e)})
                rewards.append(0.0)
        return (results, rewards, arch_id_list)
    logger.logger.info(f'\n=== 获取评估结果 ===')
    final_accuracies = train_out.get('final_accuracies', {})
    timing_stats = train_out.get('timing_stats', {})
    statistics = train_out.get('statistics', {})
    backend_train_time = 0.0
    if isinstance(train_out, dict) and 'statistics' in train_out:
        backend_train_time = float(statistics.get('total_time', 0.0))
    if backend_train_time == 0.0:
        backend_train_time = training_duration
    eval_time = max(0, training_duration - backend_train_time)
    cpu_accumulated_time = sum(timing_stats.values()) if timing_stats else 0.0
    logger.logger.debug(f'训练时间统计: training_duration={training_duration:.2f}s, backend_train_time={backend_train_time:.2f}s, eval_time={eval_time:.2f}s, cpu_accumulated_time={cpu_accumulated_time:.2f}s')
    logger.logger.debug(f'timing_stats: {timing_stats}')
    if not final_accuracies:
        logger.logger.warning('训练返回的 final_accuracies 为空，尝试从 Registry 查询...')
        try:
            all_results_dict = ray.get(dispatcher.get_results.remote(format='detailed'))
            registry_entries = all_results_dict.get('all_entries', {})
            final_accuracies = {arch_id: entry.get('best_accuracy', 0.0) for arch_id, entry in registry_entries.items()}
            logger.logger.info(f'从 Registry 获取了 {len(final_accuracies)} 个架构的准确率')
        except Exception as e:
            logger.logger.error(f'从 Registry 获取结果失败: {e}')
            final_accuracies = {}
    try:
        for arch in architectures_with_load:
            arch_id = arch['arch_id']
            best_accuracy = 0.0
            if arch_id in final_accuracies:
                best_accuracy = final_accuracies[arch_id]
            else:
                for key, accuracy in final_accuracies.items():
                    if key.startswith(arch_id):
                        best_accuracy = max(best_accuracy, accuracy)
            logger.logger.info(f'架构 {arch_id}: 最佳准确率 = {best_accuracy:.4f}')
            training_history = [{'round': num_rounds, 'accuracy': best_accuracy, 'accumulated_time': backend_train_time, 'duration': backend_train_time}]
            result_timing_stats = dict(timing_stats) if timing_stats else {}
            if 'sync_overhead' in statistics:
                result_timing_stats['sync_overhead'] = statistics['sync_overhead']
            results.append({'architecture_name': arch_id, 'best_accuracy': best_accuracy, 'best_round': num_rounds, 'total_training_time': backend_train_time, 'converged': best_accuracy > 0, 'training_history': training_history, 'timing_stats': result_timing_stats, 'architecture_spec': arch['original_dgnap'], 'layer_wise_spec': arch['arch_spec']})
            rewards.append(best_accuracy)
    except Exception as e:
        logger.logger.error(f'结果处理失败: {e}')
        import traceback
        logger.logger.error(f'详细错误:\n{traceback.format_exc()}')
        for arch in architectures_with_load:
            if arch['arch_id'] not in [r['architecture_name'] for r in results]:
                results.append({'architecture_name': arch['arch_id'], 'best_accuracy': 0.0, 'best_round': 0, 'total_training_time': 0.0, 'converged': False, 'training_history': [], 'timing_stats': {}, 'architecture_spec': arch['original_dgnap'], 'layer_wise_spec': arch['arch_spec'], 'error': str(e)})
                rewards.append(0.0)
    return (results, rewards, arch_id_list)

def main():
    parser = argparse.ArgumentParser(description='FastGNAS-Shapley 实验')
    parser.add_argument('--config', type=str, default='experiment/FastGNAS-shapley/config.yaml', help='配置文件路径')
    parser.add_argument('--dataset', type=str, help='数据集名称（从 dataset.yaml 读取配置）', required=True)
    parser.add_argument('--workers', type=int, default=None, help='覆盖配置中的worker数量')
    parser.add_argument('--rounds', type=int, default=None, help='覆盖NAS搜索轮次(num_search_rounds)')
    args = parser.parse_args()
    print(f'加载配置文件: {args.config}')
    config = load_config(args.config)
    if args.dataset:
        print(f'加载数据集配置: {args.dataset}')
        config = merge_dataset_config(config, args.dataset)
    else:
        raise ValueError('未指定数据集名称')
    if args.workers:
        config['fastgnas_backends']['num_workers'] = args.workers
    if args.rounds is not None:
        if 'nas' not in config:
            config['nas'] = {}
        config['nas']['num_search_rounds'] = int(args.rounds)
        print(f'应用--rounds参数: 搜索轮次={args.rounds}')
    fastgnas_config = config.get('fastgnas_backends', {})
    nas_config = config.get('nas', {})
    training_config = config.get('training', {})
    if fastgnas_config.get('device', 'cuda') == 'cuda':
        worker_list = fastgnas_config.get('workers', [])
        if worker_list:
            visible_devices = ','.join(map(str, worker_list))
            os.environ['CUDA_VISIBLE_DEVICES'] = visible_devices
            print(f'设置 CUDA_VISIBLE_DEVICES={visible_devices}')
        else:
            os.environ.pop('CUDA_VISIBLE_DEVICES', None)
            print('未指定 workers，使用系统默认可见GPU')
    exp_cfg = fastgnas_config.get('experiment', {})
    use_random_seed = bool(exp_cfg.get('use_random_seed', False))
    if use_random_seed:
        seed = exp_cfg.get('random_seed', 42)
        set_random_seed(seed)
        print(f'设置随机种子: {seed}')
    else:
        print('未启用随机种子(use_random_seed=false)')
    log_config = fastgnas_config.get('logging', {})
    if log_config.get('save_logs', True):
        log_dir = create_log_directory(log_config.get('log_dir', 'experiment/FastGNAS-shapley/log'))
        print(f'日志目录: {log_dir}')
    else:
        log_dir = None
    logger = GraphNASLogger(name=f'fastgnas_shapley_experiment_{args.dataset}', log_dir=log_dir, log_level=log_config.get('level', 'INFO'))
    logger.logger.info('=' * 80)
    logger.logger.info(f'FastGNAS-Shapley 实验开始, 数据集: {args.dataset}')
    logger.logger.info('=' * 80)
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)
        logger.logger.info('Ray 初始化完成')
    try:
        logger.logger.info('\n步骤 1: 数据分区...')
        partition_start = time.perf_counter()
        dataset_name = args.dataset
        num_workers = fastgnas_config.get('num_workers', 8)
        max_hops = nas_config.get('gnn_layers', 2)
        partition_strategy = fastgnas_config.get('partition_strategy', 'balanced_stratified_khop')
        logger.logger.info(f'加载数据集: {dataset_name} | 分区数: {num_workers} | 层数: {max_hops}')
        logger.logger.info(f'使用分区策略: {partition_strategy}')
        partitions = PartitionGraph.get_partitions(dataset_name=dataset_name, num_partitions=num_workers, max_hops=max_hops, strategy=partition_strategy)
        for i, p in enumerate(partitions):
            logger.logger.info(f'分区 {i}: Train={p.train_mask.sum().item()}, Test={p.test_mask.sum().item()}, Val={p.val_mask.sum().item()}')
        partition_duration = time.perf_counter() - partition_start
        logger.logger.info(f'数据分区完成，耗时: {partition_duration:.2f}s')
        logger.logger.info('\n步骤 2: 创建 Workers...')
        worker_setup_start = time.perf_counter()
        workers = []
        for i in range(num_workers):
            use_gpu = torch.cuda.is_available()
            worker_actor = TrainerWorker.options(num_gpus=1 if use_gpu else 0, num_cpus=4).remote(worker_id=i, gpu_id=i if use_gpu else -1, data_partition=partitions[i] if i < len(partitions) else partitions[0], config=training_config, collector_ref=None)
            workers.append(worker_actor)
        logger.logger.info(f'-> {len(workers)} 个Worker创建成功')
        worker_setup_duration = time.perf_counter() - worker_setup_start
        logger.logger.info(f'Workers 创建完成，耗时: {worker_setup_duration:.2f}s')
        logger.logger.info('\n步骤 3: 创建负载预测器...')
        balance_setup_start = time.perf_counter()
        enable_balancing = nas_config.get('balancing_strategy', {}).get('enable', True)
        load_predictor = None
        if enable_balancing:
            model_file_path = nas_config.get('balancing_strategy', {}).get('model_path', 'mload_checkpoint.pth')
            if os.path.exists(model_file_path):
                load_predictor = MloadPredictor.load_from_file(model_file_path)
                logger.logger.info(f'负载预测器加载成功: {model_file_path}')
            else:
                logger.logger.warning(f'负载预测器文件不存在: {model_file_path}')
                logger.logger.warning('将继续运行，但不使用负载预测功能')
        else:
            logger.logger.info('负载均衡已禁用，跳过负载预测器创建')
        balance_setup_duration = time.perf_counter() - balance_setup_start
        logger.logger.info(f'负载预测器创建完成，耗时: {balance_setup_duration:.2f}s')
        logger.logger.info('\n步骤 4: 创建 Shapley 搜索器...')
        searcher_setup_start = time.perf_counter()
        shapley_config = nas_config.get('shapley', {})
        searcher = ShapleySearcher(config=shapley_config, search_space=None)
        searcher_setup_duration = time.perf_counter() - searcher_setup_start
        logger.logger.info(f'Shapley 搜索器创建完成，耗时: {searcher_setup_duration:.2f}s')
        logger.logger.info('\n步骤 5: 创建 Dispatcher...')
        dispatcher_setup_start = time.perf_counter()
        dispatcher_scheduler_type = nas_config.get('balancing_strategy', {}).get('rotation_mode', 'subtask')
        logger.logger.info(f'启用的轮转模式: {dispatcher_scheduler_type}')
        dispatcher = Dispatcher.remote(worker_refs=workers, train_config=training_config, scheduler_type=dispatcher_scheduler_type, balance_controller=None, load_predictor=load_predictor, enable_balancing=enable_balancing)
        logger.logger.info('\n步骤 6: 预热 Dispatcher...')
        warmup_start = time.perf_counter()
        warmup_ret = ray.get(dispatcher.warmup_system.remote(batch_size=4, min_architectures=num_workers))
        if warmup_ret['ok']:
            logger.logger.info('Dispatcher预热成功')
        else:
            logger.logger.error(f"Dispatcher预热失败: {warmup_ret['reason']}")
            raise ValueError(f"Dispatcher预热失败: {warmup_ret['reason']}")
        warmup_duration = time.perf_counter() - warmup_start
        logger.logger.info(f'Dispatcher 预热耗时: {warmup_duration:.2f}s')
        dispatcher_setup_duration = time.perf_counter() - dispatcher_setup_start
        logger.logger.info(f'Dispatcher 创建完成，耗时: {dispatcher_setup_duration:.2f}s')
        logger.logger.info('\n步骤 7: 执行 FastGNAS 多轮批量采样 + Shapley 搜索...')
        total_experiment_start = time.perf_counter()
        graph_info = {'vertices': sum((p.num_nodes for p in partitions)), 'edges': sum((p.num_edges for p in partitions)), 'avg_degree': 3.9, 'diameter': 10}
        num_search_rounds = nas_config.get('num_search_rounds', 3)
        batch_size = nas_config.get('balancing_strategy', {}).get('batch_size', 24)
        min_architectures = nas_config.get('balancing_strategy', {}).get('min_architectures', 8)
        tolerance = nas_config.get('balancing_strategy', {}).get('tolerance', 0.1)
        logger.logger.info(f'\n开始执行 {num_search_rounds} 轮搜索，每轮批量生成 {batch_size} 个架构')
        logger.logger.info(f'负载均衡最小架构数: {min_architectures}')
        all_results = []
        all_rewards = []
        model_counter = 0
        total_search_time = 0.0
        total_train_time_wallclock = 0.0
        total_train_time_accumulated = 0.0
        total_eval_time = 0.0
        total_predictor_time = 0.0
        total_pruning_time = 0.0
        total_sync_time = 0.0
        global_timeline = []
        global_best_accuracy = 0.0
        for round_num in range(1, num_search_rounds + 1):
            logger.logger.info(f'\n--- 开始第 {round_num}/{num_search_rounds} 轮搜索 ---')
            round_start = time.perf_counter()
            search_start = time.perf_counter()
            if enable_balancing and load_predictor:
                logger.logger.info(f'使用Shapley负载均衡生成架构...')
                device_info = {'gpu_cores': 3584, 'core_frequency': 1.328, 'fp_capability': 6.0}
                balanced_architectures = generate_balanced_shapley_architectures(searcher=searcher, load_predictor=load_predictor, batch_size=batch_size, min_architectures=min_architectures, tolerance=tolerance, num_workers=num_workers, graph_info=graph_info, device_info=device_info, logger=logger)
                batch_architectures = balanced_architectures
                batch_arch_specs = [arch['arch_spec'] for arch in batch_architectures]
                logger.logger.info(f'负载均衡选择了 {len(batch_arch_specs)} 个架构进行评估')
            else:
                logger.logger.info(f'使用标准Shapley生成 {batch_size} 个架构...')
                batch_architectures = searcher.sample_architectures(batch_size, stage='initial')
                batch_arch_specs = [arch['arch_spec'] for arch in batch_architectures]
                logger.logger.info(f'生成了 {len(batch_arch_specs)} 个架构进行评估')
            search_time = time.perf_counter() - search_start
            total_search_time += search_time
            logger.logger.info(f'架构生成完成，耗时: {search_time:.2f}s')
            logger.logger.info(f'开始评估 {len(batch_arch_specs)} 个架构...')
            eval_start = time.perf_counter()
            batch_results, batch_rewards, batch_arch_ids = evaluate_architecture_batch(batch_arch_specs, dispatcher, training_config, logger, load_predictor=load_predictor, graph_info=graph_info, start_arch_index=model_counter + 1, stage_name=f'第{round_num}轮')
            eval_duration = time.perf_counter() - eval_start
            batch_train_time_wallclock = 0.0
            batch_train_time_accumulated = 0.0
            for result in batch_results:
                if batch_train_time_wallclock == 0.0:
                    batch_train_time_wallclock = result.get('total_training_time', 0.0)
                training_history = result.get('training_history', [])
                if training_history:
                    last_round = training_history[-1]
                    cpu_time = last_round.get('accumulated_time', 0.0)
                    batch_train_time_accumulated += cpu_time
            total_train_time_wallclock += batch_train_time_wallclock
            total_train_time_accumulated += batch_train_time_accumulated
            batch_sync_time = 0.0
            if batch_results:
                first_result = batch_results[0]
                timing_stats = first_result.get('timing_stats', {})
                if 'sync_overhead' in timing_stats:
                    sync_overhead = timing_stats.get('sync_overhead', {})
                    if isinstance(sync_overhead, dict):
                        batch_sync_time = sum(sync_overhead.values())
            total_sync_time += batch_sync_time
            batch_eval_time = max(0, eval_duration - batch_train_time_wallclock)
            total_eval_time += batch_eval_time
            logger.logger.info(f'注册第 {round_num} 轮评估结果到 Shapley 搜索器...')
            for i, (arch_spec, reward) in enumerate(zip(batch_arch_specs, batch_rewards)):
                model_counter += 1
                arch_id = batch_arch_ids[i] if i < len(batch_arch_ids) else f'shap_arch_{model_counter:05d}'
                standardized_arch_spec = arch_spec.copy()
                if 'activation1' in standardized_arch_spec:
                    if standardized_arch_spec['activation1'] == 'relu':
                        standardized_arch_spec['activation1'] = 'Relu'
                if 'activation2' in standardized_arch_spec:
                    if standardized_arch_spec['activation2'] == 'relu':
                        standardized_arch_spec['activation2'] = 'Relu'
                searcher.register_evaluation(arch_id, standardized_arch_spec, reward)
                logger.logger.info(f'注册架构 {arch_id}: 准确率={reward:.4f}')
            round_best_idx = np.argmax(batch_rewards)
            round_best_accuracy = batch_rewards[round_best_idx]
            if round_best_accuracy > global_best_accuracy:
                global_best_accuracy = round_best_accuracy
                logger.logger.info(f'发现新的全局最佳准确率: {global_best_accuracy:.4f}')
            round_duration = time.perf_counter() - round_start
            global_timeline.append({'round': round_num, 'cumulative_time': time.perf_counter() - total_experiment_start, 'best_accuracy': global_best_accuracy, 'sampled_architectures': len(batch_architectures), 'evaluated_architectures': len(batch_results)})
            all_results.extend(batch_results)
            all_rewards.extend(batch_rewards)
            logger.logger.info(f'第 {round_num} 轮完成，耗时: {round_duration:.2f}s')
            logger.logger.info(f'  采样架构数: {len(batch_architectures)}')
            logger.logger.info(f'  评估架构数: {len(batch_results)}')
            logger.logger.info(f'  轮次最佳准确率: {round_best_accuracy:.4f}')
            logger.logger.info(f'  全局最佳准确率: {global_best_accuracy:.4f}')
        logger.logger.info('\n' + '=' * 60)
        logger.logger.info('跳过 Shapley 预测器训练与剪枝（统一为仅阶段1）')
        logger.logger.info('=' * 60)
        shapley_duration = 0.0
        total_experiment_duration = time.perf_counter() - total_experiment_start
        best_idx = np.argmax(all_rewards)
        best_architecture = all_results[best_idx]
        best_accuracy = all_rewards[best_idx]
        logger.logger.info('\n' + '=' * 80)
        logger.logger.info('🎉 FastGNAS-Shapley 实验完成!')
        logger.logger.info('=' * 80)
        logger.logger.info(f'\n🏆 最佳架构结果:')
        logger.logger.info(f"  架构ID: {best_architecture['architecture_name']}")
        logger.logger.info(f'  最佳准确率: {best_accuracy:.4f}')
        logger.logger.info(f"  最佳轮次: {best_architecture['best_round']}")
        logger.logger.info(f"  是否收敛: {('是' if best_architecture['converged'] else '否')}")
        logger.logger.info(f'\n⏱️  详细时间统计:')
        logger.logger.info(f'  总实验时间: {total_experiment_duration:.2f}s')
        logger.logger.info(f'  数据分区: {partition_duration:.2f}s ({partition_duration / total_experiment_duration * 100:.1f}%)')
        logger.logger.info(f'  Workers创建: {worker_setup_duration:.2f}s ({worker_setup_duration / total_experiment_duration * 100:.1f}%)')
        logger.logger.info(f'  负载预测器: {balance_setup_duration:.2f}s ({balance_setup_duration / total_experiment_duration * 100:.1f}%)')
        logger.logger.info(f'  Shapley搜索器: {searcher_setup_duration:.2f}s ({searcher_setup_duration / total_experiment_duration * 100:.1f}%)')
        logger.logger.info(f'  Dispatcher: {dispatcher_setup_duration:.2f}s ({dispatcher_setup_duration / total_experiment_duration * 100:.1f}%)')
        logger.logger.info(f'  Dispatcher预热: {warmup_duration:.2f}s ({warmup_duration / total_experiment_duration * 100:.1f}%)')
        total_search_loop_duration = total_search_time + total_eval_time + total_predictor_time + total_pruning_time
        logger.logger.info(f'\n📊 多轮搜索时间分解:')
        logger.logger.info(f'  总搜索循环时间: {total_search_loop_duration:.2f}s')
        logger.logger.info(f'  ├─ 架构采样时间: {total_search_time:.2f}s ({total_search_time / total_search_loop_duration * 100:.1f}%)')
        logger.logger.info(f'  ├─ 架构训练时间（墙上时钟）: {total_train_time_wallclock:.2f}s ({total_train_time_wallclock / total_search_loop_duration * 100:.1f}%)')
        logger.logger.info(f'      (Backend层墙上时钟时间，含调度不含评估，可与Ad-rl/Ad_OD-rl/FastGNAS-rl公平对比)')
        logger.logger.info(f'  ├─ 架构训练时间（串行累加）: {total_train_time_accumulated:.2f}s (保留旧方案统计)')
        logger.logger.info(f'  ├─ 架构评估时间: {total_eval_time:.2f}s ({total_eval_time / total_search_loop_duration * 100:.1f}%)')
        if total_sync_time > 0:
            logger.logger.info(f'  ├─ 轮转同步时间: {total_sync_time:.2f}s ({total_sync_time / total_search_loop_duration * 100:.1f}%)')
            logger.logger.info(f'      (对标Ca-rl的sync_time，包含checkpoint I/O和ray.get)')
        if total_train_time_wallclock > 0 and total_train_time_accumulated > 0:
            num_workers = 8
            parallelism = total_train_time_accumulated / total_train_time_wallclock
            theoretical_max = num_workers
            efficiency = parallelism / theoretical_max * 100 if theoretical_max > 0 else 0
            logger.logger.info(f'\n⚡ 并行效率分析:')
            logger.logger.info(f'  Worker数量: {num_workers}')
            logger.logger.info(f'  架构训练时间（墙上时钟）: {total_train_time_wallclock:.2f}s')
            logger.logger.info(f'  CPU累计时间: {total_train_time_accumulated:.2f}s')
            logger.logger.info(f'  实际并行倍率: {parallelism:.2f}x')
            logger.logger.info(f'  理论最大倍率: {theoretical_max}x')
            logger.logger.info(f'  并行效率: {efficiency:.1f}%')
            if efficiency < 50:
                logger.logger.info(f'  ⚠️  并行效率较低')
            elif efficiency < 80:
                logger.logger.info(f'  ✅ 并行效率正常')
            else:
                logger.logger.info(f'  🎉 并行效率优秀')
        logger.logger.info(f'\n📊 其他时间开销:')
        logger.logger.info(f'  ├─ 预测器训练时间: {total_predictor_time:.2f}s ({total_predictor_time / total_search_loop_duration * 100:.1f}%)')
        logger.logger.info(f'  └─ 搜索空间剪枝时间: {total_pruning_time:.2f}s ({total_pruning_time / total_search_loop_duration * 100:.1f}%)')
        logger.logger.info(f'\n📊 架构统计:')
        logger.logger.info(f'  搜索轮次: {num_search_rounds}')
        logger.logger.info(f'  每轮批量大小: {batch_size}')
        logger.logger.info(f'  总评估架构数: {len(all_results)}')
        logger.logger.info(f"  成功评估架构数: {sum((1 for r in all_results if r.get('converged', False)))}")
        logger.logger.info(f"  失败架构数: {sum((1 for r in all_results if not r.get('converged', False)))}")
        logger.logger.info(f'  平均准确率: {np.mean(all_rewards):.4f}')
        logger.logger.info(f'  最高准确率: {np.max(all_rewards):.4f}')
        logger.logger.info(f'  最低准确率: {np.min(all_rewards):.4f}')
        logger.logger.info(f'  准确率标准差: {np.std(all_rewards):.4f}')
        logger.logger.info('\n' + '=' * 80)
        logger.logger.info('📊 实验汇总（统一格式）')
        logger.logger.info('=' * 80)
        logger.logger.info(f'实验-训练数据集: {args.dataset}')
        logger.logger.info(f'实验-纯训练时间: {total_train_time_wallclock:.6f}s (不含评估)')
        logger.logger.info(f'实验-训练架构数: {len(all_results)}')
        if all_results and all_rewards and (len(all_results) > 0):
            avg_train_time = total_train_time_wallclock / len(all_results)
            logger.logger.info(f'实验-平均架构训练时间: {avg_train_time:.6f}s')
            logger.logger.info(f'实验-最优架构精度: {best_accuracy:.6f}')
            logger.logger.info(f"实验-最优架构编号: {best_architecture['architecture_name']}")
            logger.logger.info(f"实验-最优架构出现轮次: {best_architecture['best_round']}")
        logger.logger.info(f'\n📈 搜索时间线:')
        for timeline_entry in global_timeline:
            logger.logger.info(f"  第{timeline_entry['round']}轮: 时间={timeline_entry['cumulative_time']:.1f}s, 最佳准确率={timeline_entry['best_accuracy']:.4f}, 采样={timeline_entry['sampled_architectures']}, 评估={timeline_entry['evaluated_architectures']}")
        if log_dir:
            results_file = os.path.join(log_dir, 'experiment_results.yaml')
            results_data = {'best_architecture': best_architecture, 'best_accuracy': float(best_accuracy), 'all_results': all_results, 'timing_stats': {'total_duration': total_experiment_duration, 'total_search_duration': total_experiment_duration, 'shapley_duration': shapley_duration, 'partition_duration': partition_duration, 'worker_setup_duration': worker_setup_duration, 'balance_setup_duration': balance_setup_duration, 'searcher_setup_duration': searcher_setup_duration, 'dispatcher_setup_duration': dispatcher_setup_duration, 'warmup_duration': warmup_duration}}
            with open(results_file, 'w', encoding='utf-8') as f:
                yaml.dump(results_data, f, default_flow_style=False, allow_unicode=True)
            logger.logger.info(f'\n💾 实验结果已保存到: {results_file}')
            try:
                from utils.export_timeline import normalize_and_save_timeline
                paths = normalize_and_save_timeline(global_timeline, log_dir)
                logger.logger.info(f"搜索时间线已导出: {paths.get('csv')} {paths.get('xlsx')}")
            except Exception as e:
                logger.logger.error(f'导出搜索时间线失败: {e}')
            try:
                logger.save_metrics_to_excel(save_csv=True)
            except Exception as e:
                logger.logger.error(f'导出 metrics 失败: {e}')
    except Exception as e:
        logger.logger.error(f'实验执行失败: {e}')
        raise
    finally:
        if ray.is_initialized():
            ray.shutdown()
            logger.logger.info('Ray 资源已清理')
if __name__ == '__main__':
    main()