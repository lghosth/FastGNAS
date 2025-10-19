import os
import sys
import yaml
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
import torch
import ray
import argparse
import pprint
import copy
from tqdm import tqdm
from datetime import datetime
import time
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)
from core.controller import ControllerManager
from core.dispatcher_v2 import Dispatcher
from trainer.worker import TrainerWorker
from data.data_utils import get_dataset
from data.partition import PartitionGraph
from utils.logger import GraphNASLogger
from core.balancing import evaluate_balance
from core.load_estimator import MloadPredictor, DataCollectorActor
from core.balance_controller import BalanceController
from utils.utils import format_architecture_for_display, load_config
from utils.precision_explorer import HyperparameterExplorer
from baseline.components.backends.ca_backends import CABackends
from baseline.gnas_ca_rl.run import run_single_architecture_experiment

def main():
    parser = argparse.ArgumentParser(description='GraphNAS - Distributed Graph Neural Network Architecture Search')
    parser.add_argument('--config', type=str, default='configs/train.yaml', help='Path to configuration file')
    args = parser.parse_args()
    workflow_start = time.perf_counter()
    run_timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    log_dir = os.path.join('logs', 'logings', f'run_{run_timestamp}')
    logger = GraphNASLogger(log_dir=log_dir, name='main_process')
    logger.logger.info('GraphNAS search workflow starting...')
    config = load_config(args.config)
    nas_config = config['nas']
    train_config = config['training']
    logger.logger.info(f'Loaded configuration: \n{yaml.dump(config, indent=2)}')
    logger.logger.info('Step 1: Initializing Ray...')
    ray_init_start = time.perf_counter()
    device_cfg = train_config.get('devie_config') or {}
    num_gpus: int = 0
    worker_list: list[int] = []
    if int(device_cfg.get('device_GPU', 0)) == int(device_cfg.get('num_workers', 2)):
        logger.logger.info(f"Using all GPUs, GPU count: {device_cfg.get('device_GPU', 0)}")
        num_gpus = int(device_cfg.get('device_GPU', 0))
    elif int(device_cfg.get('device_GPU', 0)) > int(device_cfg.get('num_workers', 2)) and int(device_cfg.get('num_workers', 2)) > 0:
        logger.logger.info(f"Using GPU count: {int(device_cfg.get('num_workers', 2))}")
        num_gpus = int(device_cfg.get('num_workers', 2))
        worker_list = device_cfg.get('workers', [])
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, worker_list))
        logger.logger.info(f'Set visible GPUs: {worker_list}')
    else:
        logger.logger.warning('GPU configuration error')
        raise ValueError('GPU configuration error')
    if num_gpus == 0:
        logger.logger.warning('No GPU detected, will use CPU for simulation')
        ray.init(num_cpus=2, ignore_reinit_error=True)
    else:
        ray.init(num_gpus=num_gpus, ignore_reinit_error=True)
    logger.logger.info(f'-> Ray initialized successfully. Available resources: {ray.available_resources()}')
    ray_init_duration = time.perf_counter() - ray_init_start
    logger.logger.info(f'Ray initialization took {ray_init_duration:.2f} seconds')
    logger.logger.info('Step 2: Preparing and partitioning dataset...')
    data_prep_start = time.perf_counter()
    dataset_name = train_config['dataset']
    num_workers = num_gpus
    max_hops = nas_config['gnn_layers']
    logger.logger.info(f'Preparing to load dataset:{dataset_name}| Partition into {num_workers} graph partitions | Model layers: {max_hops}')
    partition_strategy = train_config.get('train_partition_strategy', 'hybrid_fixed_random')
    logger.logger.info(f'Using partition strategy: {partition_strategy}')
    partitions = PartitionGraph.get_partitions(dataset_name=dataset_name, num_partitions=num_workers, max_hops=max_hops, strategy=partition_strategy)
    for i, p in enumerate(partitions):
        logger.logger.info(f'Partition {i}: Train={p.train_mask.sum().item()}, Test={p.test_mask.sum().item()}, Val={p.val_mask.sum().item()}')
    logger.logger.info('Graph partitioning completed.')
    data_prep_duration = time.perf_counter() - data_prep_start
    logger.logger.info(f'Data preparation and partitioning took {data_prep_duration:.2f} seconds')
    global_num_classes = 0
    try:
        for p in partitions:
            if hasattr(p, 'y') and isinstance(p.y, torch.Tensor):
                val = int(p.y.max().item()) + 1 if p.y.numel() > 0 else 0
                global_num_classes = max(global_num_classes, val)
    except Exception as _:
        logger.logger.warning('Failed to calculate global class count, will use class count from configuration file')
        pass
    if global_num_classes <= 0:
        global_num_classes = int(train_config.get('num_classes', 7))
    train_config['num_classes'] = int(global_num_classes)
    logger.logger.info(f"Unified class count num_classes={train_config['num_classes']}")
    logger.logger.info('Step 3: Creating backend evaluation components (Workers & Dispatcher)...')
    worker_setup_start = time.perf_counter()
    collector_actor = None
    if train_config['data_collection'].get('enable_mload_collector', False):
        collector_dataset_dir = train_config['data_collection'].get('dataset_dir', 'dataset/Mload_dataset')
        file_prefix = train_config['data_collection'].get('file_prefix', 'test')
        time_window_mode = train_config['data_collection'].get('time_window_mode', 'task')
        device_type = train_config['data_collection'].get('device_type', 'a6000')
        filename = f'{file_prefix}-{device_type}-{time_window_mode}.yaml'
        flush_every = train_config['data_collection'].get('flush_every', 1)
        data_collection_cfg = train_config.get('data_collection', {})
        collector_actor = DataCollectorActor.options(num_gpus=0).remote(dataset_dir=collector_dataset_dir, filename=filename, flush_every=flush_every, data_collection_cfg=data_collection_cfg)
    workers = []
    for i in range(num_workers):
        use_gpu = torch.cuda.is_available()
        worker_actor = TrainerWorker.options(num_gpus=1 if use_gpu else 0, num_cpus=4).remote(worker_id=i, gpu_id=i if use_gpu else -1, data_partition=partitions[i], config=train_config, collector_ref=collector_actor)
        workers.append(worker_actor)
    logger.logger.info(f'-> {len(workers)} Workers successfully created')
    worker_setup_duration = time.perf_counter() - worker_setup_start
    logger.logger.info(f'Backend component creation took {worker_setup_duration:.2f} seconds')
    logger.logger.info('Step 4: Initializing reinforcement learning controller...')
    controller = ControllerManager(search_strategy=nas_config['search_strategy'], num_layers=nas_config['gnn_layers'], num_intermediate_nodes=nas_config['num_intermediate_nodes'], learning_rate=nas_config['controller_lr'], baseline_decay=nas_config['controller_baseline_decay'], hidden_size=nas_config.get('controller_hidden_size', 100), tanh_constant=nas_config.get('tanh_constant', 2.5), temperature=nas_config.get('temperature', 5.0), entropy_weight=nas_config.get('entropy_weight', 0.0001), weight_init_range=nas_config.get('weight_init_range', 0.1))
    logger.logger.info('-> Controller initialized successfully')
    load_predictor = None
    balance_controller = None
    enable_balancing = nas_config.get('balancing_strategy', {}).get('enable', False)
    if enable_balancing:
        logger.logger.info('Step 5a: Initializing load balancing component...')
        model_file_path = 'mload_checkpoint.pth'
        if os.path.exists(model_file_path):
            load_predictor = MloadPredictor.load_from_file(model_file_path)
            logger.logger.info(f'Successfully loaded load predictor, model checkpoint: {model_file_path}')
        else:
            logger.logger.warning(f'Load predictor file does not exist: {model_file_path}')
            raise FileNotFoundError(f'Model loading failed, path does not exist: {model_file_path}')
        balance_strategy_config = nas_config.get('balancing_strategy', {})
        balance_controller = BalanceController(controller_manager=controller, load_predictor=load_predictor, batch_size=balance_strategy_config.get('batch_size', 10), min_architectures=balance_strategy_config.get('min_architectures', 3), min_samples=balance_strategy_config.get('min_samples', 3), max_samples=balance_strategy_config.get('max_samples', 10), tolerance=balance_strategy_config.get('tolerance', 0.1), num_workers=num_workers, graph_info={'vertices': 2708, 'edges': 5429, 'avg_degree': 3.9, 'diameter': 10})
        logger.logger.info('Load balancing controller initialization completed')
    dispatcher_scheduler_type = nas_config['balancing_strategy'].get('rotation_mode', 'bucket')
    logger.logger.info(f'Enabled rotation mode: {dispatcher_scheduler_type}')
    dispatcher = Dispatcher.remote(worker_refs=workers, train_config=train_config, scheduler_type=dispatcher_scheduler_type, balance_controller=balance_controller, load_predictor=load_predictor, enable_balancing=enable_balancing)
    logger.logger.info('Step 6b: Warming up Dispatcher...')
    warmup_start = time.perf_counter()
    warmup_ret: dict = ray.get(dispatcher.warmup_system.remote(batch_size=4, min_architectures=num_workers))
    if warmup_ret['ok']:
        logger.logger.info('Dispatcher warmup successful')
    else:
        logger.logger.error(f"Dispatcher warmup failed: {warmup_ret['reason']}")
        raise ValueError(f"Dispatcher warmup failed: {warmup_ret['reason']}")
    logger.logger.info('Dispatcher warmup completed')
    warmup_duration = time.perf_counter() - warmup_start
    logger.logger.info(f'Dispatcher warmup took {warmup_duration:.2f} seconds')
    logger.logger.info(f"Step 5: Starting {nas_config['num_search_rounds']} rounds of architecture search...")
    search_loop_start = time.perf_counter()
    all_candidate_models = {}
    best_model_record = {'model_id': None, 'arch_spec': None, 'accuracy': -1.0, 'stage': '', 'config': copy.deepcopy(train_config), 'best_round': None}
    model_counter = 0
    for round_num in tqdm(range(1, nas_config['num_search_rounds'] + 1), desc='GraphNAS Search Progress'):
        logger.logger.info(f"--- Starting round {round_num}/{nas_config['num_search_rounds']} of search ---")
        round_start = time.perf_counter()
        search_stage_start = round_start
        if enable_balancing and balance_controller:
            logger.logger.info('Using load balancing strategy to generate architectures...')
            balance_strategy_config = nas_config.get('balancing_strategy', {})
            batch_size = balance_strategy_config.get('batch_size', 10)
            min_architectures = balance_strategy_config.get('min_architectures', 3)
            if hasattr(balance_controller, 'batch_size'):
                balance_controller.batch_size = batch_size
            if hasattr(balance_controller, 'min_architectures'):
                balance_controller.min_architectures = min_architectures
            architectures_to_eval = balance_controller.generate_balanced_architectures()
            ray.get(dispatcher.submit_architectures.remote(architectures=architectures_to_eval, strategy=dispatcher_scheduler_type))
        else:
            logger.logger.info('Using original normal strategy to generate architectures...')
            architectures_to_eval = controller.sample_architectures(num_samples=nas_config['samples_per_round'])
            logger.logger.info(f'Sampled {len(architectures_to_eval)} new architectures for evaluation')
            ray.get(dispatcher.submit_architectures.remote(architectures=architectures_to_eval, strategy='auto'))
        search_duration = time.perf_counter() - search_stage_start
        logger.logger.info(f'第 {round_num} 轮搜索阶段耗时 {search_duration:.2f} 秒')
        allocation_start = time.perf_counter()
        current_round_models = {}
        for arch in architectures_to_eval:
            model_counter += 1
            model_id = f'model_{model_counter:04d}'
            current_round_models[arch['arch_id']] = model_id
            all_candidate_models[model_id] = {'arch_spec': arch['arch_spec'], 'best_accuracy': 0.0, 'round': round_num, 'best_round': None, 'original_arch_id': arch['arch_id'], 'load': arch.get('load')}
        allocation_duration = time.perf_counter() - allocation_start
        logger.logger.debug(f'第 {round_num} 轮模型ID分配耗时 {allocation_duration:.2f} 秒')
        num_eval_rounds = train_config.get('num_eval_rounds', 5)
        enable_global_evaluation = nas_config.get('enable_global_evaluation', False)
        training_start = time.perf_counter()
        train_out = ray.get(dispatcher.run_training.remote(num_rounds=num_eval_rounds, epochs_per_round=train_config['epochs_per_rotation'], enable_global_evaluation=enable_global_evaluation))
        training_duration = time.perf_counter() - training_start
        logger.logger.info(f'第 {round_num} 轮评估训练耗时 {training_duration:.2f} 秒')
        postprocess_start = time.perf_counter()
        eval_rewards = {}
        if isinstance(train_out, dict):
            eval_rewards = train_out.get('final_accuracies', {}) or {}
        else:
            logger.logger.warning(f'未能获取有效的评估结果: {train_out}')
            raise ValueError(f'未能获取有效的评估结果: {train_out}')
        logger.logger.info(f'第 {round_num} 轮评估了 {len(eval_rewards)} 个模型，生成了 {len(architectures_to_eval)} 个架构')
        if architectures_to_eval:
            preview = [a.get('arch_id') for a in architectures_to_eval[:min(5, len(architectures_to_eval))]]
            logger.logger.debug(f'本轮架构ID示例: {preview}')
        for arch_id, accuracy in eval_rewards.items():
            if arch_id in current_round_models:
                model_id = current_round_models[arch_id]
                all_candidate_models[model_id]['best_accuracy'] = accuracy
                all_candidate_models[model_id]['best_round'] = round_num
                logger.logger.info(f'模型 {model_id} (轮次 {round_num}) 准确率: {accuracy:.4f}')
                if accuracy > best_model_record['accuracy']:
                    best_model_record = {'model_id': model_id, 'arch_spec': copy.deepcopy(all_candidate_models[model_id]['arch_spec']), 'accuracy': accuracy, 'stage': 'search', 'config': copy.deepcopy(train_config), 'best_round': round_num}
        if eval_rewards:
            logger.log_metrics(eval_rewards, step=round_num, prefix='nas_rewards/')
        controller_start = time.perf_counter()
        controller.train_step(rewards=eval_rewards)
        controller_duration = time.perf_counter() - controller_start
        postprocess_duration = time.perf_counter() - postprocess_start
        logger.logger.debug(f'第 {round_num} 轮结果处理耗时 {postprocess_duration:.2f} 秒 (含控制器更新 {controller_duration:.2f} 秒)')
        round_duration = time.perf_counter() - round_start
        num_models_evaluated = len(eval_rewards)
        avg_time_per_model = training_duration / num_models_evaluated if num_models_evaluated > 0 else 0.0
        accounted_time = search_duration + allocation_duration + training_duration + postprocess_duration
        unaccounted_time = round_duration - accounted_time
        logger.logger.info(f'第 {round_num} 轮总耗时 {round_duration:.2f}s (搜索{search_duration:.2f}s + 分配{allocation_duration:.2f}s + 训练{training_duration:.2f}s + 后处理{postprocess_duration:.2f}s + 其他{unaccounted_time:.2f}s)')
        logger.logger.info(f'第 {round_num} 轮评估了 {num_models_evaluated} 个模型，平均每个模型训练耗时 {avg_time_per_model:.2f}s\n')
    logger.logger.info('步骤 6: 架构搜索完成！')
    total_search_duration = time.perf_counter() - search_loop_start
    num_total_models = len(all_candidate_models)
    avg_time_per_model_overall = total_search_duration / num_total_models if num_total_models > 0 else 0.0
    logger.logger.info(f'搜索阶段总耗时 {total_search_duration:.2f} 秒')
    logger.logger.info(f'总共评估了 {num_total_models} 个模型')
    logger.logger.info(f'平均每个模型搜索评估耗时 {avg_time_per_model_overall:.2f}s (包含所有开销)')
    logger.logger.info('步骤 7: 选择Top K候选模型进行最终重训练...')
    sorted_models = sorted(all_candidate_models.items(), key=lambda x: x[1]['best_accuracy'], reverse=True)
    logger.logger.info(f'总共评估了 {len(all_candidate_models)} 个模型')
    logger.logger.info('=== 所有模型排名（Top 10）===')
    for i, (model_id, model_info) in enumerate(sorted_models[:10]):
        best_round = model_info.get('best_round', model_info['round'])
        logger.logger.info(f"第{i + 1}名: {model_id} (首次出现于轮次{model_info['round']}) - 历史最佳准确率: {model_info['best_accuracy']:.4f} (发生于轮次{best_round})")
    final_candidates_count = nas_config.get('final_candidates', 5)
    top_k_models = sorted_models[:final_candidates_count]
    logger.logger.info(f'=== 选择Top {final_candidates_count}模型进行最终重训练 ===')
    for i, (model_id, model_info) in enumerate(top_k_models):
        logger.logger.info(f"候选{i + 1}: {model_id} - 历史最佳准确率: {model_info['best_accuracy']:.4f} (轮次{model_info.get('best_round', model_info['round'])})")
    raise ValueError('测试终点')
    final_retrain_epochs = nas_config.get('final_retrain_epochs', 20)
    logger.logger.info(f'步骤 8: 对Top {final_candidates_count}模型进行完整重训练 ({final_retrain_epochs} rounds)...')
    final_train_config = train_config.copy()
    logger.logger.info('步骤 8a: 释放搜索阶段的 Dispatcher 实例，准备新的CA后端...')
    try:
        ray.kill(dispatcher, no_restart=True)
        logger.logger.info('旧 Dispatcher 已成功释放')
    except Exception as e:
        logger.logger.warning(f'释放旧 Dispatcher 时遇到问题: {e}')
    logger.logger.info('步骤 8b: 释放搜索阶段的 TrainerWorker ...')
    for w in workers:
        try:
            ray.kill(w, no_restart=True)
        except Exception as e:
            logger.logger.warning(f'释放 TrainerWorker 时遇到问题: {e}')
    ca_backend_settings = {'num_workers': num_workers, 'dataset': dataset_name, 'device': 'cuda' if torch.cuda.is_available() else 'cpu', 'learning_rate': final_train_config.get('lr', train_config.get('lr', 0.01)), 'weight_decay': final_train_config.get('weight_decay', train_config.get('weight_decay', 0.0005)), 'dropout': final_train_config.get('dropout', train_config.get('dropout', 0.6)), 'num_classes': final_train_config.get('num_classes', train_config.get('num_classes', 7)), 'partition_strategy': train_config.get('train_partition_strategy', 'hybrid_fixed_random')}
    ca_training_config = {'training': {'epochs_per_round': final_train_config.get('epochs_per_rotation', 5), 'num_rounds': final_retrain_epochs, 'batch_size': final_train_config.get('batch_size'), 'steps_per_epoch': final_train_config.get('steps_per_epoch')}}
    final_results = {}
    retrain_start = time.perf_counter()
    ca_backends = CABackends(ca_backend_settings)
    ca_backends.setup_distributed_training(num_workers=num_workers)
    for idx, (model_id, model_info) in enumerate(top_k_models, start=1):
        arch_name = f'{model_id}_final'
        logger.logger.info(f'[CA-Retrain] 开始重训练模型 {idx}/{len(top_k_models)}: {arch_name}')
        try:
            retrain_result = run_single_architecture_experiment(ca_backends=ca_backends, arch_spec=model_info['arch_spec'], arch_name=arch_name, config=ca_training_config, logger=logger)
            final_accuracy = float(retrain_result.get('best_accuracy', 0.0))
            final_results[model_id] = {'model_info': model_info, 'initial_accuracy': model_info['best_accuracy'], 'best_round': model_info.get('best_round'), 'final_accuracy': final_accuracy, 'improvement': final_accuracy - model_info['best_accuracy']}
            logger.logger.info(f"模型 {model_id} 重训练完成: 初始={model_info['best_accuracy']:.4f} 最终={final_accuracy:.4f} 提升={final_accuracy - model_info['best_accuracy']:.4f}")
            if final_accuracy > best_model_record['accuracy']:
                best_model_record = {'model_id': model_id, 'arch_spec': copy.deepcopy(model_info['arch_spec']), 'accuracy': final_accuracy, 'stage': 'retrain-ca', 'config': copy.deepcopy(final_train_config), 'best_round': model_info.get('best_round')}
        except Exception as e:
            logger.logger.error(f'模型 {model_id} 重训练失败: {e}')
            final_results[model_id] = {'model_info': model_info, 'initial_accuracy': model_info['best_accuracy'], 'best_round': model_info.get('best_round'), 'final_accuracy': model_info['best_accuracy'], 'improvement': 0.0}
    retrain_duration = time.perf_counter() - retrain_start
    logger.logger.info(f'CA 重训练阶段耗时 {retrain_duration:.2f} 秒')
    try:
        for worker in ca_backends.workers:
            ray.kill(worker, no_restart=True)
    except Exception:
        pass
    logger.logger.info('步骤 8c: 为精度探索阶段重新配置 CA 后端...')
    ca_backends.setup_distributed_training(num_workers=num_workers)
    ca_backends.update_training_config({'learning_rate': final_train_config.get('lr', train_config.get('lr', 0.01)), 'lr': final_train_config.get('lr', train_config.get('lr', 0.01)), 'weight_decay': final_train_config.get('weight_decay', train_config.get('weight_decay', 0.0005)), 'dropout': final_train_config.get('dropout', train_config.get('dropout', 0.6))})
    logger.logger.info(f'CA 后端已准备 {len(ca_backends.workers)} 个 Worker 供精度探索使用')
    workers = []
    logger.logger.info('步骤 9: 输出最终结果...')
    final_results_header = '=' * 25 + ' GraphNAS 最终结果 ' + '=' * 25
    logger.logger.info(final_results_header)
    best_final_model = max(final_results.items(), key=lambda x: x[1]['final_accuracy'])
    best_model_id, best_model_result = best_final_model
    logger.logger.info(f'数据集名称: {dataset_name}')
    logger.logger.info(f"搜索策略: {nas_config['search_strategy']}")
    logger.logger.info(f"总搜索轮次: {nas_config['num_search_rounds']}")
    logger.logger.info(f'总评估模型数: {len(all_candidate_models)}')
    logger.logger.info(f'最终候选数: {final_candidates_count}')
    logger.logger.info(f'\n🔥 最佳模型: {best_model_id}')
    logger.logger.info(f"   来源轮次: {best_model_result['model_info']['round']}")
    logger.logger.info(f"   初始准确率: {best_model_result['initial_accuracy']:.4f}")
    logger.logger.info(f"   最终准确率: {best_model_result['final_accuracy']:.4f}")
    logger.logger.info(f"   准确率提升: {best_model_result['improvement']:.4f}")
    formatted_arch = format_architecture_for_display(best_model_result['model_info']['arch_spec'], nas_config['search_strategy'])
    logger.logger.info(f'\n🏆 最佳GNN架构:\n{formatted_arch}')
    logger.logger.info(f"🚀🚀🚀 最佳验证准确率: {best_model_result['final_accuracy']:.4f}")
    logger.logger.info(f'\n=== 所有最终候选模型结果 ===')
    sorted_final = sorted(final_results.items(), key=lambda x: x[1]['initial_accuracy'], reverse=True)
    for i, (model_id, result) in enumerate(sorted_final):
        logger.logger.info(f'第{i + 1}名: {model_id}')
        logger.logger.info(f"  历史最佳准确率: {result['initial_accuracy']:.4f} (轮次{result.get('best_round', result['model_info']['round'])})")
        logger.logger.info(f"  最终准确率: {result['final_accuracy']:.4f}")
        logger.logger.info(f"  准确率提升: {result['improvement']:.4f}")
    logger.logger.info('=' * len(final_results_header))
    logger.logger.info('步骤 10: 探寻最佳模型的精度上限 (贝叶斯优化)...')
    if best_model_record['arch_spec'] is not None:
        hyperparam_config = nas_config.get('hyperparameter_optimization', {})
        explorer = HyperparameterExplorer(base_config=best_model_record.get('config', train_config), logger=logger, ca_backend=ca_backends, optimization_strategy=hyperparam_config.get('strategy', 'bayesian'), n_initial_points=hyperparam_config.get('n_initial_points', 10), n_calls=hyperparam_config.get('n_calls', 50), acquisition_func=hyperparam_config.get('acquisition_func', 'EI'), early_stop_patience=hyperparam_config.get('early_stop_patience', 3))
        precision_result = explorer.run({'model_id': best_model_record['model_id'], 'arch_spec': best_model_record['arch_spec']})
        logger.logger.info(f"超参搜索最佳组合: {precision_result['params']}，平均精度 {precision_result['accuracy']:.4f}")
        logger.logger.info(f"策略: {hyperparam_config.get('strategy', 'bayesian')}, 评估次数: {hyperparam_config.get('n_calls', 50)}, 采集函数: {hyperparam_config.get('acquisition_func', 'EI')}")
    else:
        logger.logger.warning('未找到可用的最佳模型记录，跳过精度上限探寻阶段')
    csv_path = logger.save_metrics_to_csv()
    logger.logger.info(f'所有搜索指标已保存到: {csv_path}')
    logger.logger.info('步骤 7: 关闭Ray，释放资源...')
    ray.shutdown()
    total_duration = time.perf_counter() - workflow_start
    logger.logger.info(f'Ray初始化耗时: {ray_init_duration:.2f} 秒')
    logger.logger.info(f'数据准备与分区耗时: {data_prep_duration:.2f} 秒')
    logger.logger.info(f'后端组件创建耗时: {worker_setup_duration:.2f} 秒')
    logger.logger.info(f'Dispatcher预热耗时: {warmup_duration:.2f} 秒')
    logger.logger.info(f'搜索阶段总耗时: {total_search_duration:.2f} 秒')
    logger.logger.info(f'重训练阶段总耗时: {retrain_duration:.2f} 秒')
    total_work_duration = total_search_duration + retrain_duration
    logger.logger.info(f'系统真正工作时间: {total_work_duration:.2f} 秒')
    total_plus = ray_init_duration + data_prep_duration + worker_setup_duration + warmup_duration + total_search_duration + retrain_duration
    logger.logger.info(f'分部计时和: {total_plus:.2f} 秒')
    logger.logger.info(f'总运行耗时: {total_duration:.2f} 秒')
    logger.logger.info(f'Ray初始化占比: {ray_init_duration / total_duration * 100:.2f}%')
    logger.logger.info(f'数据准备与分区占比: {data_prep_duration / total_duration * 100:.2f}%')
    logger.logger.info(f'后端组件创建占比: {worker_setup_duration / total_duration * 100:.2f}%')
    logger.logger.info(f'Dispatcher预热占比: {warmup_duration / total_duration * 100:.2f}%')
    logger.logger.info(f'搜索阶段占比: {total_search_duration / total_duration * 100:.2f}%')
    logger.logger.info(f'重训练阶段占比: {retrain_duration / total_duration * 100:.2f}%')
    logger.logger.info(f'系统真正工作时间占比: {total_work_duration / total_duration * 100:.2f}%')
    logger.logger.info('-> 流程结束')
if __name__ == '__main__':
    main()