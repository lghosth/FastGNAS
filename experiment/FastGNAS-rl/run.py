import os
import sys
import yaml
import torch
import ray
import argparse
import numpy as np
from datetime import datetime
from typing import Dict, Any, List, Optional
import time
import copy
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(project_root)
from core.controller import ControllerManager
from core.dispatcher_v2 import Dispatcher
from trainer.worker import TrainerWorker
from data.partition import PartitionGraph
from utils.logger import GraphNASLogger
from core.load_estimator import MloadPredictor, DataCollectorActor
from core.balance_controller import BalanceController
from utils.utils import load_dataset_config, format_architecture_for_display

def load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config

def merge_dataset_config(config: Dict[str, Any], dataset_name: str) -> Dict[str, Any]:
    dataset_config_path = os.path.join(project_root, 'experiment/dataset.yaml')
    dataset_cfg = load_dataset_config(dataset_name, dataset_config_path)
    fastgnas_backends = config.get('fastgnas_backends', {})
    fastgnas_backends['dataset'] = dataset_cfg.get('name', dataset_name)
    if 'training' not in config:
        config['training'] = {}
    training_cfg = config['training']
    training_cfg['dataset'] = dataset_cfg.get('name', dataset_name)
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
        fastgnas_backends['num_classes'] = dataset_cfg['num_classes']
    if 'gnn_layers' in dataset_cfg:
        if 'nas' not in config:
            config['nas'] = {}
        config['nas']['gnn_layers'] = dataset_cfg['gnn_layers']
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

def _print_fastgnas_round_timing_analysis(round_num: int, train_out: Dict[str, Any], logger: GraphNASLogger) -> None:
    if not isinstance(train_out, dict):
        return
    statistics = train_out.get('statistics', {})
    timing_stats = train_out.get('timing_stats', {})
    if not statistics:
        return
    total_time = statistics.get('total_time', 0.0)
    sync_overhead = statistics.get('sync_overhead', {})
    logger.logger.info(f'\n⏱️  第 {round_num} 轮时间详细分析:')
    logger.logger.info(f'  训练时间: {total_time:.2f}s')
    if total_time > 0:
        if timing_stats:
            cpu_total = sum(timing_stats.values())
            if cpu_total > 0:
                logger.logger.info(f'\n  CPU累计时间分解 (所有Worker之和):')
                logger.logger.info(f"    ├─ 模型迁移: {timing_stats.get('model_migration', 0):.4f}s ({timing_stats.get('model_migration', 0) / cpu_total * 100:.1f}%)")
                batch_prep = timing_stats.get('batch_preparation', 0)
                if batch_prep > 0:
                    logger.logger.info(f'    ├─ 批次准备: {batch_prep:.2f}s ({batch_prep / cpu_total * 100:.1f}%)')
                    logger.logger.info(f'    │      (所有Worker×所有epoch×所有batch的累计时间)')
                    logger.logger.info(f'    │      包含: 邻居采样 + 特征提取 + 子图构建')
                compute_time = timing_stats.get('forward', 0) + timing_stats.get('backward', 0)
                logger.logger.info(f'    ├─ 计算时间: {compute_time:.2f}s ({compute_time / cpu_total * 100:.1f}%)')
                logger.logger.info(f"    │   ├─ 前向传播: {timing_stats.get('forward', 0):.2f}s")
                logger.logger.info(f"    │   └─ 反向传播: {timing_stats.get('backward', 0):.2f}s")
                logger.logger.info(f"    ├─ 参数更新: {timing_stats.get('optimizer_step', 0):.2f}s ({timing_stats.get('optimizer_step', 0) / cpu_total * 100:.1f}%)")
                memory_cleanup = timing_stats.get('memory_cleanup', 0)
                if memory_cleanup > 0:
                    logger.logger.info(f'    └─ 显存清理: {memory_cleanup:.4f}s ({memory_cleanup / cpu_total * 100:.1f}%)')
                    logger.logger.info(f'        (包含: batch释放 + GPU缓存清理 + 优化器清理)')
        if sync_overhead:
            sync_total = sum(sync_overhead.values())
            if sync_total > 0:
                logger.logger.info(f'\n  轮转同步开销分解:')
                logger.logger.info(f"    ├─ checkpoint加载: {sync_overhead.get('checkpoint_load', 0):.4f}s ({sync_overhead.get('checkpoint_load', 0) / sync_total * 100:.1f}%)")
                logger.logger.info(f"    ├─ 通信开销: {sync_overhead.get('rotation_overhead', 0):.4f}s ({sync_overhead.get('rotation_overhead', 0) / sync_total * 100:.1f}%)")
                logger.logger.info(f'    │      (Ray序列化 + 网络传输 + 反序列化)')
                logger.logger.info(f"    ├─ checkpoint保存: {sync_overhead.get('checkpoint_save', 0):.4f}s ({sync_overhead.get('checkpoint_save', 0) / sync_total * 100:.1f}%)")
                logger.logger.info(f"    └─ scheduler反馈: {sync_overhead.get('scheduler_feedback', 0):.4f}s ({sync_overhead.get('scheduler_feedback', 0) / sync_total * 100:.1f}%)")
        num_workers = 8
        avg_per_worker = cpu_total / num_workers if cpu_total > 0 else 0
        logger.logger.info(f'\n  💡 时间统计说明:')
        logger.logger.info(f'    - 训练时间: {total_time:.2f}s (墙上时钟时间，实际物理耗时)')
        logger.logger.info(f'    - CPU累计时间: {cpu_total:.2f}s (所有Worker时间之和)')
        if avg_per_worker > 0:
            logger.logger.info(f'    - 平均每个Worker: {avg_per_worker:.2f}s')
            logger.logger.info(f'    - 并行效率: {total_time / avg_per_worker * 100:.1f}%')

def main():
    parser = argparse.ArgumentParser(description='FastGNAS-RL 实验')
    parser.add_argument('--config', type=str, default='experiment/FastGNAS-rl/config.yaml', help='配置文件路径')
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
        try:
            config['fastgnas_backends']['workers'] = list(range(int(args.workers)))
        except Exception:
            pass
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
        log_dir = create_log_directory(log_config.get('log_dir', 'experiment/FastGNAS-rl/log'))
        print(f'日志目录: {log_dir}')
    else:
        log_dir = None
    logger = GraphNASLogger(name=f'fastgnas_rl_experiment_{args.dataset}', log_dir=log_dir, log_level=log_config.get('level', 'INFO'))
    logger.logger.info('=' * 80)
    logger.logger.info(f'FastGNAS-RL 实验开始, 数据集: {args.dataset}')
    logger.logger.info('=' * 80)
    logger.logger.info(f"数据集: {fastgnas_config['dataset']}")
    logger.logger.info(f"Worker数量: {fastgnas_config['num_workers']}")
    logger.logger.info(f"设备类型: {fastgnas_config.get('device', 'cuda')}")
    logger.logger.info(f"学习率: {fastgnas_config.get('learning_rate', 0.01)}")
    logger.logger.info(f"搜索策略: {nas_config.get('search_strategy', 'layer_wise')}")
    logger.logger.info(f'训练配置: {training_config}')
    logger.logger.info(f"负载均衡配置: {nas_config.get('balancing_strategy', {})}")
    workflow_start = time.perf_counter()
    try:
        logger.logger.info('\n步骤 1: 初始化 Ray...')
        ray_init_start = time.perf_counter()
        num_workers = fastgnas_config['num_workers']
        num_gpus = num_workers if torch.cuda.is_available() else 0
        if num_gpus == 0:
            logger.logger.warning('未检测到GPU，将使用CPU进行模拟')
            ray.init(num_cpus=max(2, num_workers), ignore_reinit_error=True)
        else:
            ray.init(num_gpus=num_gpus, ignore_reinit_error=True)
        logger.logger.info(f'-> Ray 初始化成功. 可用资源: {ray.available_resources()}')
        ray_init_duration = time.perf_counter() - ray_init_start
        logger.logger.info(f'Ray 初始化耗时: {ray_init_duration:.2f} 秒')
        logger.logger.info('\n步骤 2: 准备并划分数据集...')
        data_prep_start = time.perf_counter()
        dataset_name = fastgnas_config['dataset']
        max_hops = nas_config.get('gnn_layers', 2)
        partition_strategy = fastgnas_config.get('partition_strategy', 'balanced_stratified_khop')
        logger.logger.info(f'加载数据集: {dataset_name} | 分区数: {num_workers} | 层数: {max_hops}')
        logger.logger.info(f'使用分区策略: {partition_strategy}')
        partitions = PartitionGraph.get_partitions(dataset_name=dataset_name, num_partitions=num_workers, max_hops=max_hops, strategy=partition_strategy)
        for i, p in enumerate(partitions):
            logger.logger.info(f'分区 {i}: Train={p.train_mask.sum().item()}, Test={p.test_mask.sum().item()}, Val={p.val_mask.sum().item()}')
        logger.logger.info('图分区完成.')
        data_prep_duration = time.perf_counter() - data_prep_start
        logger.logger.info(f'数据准备耗时: {data_prep_duration:.2f} 秒')
        global_num_classes = 0
        try:
            for p in partitions:
                if hasattr(p, 'y') and isinstance(p.y, torch.Tensor):
                    val = int(p.y.max().item()) + 1 if p.y.numel() > 0 else 0
                    global_num_classes = max(global_num_classes, val)
        except Exception as _:
            logger.logger.warning('计算全局类别数失败，将使用配置文件中的值')
            pass
        if global_num_classes <= 0:
            global_num_classes = int(training_config.get('num_classes', 7))
        training_config['num_classes'] = int(global_num_classes)
        logger.logger.info(f"统一类别数 num_classes={training_config['num_classes']}")
        logger.logger.info('\n步骤 3: 创建后端评估组件 (Workers & Dispatcher)...')
        worker_setup_start = time.perf_counter()
        collector_actor = None
        if training_config.get('data_collection', {}).get('enable_mload_collector', False):
            collector_dataset_dir = training_config['data_collection'].get('dataset_dir', 'dataset/Mload_dataset')
            file_prefix = training_config['data_collection'].get('file_prefix', 'fastgnas')
            time_window_mode = training_config['data_collection'].get('time_window_mode', 'batch')
            device_type = training_config['data_collection'].get('device_type', 'a6000')
            filename = f'{file_prefix}-{device_type}-{time_window_mode}.yaml'
            flush_every = training_config['data_collection'].get('flush_every', 1)
            data_collection_cfg = training_config.get('data_collection', {})
            collector_actor = DataCollectorActor.options(num_gpus=0).remote(dataset_dir=collector_dataset_dir, filename=filename, flush_every=flush_every, data_collection_cfg=data_collection_cfg)
        workers = []
        for i in range(num_workers):
            use_gpu = torch.cuda.is_available()
            worker_actor = TrainerWorker.options(num_gpus=1 if use_gpu else 0, num_cpus=4).remote(worker_id=i, gpu_id=i if use_gpu else -1, data_partition=partitions[i], config=training_config, collector_ref=collector_actor)
            workers.append(worker_actor)
        logger.logger.info(f'-> {len(workers)} 个Worker创建成功')
        worker_setup_duration = time.perf_counter() - worker_setup_start
        logger.logger.info(f'后端组件创建耗时: {worker_setup_duration:.2f} 秒')
        logger.logger.info('\n步骤 4: 初始化强化学习控制器...')
        controller = ControllerManager(search_strategy=nas_config.get('search_strategy', 'layer_wise'), num_layers=nas_config.get('gnn_layers', 2), num_intermediate_nodes=nas_config.get('num_intermediate_nodes', 6), learning_rate=nas_config.get('controller_lr', 0.00035), baseline_decay=nas_config.get('controller_baseline_decay', 0.999), hidden_size=nas_config.get('controller_hidden_size', 100), tanh_constant=nas_config.get('tanh_constant', 2.5), temperature=nas_config.get('temperature', 5.0), entropy_weight=nas_config.get('entropy_weight', 0.0001), weight_init_range=nas_config.get('weight_init_range', 0.1))
        logger.logger.info('-> 控制器初始化成功')
        load_predictor = None
        balance_controller = None
        enable_balancing = nas_config.get('balancing_strategy', {}).get('enable', False)
        if enable_balancing:
            logger.logger.info('\n步骤 5: 初始化负载均衡组件...')
            model_file_path = nas_config.get('balancing_strategy', {}).get('model_path', 'mload_checkpoint.pth')
            if os.path.exists(model_file_path):
                load_predictor = MloadPredictor.load_from_file(model_file_path)
                logger.logger.info(f'成功加载负载预测器: {model_file_path}')
            else:
                logger.logger.warning(f'负载预测器文件不存在: {model_file_path}')
                logger.logger.warning('将继续运行，但不使用负载预测功能')
            balance_strategy_config = nas_config.get('balancing_strategy', {})
            default_batch_size = balance_strategy_config.get('batch_size', 10)
            default_min_architectures = balance_strategy_config.get('min_architectures', 3)
            if num_workers == 1:
                logger.logger.info('⚠️  检测到单GPU配置，自动设置采样8个架构（忽略负载均衡配置）')
                adjusted_batch_size = 8
                adjusted_min_architectures = 8
            else:
                adjusted_batch_size = default_batch_size
                adjusted_min_architectures = default_min_architectures
            balance_controller = BalanceController(controller_manager=controller, load_predictor=load_predictor, batch_size=adjusted_batch_size, min_architectures=adjusted_min_architectures, min_samples=balance_strategy_config.get('min_samples', 3), max_samples=balance_strategy_config.get('max_samples', 10), tolerance=balance_strategy_config.get('tolerance', 0.1), num_workers=num_workers, graph_info={'vertices': sum((p.num_nodes for p in partitions)), 'edges': sum((p.num_edges for p in partitions)), 'avg_degree': 3.9, 'diameter': 10})
            logger.logger.info(f'负载均衡控制器初始化完成 (batch_size={adjusted_batch_size}, min_architectures={adjusted_min_architectures})')
        dispatcher_scheduler_type = nas_config.get('balancing_strategy', {}).get('rotation_mode', 'subtask')
        logger.logger.info(f'启用的轮转模式: {dispatcher_scheduler_type}')
        dispatcher = Dispatcher.remote(worker_refs=workers, train_config=training_config, scheduler_type=dispatcher_scheduler_type, balance_controller=balance_controller, load_predictor=load_predictor, enable_balancing=enable_balancing)
        logger.logger.info('\n步骤 6: 预热 Dispatcher...')
        warmup_start = time.perf_counter()
        warmup_ret = ray.get(dispatcher.warmup_system.remote(batch_size=4, min_architectures=num_workers))
        if warmup_ret['ok']:
            logger.logger.info('Dispatcher预热成功')
        else:
            logger.logger.error(f"Dispatcher预热失败: {warmup_ret['reason']}")
            raise ValueError(f"Dispatcher预热失败: {warmup_ret['reason']}")
        warmup_duration = time.perf_counter() - warmup_start
        logger.logger.info(f'Dispatcher 预热耗时: {warmup_duration:.2f} 秒')
        num_search_rounds = nas_config.get('num_search_rounds', 10)
        logger.logger.info(f'\n步骤 7: 开始执行 {num_search_rounds} 轮架构搜索...')
        search_loop_start = time.perf_counter()
        all_candidate_models = {}
        best_model_record = {'model_id': None, 'arch_spec': None, 'accuracy': -1.0, 'stage': '', 'config': copy.deepcopy(training_config), 'best_round': None}
        model_counter = 0
        total_search_time = 0.0
        total_train_time = 0.0
        total_eval_time = 0.0
        total_controller_time = 0.0
        total_sync_time = 0.0
        total_overhead_time = 0.0
        cumulative_time = 0.0
        cpu_accumulated_timing = {'model_migration': 0.0, 'batch_preparation': 0.0, 'forward': 0.0, 'backward': 0.0, 'optimizer_step': 0.0, 'memory_cleanup': 0.0}
        global_timeline = []
        global_best_accuracy = 0.0
        best_arch_index = 0
        for round_num in range(1, num_search_rounds + 1):
            logger.logger.info(f'\n--- 开始第 {round_num}/{num_search_rounds} 轮搜索 ---')
            round_start = time.perf_counter()
            search_start = time.perf_counter()
            if dispatcher_scheduler_type == 'rotation':
                logger.logger.info('使用Rotation模式：直接采样架构（不预测负载）...')
                num_samples = num_workers
                architectures_to_eval = controller.sample_architectures(num_samples=num_samples)
                logger.logger.info(f'Rotation模式采样了 {len(architectures_to_eval)} 个架构（与worker数一致）')
                ray.get(dispatcher.submit_architectures.remote(architectures=architectures_to_eval, strategy='rotation'))
            elif dispatcher_scheduler_type in ('bucket', 'subtask') and enable_balancing and balance_controller:
                logger.logger.info(f'使用{dispatcher_scheduler_type}模式：负载均衡策略...')
                balance_strategy_config = nas_config.get('balancing_strategy', {})
                batch_size = balance_strategy_config.get('batch_size', 10)
                min_architectures = balance_strategy_config.get('min_architectures', 3)
                if num_workers == 1:
                    logger.logger.info('⚠️  检测到单GPU/单桶配置，强制设置生成8个架构')
                    batch_size = 8
                    min_architectures = 8
                if hasattr(balance_controller, 'batch_size'):
                    balance_controller.batch_size = batch_size
                if hasattr(balance_controller, 'min_architectures'):
                    balance_controller.min_architectures = min_architectures
                architectures_to_eval = balance_controller.generate_balanced_architectures()
                logger.logger.info(f'负载均衡生成了 {len(architectures_to_eval)} 个架构')
                if len(architectures_to_eval) < num_workers and num_workers > 1:
                    logger.logger.warning(f'⚠️  生成的架构数({len(architectures_to_eval)}) < Worker数({num_workers})，可能导致负载不均衡！')
                elif len(architectures_to_eval) < 8 and num_workers == 1:
                    logger.logger.warning(f'⚠️  单GPU模式下架构数不足8个（当前{len(architectures_to_eval)}个），重新生成架构...')
                    balance_controller.batch_size = 8
                    balance_controller.min_architectures = 8
                    architectures_to_eval = balance_controller.generate_balanced_architectures()
                    logger.logger.info(f'重新生成后得到 {len(architectures_to_eval)} 个架构')
                ray.get(dispatcher.submit_architectures.remote(architectures=architectures_to_eval, strategy=dispatcher_scheduler_type))
            else:
                logger.logger.info(f'使用标准策略生成架构（{dispatcher_scheduler_type}模式）...')
                samples_per_round = nas_config.get('samples_per_round', 2)
                architectures_to_eval = controller.sample_architectures(num_samples=samples_per_round)
                logger.logger.info(f'采样了 {len(architectures_to_eval)} 个新架构进行评估')
                ray.get(dispatcher.submit_architectures.remote(architectures=architectures_to_eval, strategy=dispatcher_scheduler_type))
            search_time = time.perf_counter() - search_start
            total_search_time += search_time
            logger.logger.info(f'架构采样耗时: {search_time:.2f}s')
            current_round_models = {}
            for arch in architectures_to_eval:
                model_counter += 1
                model_id = f'model_{model_counter:04d}'
                current_round_models[arch['arch_id']] = model_id
                all_candidate_models[model_id] = {'arch_spec': arch['arch_spec'], 'best_accuracy': 0.0, 'round': round_num, 'best_round': None, 'original_arch_id': arch['arch_id'], 'load': arch.get('load')}
            num_eval_rounds = training_config.get('num_rounds', training_config.get('num_eval_rounds', 2))
            enable_global_evaluation = nas_config.get('enable_global_evaluation', False)
            training_start = time.perf_counter()
            train_out = ray.get(dispatcher.run_training.remote(num_rounds=num_eval_rounds, epochs_per_round=training_config.get('epochs_per_round', training_config.get('epochs_per_rotation', 5)), enable_global_evaluation=enable_global_evaluation))
            training_duration = time.perf_counter() - training_start
            cumulative_time += training_duration
            backend_train_time = 0.0
            if isinstance(train_out, dict) and 'statistics' in train_out:
                backend_train_time = float(train_out['statistics'].get('total_time', 0.0))
                total_train_time += backend_train_time
            else:
                backend_train_time = training_duration
                total_train_time += training_duration
                logger.logger.warning(f'⚠️  train_out格式异常，使用training_duration作为回退')
            eval_time = max(0, training_duration - backend_train_time)
            total_eval_time += eval_time
            logger.logger.info(f'第 {round_num} 轮耗时: {training_duration:.2f}s (训练: {backend_train_time:.2f}s, 评估: {eval_time:.2f}s), 累计: {cumulative_time:.2f}s')
            if isinstance(train_out, dict) and 'timing_stats' in train_out:
                timing_stats = train_out['timing_stats']
                cpu_detailed_time = sum(timing_stats.values())
                logger.logger.debug(f'\n  ⏱️  第 {round_num} 轮 CPU累计时间分解:')
                logger.logger.debug(f'    墙上时钟: {training_duration:.2f}s (实际耗时)')
                logger.logger.debug(f'    CPU累计: {cpu_detailed_time:.2f}s (所有Worker之和)')
                logger.logger.debug(f"    ├─ 模型迁移: {timing_stats.get('model_migration', 0):.4f}s")
                logger.logger.debug(f"    ├─ 批次准备: {timing_stats.get('batch_preparation', 0):.2f}s")
                logger.logger.debug(f"    ├─ 前向传播: {timing_stats.get('forward', 0):.2f}s")
                logger.logger.debug(f"    ├─ 反向传播: {timing_stats.get('backward', 0):.2f}s")
                logger.logger.debug(f"    ├─ 参数更新: {timing_stats.get('optimizer_step', 0):.2f}s")
                memory_cleanup_time = timing_stats.get('memory_cleanup', 0.0)
                if memory_cleanup_time > 0:
                    logger.logger.debug(f'    └─ 显存清理: {memory_cleanup_time:.4f}s')
                for key in cpu_accumulated_timing.keys():
                    cpu_accumulated_timing[key] += timing_stats.get(key, 0.0)
            if isinstance(train_out, dict) and 'statistics' in train_out:
                sync_overhead = train_out['statistics'].get('sync_overhead', {})
                if sync_overhead:
                    round_sync_time = sum(sync_overhead.values())
                    total_sync_time += round_sync_time
            eval_rewards = {}
            if isinstance(train_out, dict):
                eval_rewards = train_out.get('final_accuracies', {}) or {}
            else:
                logger.logger.warning(f'未能获取有效的评估结果: {train_out}')
            logger.logger.info(f'第 {round_num} 轮评估了 {len(eval_rewards)} 个模型，生成了 {len(architectures_to_eval)} 个架构')
            for arch_id, accuracy in eval_rewards.items():
                if arch_id in current_round_models:
                    model_id = current_round_models[arch_id]
                    all_candidate_models[model_id]['best_accuracy'] = accuracy
                    all_candidate_models[model_id]['best_round'] = round_num
                    logger.logger.info(f'模型 {model_id} (轮次 {round_num}) 准确率: {accuracy:.4f}')
                    if accuracy > global_best_accuracy:
                        global_best_accuracy = accuracy
                        best_arch_index = model_counter
                        logger.logger.info(f'🎉 发现新的全局最优! 准确率: {global_best_accuracy:.4f}')
                    global_timeline.append({'cumulative_time': cumulative_time, 'global_best_accuracy': global_best_accuracy, 'sampled_arch_count': model_counter, 'best_arch_index': best_arch_index, 'current_arch_name': arch_id, 'current_round': round_num, 'current_accuracy': accuracy})
                    if accuracy > best_model_record['accuracy']:
                        best_model_record = {'model_id': model_id, 'arch_spec': copy.deepcopy(all_candidate_models[model_id]['arch_spec']), 'accuracy': accuracy, 'stage': 'search', 'config': copy.deepcopy(training_config), 'best_round': round_num}
            if eval_rewards:
                logger.log_metrics(eval_rewards, step=round_num, prefix='nas_rewards/')
            controller_train_start = time.perf_counter()
            controller.train_step(rewards=eval_rewards)
            controller_train_time = time.perf_counter() - controller_train_start
            total_controller_time += controller_train_time
            round_duration = time.perf_counter() - round_start
            logger.logger.info(f'第 {round_num} 轮总耗时: {round_duration:.2f}s\n')
            _print_fastgnas_round_timing_analysis(round_num, train_out, logger)
        logger.logger.info('\n步骤 8: 架构搜索完成！')
        total_search_duration = time.perf_counter() - search_loop_start
        logger.logger.info(f'搜索阶段总耗时: {total_search_duration:.2f} 秒')
        logger.logger.info(f'总共评估了 {len(all_candidate_models)} 个模型')
        logger.logger.info(f'\n' + '=' * 80)
        logger.logger.info('⏱️  时间统计汇总')
        logger.logger.info('=' * 80)
        logger.logger.info(f'搜索阶段总耗时: {total_search_duration:.2f}s')
        accounted_search_time = total_search_time + total_train_time + total_eval_time + total_controller_time
        unaccounted_search_time = total_search_duration - accounted_search_time
        logger.logger.info(f'  ├─ 架构采样时间: {total_search_time:.2f}s ({total_search_time / total_search_duration * 100:.1f}%)')
        logger.logger.info(f'  ├─ 架构训练时间: {total_train_time:.2f}s ({total_train_time / total_search_duration * 100:.1f}%)')
        logger.logger.info(f'      (Backend层墙上时钟时间，含调度不含评估)')
        if total_sync_time > 0:
            sync_ratio = total_sync_time / total_train_time * 100 if total_train_time > 0 else 0
            logger.logger.info(f'  │   └─ 其中轮转同步开销: {total_sync_time:.2f}s (占训练时间{sync_ratio:.1f}%)')
            logger.logger.info(f'          (checkpoint I/O + Ray通信 + 调度反馈)')
        logger.logger.info(f'  ├─ 架构评估时间: {total_eval_time:.2f}s ({total_eval_time / total_search_duration * 100:.1f}%)')
        logger.logger.info(f'  ├─ 控制器训练时间: {total_controller_time:.2f}s ({total_controller_time / total_search_duration * 100:.1f}%)')
        if abs(unaccounted_search_time) > 0.01:
            logger.logger.info(f'  └─ 其他开销: {unaccounted_search_time:.2f}s ({unaccounted_search_time / total_search_duration * 100:.1f}%)')
            logger.logger.info(f'      (包含: 任务分发、结果处理、日志输出等)')
        total_cpu_time = sum(cpu_accumulated_timing.values())
        logger.logger.info(f'\n📊 训练时间详细分析:')
        if total_cpu_time > 0:
            logger.logger.info(f'\n💻 CPU累计时间（所有Worker之和）:')
            logger.logger.info(f'  总CPU时间: {total_cpu_time:.2f}s')
            logger.logger.info(f"  ├─ 模型迁移: {cpu_accumulated_timing['model_migration']:.4f}s ({cpu_accumulated_timing['model_migration'] / total_cpu_time * 100:.1f}%)")
            logger.logger.info(f"  ├─ 批次准备: {cpu_accumulated_timing['batch_preparation']:.2f}s ({cpu_accumulated_timing['batch_preparation'] / total_cpu_time * 100:.1f}%)")
            logger.logger.info(f'  │      (包含: 邻居采样 + 特征提取 + 子图构建)')
            compute_time = cpu_accumulated_timing['forward'] + cpu_accumulated_timing['backward']
            logger.logger.info(f'  ├─ 计算时间: {compute_time:.2f}s ({compute_time / total_cpu_time * 100:.1f}%)')
            logger.logger.info(f"  │  ├─ 前向传播: {cpu_accumulated_timing['forward']:.2f}s")
            logger.logger.info(f"  │  └─ 反向传播: {cpu_accumulated_timing['backward']:.2f}s")
            logger.logger.info(f"  ├─ 参数更新: {cpu_accumulated_timing['optimizer_step']:.2f}s ({cpu_accumulated_timing['optimizer_step'] / total_cpu_time * 100:.1f}%)")
            if cpu_accumulated_timing.get('memory_cleanup', 0.0) > 0:
                logger.logger.info(f"  └─ 显存清理: {cpu_accumulated_timing['memory_cleanup']:.4f}s ({cpu_accumulated_timing['memory_cleanup'] / total_cpu_time * 100:.1f}%)")
                logger.logger.info(f'      (包含: batch释放 + GPU缓存清理 + 优化器清理)')
            if total_train_time > 0 and total_cpu_time > 0:
                parallelism = total_cpu_time / total_train_time
                theoretical_max = num_workers
                efficiency = parallelism / theoretical_max * 100 if theoretical_max > 0 else 0
                logger.logger.info(f'\n⚡ 并行效率分析:')
                logger.logger.info(f'  Worker数量: {num_workers}')
                logger.logger.info(f'  架构训练时间（墙上时钟）: {total_train_time:.2f}s')
                logger.logger.info(f'  CPU累计时间: {total_cpu_time:.2f}s')
                logger.logger.info(f'  实际并行倍率: {parallelism:.2f}x')
                logger.logger.info(f'  理论最大倍率: {theoretical_max}x')
                logger.logger.info(f'  并行效率: {efficiency:.1f}%')
                if efficiency < 50:
                    logger.logger.info(f'  ⚠️  并行效率较低，可能存在：')
                    logger.logger.info(f'      - Worker负载不均衡')
                    logger.logger.info(f'      - 同步开销过大')
                    logger.logger.info(f'      - Worker等待时间过长')
                elif efficiency < 80:
                    logger.logger.info(f'  ✅ 并行效率正常')
                else:
                    logger.logger.info(f'  🎉 并行效率优秀')
            if total_sync_time > 0:
                sync_overhead = {}
                if isinstance(train_out, dict) and 'statistics' in train_out:
                    sync_overhead = train_out['statistics'].get('sync_overhead', {})
                if sync_overhead:
                    logger.logger.info(f'\n🔄 轮转同步开销详细分解:')
                    logger.logger.info(f'  总轮转同步开销: {total_sync_time:.2f}s (已包含在上方统计中)')
                    checkpoint_load = sync_overhead.get('checkpoint_load', 0)
                    communication_overhead = sync_overhead.get('rotation_overhead', 0)
                    checkpoint_save = sync_overhead.get('checkpoint_save', 0)
                    scheduler_feedback = sync_overhead.get('scheduler_feedback', 0)
                    sync_total = checkpoint_load + communication_overhead + checkpoint_save + scheduler_feedback
                    if sync_total > 0:
                        logger.logger.info(f'  ├─ checkpoint加载: {checkpoint_load:.4f}s ({checkpoint_load / sync_total * 100:.1f}%)')
                        logger.logger.info(f'  ├─ 通信开销: {communication_overhead:.4f}s ({communication_overhead / sync_total * 100:.1f}%)')
                        logger.logger.info(f'  │      (Ray序列化 + 网络传输 + 反序列化)')
                        logger.logger.info(f'  ├─ checkpoint保存: {checkpoint_save:.4f}s ({checkpoint_save / sync_total * 100:.1f}%)')
                        logger.logger.info(f'  └─ scheduler反馈: {scheduler_feedback:.4f}s ({scheduler_feedback / sync_total * 100:.1f}%)')
            logger.logger.info(f'\n💡 时间统计说明:')
            logger.logger.info(f'  - 架构训练时间：Backend层墙上时钟时间（含调度，不含评估）')
            logger.logger.info(f'  - CPU累计时间：所有Worker的时间总和（用于占比分析）')
            logger.logger.info(f'  - CPU累计 > 架构训练时间：正常现象（多Worker并行训练）')
            logger.logger.info(f'  - 轮转同步开销：checkpoint I/O + Ray通信 + 调度反馈（不包含Worker训练）')
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
        logger.logger.info('\n步骤 9: 输出最终结果...')
        logger.logger.info('=' * 80)
        logger.logger.info(' FastGNAS-RL 最终结果 ')
        logger.logger.info('=' * 80)
        sorted_models = sorted(all_candidate_models.items(), key=lambda x: x[1]['best_accuracy'], reverse=True)
        logger.logger.info(f'总共评估了 {len(all_candidate_models)} 个模型')
        logger.logger.info('\n=== 所有模型排名（Top 10）===')
        for i, (model_id, model_info) in enumerate(sorted_models[:10]):
            best_round = model_info.get('best_round', model_info['round'])
            logger.logger.info(f"第{i + 1}名: {model_id} (首次出现于轮次{model_info['round']}) - 最佳准确率: {model_info['best_accuracy']:.4f} (发生于轮次{best_round})")
        if sorted_models:
            best_model_id, best_model_info = sorted_models[0]
            logger.logger.info(f'\n🔥 最佳模型: {best_model_id}')
            logger.logger.info(f"   来源轮次: {best_model_info['round']}")
            logger.logger.info(f"   最佳准确率: {best_model_info['best_accuracy']:.4f}")
            formatted_arch = format_architecture_for_display(best_model_info['arch_spec'], nas_config.get('search_strategy', 'layer_wise'))
            logger.logger.info(f'\n🏆 最佳GNN架构:\n{formatted_arch}')
            logger.logger.info(f"🚀 最佳验证准确率: {best_model_info['best_accuracy']:.4f}")
        csv_path = logger.save_metrics_to_csv()
        logger.logger.info(f'\n所有搜索指标已保存到: {csv_path}')
    except Exception as e:
        logger.logger.error(f'实验执行失败: {e}')
        import traceback
        traceback.print_exc()
        return 1
    finally:
        logger.logger.info('\n步骤 10: 关闭Ray，释放资源...')
        ray.shutdown()
    total_duration = time.perf_counter() - workflow_start
    logger.logger.info(f'\n总运行耗时: {total_duration:.2f} 秒')
    logger.logger.info(f'  ├─ Ray初始化: {ray_init_duration:.2f}s ({ray_init_duration / total_duration * 100:.1f}%)')
    logger.logger.info(f'  ├─ 数据准备: {data_prep_duration:.2f}s ({data_prep_duration / total_duration * 100:.1f}%)')
    logger.logger.info(f'  ├─ Worker创建: {worker_setup_duration:.2f}s ({worker_setup_duration / total_duration * 100:.1f}%)')
    logger.logger.info(f'  ├─ Dispatcher预热: {warmup_duration:.2f}s ({warmup_duration / total_duration * 100:.1f}%)')
    logger.logger.info(f'  └─ 架构搜索: {total_search_duration:.2f}s ({total_search_duration / total_duration * 100:.1f}%)')
    if total_search_duration > 0:
        logger.logger.info(f'\n📊 搜索阶段时间分解:')
        logger.logger.info(f'  ├─ 架构采样时间: {total_search_time:.2f}s ({total_search_time / total_duration * 100:.1f}%)')
        logger.logger.info(f'  ├─ 架构训练时间: {total_train_time:.2f}s ({total_train_time / total_duration * 100:.1f}%)')
        logger.logger.info(f'      (Backend层墙上时钟时间，含调度不含评估)')
        if total_sync_time > 0:
            sync_ratio = total_sync_time / total_train_time * 100 if total_train_time > 0 else 0
            logger.logger.info(f'  │   └─ 其中轮转同步开销: {total_sync_time:.2f}s (占训练时间{sync_ratio:.1f}%)')
            logger.logger.info(f'          (checkpoint I/O + Ray通信 + 调度反馈)')
        logger.logger.info(f'  ├─ 架构评估时间: {total_eval_time:.2f}s ({total_eval_time / total_duration * 100:.1f}%)')
        logger.logger.info(f'  ├─ 控制器训练时间: {total_controller_time:.2f}s ({total_controller_time / total_duration * 100:.1f}%)')
        accounted_time = ray_init_duration + data_prep_duration + worker_setup_duration + warmup_duration + total_search_time + total_train_time + total_eval_time + total_controller_time
        unaccounted_time = total_duration - accounted_time
        logger.logger.info(f'\n💡 时间验证:')
        logger.logger.info(f'  已统计时间: {accounted_time:.2f}s')
        logger.logger.info(f'  未统计时间: {unaccounted_time:.2f}s ({abs(unaccounted_time) / total_duration * 100:.1f}%)')
        if abs(unaccounted_time) > 1.0:
            logger.logger.info(f'  ⚠️  注意：未统计时间可能包含Ray通信、结果处理、日志输出等开销')
    logger.logger.info('\n' + '=' * 80)
    logger.logger.info('📊 实验汇总（统一格式）')
    logger.logger.info('=' * 80)
    logger.logger.info(f'实验-训练数据集: {args.dataset}')
    logger.logger.info(f'实验-纯训练时间: {total_train_time:.6f}s (不含评估)')
    logger.logger.info(f'实验-训练架构数: {len(all_candidate_models)}')
    if sorted_models and len(all_candidate_models) > 0:
        avg_train_time = total_train_time / len(all_candidate_models)
        logger.logger.info(f'实验-平均架构训练时间: {avg_train_time:.6f}s')
        best_model_id, best_model_info = sorted_models[0]
        logger.logger.info(f"实验-最优架构精度: {best_model_info['best_accuracy']:.6f}")
        logger.logger.info(f'实验-最优架构编号: {best_model_id}')
        logger.logger.info(f"实验-最优架构出现轮次: {best_model_info.get('best_round', best_model_info['round'])}")
    logger.logger.info('\n' + '=' * 80)
    logger.logger.info(f'FastGNAS-RL 实验完成，数据集: {args.dataset}')
    logger.logger.info(f"使用了 {config['fastgnas_backends']['num_workers']} 个GPU")
    logger.logger.info('=' * 80)
    return 0
if __name__ == '__main__':
    exit(main())