# file: run.py

import os
import sys
import yaml
import torch
import ray
import argparse
import pprint
from tqdm import tqdm
from datetime import datetime

# 动态添加项目根目录到Python路径，确保所有模块都能被正确导入
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)

from core.controller import ControllerManager
from core.dispatcher import Dispatcher
from trainer.worker import TrainerWorker
from data.data_utils import get_dataset
from data.partition import partition_training_graph
from utils.logger import GraphNASLogger
from core.balancing import evaluate_balance
from core.load_estimator import MloadPredictor, Config
from core.balance_controller import BalanceController

def format_architecture_for_display(arch_spec: dict, search_strategy: str) -> str:
    """
    将架构规范格式化为人类可读的形式
    """
    lines = []
    lines.append("🏗️  GNN架构详情:")
    lines.append("─" * 50)
    
    if search_strategy == "layer_wise":
        lines.append("📋 架构类型: Layer-wise搜索策略")
        if 'layers' in arch_spec:
            lines.append(f"📊 总层数: {len(arch_spec['layers'])}")
            lines.append("")
            
            for i, layer in enumerate(arch_spec['layers']):
                layer_num = i + 1
                lines.append(f"🔸 第 {layer_num} 层:")
                
                # 确定GNN类型
                encoding = layer.get('encoding', 'unknown')
                gnn_type_map = {
                    'gcn': 'GCN (图卷积网络)',
                    'gat': 'GAT (图注意力网络)', 
                    'sym-gat': 'Symmetric GAT',
                    'sage': 'GraphSAGE',
                    'cheb': 'ChebNet (切比雪夫网络)',
                    'arma': 'ARMA',
                    'sgc': 'SGC (简化图卷积)',
                    'const': 'Constant',
                    'cos': 'Cosine',
                    'linear': 'Linear',
                    'gene-linear': 'Generalized Linear'
                }
                gnn_name = gnn_type_map.get(encoding, f"Unknown ({encoding})")
                lines.append(f"   🧠 模型类型: {gnn_name}")
                
                # 其他参数
                if 'hidden_dim' in layer:
                    lines.append(f"   📐 隐藏维度: {layer['hidden_dim']}")
                if 'num_heads' in layer and layer.get('num_heads', 1) > 1:
                    lines.append(f"   👁️  注意力头数: {layer['num_heads']}")
                if 'activation' in layer:
                    act_map = {
                        'relu': 'ReLU', 'tanh': 'Tanh', 'sigmoid': 'Sigmoid',
                        'elu': 'ELU', 'leaky_relu': 'LeakyReLU', 'none': '无激活',
                        'linear': '线性', 'softplus': 'Softplus', 'relu6': 'ReLU6'
                    }
                    act_name = act_map.get(layer['activation'], layer['activation'])
                    lines.append(f"   ⚡ 激活函数: {act_name}")
                if 'aggregation' in layer:
                    agg_map = {
                        'mean': '平均聚合', 'sum': '求和聚合', 'max': '最大聚合', 'mlp': 'MLP聚合'
                    }
                    agg_name = agg_map.get(layer['aggregation'], layer['aggregation'])
                    lines.append(f"   🔄 聚合方式: {agg_name}")
                lines.append("")
                
    elif search_strategy == "cell_based":
        lines.append("📋 架构类型: Cell-based搜索策略")
        if 'cell_spec' in arch_spec:
            cell_spec = arch_spec['cell_spec']
            num_cells = arch_spec.get('num_cells_to_stack', 2)
            lines.append(f"📊 堆叠单元数: {num_cells}")
            lines.append(f"🔗 中间节点数: {cell_spec.get('intermediate_nodes', 'unknown')}")
            lines.append("")
            
            if 'connections' in cell_spec:
                lines.append("🔸 单元内连接:")
                for i, conn in enumerate(cell_spec['connections']):
                    op = conn.get('op', 'unknown')
                    from_node = conn.get('from_node', 'unknown')
                    
                    # 映射操作到GNN类型
                    op_type_map = {
                        'gcn': 'GCN (图卷积网络)',
                        'gat_1_head': 'GAT (1头)',
                        'gat_2_head': 'GAT (2头)',
                        'gat_4_head': 'GAT (4头)',
                        'gat_6_head': 'GAT (6头)',
                        'gat_8_head': 'GAT (8头)',
                        'mean_sage': 'GraphSAGE (平均聚合)',
                        'chebnet': 'ChebNet (切比雪夫网络)',
                        'arma': 'ARMA',
                        'sgc': 'SGC (简化图卷积)',
                        'identity': 'Identity (恒等映射)',
                        'zero': 'Zero (零操作)'
                    }
                    op_name = op_type_map.get(op, f"Unknown ({op})")
                    lines.append(f"   连接 {i+1}: 节点{from_node} → {op_name}")
            
            if 'readout_op' in cell_spec:
                readout_map = {'add': '加法', 'concat': '拼接', 'multiply': '乘法'}
                readout_name = readout_map.get(cell_spec['readout_op'], cell_spec['readout_op'])
                lines.append(f"   🔄 读出操作: {readout_name}")
            
            if 'final_act' in cell_spec:
                act_map = {
                    'relu': 'ReLU', 'tanh': 'Tanh', 'sigmoid': 'Sigmoid',
                    'elu': 'ELU', 'leaky_relu': 'LeakyReLU', 'none': '无激活',
                    'linear': '线性', 'softplus': 'Softplus', 'relu6': 'ReLU6'
                }
                act_name = act_map.get(cell_spec['final_act'], cell_spec['final_act'])
                lines.append(f"   ⚡ 最终激活: {act_name}")
    else:
        lines.append(f"📋 架构类型: 未知策略 ({search_strategy})")
        lines.append("原始架构规范:")
        lines.append(pprint.pformat(arch_spec, width=60))
    
    lines.append("─" * 50)
    return "\n".join(lines)

def load_config(config_path: str) -> dict:
    """
    加载YAML配置文件。如果文件不存在或不完整，则创建并写入默认配置。
    """
    default_config = {
        'nas': {
            'num_search_rounds': 200,
            'samples_per_round': 5,
            'gnn_layers': 3,
            'controller_lr': 0.001,
            'controller_baseline_decay': 0.9,
        },
        'training': {
            'dataset': 'Cora',
            'lr': 0.01,
            'dropout': 0.6,
            'epochs_per_rotation': 10,
            'batch_size': 64,
            'steps_per_epoch': 50,
        }
    }

    if not os.path.exists(config_path):
        print(f"警告: 配置文件 '{config_path}' 不存在。将创建并使用默认配置。")
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        with open(config_path, 'w') as f:
            yaml.dump(default_config, f, sort_keys=False)
        return default_config

    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f) or {}
        # 深度合并默认配置
        for key, value in default_config.items():
            if key not in config:
                config[key] = value
            elif isinstance(value, dict):
                config[key] = {**value, **config[key]}
        print(f"成功从 '{config_path}' 加载并合并配置。")
        return config
    except Exception as e:
        print(f"错误: 加载配置文件失败: {e}。将使用默认配置。")
        return default_config

def main():
    """
    主执行函数，串联整个GraphNAS搜索流程。
    """
    parser = argparse.ArgumentParser(description="GraphNAS - 分布式图神经网络架构搜索")
    parser.add_argument('--config', type=str, default='configs/train.yaml',
                        help='配置文件的路径')
    args = parser.parse_args()

    # 0. 初始化日志记录器
    run_timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    log_dir = os.path.join("logs", "logings", f"run_{run_timestamp}")
    logger = GraphNASLogger(log_dir=log_dir, name="main_process")
    logger.logger.info("GraphNAS 搜索流程启动...")

    # 1. 加载配置
    config = load_config(args.config)
    nas_config = config['nas']
    train_config = config['training']
    logger.logger.info(f"加载配置: \n{yaml.dump(config, indent=2)}")

    # 2. 初始化Ray
    logger.logger.info("步骤 1: 初始化 Ray...")
    num_gpus = int(torch.cuda.device_count())
    if num_gpus == 0:
        logger.logger.warning("未检测到GPU。将使用CPU进行模拟。")
        ray.init(num_cpus=2, ignore_reinit_error=True) # 至少保证2个worker
    else:
        ray.init(num_gpus=num_gpus, ignore_reinit_error=True)
    logger.logger.info(f"-> Ray 初始化成功. 可用资源: {ray.available_resources()}")

    # 3. 数据准备和分区
    logger.logger.info("步骤 2: 准备并划分数据集...")
    dataset_name = train_config['dataset']
    num_workers = int(ray.available_resources().get("GPU", 2))
    max_hops = nas_config['gnn_layers'] 
    
    partitions = partition_training_graph(dataset_name, num_workers, max_hops)
    logger.logger.info(f"-> 数据集 '{dataset_name}' 已划分为 {len(partitions)} 个部分。")

    # 4. 创建后端评估组件 (Workers 和 Dispatcher)
    logger.logger.info("步骤 3: 创建后端评估组件 (Workers & Dispatcher)...")
    workers = []
    for i in range(num_workers):
        use_gpu = torch.cuda.is_available()
        worker_actor = TrainerWorker.options(num_gpus=1 if use_gpu else 0).remote(
            worker_id=i,
            gpu_id=i if use_gpu else -1, # CPU worker use -1
            data_partition=partitions[i]
        )
        workers.append(worker_actor)
    
    logger.logger.info(f"-> {len(workers)} 个Worker已成功创建。")

    # 5. 初始化前端搜索控制器
    logger.logger.info("步骤 4: 初始化强化学习控制器...")
    controller = ControllerManager(
        search_strategy=nas_config['search_strategy'],
        num_layers=nas_config['gnn_layers'],
        num_intermediate_nodes=nas_config['num_intermediate_nodes'],
        learning_rate=nas_config['controller_lr'],
        baseline_decay=nas_config['controller_baseline_decay'],
        hidden_size=nas_config.get('controller_hidden_size', 100),
        tanh_constant=nas_config.get('tanh_constant', 2.5),
        temperature=nas_config.get('temperature', 5.0),
        entropy_weight=nas_config.get('entropy_weight', 0.0001),
        weight_init_range=nas_config.get('weight_init_range', 0.1)
    )
    logger.logger.info("-> 控制器初始化成功。")

    # 6. 初始化负载均衡组件（如果启用）
    load_predictor = None
    balance_controller = None
    enable_balancing = nas_config.get('balancing_strategy', {}).get('enable', False)
    
    if enable_balancing:
        logger.logger.info("步骤 5a: 初始化负载均衡组件...")
        
        # 尝试加载负载预测器
        model_file_path = "mload_checkpoint.pth"
        if os.path.exists(model_file_path):
            load_predictor = MloadPredictor.load_from_file(model_file_path)
            logger.logger.info(f"成功加载负载预测器: {model_file_path}")
        else:
            logger.logger.warning(f"负载预测器文件不存在: {model_file_path}")
            logger.logger.warning("将使用模拟负载进行测试")
        
        # 创建负载均衡控制器
        balance_strategy_config = nas_config.get('balancing_strategy', {})
        balance_controller = BalanceController(
            controller_manager=controller,
            load_predictor=load_predictor,
            min_samples=balance_strategy_config.get('min_samples', 3),
            max_samples=balance_strategy_config.get('max_samples', 10),
            tolerance=balance_strategy_config.get('tolerance', 0.1),
            num_workers=num_workers,
            graph_info={
                "vertices": 2708,  # 默认Cora数据集
                "edges": 5429,
                "avg_degree": 3.9,
                "diameter": 10
            }
        )
        logger.logger.info("负载均衡控制器初始化完成")

    # 7. 执行主搜索循环 - 架构发现阶段
    logger.logger.info(f"步骤 5: 开始执行 {nas_config['num_search_rounds']} 轮架构搜索...")
    
    # 新增：存储所有采样模型的信息
    all_candidate_models = {}  # {model_id: {'arch_spec': ..., 'best_accuracy': ..., 'round': ...}}
    model_counter = 0  # 模型编号计数器
    
    for round_num in tqdm(range(1, nas_config['num_search_rounds'] + 1), desc="GraphNAS 搜索进度"):
        logger.logger.info(f"--- 开始第 {round_num}/{nas_config['num_search_rounds']} 轮搜索 ---")
        
        # a. 根据策略生成架构
        if enable_balancing and balance_controller:
            # 使用负载均衡策略
            logger.logger.info("使用负载均衡策略生成架构...")
            # 创建带负载均衡的Dispatcher
            dispatcher = Dispatcher.remote(
                worker_refs=workers, 
                train_config=train_config,
                balance_controller=balance_controller,
                load_predictor=load_predictor,
                enable_balancing=True
            )
            # 生成并提交负载均衡的架构
            num_generated = ray.get(dispatcher.submit_balanced_architectures.remote())
            logger.logger.info(f"负载均衡策略生成了 {num_generated} 个架构")
            
            # 获取生成的架构信息
            arch_results = ray.get(dispatcher.get_architectures.remote(format_type='pool_only'))
            if isinstance(arch_results, dict):
                architectures_to_eval = arch_results.get('model_pool', [])
            else:
                architectures_to_eval = []
        else:
            # 使用原有的普通策略
            architectures_to_eval = controller.sample_architectures(
                num_samples=nas_config['samples_per_round']
            )
            logger.logger.info(f"采样了 {len(architectures_to_eval)} 个新架构进行评估。")
            
            # 创建普通的Dispatcher
            dispatcher = Dispatcher.remote(
                worker_refs=workers, 
                train_config=train_config
            )
            ray.get(dispatcher.submit_architectures.remote(architectures_to_eval))
        
        # b. 为每个架构分配全局唯一编号
        current_round_models = {}
        for arch in architectures_to_eval:
            model_counter += 1
            model_id = f"model_{model_counter:04d}"
            current_round_models[arch['arch_id']] = model_id
            all_candidate_models[model_id] = {
                'arch_spec': arch['arch_spec'],
                'best_accuracy': 0.0,
                'round': round_num,
                'original_arch_id': arch['arch_id']
            }
        
        # c. Dispatcher已在上面创建，无需重复创建

        # d. Dispatcher执行模型轮转评估
        num_eval_rounds = 1  # 执行一次轮转
        ray.get(dispatcher.run_training_rounds.remote(num_complete_rounds=num_eval_rounds))
        
        # e. 从Dispatcher获取评估结果（奖励）
        results_summary = ray.get(dispatcher.get_architectures.remote(format_type='summary'))
        eval_rewards = {}
        if isinstance(results_summary, dict) and hasattr(results_summary, 'get'):
            architectures_dict = results_summary.get('architectures', {})
            if isinstance(architectures_dict, dict):
                eval_rewards = {
                    arch_id: data.get('best_accuracy', 0.0) 
                    for arch_id, data in architectures_dict.items()
                    if isinstance(data, dict)
                }
        else:
            logger.logger.warning(f"未能获取有效的评估结果: {results_summary}")
        
        # f. 更新候选模型的最佳准确率
        for arch_id, accuracy in eval_rewards.items():
            if arch_id in current_round_models:
                model_id = current_round_models[arch_id]
                all_candidate_models[model_id]['best_accuracy'] = accuracy
                logger.logger.info(f"模型 {model_id} (轮次 {round_num}) 准确率: {accuracy:.4f}")
        
        # g. 新增：使用logger记录本轮所有模型的奖励
        if eval_rewards:
            logger.log_metrics(eval_rewards, step=round_num, prefix='nas_rewards/')
        
        # h. 使用奖励来训练控制器
        controller.train_step(rewards=eval_rewards)
        
        # i. 释放Dispatcher Actor，为下一轮做准备
        ray.kill(dispatcher)

    logger.logger.info("步骤 6: 架构搜索完成！")

    # 7. 选择Top K候选模型进行最终重训练
    logger.logger.info("步骤 7: 选择Top K候选模型进行最终重训练...")
    
    # 按准确率降序排序所有候选模型
    sorted_models = sorted(
        all_candidate_models.items(), 
        key=lambda x: x[1]['best_accuracy'], 
        reverse=True
    )
    
    # 记录所有模型的排名信息
    logger.logger.info(f"总共评估了 {len(all_candidate_models)} 个模型")
    logger.logger.info("=== 所有模型排名（Top 10）===")
    for i, (model_id, model_info) in enumerate(sorted_models[:10]):
        logger.logger.info(f"第{i+1}名: {model_id} (轮次{model_info['round']}) - 准确率: {model_info['best_accuracy']:.4f}")
    
    # 选择Top K模型
    final_candidates_count = nas_config.get('final_candidates', 5)
    top_k_models = sorted_models[:final_candidates_count]
    
    logger.logger.info(f"=== 选择Top {final_candidates_count}模型进行最终重训练 ===")
    for i, (model_id, model_info) in enumerate(top_k_models):
        logger.logger.info(f"候选{i+1}: {model_id} - 初始准确率: {model_info['best_accuracy']:.4f}")
    
    # 8. 对Top K模型进行完整重训练
    final_retrain_epochs = nas_config.get('final_retrain_epochs', 20)
    logger.logger.info(f"步骤 8: 对Top {final_candidates_count}模型进行完整重训练 ({final_retrain_epochs} epochs)...")
    
    # 创建完整训练的配置 TODO: 训练步数可以后续有时间详细研究
    final_train_config = train_config.copy()
    final_train_config['epochs_per_rotation'] =  5 # 5
    final_train_config['steps_per_epoch'] = 5  # 增加训练步数 
    
    final_results = {}
    
    for i, (model_id, model_info) in enumerate(top_k_models):
        logger.logger.info(f"--- 重训练候选模型 {i+1}/{final_candidates_count}: {model_id} ---")
        
        # 准备架构信息
        final_arch = {
            'arch_id': f"{model_id}_final",
            'arch_spec': model_info['arch_spec']
        }
        
        # 创建Dispatcher进行完整训练
        final_dispatcher = Dispatcher.remote(
            worker_refs=workers, 
            train_config=final_train_config
        )
        ray.get(final_dispatcher.submit_architectures.remote([final_arch]))
        
        # 执行完整训练 (多轮以确保充分训练)
        complete_rounds = 5  # 每个模型在所有数据分区上训练4轮
        ray.get(final_dispatcher.run_training_rounds.remote(num_complete_rounds=complete_rounds))
        
        # 获取最终结果
        final_summary = ray.get(final_dispatcher.get_architectures.remote(format_type='summary'))
        final_accuracy = 0.0
        if isinstance(final_summary, dict):
            arch_dict = final_summary.get('architectures', {})
            if isinstance(arch_dict, dict) and f"{model_id}_final" in arch_dict:
                final_accuracy = arch_dict[f"{model_id}_final"].get('best_accuracy', 0.0)
        
        final_results[model_id] = {
            'model_info': model_info,
            'initial_accuracy': model_info['best_accuracy'],
            'final_accuracy': final_accuracy,
            'improvement': final_accuracy - model_info['best_accuracy']
        }
        
        logger.logger.info(f"模型 {model_id} 重训练完成:")
        logger.logger.info(f"  初始准确率: {model_info['best_accuracy']:.4f}")
        logger.logger.info(f"  最终准确率: {final_accuracy:.4f}")
        logger.logger.info(f"  提升幅度: {final_accuracy - model_info['best_accuracy']:.4f}")
        
        # 释放Dispatcher
        ray.kill(final_dispatcher)
    
    # 9. 输出最终结果
    logger.logger.info("步骤 9: 输出最终结果...")
    final_results_header = "="*25 + " GraphNAS 最终结果 " + "="*25
    logger.logger.info(final_results_header)
    
    # 找出最佳模型
    best_final_model = max(final_results.items(), key=lambda x: x[1]['final_accuracy'])
    best_model_id, best_model_result = best_final_model
    
    logger.logger.info(f"数据集名称: {dataset_name}")
    logger.logger.info(f"搜索策略: {nas_config['search_strategy']}")
    logger.logger.info(f"总搜索轮次: {nas_config['num_search_rounds']}")
    logger.logger.info(f"总评估模型数: {len(all_candidate_models)}")
    logger.logger.info(f"最终候选数: {final_candidates_count}")
    
    logger.logger.info(f"\n🔥 最佳模型: {best_model_id}")
    logger.logger.info(f"   来源轮次: {best_model_result['model_info']['round']}")
    logger.logger.info(f"   初始准确率: {best_model_result['initial_accuracy']:.4f}")
    logger.logger.info(f"   最终准确率: {best_model_result['final_accuracy']:.4f}")
    logger.logger.info(f"   准确率提升: {best_model_result['improvement']:.4f}")
    
    # 输出架构详情（使用新的格式化函数）
    formatted_arch = format_architecture_for_display(
        best_model_result['model_info']['arch_spec'], 
        nas_config['search_strategy']
    )
    logger.logger.info(f"\n🏆 最佳GNN架构:\n{formatted_arch}")
    logger.logger.info(f"🚀🚀🚀 最佳验证准确率: {best_model_result['final_accuracy']:.4f}")
    
    # 记录所有最终候选的详细结果
    logger.logger.info(f"\n=== 所有最终候选模型结果 ===")
    sorted_final = sorted(final_results.items(), key=lambda x: x[1]['final_accuracy'], reverse=True)
    for i, (model_id, result) in enumerate(sorted_final):
        logger.logger.info(f"第{i+1}名: {model_id}")
        logger.logger.info(f"  最终准确率: {result['final_accuracy']:.4f}")
        logger.logger.info(f"  初始准确率: {result['initial_accuracy']:.4f}")
        logger.logger.info(f"  准确率提升: {result['improvement']:.4f}")
    
    logger.logger.info("="*len(final_results_header))

    # 新增：将所有记录的指标保存到CSV文件
    csv_path = logger.save_metrics_to_csv()
    logger.logger.info(f"所有搜索指标已保存到: {csv_path}")

    # 8. 清理资源
    logger.logger.info("步骤 7: 关闭Ray，释放资源...")
    ray.shutdown()
    logger.logger.info("-> 流程结束。")

if __name__ == "__main__":
    main()
