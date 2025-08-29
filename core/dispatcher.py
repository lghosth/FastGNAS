# graphnas_dist/core/dispatcher.py (修改版)

import torch
import ray
from typing import List, Dict, Any, Tuple, Optional

# 导入项目模块
from data.partition import partition_training_graph
from trainer.worker import TrainerWorker
from core.balance_controller import BalanceController
from core.load_estimator import MloadPredictor

@ray.remote(num_gpus=0)  
class Dispatcher:
    """
    新版Dispatcher，负责编排"数据分区+模型轮转"流程。
    """
    
    def __init__(self, 
                 worker_refs: Optional[List[ray.ObjectRef]] = None,
                 train_config: Optional[Dict[str, Any]] = None,
                 balance_controller: Optional[BalanceController] = None,
                 load_predictor: Optional[MloadPredictor] = None,
                 enable_balancing: bool = False): # 训练配置
                 
        self.workers = worker_refs or []
        self.num_workers = len(self.workers)
        self.config = train_config or {}
        
        # 负载均衡相关
        self.balance_controller = balance_controller
        self.load_predictor = load_predictor
        self.enable_balancing = enable_balancing
        
        # model_pool 存储所有要训练的架构
        self.model_pool: List[Dict] = []
        # model_states 存储每个模型的最新参数和最佳表现
        # 格式: {arch_id: {'spec': ..., 'state': ..., 'best_acc': ..., 'load': ...}}
        self.model_states: Dict[str, Dict] = {}
        
        print(f"Dispatcher initialized for model rotation with {self.num_workers} workers.")
        print(f"Load balancing enabled: {self.enable_balancing}")
        if self.enable_balancing:
            print(f"Balance controller available: {self.balance_controller is not None}")
            print(f"Load predictor available: {self.load_predictor is not None}")

    def get_architectures(self, format_type: str = 'detailed', sort_by: str = 'best_acc') -> Dict[str, Any]:
        """
        返回所有架构的当前状态信息。
        
        Args:
            format_type (str): 返回格式类型
                - 'detailed': 详细信息（默认）
                - 'summary': 简要摘要信息  
                - 'states_only': 仅返回model_states
                - 'pool_only': 仅返回model_pool
            sort_by (str): 排序方式
                - 'best_acc': 按最佳准确率降序排列（默认）
                - 'arch_id': 按架构ID字母顺序排列
                - 'none': 不排序
        
        Returns:
            Dict[str, Any]: 包含架构状态信息的字典
        
        Examples:
            >>> dispatcher.get_architectures('summary')
            >>> dispatcher.get_architectures('detailed', sort_by='arch_id')
        """
        
        if format_type == 'states_only':
            return self.model_states
        
        if format_type == 'pool_only':
            return {'model_pool': self.model_pool}
        
        # 创建基础信息
        result = {
            'total_architectures': len(self.model_states),
            'active_models_in_pool': len(self.model_pool),
            'num_workers': self.num_workers,
            'architectures': {}
        }
        
        # 获取架构列表并排序
        arch_items = list(self.model_states.items())
        
        if sort_by == 'best_acc':
            arch_items.sort(key=lambda x: x[1]['best_acc'], reverse=True)
        elif sort_by == 'arch_id':
            arch_items.sort(key=lambda x: x[0])
        # 'none' 情况下不排序
        
        # 根据格式类型生成输出
        if format_type == 'summary':
            # 简要摘要格式
            result['architectures'] = {
                arch_id: {
                    'best_accuracy': info['best_acc'],
                    'has_trained_state': info['state'] is not None,
                    'is_in_pool': any(model['arch_id'] == arch_id for model in self.model_pool)
                }
                for arch_id, info in arch_items
            }
            
            # 添加统计信息
            result['statistics'] = {
                'best_overall_accuracy': max([info['best_acc'] for info in self.model_states.values()], default=0.0),
                'average_accuracy': sum([info['best_acc'] for info in self.model_states.values()]) / len(self.model_states) if self.model_states else 0.0,
                'trained_models': sum(1 for info in self.model_states.values() if info['state'] is not None),
                'untrained_models': sum(1 for info in self.model_states.values() if info['state'] is None)
            }
            
        else:  # 'detailed' 格式（默认）
            # 详细信息格式
            result['architectures'] = {
                arch_id: {
                    'architecture_spec': info['spec'],
                    'best_accuracy': info['best_acc'],
                    'has_trained_state': info['state'] is not None,
                    'state_dict_size': len(info['state']) if info['state'] is not None else 0,
                    'is_in_pool': any(model['arch_id'] == arch_id for model in self.model_pool),
                    'pool_position': next((i for i, model in enumerate(self.model_pool) if model['arch_id'] == arch_id), -1)
                }
                for arch_id, info in arch_items
            }
            
            # 添加详细统计信息
            accuracies = [info['best_acc'] for info in self.model_states.values()]
            result['statistics'] = {
                'best_overall_accuracy': max(accuracies, default=0.0),
                'worst_accuracy': min(accuracies, default=0.0),
                'average_accuracy': sum(accuracies) / len(accuracies) if accuracies else 0.0,
                'accuracy_std': torch.std(torch.tensor(accuracies)).item() if len(accuracies) > 1 else 0.0,
                'trained_models': sum(1 for info in self.model_states.values() if info['state'] is not None),
                'untrained_models': sum(1 for info in self.model_states.values() if info['state'] is None)
            }
            
            # 添加模型池信息
            result['model_pool_info'] = {
                'current_rotation_order': [model['arch_id'] for model in self.model_pool],
                'next_to_train': self.model_pool[0]['arch_id'] if self.model_pool else None
            }
            
            # 添加配置信息
            result['training_config'] = self.config.copy() if self.config else {}
            
        return result

    def submit_architectures(self, architectures: List[Dict]):
        """API: 提交一批要训练的架构 (每个包含 'arch_id', 'arch_spec')"""
        for arch in architectures:
            if arch['arch_id'] not in self.model_states:
                self.model_pool.append(arch)
                self.model_states[arch['arch_id']] = {
                    'spec': arch['arch_spec'],
                    'state': None, # 初始状态为空
                    'best_acc': 0.0,
                    'load': arch.get('load', 0.0)  # 支持负载信息
                }
        # 将模型数量扩展到与worker数量一致，便于轮转
        while len(self.model_pool) > 0 and len(self.model_pool) < self.num_workers:
            self.model_pool.extend(self.model_pool)
        self.model_pool = self.model_pool[:self.num_workers]
    
    def submit_balanced_architectures(self) -> int:
        """
        使用负载均衡策略生成并提交架构
        
        Returns:
            int: 实际生成的架构数量
        """
        if not self.enable_balancing or self.balance_controller is None:
            print("警告: 负载均衡未启用或BalanceController未设置")
            return 0
        
        print("\n=== 使用负载均衡策略生成架构 ===")
        
        # 使用BalanceController生成负载均衡的架构
        balanced_architectures = self.balance_controller.generate_balanced_architectures()
        
        # 转换格式并提交
        architectures_to_submit = []
        for arch in balanced_architectures:
            arch_dict = {
                'arch_id': arch['arch_id'],
                'arch_spec': arch['arch_spec'],
                'load': arch['load']
            }
            architectures_to_submit.append(arch_dict)
        
        self.submit_architectures(architectures_to_submit)
        
        print(f"已提交 {len(architectures_to_submit)} 个负载均衡的架构")
        return len(architectures_to_submit)

    def run_training_rounds(self, num_complete_rounds: int):
        """
        执行指定轮数的完整训练。
        
        完整训练的定义：每个模型都在所有数据分区上训练一次。
        例如：有2个Worker时，一次完整训练意味着每个模型都在2个数据分区上各训练一次。
        
        Args:
            num_complete_rounds (int): 完整训练的轮数。每轮完整训练后，所有模型都会在所有分区上训练一次。
        """
        if not self.model_pool:
            print("No models in the pool to train.")
            return

        # 计算总的底层轮转次数: 完整轮数 × Worker数量
        total_rotation_rounds = num_complete_rounds * self.num_workers
        
        print(f"\n=== 开始 {num_complete_rounds} 轮完整训练 ===")
        print(f"每轮完整训练包含 {self.num_workers} 次轮转")
        print(f"总计将执行 {total_rotation_rounds} 次底层轮转")
        
        # 跟踪每个完整轮次的完成情况
        for complete_round in range(1, num_complete_rounds + 1):
            print(f"\n{'='*60}")
            print(f"🚀 开始第 {complete_round}/{num_complete_rounds} 轮完整训练")
            print(f"{'='*60}")
            
            # 在一轮完整训练中，需要进行 num_workers 次底层轮转
            # 这样每个模型都能在每个Worker的数据分区上训练一次
            for rotation_step in range(self.num_workers):
                rotation_round = (complete_round - 1) * self.num_workers + rotation_step + 1
                print(f"\n--- 底层轮转 {rotation_round}/{total_rotation_rounds} (完整轮次 {complete_round}, 步骤 {rotation_step + 1}/{self.num_workers}) ---")
                
                # 1. 准备并分发任务
                futures = []
                for i, worker in enumerate(self.workers):
                    # 根据当前轮转位置确定要训练的模型
                    model_to_train = self.model_pool[i]
                    arch_id = model_to_train['arch_id']
                    model_info = self.model_states[arch_id]
                    
                    print(f"  - Worker {i} will train model '{arch_id}' on partition {i}.")
                    
                    future = worker.train_model_on_partition.remote(
                        arch_spec=model_info['spec'],
                        state_dict=model_info['state'],
                        config=self.config
                    )
                    futures.append((arch_id, future))
                
                # 2. 收集结果并更新模型状态
                for arch_id, future in futures:
                    val_acc, updated_state = ray.get(future)
                    self.model_states[arch_id]['state'] = updated_state
                    if val_acc > self.model_states[arch_id]['best_acc']:
                        self.model_states[arch_id]['best_acc'] = val_acc
                        print(f"  🎉 New best accuracy for '{arch_id}': {val_acc:.4f}")

                # 3. 执行模型轮转 (环形轮转) - 但不在完整轮次的最后一步轮转
                if rotation_step < self.num_workers - 1:
                    self.model_pool.insert(0, self.model_pool.pop())
                    print(f"  - Model rotation complete. Next assignment: {self.model_pool[0]['arch_id']} -> Worker 0.")
            
            print(f"\n✅ 第 {complete_round}/{num_complete_rounds} 轮完整训练完成！")
            print(f"   所有模型都已在 {self.num_workers} 个数据分区上训练过一次。")
            
            # 完整轮次结束后，恢复到初始轮转状态
            if complete_round < num_complete_rounds:
                # 确保下一轮完整训练从相同的模型分配开始
                print(f"   准备下一轮完整训练...")
        
        print(f"\n{'='*60}")
        # 训练轮次
        final_epochs = num_complete_rounds * int(self.config.get('epochs_per_rotation', 5)) * int(self.config.get('steps_per_epoch', 1))
        print(f"🎉 所有 {num_complete_rounds} 轮完整训练已完成！")
        print(f"   每个模型总共训练了 {num_complete_rounds * self.num_workers * final_epochs} 次")
        print(f"   每个模型在每个数据分区上训练了 {int(self.config.get('epochs_per_rotation', 5)) * int(self.config.get('steps_per_epoch', 1))} 次")
        print(f"{'='*60}")


if __name__ == '__main__':
    import sys
    import os
    import ray
    import torch
    import pprint  # 用于美化输出结果

    # 动态添加项目根目录到Python路径，确保跨文件夹的导入能够正常工作
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.append(project_root)

    from data.partition import partition_training_graph
    from trainer.worker import TrainerWorker

    print("===== 运行 Dispatcher 独立集成测试 =====")

    # --- 测试配置 ---
    DATASET = 'Cora'
    # 自动检测可用GPU数量，若无GPU则模拟2个CPU Worker
    NUM_WORKERS = int(torch.cuda.device_count()) if torch.cuda.is_available() else 2
    MAX_HOPS = 3  # 必须大于或等于所有模型中的最大层数
    TRAINING_ROUNDS = 10  # 总共执行10轮训练
    EPOCHS_PER_ROTATION = 5  # 每个模型在单个GPU上训练的轮数
    LEARNING_RATE = 0.01

    if NUM_WORKERS == 0:
        print("警告: 未检测到GPU。将使用2个CPU Worker进行模拟。")
        NUM_WORKERS = 2

    # --- 定义要测试的两个模型架构 ---
    # 这个结构必须与你的 `builder.py` 实现相匹配
    architectures_to_test = [
        {'arch_id': 'GCN-2-layer', 'arch_spec': {'layers': [{'type': 'gcn'}] * 2}},
        {'arch_id': 'GCN-3-layer', 'arch_spec': {'layers': [{'type': 'gcn'}] * 3}},
    ]

    try:
        # 1. 初始化 Ray
        print(f"\n[步骤 1] 初始化 Ray ({NUM_WORKERS} 个 Worker)...")
        # 如果有GPU，为Ray分配GPU资源
        if torch.cuda.is_available():
            ray.init(num_gpus=NUM_WORKERS, ignore_reinit_error=True)
        else:
            ray.init(num_cpus=NUM_WORKERS, ignore_reinit_error=True)
        print(f"   -> Ray 初始化成功. 可用资源: {ray.available_resources()}")

        # 2. 对数据集进行静态分区
        print(f"\n[步骤 2] 为 {NUM_WORKERS} 个 Worker 划分 '{DATASET}' 数据集...")
        partitions = partition_training_graph(DATASET, NUM_WORKERS, MAX_HOPS)
        print(f"   -> 数据集已划分为 {len(partitions)} 个部分。")

        # 3. 创建 TrainerWorker Actors
        print(f"\n[步骤 3] 创建 {NUM_WORKERS} 个 TrainerWorker...")
        workers = []
        for i in range(NUM_WORKERS):
            use_gpu = torch.cuda.is_available()
            # 为每个Worker指定一个GPU（如果可用）
            worker_actor = TrainerWorker.options(num_gpus=1 if use_gpu else 0).remote(
                worker_id=i,
                # 正确传递device_id，CPU模式下为'cpu'
                gpu_id=i if use_gpu else -1, # CPU worker 使用 -1
                data_partition=partitions[i]
            )
            workers.append(worker_actor)
        print(f"   -> {len(workers)} 个 Worker 创建成功。")

        # 4. 创建 Dispatcher Actor
        print("\n[步骤 4] 创建 Dispatcher...")
        train_config = {'epochs_per_rotation': EPOCHS_PER_ROTATION, 'lr': LEARNING_RATE}
        # 使用关键字参数来正确初始化Dispatcher
        dispatcher = Dispatcher.remote(worker_refs=workers, train_config=train_config)
        print("   -> Dispatcher 创建成功。")

        # 5. 向 Dispatcher 提交待评估的架构
        print(f"\n[步骤 5] 提交 {len(architectures_to_test)} 个模型架构...")
        ray.get(dispatcher.submit_architectures.remote(architectures_to_test))
        print("   -> 模型架构提交成功。")

        # 6. 运行10轮训练
        print(f"\n[步骤 6] 开始执行 {TRAINING_ROUNDS} 轮训练...")
        # 这10轮训练任务将被分发给池中的2个模型。
        # 每个模型大约会被调度训练5次，每次都在不同的Worker上。
        ray.get(dispatcher.run_training_rounds.remote(num_complete_rounds=TRAINING_ROUNDS))
        print("   -> 训练执行完毕。")

        # 7. 获取并展示最终结果
        print("\n[步骤 7] 获取并展示最终评估结果...")
        final_results = ray.get(dispatcher.get_architectures.remote())

        print("\n" + "="*50)
        print("           最终模型性能评估结果")
        print("="*50)
        pprint.pprint(final_results)
        print("="*50)

    except Exception as e:
        print(f"\n❌ 测试过程中发生错误: {e}")
        import traceback
        traceback.print_exc()

    finally:
        # 8. 关闭 Ray
        print("\n[步骤 8] 关闭 Ray...")
        ray.shutdown()
        print("   -> Ray 已关闭。测试结束。")



# # ====================
# #  独立的测试模块
# # ====================
# if __name__ == '__main__':
#     print("===== Running Dispatcher Test =====")
    
#     try:
#         # 初始化Ray (假设有2个GPU)
#         print("1. 初始化Ray...")
#         ray.init(num_gpus=2, ignore_reinit_error=True)
#         print(f"   Ray initialized. Available resources: {ray.available_resources()}")
        
#         # 1. 数据分区
#         print("\n2. 数据分区...")
#         num_gpus = int(ray.available_resources().get("GPU", 0))
#         if num_gpus == 0:
#             print("   警告: 没有检测到GPU，使用CPU模式")
#             num_gpus = 2  # 强制使用2个worker进行测试
        
#         dataset_name = 'Cora'
#         print(f"   分区数据集 '{dataset_name}' 为 {num_gpus} 个分区...")
#         partitions = partition_training_graph(dataset_name, num_gpus, 2)
#         print(f"   数据分区完成: {len(partitions)} 个分区")
        
#         # 2. 创建Workers，每个Worker拥有一个数据分区
#         print("\n3. 创建Workers...")
#         workers = []
#         for i in range(num_gpus):
#             print(f"   创建 Worker {i}...")
#             worker = TrainerWorker.remote(i, i, partitions[i])
#             workers.append(worker)
#         print(f"   {len(workers)} 个Workers创建完成")
        
#         # 3. 初始化并配置Dispatcher
#         print("\n4. 初始化Dispatcher...")
#         config = {'epochs_per_rotation': 5, 'lr': 0.01}  # 减少epoch数量便于测试
#         dispatcher = Dispatcher.remote(None, workers, config)
#         print("   Dispatcher初始化完成")
        
#         # 4. 提交架构
#         print("\n5. 提交架构...")
#         architectures = [
#             {'arch_id': 'GCN-1', 'arch_spec': {'layers': [{'type':'gcn'}]*2}},
#             {'arch_id': 'GCN-2', 'arch_spec': {'layers': [{'type':'gcn'}]*2}},
#         ]
#         ray.get(dispatcher.submit_architectures.remote(architectures))
#         print(f"   {len(architectures)} 个架构提交完成")
        
#         # 5. 运行训练
#         print(f"\n6. 运行训练 (共{num_gpus}轮)...")
#         ray.get(dispatcher.run_training_rounds.remote(num_complete_rounds=num_gpus))
#         print("   训练完成!")
        
#         print("\n===== 测试成功完成! =====")
        
#     except Exception as e:
#         print(f"\n❌ 测试过程中发生错误: {e}")
#         import traceback
#         traceback.print_exc()
        
#     finally:
#         print("\n7. 清理资源...")
#         try:
#             ray.shutdown()
#             print("   Ray已关闭")
#         except Exception as e:
#             print(f"   关闭Ray时出错: {e}")
        
#         print("===== 测试结束 =====")