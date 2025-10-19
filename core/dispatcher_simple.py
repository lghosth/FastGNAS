import ray
import time
import warnings
from typing import List, Dict, Any, Optional
from datetime import datetime
from .scheduling.unified_scheduler import UnifiedScheduler
from .registry.model_registry import ModelRegistry
from .state.state_manager import StateManager
from .balance_controller import BalanceController
from .scheduling.base import TrainingResult

@ray.remote(num_gpus=0)
class SimpleDispatcher:

    def __init__(self, worker_refs: List[ray.ObjectRef], train_config: Dict[str, Any], use_buckets: bool=True, enable_state_management: bool=True, balance_controller: Optional[BalanceController]=None):
        self.workers = worker_refs
        self.num_workers = len(worker_refs)
        self.config = train_config
        self.balance_controller = balance_controller
        self.scheduler = UnifiedScheduler(self.num_workers, use_buckets)
        self.registry = ModelRegistry()
        self.state_manager = StateManager() if enable_state_management else None
        self.is_initialized = False
        self.total_tasks = 0
        self.failed_tasks = 0
        print(f'SimpleDispatcher初始化:')
        print(f'  - Workers: {self.num_workers}')
        print(f"  - 模式: {('桶式' if use_buckets else '轮转')}")
        print(f"  - 状态管理: {('启用' if enable_state_management else '禁用')}")

    def submit_architectures(self, architectures: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not architectures:
            return {'error': '架构列表为空'}
        for arch in architectures:
            self.registry.register(arch_id=arch['arch_id'], arch_spec=arch['arch_spec'], load=arch.get('load', 1.0))
        self.scheduler.initialize(architectures)
        self.is_initialized = True
        return {'submitted': len(architectures), 'mode': 'bucket' if self.scheduler.use_buckets else 'rotation'}

    def run_training_round(self, epochs: int=1) -> Dict[str, float]:
        if not self.is_initialized:
            raise RuntimeError('请先提交架构')
        tasks = self.scheduler.schedule_next()
        if not tasks:
            return {}
        print(f'\n执行训练轮次: {len(tasks)} 个任务')
        futures = []
        for task in tasks:
            state_dict = None
            if self.state_manager:
                state_dict = self.state_manager.load_checkpoint(task.arch_id)
            future = self.workers[task.worker_id].train_model_on_partition.remote(arch_spec=task.arch_spec, state_dict=state_dict, config={'epochs_per_rotation': epochs})
            futures.append((task, future))
        results = []
        accuracies = {}
        for task, future in futures:
            try:
                val_acc, updated_state, correct, incorrect = ray.get(future)
                if self.state_manager and updated_state:
                    self.state_manager.save_checkpoint(task.arch_id, updated_state, self.total_tasks)
                self.registry.update_performance(task.arch_id, val_acc, epochs)
                results.append(TrainingResult(arch_id=task.arch_id, accuracy=val_acc, training_time=0.0, worker_id=task.worker_id))
                accuracies[task.arch_id] = val_acc
                self.total_tasks += 1
                print(f'  ✓ {task.arch_id}: {val_acc:.4f}')
            except Exception as e:
                print(f'  ✗ {task.arch_id}: {e}')
                self.failed_tasks += 1
                accuracies[task.arch_id] = 0.0
        self.scheduler.update_results(results)
        return accuracies

    def run_multiple_rounds(self, num_rounds: int, epochs_per_round: int=1) -> Dict[str, Any]:
        print(f'\n开始 {num_rounds} 轮训练')
        all_results = {}
        best_accuracies = {}
        for round_num in range(1, num_rounds + 1):
            print(f'\n--- 第 {round_num}/{num_rounds} 轮 ---')
            round_results = self.run_training_round(epochs_per_round)
            for arch_id, acc in round_results.items():
                if arch_id not in all_results:
                    all_results[arch_id] = []
                all_results[arch_id].append(acc)
                if arch_id not in best_accuracies or acc > best_accuracies[arch_id]:
                    best_accuracies[arch_id] = acc
        return {'best_accuracies': best_accuracies, 'all_results': all_results, 'total_tasks': self.total_tasks, 'failed_tasks': self.failed_tasks, 'registry_summary': self.registry.get_summary()}

    def get_results(self) -> Dict[str, Any]:
        return {'top_models': self.registry.get_top_k(10), 'summary': self.registry.get_summary(), 'scheduler_progress': self.scheduler.get_progress(), 'state_stats': self.state_manager.get_statistics() if self.state_manager else None}

    def reset(self) -> None:
        self.scheduler.reset()
        self.registry.clear()
        if self.state_manager:
            self.state_manager.clear_cache()
        self.is_initialized = False
        self.total_tasks = 0
        self.failed_tasks = 0

    def submit_balanced_architectures_with_buckets(self, **kwargs) -> Dict[str, Any]:
        warnings.warn('使用submit_architectures代替', DeprecationWarning)
        if self.balance_controller:
            architectures = self.balance_controller.generate_balanced_architectures()
            return self.submit_architectures(architectures)
        return {'error': '未设置BalanceController'}

    def run_bucket_based_training_rounds(self, num_complete_rounds: int, epochs_per_bucket: int=5) -> None:
        warnings.warn('使用run_multiple_rounds代替', DeprecationWarning)
        self.run_multiple_rounds(num_complete_rounds * self.num_workers, epochs_per_bucket)

    def get_architectures(self, **kwargs) -> Dict[str, Any]:
        warnings.warn('使用get_results代替', DeprecationWarning)
        return self.get_results()