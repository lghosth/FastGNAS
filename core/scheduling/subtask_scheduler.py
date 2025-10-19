import math
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple
from abc import ABC, abstractmethod
from .base import SchedulerInterface, TrainingTask, TrainingResult
from ..balancing import assign_to_buckets

@dataclass
class ModelPartitionQuota:
    arch_id: str
    arch_spec: Dict[str, Any]
    total_load: float
    standard_batches_per_partition: int
    completed_batches_this_partition: int
    batch_load: float
    batches_per_round: int

    @property
    def remaining_batches_this_partition(self) -> int:
        return max(0, self.standard_batches_per_partition - self.completed_batches_this_partition)

    @property
    def rounds_needed_this_partition(self) -> int:
        if self.batches_per_round <= 0:
            return 0
        return math.ceil(self.remaining_batches_this_partition / self.batches_per_round)

    @property
    def is_completed_this_partition(self) -> bool:
        return self.completed_batches_this_partition >= self.standard_batches_per_partition

    def get_next_round_batches(self) -> int:
        remaining = self.remaining_batches_this_partition
        planned = self.batches_per_round
        return min(remaining, planned)

@dataclass
class RoundTrainingPlan:
    round_number: int
    partition_id: int
    active_models: List[str]
    model_batch_allocation: Dict[str, int]
    expected_load_distribution: Dict[str, float]
    total_expected_load: float
    load_balance_score: float

class ConfigBasedSplitter:

    def __init__(self, epochs_per_rotation: int=5, steps_per_epoch: int=10, load_variance_threshold: float=0.3, num_rounds: int=5, load_tolerance: float=0.1, st_cfg: Optional[Dict[str, Any]]=None):
        self.epochs_per_rotation = epochs_per_rotation
        self.steps_per_epoch = steps_per_epoch
        self.base_batches = 1
        self.eval_rounds = num_rounds
        self.total_training_batches = epochs_per_rotation * steps_per_epoch * num_rounds
        self.load_variance_threshold = load_variance_threshold
        self.num_rounds = num_rounds
        self.load_tolerance = load_tolerance
        self.min_batch_per_round = st_cfg.get('min_batch_per_round', 3) if st_cfg else 3
        self.max_batch_per_round = st_cfg.get('max_batch_per_round', 25) if st_cfg else 25

    def analyze_load_variance(self, models: List[Dict[str, Any]]) -> Tuple[bool, float, Dict[str, float]]:
        if not models:
            return (False, 0.0, {})
        batch_loads = {}
        total_loads = []
        for model in models:
            total_load = model.get('load', 1.0)
            batch_load = total_load / self.total_training_batches
            batch_loads[model['arch_id']] = batch_load
            total_loads.append(total_load)
        max_load = max(total_loads)
        min_load = min(total_loads)
        avg_load = sum(total_loads) / len(total_loads)
        load_variance_ratio = (max_load - min_load) / avg_load if avg_load > 0 else 0
        should_split = load_variance_ratio > self.load_variance_threshold
        return (should_split, load_variance_ratio, batch_loads)

    def calculate_batch_allocation(self, models: List[Dict[str, Any]], num_workers: int=1) -> List[ModelPartitionQuota]:
        if not models:
            return []
        total_loads = []
        relative_loads = {}
        for model in models:
            total_load = model.get('load', 1.0)
            total_loads.append(total_load)
        min_load = min(total_loads)
        for model in models:
            arch_id = model['arch_id']
            total_load = model.get('load', 1.0)
            relative_load = total_load / min_load
            relative_loads[arch_id] = relative_load
        tolerance = 0.1
        groups = self._group_by_tolerance(relative_loads, tolerance)
        for gid, arch_ids in groups.items():
            avg_load = sum((relative_loads[aid] for aid in arch_ids)) / len(arch_ids)
        optimal_allocations = self._calculate_optimal_allocation(groups, relative_loads)
        batches_per_partition = math.ceil(self.total_training_batches / num_workers)
        quotas = []
        for model in models:
            arch_id = model['arch_id']
            total_load = model.get('load', 1.0)
            batch_load = total_load / self.total_training_batches
            batches_per_round = optimal_allocations[arch_id]
            quota = ModelPartitionQuota(arch_id=arch_id, arch_spec=model.get('arch_spec', {}), total_load=total_load, standard_batches_per_partition=batches_per_partition, completed_batches_this_partition=0, batch_load=batch_load, batches_per_round=batches_per_round)
            quotas.append(quota)
        return quotas

    def _group_by_tolerance(self, relative_loads: Dict[str, float], tolerance: float) -> Dict[int, List[str]]:
        groups = {}
        group_id = 0
        sorted_items = sorted(relative_loads.items(), key=lambda x: x[1])
        for arch_id, load in sorted_items:
            assigned = False
            for gid, arch_ids in groups.items():
                group_avg = sum((relative_loads[aid] for aid in arch_ids)) / len(arch_ids)
                if abs(load - group_avg) / group_avg <= tolerance:
                    groups[gid].append(arch_id)
                    assigned = True
                    break
            if not assigned:
                groups[group_id] = [arch_id]
                group_id += 1
        return groups

    def _calculate_optimal_allocation(self, groups: Dict[int, List[str]], relative_loads: Dict[str, float]) -> Dict[str, int]:
        group_avg_loads = {}
        for gid, arch_ids in groups.items():
            avg_load = sum((relative_loads[aid] for aid in arch_ids)) / len(arch_ids)
            group_avg_loads[gid] = avg_load
        min_avg_load = min(group_avg_loads.values())
        allocation_ratios = {}
        for gid, avg_load in group_avg_loads.items():
            ratio = min_avg_load / avg_load
            allocation_ratios[gid] = ratio
        min_batches = max(1, self.min_batch_per_round)
        max_batches = max(min_batches, self.max_batch_per_round)
        allocations = {}
        for gid, arch_ids in groups.items():
            ratio = allocation_ratios[gid]
            raw_batches = max_batches * ratio
            if raw_batches >= max_batches:
                batch_count = max_batches
            else:
                batch_count = math.floor(raw_batches)
                if batch_count < min_batches:
                    batch_count = min_batches
            for arch_id in arch_ids:
                allocations[arch_id] = batch_count
        return allocations

class PartitionProgressTracker:

    def __init__(self):
        self.partition_quotas: Dict[int, Dict[str, ModelPartitionQuota]] = {}
        self.round_histories: List[RoundTrainingPlan] = []

    def initialize_partition(self, partition_id: int, quotas: List[ModelPartitionQuota]):
        if partition_id not in self.partition_quotas:
            self.partition_quotas[partition_id] = {}
        for quota in quotas:
            partition_quota = ModelPartitionQuota(arch_id=quota.arch_id, arch_spec=quota.arch_spec, total_load=quota.total_load, standard_batches_per_partition=quota.standard_batches_per_partition, completed_batches_this_partition=0, batch_load=quota.batch_load, batches_per_round=quota.batches_per_round)
            self.partition_quotas[partition_id][quota.arch_id] = partition_quota

    def update_progress(self, partition_id: int, model_id: str, completed_batches: int):
        if partition_id in self.partition_quotas and model_id in self.partition_quotas[partition_id]:
            quota = self.partition_quotas[partition_id][model_id]
            old_completed = quota.completed_batches_this_partition
            quota.completed_batches_this_partition = completed_batches
        else:
            print(f'警告: 分区 {partition_id} 中找不到模型 {model_id}')

    def get_partition_quotas(self, partition_id: int) -> Dict[str, ModelPartitionQuota]:
        return self.partition_quotas.get(partition_id, {})

    def get_active_models_in_partition(self, partition_id: int) -> List[ModelPartitionQuota]:
        if partition_id not in self.partition_quotas:
            return []
        active_models = []
        for quota in self.partition_quotas[partition_id].values():
            if not quota.is_completed_this_partition:
                active_models.append(quota)
        return active_models

    def record_round_plan(self, plan: RoundTrainingPlan):
        self.round_histories.append(plan)

    def get_partition_progress_summary(self, partition_id: int) -> Dict[str, Any]:
        if partition_id not in self.partition_quotas:
            return {'error': f'分区 {partition_id} 未初始化'}
        quotas = self.partition_quotas[partition_id]
        total_models = len(quotas)
        completed_models = sum((1 for q in quotas.values() if q.is_completed_this_partition))
        progress_details = {}
        for model_id, quota in quotas.items():
            progress_details[model_id] = {'completed_batches': quota.completed_batches_this_partition, 'total_batches': quota.standard_batches_per_partition, 'completion_ratio': quota.completed_batches_this_partition / quota.standard_batches_per_partition, 'is_completed': quota.is_completed_this_partition, 'batches_per_round': quota.batches_per_round, 'rounds_needed': quota.rounds_needed_this_partition}
        return {'partition_id': partition_id, 'total_models': total_models, 'completed_models': completed_models, 'completion_ratio': completed_models / total_models if total_models > 0 else 0, 'model_details': progress_details}

class DynamicBatchScheduler(SchedulerInterface):

    def __init__(self, num_workers: int, epochs_per_rotation: int=5, steps_per_epoch: int=10, load_variance_threshold: float=0.3, num_rounds: int=5, st_cfg: Optional[Dict[str, Any]]=None):
        super().__init__(num_workers)
        self.splitter = ConfigBasedSplitter(epochs_per_rotation=epochs_per_rotation, steps_per_epoch=steps_per_epoch, load_variance_threshold=load_variance_threshold, num_rounds=num_rounds, load_tolerance=st_cfg.get('load_tolerance', 0.1) if st_cfg else 0.1, st_cfg=st_cfg)
        self.progress_tracker = PartitionProgressTracker()
        self.current_round = 0
        self.partition_assignments = {}
        self.bucket_arch_ids: List[set] = []
        self.bucket_loads: List[float] = []
        self.min_batch_per_round = st_cfg.get('min_batch_per_round', 3) if st_cfg else 3
        self.max_batch_per_round = st_cfg.get('max_batch_per_round', 25) if st_cfg else 25

    def initialize(self, architectures: List[Dict[str, Any]], train_config: Dict[str, Any]=None) -> None:
        self.validate_architectures(architectures)
        if train_config:
            num_rounds_config = train_config.get('num_rounds', train_config.get('num_eval_rounds', 2))
            steps_per_epoch = train_config.get('steps_per_epoch', 5)
            epochs_per_round = train_config.get('epochs_per_round', train_config.get('epochs_per_rotation', 5))
            self.splitter.base_batches = 1
            self.splitter.eval_rounds = num_rounds_config
            self.splitter.total_training_batches = steps_per_epoch * epochs_per_round * num_rounds_config
        quotas = self.splitter.calculate_batch_allocation(architectures, num_workers=self.num_workers)
        buckets, bucket_loads = assign_to_buckets(architectures, self.num_workers)
        self.bucket_arch_ids = [set((a['arch_id'] for a in bucket)) for bucket in buckets]
        self.bucket_loads = bucket_loads
        for partition_id in range(self.num_workers):
            self.progress_tracker.initialize_partition(partition_id, quotas)
            self.partition_assignments[partition_id] = [arch['arch_id'] for arch in architectures]
        self.current_round = 0
        self.is_initialized = True

    def schedule_next(self) -> List[TrainingTask]:
        if not self.is_initialized:
            raise RuntimeError('调度器未初始化')
        if not self.has_more_tasks():
            return []
        tasks = []
        self.current_round += 1
        for partition_id in range(self.num_workers):
            active_models = self.progress_tracker.get_active_models_in_partition(partition_id)
            rotation_step = self.current_round - 1
            bucket_id = (partition_id + rotation_step) % self.num_workers
            if self.bucket_arch_ids:
                active_models = [q for q in active_models if q.arch_id in self.bucket_arch_ids[bucket_id]]
            if not active_models:
                print(f'  分区 {partition_id}: 无活跃模型')
                continue
            partition_tasks = []
            model_batch_allocation = {}
            expected_load_distribution = {}
            total_load = 0.0
            for quota in active_models:
                batch_count = quota.get_next_round_batches()
                expected_load = batch_count * quota.batch_load
                if batch_count > 0:
                    task = TrainingTask(arch_id=quota.arch_id, arch_spec=quota.arch_spec, worker_id=partition_id, epochs=1, load=expected_load, metadata={'batch_count': int(batch_count), 'partition_id': partition_id, 'round_number': self.current_round, 'bucket_id': bucket_id, 'is_subtask_split': True, 'config_overrides': {'steps_per_epoch': int(batch_count), 'epochs_per_rotation': 1}})
                    partition_tasks.append(task)
                    model_batch_allocation[quota.arch_id] = batch_count
                    expected_load_distribution[quota.arch_id] = expected_load
                    total_load += expected_load
            tasks.extend(partition_tasks)
            loads = list(expected_load_distribution.values())
            if loads:
                load_balance_score = min(loads) / max(loads) if max(loads) > 0 else 1.0
            else:
                load_balance_score = 1.0
            plan = RoundTrainingPlan(round_number=self.current_round, partition_id=partition_id, active_models=list(model_batch_allocation.keys()), model_batch_allocation=model_batch_allocation, expected_load_distribution=expected_load_distribution, total_expected_load=total_load, load_balance_score=load_balance_score)
            self.progress_tracker.record_round_plan(plan)
        return tasks

    def update_results(self, results: List[TrainingResult]) -> None:
        if not results:
            print('警告: 训练结果列表为空')
            raise ValueError('训练结果列表为空')
            return
        for result in results:
            metadata = getattr(result, 'metadata', {})
            if not metadata:
                raise ValueError('训练结果缺少metadata')
                continue
            partition_id = metadata.get('partition_id')
            batch_count = metadata.get('batch_count', 0)
            if partition_id is not None and batch_count > 0:
                quotas = self.progress_tracker.get_partition_quotas(partition_id)
                if result.arch_id in quotas:
                    old_completed = quotas[result.arch_id].completed_batches_this_partition
                    new_completed = old_completed + batch_count
                    self.progress_tracker.update_progress(partition_id, result.arch_id, new_completed)

    def has_more_tasks(self) -> bool:
        if not self.is_initialized:
            return False
        for partition_id in range(self.num_workers):
            active_models = self.progress_tracker.get_active_models_in_partition(partition_id)
            if active_models:
                return True
        return False

    def get_progress(self) -> Dict[str, Any]:
        progress_info = {'scheduler_type': 'dynamic_batch_subtask', 'current_round': self.current_round, 'num_workers': self.num_workers, 'is_initialized': self.is_initialized, 'partition_progress': {}}
        for partition_id in range(self.num_workers):
            partition_summary = self.progress_tracker.get_partition_progress_summary(partition_id)
            progress_info['partition_progress'][partition_id] = partition_summary
        total_models = 0
        total_completed = 0
        for partition_info in progress_info['partition_progress'].values():
            if 'total_models' in partition_info:
                total_models += partition_info['total_models']
                total_completed += partition_info['completed_models']
        progress_info['global_completion_ratio'] = total_completed / total_models if total_models > 0 else 0
        return progress_info

    def reset(self) -> None:
        self.current_round = 0
        self.progress_tracker = PartitionProgressTracker()
        self.partition_assignments = {}
        self.is_initialized = False

    def get_configuration_summary(self) -> Dict[str, Any]:
        return {'scheduler_type': 'DynamicBatchScheduler', 'num_workers': self.num_workers, 'epochs_per_rotation': self.splitter.epochs_per_rotation, 'steps_per_epoch': self.splitter.steps_per_epoch, 'total_training_batches': self.splitter.total_training_batches, 'base_batches': self.splitter.base_batches, 'load_variance_threshold': self.splitter.load_variance_threshold, 'num_rounds': self.splitter.num_rounds, 'load_tolerance': self.splitter.load_tolerance}

def create_test_models() -> List[Dict[str, Any]]:
    return [{'arch_id': 'model_high_load', 'arch_spec': {'layers': [{'type': 'gat'}] * 3}, 'load': 100.0}, {'arch_id': 'model_medium_load', 'arch_spec': {'layers': [{'type': 'gcn'}] * 2}, 'load': 50.0}, {'arch_id': 'model_low_load', 'arch_spec': {'layers': [{'type': 'gcn'}] * 1}, 'load': 30.0}]

def demo_intelligent_splitting():
    print('=' * 60)
    print('智能子任务切分模块演示')
    print('=' * 60)
    test_models = create_test_models()
    scheduler = DynamicBatchScheduler(num_workers=3, epochs_per_rotation=5, steps_per_epoch=10, load_variance_threshold=0.3, num_rounds=40.0)
    scheduler.initialize(test_models)
    config_summary = scheduler.get_configuration_summary()
    round_count = 0
    max_rounds = 10
    while scheduler.has_more_tasks() and round_count < max_rounds:
        round_count += 1
        tasks = scheduler.schedule_next()
        if not tasks:
            break
        results = []
        for task in tasks:
            batch_count = task.metadata.get('batch_count', 0)
            result = TrainingResult(arch_id=task.arch_id, accuracy=0.85 + round_count * 0.01, training_time=batch_count * 0.1, worker_id=task.worker_id)
            result.metadata = task.metadata
            results.append(result)
        scheduler.update_results(results)
    final_progress = scheduler.get_progress()
if __name__ == '__main__':
    demo_intelligent_splitting()