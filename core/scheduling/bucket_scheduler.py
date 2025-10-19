from typing import List, Dict, Any
from .base import SchedulerInterface, TrainingTask, TrainingResult
from ..balancing import assign_to_buckets

class BucketScheduler(SchedulerInterface):

    def __init__(self, num_workers: int, target_rotations: int=None):
        super().__init__(num_workers)
        self.buckets: List[List[Dict]] = []
        self.bucket_loads: List[float] = []
        self.current_rotation = 0
        self.target_rotations = target_rotations
        self.completed_rotations = 0
        self.architecture_count = 0
        self.num_rounds_config = 0
        self.steps_per_epoch = 0
        self.epochs_per_round = 0
        self.total_batches_per_arch = 0
        self.batches_per_worker: List[int] = []

    def initialize(self, architectures: List[Dict[str, Any]], train_config: Dict[str, Any]=None) -> None:
        self.validate_architectures(architectures)
        self.buckets, self.bucket_loads = assign_to_buckets(architectures, self.num_workers)
        if train_config:
            self.num_rounds_config = train_config.get('num_rounds', train_config.get('num_eval_rounds', 2))
            self.steps_per_epoch = train_config.get('steps_per_epoch', 5)
            self.epochs_per_round = train_config.get('epochs_per_round', train_config.get('epochs_per_rotation', 5))
            self.total_batches_per_arch = self.num_rounds_config * self.steps_per_epoch * self.epochs_per_round
            base_batches = self.total_batches_per_arch // self.num_workers
            remainder = self.total_batches_per_arch % self.num_workers
            self.batches_per_worker = [base_batches] * self.num_workers
            for i in range(remainder):
                self.batches_per_worker[i] += 1
            print(f'\n=== Bucket模式固定训练量分配方案 ===')
            print(f'  - 配置参数: num_rounds={self.num_rounds_config}, steps_per_epoch={self.steps_per_epoch}, epochs_per_round={self.epochs_per_round}')
            print(f'  - 每个架构总batch数: {self.total_batches_per_arch}')
            print(f'  - 每个worker的batch分配: {self.batches_per_worker}')
            print(f'  - 桶轮转实现完整训练覆盖')
        else:
            self.batches_per_worker = [1] * self.num_workers
            print(f'\n⚠️  未提供训练配置，使用默认batch分配')
        self.current_rotation = 0
        self.completed_rotations = 0
        self.architecture_count = len(architectures)
        self.is_initialized = True
        print(f'\n桶式调度器已初始化:')
        print(f'  - 架构总数: {self.architecture_count}')
        print(f'  - 桶数量: {len(self.buckets)}')
        print(f"  - 桶负载分布: {[f'{load:.2f}' for load in self.bucket_loads]}")
        if self.target_rotations is not None:
            total_steps = self.target_rotations * self.num_workers
            print(f'  - 目标轮转次数: {self.target_rotations}（总调度步骤: {total_steps}）')
        else:
            print(f'  - 目标轮转次数: 无限制（由外部控制）')

    def schedule_next(self) -> List[TrainingTask]:
        if not self.is_initialized:
            raise RuntimeError('调度器未初始化')
        if not self.has_more_tasks():
            return []
        tasks = []
        for worker_id in range(self.num_workers):
            bucket_id = (worker_id + self.current_rotation) % self.num_workers
            bucket = self.buckets[bucket_id]
            batches_for_this_worker = self.batches_per_worker[worker_id] if self.batches_per_worker else 1
            for arch in bucket:
                task = TrainingTask(arch_id=arch['arch_id'], arch_spec=arch['arch_spec'], worker_id=worker_id, epochs=1, load=arch.get('load', 1.0), metadata={'bucket_id': bucket_id, 'rotation_step': self.current_rotation, 'complete_rotation_progress': f'{self.current_rotation % self.num_workers + 1}/{self.num_workers}', 'config_overrides': {'steps_per_epoch': int(batches_for_this_worker), 'epochs_per_rotation': 1}})
                tasks.append(task)
        print(f'调度第 {self.current_rotation + 1} 轮任务: {len(tasks)} 个任务 (桶轮转进度: {self.current_rotation % self.num_workers + 1}/{self.num_workers})')
        return tasks

    def update_results(self, results: List[TrainingResult]) -> None:
        if not results:
            return
        self.current_rotation += 1
        if self.current_rotation >= self.num_workers:
            self.current_rotation = 0
            self.completed_rotations += 1
        print(f'桶轮转进度: 步骤 {self.current_rotation}/{self.num_workers}, 完成轮次: {self.completed_rotations}')

    def has_more_tasks(self) -> bool:
        if not self.is_initialized or self.architecture_count == 0:
            return False
        if self.target_rotations is not None:
            return self.completed_rotations < self.target_rotations
        return True

    def get_progress(self) -> Dict[str, Any]:
        progress = {'scheduler_type': 'bucket', 'total_architectures': self.architecture_count, 'num_buckets': len(self.buckets), 'current_rotation_step': self.current_rotation, 'completed_rotations': self.completed_rotations, 'bucket_loads': self.bucket_loads, 'bucket_architecture_counts': [len(bucket) for bucket in self.buckets]}
        if self.total_batches_per_arch > 0:
            progress.update({'training_mode': 'fixed_total_batches', 'total_batches_per_arch': self.total_batches_per_arch, 'batches_per_worker': self.batches_per_worker, 'rotation_for_complete_coverage': self.num_workers, 'rotation_progress': f'{self.current_rotation % self.num_workers}/{self.num_workers}'})
        return progress

    def reset(self) -> None:
        self.current_rotation = 0
        self.completed_rotations = 0
        print('桶式调度器状态已重置')

    def get_bucket_assignment(self) -> List[List[Dict]]:
        return self.buckets.copy()

    def get_bucket_loads(self) -> List[float]:
        return self.bucket_loads.copy()

    def advance_rotation(self, steps: int=1) -> None:
        for _ in range(steps):
            self.current_rotation = (self.current_rotation + 1) % self.num_workers
            if self.current_rotation == 0:
                self.completed_rotations += 1
        print(f'手动推进轮转: 当前步骤 {self.current_rotation}, 完成轮次 {self.completed_rotations}')

    def get_current_assignment(self) -> Dict[int, int]:
        assignment = {}
        for worker_id in range(self.num_workers):
            bucket_id = (worker_id + self.current_rotation) % self.num_workers
            assignment[worker_id] = bucket_id
        return assignment