from typing import List, Dict, Any
from .base import SchedulerInterface, TrainingTask, TrainingResult
from ..balancing import assign_to_buckets

class UnifiedScheduler(SchedulerInterface):

    def __init__(self, num_workers: int, use_buckets: bool=True):
        super().__init__(num_workers)
        self.use_buckets = use_buckets
        self.architectures: List[Dict] = []
        self.current_step = 0
        self.completed_rounds = 0
        self.buckets: List[List[Dict]] = []
        self.bucket_loads: List[float] = []
        self.rotation_queue: List[Dict] = []

    def initialize(self, architectures: List[Dict[str, Any]]) -> None:
        self.validate_architectures(architectures)
        self.architectures = architectures
        if self.use_buckets:
            self.buckets, self.bucket_loads = assign_to_buckets(architectures, self.num_workers)
            print(f'桶模式初始化: {len(architectures)} 个架构分配到 {len(self.buckets)} 个桶')
            print(f"桶负载: {[f'{load:.2f}' for load in self.bucket_loads]}")
        else:
            self.rotation_queue = architectures.copy()
            while len(self.rotation_queue) < self.num_workers and self.rotation_queue:
                self.rotation_queue.extend(self.rotation_queue[:self.num_workers - len(self.rotation_queue)])
            print(f'轮转模式初始化: {len(architectures)} 个架构，队列长度 {len(self.rotation_queue)}')
        self.current_step = 0
        self.completed_rounds = 0
        self.is_initialized = True

    def schedule_next(self) -> List[TrainingTask]:
        if not self.is_initialized:
            raise RuntimeError('调度器未初始化')
        tasks = []
        if self.use_buckets:
            for worker_id in range(self.num_workers):
                bucket_id = (worker_id + self.current_step) % self.num_workers
                if bucket_id < len(self.buckets):
                    bucket = self.buckets[bucket_id]
                    for arch in bucket:
                        task = TrainingTask(arch_id=arch['arch_id'], arch_spec=arch['arch_spec'], worker_id=worker_id, epochs=1, load=arch.get('load', 1.0), metadata={'bucket_id': bucket_id, 'rotation_step': self.current_step})
                        tasks.append(task)
        else:
            for worker_id in range(min(self.num_workers, len(self.rotation_queue))):
                arch = self.rotation_queue[worker_id]
                task = TrainingTask(arch_id=arch['arch_id'], arch_spec=arch['arch_spec'], worker_id=worker_id, epochs=1, load=arch.get('load', 1.0), metadata={'queue_position': worker_id, 'round': self.completed_rounds})
                tasks.append(task)
        print(f'调度 {len(tasks)} 个任务 (步骤 {self.current_step + 1})')
        return tasks

    def update_results(self, results: List[TrainingResult]) -> None:
        if not results:
            return
        if self.use_buckets:
            self.current_step += 1
            if self.current_step >= self.num_workers:
                self.current_step = 0
                self.completed_rounds += 1
        else:
            if len(self.rotation_queue) > 1:
                self.rotation_queue.insert(0, self.rotation_queue.pop())
            self.completed_rounds += 1
        print(f"调度更新: {('桶' if self.use_buckets else '轮转')}模式, 步骤 {self.current_step}, 完成轮次 {self.completed_rounds}")

    def has_more_tasks(self) -> bool:
        if self.use_buckets:
            return self.is_initialized and len(self.buckets) > 0
        else:
            return self.is_initialized and len(self.rotation_queue) > 0

    def get_progress(self) -> Dict[str, Any]:
        base_info = {'scheduler_type': 'unified', 'mode': 'bucket' if self.use_buckets else 'rotation', 'total_architectures': len(self.architectures), 'current_step': self.current_step, 'completed_rounds': self.completed_rounds}
        if self.use_buckets:
            base_info.update({'num_buckets': len(self.buckets), 'bucket_loads': self.bucket_loads, 'bucket_sizes': [len(bucket) for bucket in self.buckets]})
        else:
            base_info.update({'queue_length': len(self.rotation_queue), 'queue_head': self.rotation_queue[0]['arch_id'] if self.rotation_queue else None})
        return base_info

    def reset(self) -> None:
        self.current_step = 0
        self.completed_rounds = 0
        print(f"{('桶' if self.use_buckets else '轮转')}调度器已重置")

    def get_scheduler_type(self) -> str:
        return f"UnifiedScheduler({('Bucket' if self.use_buckets else 'Rotation')}Mode)"