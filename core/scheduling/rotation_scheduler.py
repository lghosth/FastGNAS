from typing import List, Dict, Any
from .base import SchedulerInterface, TrainingTask, TrainingResult

class RotationScheduler(SchedulerInterface):

    def __init__(self, num_workers: int):
        super().__init__(num_workers)
        self.model_queue: List[Dict] = []
        self.current_round = 0
        self.total_rounds = 0
        self.architecture_count = 0
        self.num_rounds_config = 0
        self.steps_per_epoch = 0
        self.epochs_per_round = 0
        self.total_batches_per_arch = 0
        self.batches_per_worker: List[int] = []
        self.complete_rotation_steps = 0

    def initialize(self, architectures: List[Dict[str, Any]], train_config: Dict[str, Any]=None) -> None:
        self.validate_architectures(architectures)
        self.model_queue = architectures.copy()
        self.architecture_count = len(architectures)
        while len(self.model_queue) < self.num_workers and self.model_queue:
            self.model_queue.extend(self.model_queue[:self.num_workers - len(self.model_queue)])
        if len(self.model_queue) > self.num_workers:
            self.model_queue = self.model_queue[:self.num_workers]
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
            self.complete_rotation_steps = self.num_workers
            print(f'\n=== 固定训练量分配方案 ===')
            print(f'  - 配置参数: num_rounds={self.num_rounds_config}, steps_per_epoch={self.steps_per_epoch}, epochs_per_round={self.epochs_per_round}')
            print(f'  - 每个架构总batch数: {self.total_batches_per_arch}')
            print(f'  - 每个worker的batch分配: {self.batches_per_worker}')
            print(f'  - 完整轮转需要调度步数: {self.complete_rotation_steps}')
        else:
            self.batches_per_worker = [1] * self.num_workers
            self.complete_rotation_steps = self.num_workers
            print(f'\n⚠️  未提供训练配置，使用默认batch分配')
        self.current_round = 0
        self.is_initialized = True
        print(f'\n轮转调度器已初始化:')
        print(f'  - 原始架构数: {self.architecture_count}')
        print(f'  - 队列长度: {len(self.model_queue)}')
        print(f'  - Worker数量: {self.num_workers}')

    def schedule_next(self) -> List[TrainingTask]:
        if not self.is_initialized:
            raise RuntimeError('调度器未初始化')
        if not self.model_queue:
            return []
        tasks = []
        for worker_id in range(min(self.num_workers, len(self.model_queue))):
            arch = self.model_queue[worker_id]
            rotation_offset = self.current_round % self.num_workers
            actual_worker_idx = (worker_id - rotation_offset) % self.num_workers
            batches_for_this_task = self.batches_per_worker[actual_worker_idx] if self.batches_per_worker else 1
            task = TrainingTask(arch_id=arch['arch_id'], arch_spec=arch['arch_spec'], worker_id=worker_id, epochs=1, load=arch.get('load', 1.0), metadata={'queue_position': worker_id, 'round': self.current_round, 'rotation_step': self.current_round, 'complete_rotation_progress': f'{self.current_round % self.complete_rotation_steps + 1}/{self.complete_rotation_steps}', 'config_overrides': {'steps_per_epoch': int(batches_for_this_task), 'epochs_per_rotation': 1}})
            tasks.append(task)
        print(f'调度第 {self.current_round + 1} 轮任务: {len(tasks)} 个任务 (完整轮转进度: {self.current_round % self.complete_rotation_steps + 1}/{self.complete_rotation_steps})')
        return tasks

    def update_results(self, results: List[TrainingResult]) -> None:
        if not results:
            return
        if len(self.model_queue) > 1:
            self.model_queue.insert(0, self.model_queue.pop())
        self.current_round += 1
        print(f"模型轮转完成: 轮次 {self.current_round}, 下一轮首个模型: {self.model_queue[0]['arch_id']}")

    def has_more_tasks(self) -> bool:
        if not self.is_initialized or not self.model_queue:
            return False
        if self.complete_rotation_steps > 0:
            rotation_complete = self.current_round >= self.complete_rotation_steps
            return not rotation_complete
        if self.total_rounds > 0:
            return self.current_round < self.total_rounds
        return True

    def get_progress(self) -> Dict[str, Any]:
        progress = {'scheduler_type': 'rotation', 'total_architectures': self.architecture_count, 'queue_length': len(self.model_queue), 'current_round': self.current_round, 'total_rounds': self.total_rounds, 'queue_order': [arch['arch_id'] for arch in self.model_queue]}
        if self.complete_rotation_steps > 0:
            progress.update({'training_mode': 'fixed_total_batches', 'total_batches_per_arch': self.total_batches_per_arch, 'batches_per_worker': self.batches_per_worker, 'complete_rotation_steps': self.complete_rotation_steps, 'rotation_progress': f'{self.current_round}/{self.complete_rotation_steps}', 'rotation_complete': self.current_round >= self.complete_rotation_steps})
        return progress

    def reset(self) -> None:
        self.current_round = 0
        print('轮转调度器状态已重置')

    def set_total_rounds(self, rounds: int) -> None:
        self.total_rounds = rounds
        print(f"设置总轮次限制: {(rounds if rounds > 0 else '无限制')}")

    def get_model_queue(self) -> List[Dict]:
        return self.model_queue.copy()

    def rotate_manually(self, steps: int=1) -> None:
        for _ in range(steps):
            if len(self.model_queue) > 1:
                self.model_queue.insert(0, self.model_queue.pop())
                self.current_round += 1
        print(f"手动轮转 {steps} 步: 当前轮次 {self.current_round}, 首个模型: {self.model_queue[0]['arch_id']}")

    def get_current_assignment(self) -> Dict[int, str]:
        assignment = {}
        for worker_id in range(min(self.num_workers, len(self.model_queue))):
            assignment[worker_id] = self.model_queue[worker_id]['arch_id']
        return assignment