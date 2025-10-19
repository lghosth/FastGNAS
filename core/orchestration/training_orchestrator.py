import time
import ray
from typing import List, Dict, Any, Optional
from ..scheduling.base import SchedulerInterface, TrainingTask, TrainingResult

class TrainingOrchestrator:

    def __init__(self, workers: List[ray.ObjectRef], scheduler: SchedulerInterface, state_manager=None, registry=None):
        self.workers = workers
        self.scheduler = scheduler
        self.state_manager = state_manager
        self.registry = registry
        self.num_workers = len(workers)
        self.total_tasks_executed = 0
        self.total_training_time = 0.0
        self.failed_tasks = 0
        self.accumulated_timing = {'model_migration': 0.0, 'batch_preparation': 0.0, 'forward': 0.0, 'backward': 0.0, 'optimizer_step': 0.0, 'memory_cleanup': 0.0}
        self.last_accumulated_timing = {'model_migration': 0.0, 'batch_preparation': 0.0, 'forward': 0.0, 'backward': 0.0, 'optimizer_step': 0.0, 'memory_cleanup': 0.0}
        self.accumulated_sync_timing = {'checkpoint_load': 0.0, 'rotation_overhead': 0.0, 'checkpoint_save': 0.0, 'scheduler_feedback': 0.0}
        self.last_accumulated_sync_timing = {'checkpoint_load': 0.0, 'rotation_overhead': 0.0, 'checkpoint_save': 0.0, 'scheduler_feedback': 0.0}
        self.arch_iterations = {}
        self.verbose = False
        self.enable_state_save = True
        if self.verbose:
            print(f'TrainingOrchestrator初始化完成:')
            print(f'  - Worker数量: {self.num_workers}')
            print(f'  - 调度器类型: {scheduler.get_scheduler_type()}')
            print(f"  - 状态管理: {('启用' if state_manager else '禁用')}")

    def _get_incremental_timing_stats(self) -> Dict[str, float]:
        incremental_timing = {}
        for key in self.accumulated_timing.keys():
            current_value = self.accumulated_timing.get(key, 0.0)
            last_value = self.last_accumulated_timing.get(key, 0.0)
            incremental_timing[key] = current_value - last_value
            self.last_accumulated_timing[key] = current_value
        return incremental_timing

    def _execute_task_batch(self) -> Dict[str, float]:
        tasks = self.scheduler.schedule_next()
        if not tasks:
            print('没有待调度的训练任务')
            return {}
        round_start_time = time.perf_counter()
        total_checkpoint_load_time = 0.0
        futures = []
        task_mapping = {}
        for task in tasks:
            if task.worker_id >= self.num_workers:
                print(f'警告: 任务worker_id {task.worker_id} 超出worker数量 {self.num_workers}')
                raise ValueError(f'任务worker_id {task.worker_id} 超出worker数量 {self.num_workers}')
            state_dict = None
            if self.state_manager:
                checkpoint_io_start = time.perf_counter()
                state_dict = self.state_manager.load_checkpoint(task.arch_id)
                total_checkpoint_load_time += time.perf_counter() - checkpoint_io_start
            task_cfg = {}
            if isinstance(getattr(task, 'metadata', None), dict):
                task_cfg = task.metadata.get('config_overrides', {}) or {}
            future = self.workers[task.worker_id].train_model_on_partition.remote(arch_spec=task.arch_spec, state_dict=state_dict, config=task_cfg)
            futures.append(future)
            task_mapping[future] = task
        self.accumulated_sync_timing['checkpoint_load'] += total_checkpoint_load_time
        results = []
        task_accuracies = {}
        worker_timing_stats = []
        ray_get_times = []
        batch_checkpoint_save_time = 0.0
        for future in futures:
            task = task_mapping[future]
            try:
                ray_get_start = time.perf_counter()
                val_acc, updated_state, correct_count, incorrect_count, timing_stats = ray.get(future)
                ray_get_time = time.perf_counter() - ray_get_start
                ray_get_times.append(ray_get_time)
                training_time = ray_get_time
                if timing_stats and isinstance(timing_stats, dict):
                    worker_timing_stats.append(timing_stats)
                    for key in self.accumulated_timing.keys():
                        if key in timing_stats:
                            self.accumulated_timing[key] += timing_stats[key]
                result = TrainingResult(arch_id=task.arch_id, accuracy=val_acc, training_time=training_time, worker_id=task.worker_id, state_dict=updated_state, metrics={'correct_count': correct_count, 'incorrect_count': incorrect_count, 'epochs': 0.0})
                try:
                    result.metadata = getattr(task, 'metadata', {})
                except Exception:
                    pass
                results.append(result)
                task_accuracies[task.arch_id] = val_acc
                checkpoint_save_start = time.perf_counter()
                if self.enable_state_save and self.state_manager and (updated_state is not None):
                    if task.arch_id not in self.arch_iterations:
                        self.arch_iterations[task.arch_id] = 0
                    self.arch_iterations[task.arch_id] += 1
                    if self.state_manager.save_checkpoint(task.arch_id, updated_state, accuracy=val_acc):
                        pass
                elif not self.enable_state_save:
                    print(f'❌ 未保存 {task.arch_id}: enable_state_save=False')
                elif not self.state_manager:
                    print(f'❌ 未保存 {task.arch_id}: state_manager为None')
                elif updated_state is None:
                    print(f'❌ 未保存 {task.arch_id}: updated_state为None')
                checkpoint_save_time = time.perf_counter() - checkpoint_save_start
                batch_checkpoint_save_time += checkpoint_save_time
                self.total_tasks_executed += 1
            except Exception as e:
                if self.verbose:
                    print(f'    ❌ {task.arch_id}: 训练失败 - {e}')
                result = TrainingResult(arch_id=task.arch_id, accuracy=0.0, training_time=0.0, worker_id=task.worker_id, error=str(e))
                results.append(result)
                task_accuracies[task.arch_id] = 0.0
                self.failed_tasks += 1
                raise RuntimeError(f'训练任务 {task.arch_id} 失败: {e}')
        self.accumulated_sync_timing['checkpoint_save'] += batch_checkpoint_save_time
        scheduler_feedback_start = time.perf_counter()
        self.scheduler.update_results(results)
        scheduler_feedback_time = time.perf_counter() - scheduler_feedback_start
        self.accumulated_sync_timing['scheduler_feedback'] += scheduler_feedback_time
        batch_time = time.perf_counter() - round_start_time
        if task_accuracies:
            avg_acc = sum(task_accuracies.values()) / len(task_accuracies)
        wall_clock_train_time = 0.0
        old_phase_max_sum = 0.0
        if worker_timing_stats:
            timing_keys = ['model_migration', 'batch_preparation', 'forward', 'backward', 'optimizer_step', 'memory_cleanup']
            for key in timing_keys:
                max_time = max((w.get(key, 0.0) for w in worker_timing_stats), default=0.0)
                old_phase_max_sum += max_time
            per_worker_wall = [float(w.get('wall_clock_train_time', 0.0)) for w in worker_timing_stats]
            if any((x > 0 for x in per_worker_wall)):
                wall_clock_train_time = max(per_worker_wall)
            else:
                per_worker_totals = []
                for w in worker_timing_stats:
                    total = 0.0
                    for key in timing_keys:
                        total += float(w.get(key, 0.0))
                    per_worker_totals.append(total)
                wall_clock_train_time = max(per_worker_totals) if per_worker_totals else 0.0
        max_ray_get_time = max(ray_get_times) if ray_get_times else 0.0
        communication_overhead = max(0.0, max_ray_get_time - wall_clock_train_time)
        rotation_overhead = communication_overhead
        overhead_ratio = rotation_overhead / max_ray_get_time if max_ray_get_time > 0 else 0.0
        self.total_training_time += wall_clock_train_time
        self.accumulated_sync_timing['rotation_overhead'] += rotation_overhead
        batch_sync_overhead = {'checkpoint_load': total_checkpoint_load_time, 'rotation_overhead': rotation_overhead, 'checkpoint_save': batch_checkpoint_save_time, 'scheduler_feedback': scheduler_feedback_time, 'total': total_checkpoint_load_time + rotation_overhead + batch_checkpoint_save_time + scheduler_feedback_time}
        return (task_accuracies, wall_clock_train_time, old_phase_max_sum, batch_sync_overhead)

    def run_multiple_rounds(self, num_rounds: int, epochs_per_task: int=1) -> Dict[str, Any]:
        if not self.scheduler.is_ready():
            raise RuntimeError('调度器未准备就绪')
        scheduler_type = getattr(self.scheduler, '__class__', type(self.scheduler)).__name__
        if scheduler_type == 'DynamicBatchScheduler':
            return self._run_adaptive_rounds(num_rounds)
        if num_rounds <= 1:
            return self._run_single_round()
        return self._run_fixed_rounds(num_rounds)

    def _run_single_round(self) -> Dict[str, Any]:
        if self.verbose:
            print(f'\n=== 执行单轮训练 ===')
        start_time = time.perf_counter()
        final_accuracies, wall_clock_train_time, old_phase_max_sum, batch_sync_overhead = self._execute_task_batch()
        total_time = time.perf_counter() - start_time
        if self.verbose:
            print(f'单轮训练完成: 用时 {total_time:.2f}s, 训练了 {len(final_accuracies)} 个架构')
        incremental_sync_timing = {}
        for key in self.accumulated_sync_timing.keys():
            incremental_sync_timing[key] = self.accumulated_sync_timing[key] - self.last_accumulated_sync_timing[key]
            self.last_accumulated_sync_timing[key] = self.accumulated_sync_timing[key]
        return {'final_accuracies': final_accuracies, 'wall_clock_train_time': wall_clock_train_time, 'wall_clock_old_phase_max_sum': old_phase_max_sum, 'statistics': {'total_time': total_time, 'completed_rounds': 1 if final_accuracies else 0, 'total_tasks': len(final_accuracies), 'failed_tasks': self.failed_tasks, 'average_task_time': self.total_training_time / max(1, len(final_accuracies)), 'timing_stats': self._get_incremental_timing_stats(), 'sync_overhead': incremental_sync_timing}}

    def _run_adaptive_rounds(self, suggested_rounds: int) -> Dict[str, Any]:
        all_results = {}
        round_summaries = []
        round_num = 1
        total_wall_clock_train_time = 0.0
        total_old_phase_max_sum = 0.0
        max_safe_rounds = max(suggested_rounds * 3, 50)
        start_time = time.perf_counter()
        while self.scheduler.has_more_tasks():
            if round_num > max_safe_rounds:
                print(f'⚠️  达到安全轮次限制 {max_safe_rounds}，强制结束')
                print(f'   建议检查调度器逻辑或增加安全阀限制')
                break
            round_results, wall_clock_train_time, old_phase_max_sum, batch_sync_overhead = self._execute_task_batch()
            total_wall_clock_train_time += wall_clock_train_time
            total_old_phase_max_sum += old_phase_max_sum
            if not round_results:
                print(f'第 {round_num} 轮无有效结果，跳过')
                round_num += 1
                continue
            for arch_id, accuracy in round_results.items():
                if arch_id not in all_results:
                    all_results[arch_id] = []
                all_results[arch_id].append(accuracy)
            round_summary = {'round': round_num, 'tasks_count': len(round_results), 'best_accuracy': max(round_results.values()), 'average_accuracy': sum(round_results.values()) / len(round_results), 'execution_mode': 'adaptive'}
            round_summaries.append(round_summary)
            round_num += 1
        total_time = time.perf_counter() - start_time
        actual_rounds = round_num - 1
        total_architectures = len(all_results)
        print(f"\n{'=' * 60}")
        print(f'自适应轮次训练完成！')
        print(f'  建议轮数: {suggested_rounds}')
        print(f'  实际轮数: {actual_rounds}')
        print(f'  轮次倍数: {actual_rounds / suggested_rounds:.1f}x')
        print(f'  总用时: {total_time:.2f}s')
        print(f'  平均每轮: {total_time / actual_rounds:.2f}s')
        if total_architectures > 0:
            print(f'  平均每架构: {total_time / total_architectures:.2f}s')
        print(f"{'=' * 60}")
        final_accuracies = {}
        for arch_id, accuracy_list in all_results.items():
            final_accuracies[arch_id] = max(accuracy_list)
        incremental_sync_timing = {}
        for key in self.accumulated_sync_timing.keys():
            incremental_sync_timing[key] = self.accumulated_sync_timing[key] - self.last_accumulated_sync_timing[key]
            self.last_accumulated_sync_timing[key] = self.accumulated_sync_timing[key]
        return {'final_accuracies': final_accuracies, 'wall_clock_train_time': total_wall_clock_train_time, 'wall_clock_old_phase_max_sum': total_old_phase_max_sum, 'round_summaries': round_summaries, 'execution_stats': {'suggested_rounds': suggested_rounds, 'actual_rounds': actual_rounds, 'round_multiplier': actual_rounds / suggested_rounds if suggested_rounds > 0 else 0, 'total_time': total_time, 'execution_mode': 'adaptive'}, 'batches': [{'batch_arch_ids': list(final_accuracies.keys()), 'num_complete_rounds': actual_rounds, 'epochs_per_bucket': 'adaptive', 'num_workers': getattr(self.scheduler, 'num_workers', 'unknown'), 'execution_mode': 'adaptive'}], 'statistics': {'total_time': total_time, 'completed_rounds': actual_rounds, 'total_tasks': len(final_accuracies), 'failed_tasks': self.failed_tasks, 'average_task_time': self.total_training_time / max(1, len(final_accuracies)), 'timing_stats': self._get_incremental_timing_stats(), 'sync_overhead': incremental_sync_timing}}

    def _run_fixed_rounds(self, num_rounds: int) -> Dict[str, Any]:
        if self.verbose:
            print(f"\n{'=' * 60}")
            print(f'开始执行 {num_rounds} 轮训练（固定轮数模式）')
            print(f"{'=' * 60}")
        all_results = {}
        round_summaries = []
        start_time = time.perf_counter()
        total_wall_clock_train_time = 0.0
        total_old_phase_max_sum = 0.0
        for round_num in range(1, num_rounds + 1):
            if self.verbose:
                print(f'\n--- 第 {round_num}/{num_rounds} 轮 ---')
            if not self.scheduler.has_more_tasks():
                if self.verbose:
                    print(f'调度器已无更多任务，提前结束于第 {round_num - 1} 轮')
                break
            round_results, wall_clock_train_time, old_phase_max_sum, batch_sync_overhead = self._execute_task_batch()
            total_wall_clock_train_time += wall_clock_train_time
            total_old_phase_max_sum += old_phase_max_sum
            for arch_id, accuracy in round_results.items():
                if arch_id not in all_results:
                    all_results[arch_id] = []
                all_results[arch_id].append(accuracy)
            if round_results:
                round_summary = {'round': round_num, 'tasks_count': len(round_results), 'best_accuracy': max(round_results.values()), 'average_accuracy': sum(round_results.values()) / len(round_results)}
                round_summaries.append(round_summary)
        total_time = time.perf_counter() - start_time
        final_accuracies = {}
        for arch_id, accuracies in all_results.items():
            final_accuracies[arch_id] = max(accuracies)
        if self.verbose:
            print(f"\n{'=' * 60}")
            print(f'多轮训练完成!')
            print(f'总用时: {total_time:.2f}s')
            print(f'执行轮数: {len(round_summaries)}/{num_rounds}')
            print(f'总任务数: {self.total_tasks_executed}')
            if final_accuracies:
                print(f'最佳准确率: {max(final_accuracies.values()):.4f}')
            print(f"{'=' * 60}")
        incremental_sync_timing = {}
        for key in self.accumulated_sync_timing.keys():
            incremental_sync_timing[key] = self.accumulated_sync_timing[key] - self.last_accumulated_sync_timing[key]
            self.last_accumulated_sync_timing[key] = self.accumulated_sync_timing[key]
        return {'final_accuracies': final_accuracies, 'wall_clock_train_time': total_wall_clock_train_time, 'wall_clock_old_phase_max_sum': total_old_phase_max_sum, 'all_results': all_results, 'round_summaries': round_summaries, 'statistics': {'total_time': total_time, 'completed_rounds': len(round_summaries), 'total_tasks': self.total_tasks_executed, 'failed_tasks': self.failed_tasks, 'average_task_time': self.total_training_time / max(1, self.total_tasks_executed), 'timing_stats': self._get_incremental_timing_stats(), 'sync_overhead': incremental_sync_timing}}

    def run_training_round(self, epochs_per_task: int=1) -> Dict[str, float]:
        import warnings
        warnings.warn('run_training_round()已弃用，请直接使用run_multiple_rounds(1)', DeprecationWarning, stacklevel=2)
        result = self._run_single_round()
        return result['final_accuracies']

    def get_scheduler_progress(self) -> Dict[str, Any]:
        return self.scheduler.get_progress()

    def reset_scheduler(self) -> None:
        self.scheduler.reset()

    def global_evaluate(self, arch_id: str, enable_distributed_eval: bool=True) -> float:
        if not enable_distributed_eval:
            print(f'[全图评估] 使用传统评估方式评估架构 {arch_id}')
            return self._fallback_evaluate(arch_id)
        print(f'[全图评估] 开始对架构 {arch_id} 执行分布式评估...')
        if not self.registry:
            print(f'[全图评估] 错误: 没有可用的模型注册中心')
            raise ValueError('没有可用的模型注册中心')
        arch_entry = self.registry.get_entry(arch_id)
        if not arch_entry:
            print(f'[全图评估] 错误: 找不到架构 {arch_id} 的信息')
            raise ValueError(f'找不到架构 {arch_id} 的信息')
        arch_spec = arch_entry.arch_spec
        latest_state = None
        if self.state_manager:
            latest_state = self.state_manager.load_checkpoint(arch_id)
        print(f'[全图评估] 调用 {len(self.workers)} 个Worker执行分布式推理...')
        inference_futures = []
        start_time = time.perf_counter()
        for worker in self.workers:
            future = worker.distributed_inference.remote(arch_spec=arch_spec, state_dict=latest_state or {})
            inference_futures.append(future)
        all_predictions = []
        all_global_ids = []
        all_true_labels = []
        try:
            inference_results = ray.get(inference_futures)
            for worker_id, (predictions, global_ids, true_labels) in enumerate(inference_results):
                if len(predictions) > 0:
                    all_predictions.append(predictions)
                    all_global_ids.append(global_ids)
                    all_true_labels.append(true_labels)
                    print(f'[全图评估] Worker {worker_id}: 收集到 {len(predictions)} 个验证节点')
                else:
                    print(f'[全图评估] Worker {worker_id}: 无验证数据')
        except Exception as e:
            print(f'[全图评估] 收集推理结果时出错: {e}')
            raise RuntimeError('[全图评估] 收集推理结果时出错: no_validation_data_collected')
        if not all_predictions:
            print(f'[全图评估] 错误: 没有收集到任何验证数据')
            raise ValueError('[全图评估] 错误: 没有收集到任何验证数据')
        import torch
        merged_predictions = torch.cat(all_predictions, dim=0)
        merged_global_ids = torch.cat(all_global_ids, dim=0)
        merged_true_labels = torch.cat(all_true_labels, dim=0)
        unique_global_ids, unique_indices = torch.unique(merged_global_ids, return_inverse=True)
        if len(unique_global_ids) < len(merged_global_ids):
            duplicates = len(merged_global_ids) - len(unique_global_ids)
            duplicate_ratio = duplicates / len(merged_global_ids) * 100
            print(f'[全图评估] 检测到 {duplicates} 个重复节点（{duplicate_ratio:.1f}%），正在去重...')
            print(f'[全图评估] 提示: 分区使用了 val_mask，建议使用 eval_mask 以避免去重开销')
            final_predictions = torch.zeros(len(unique_global_ids), dtype=merged_predictions.dtype)
            final_true_labels = torch.zeros(len(unique_global_ids), dtype=merged_true_labels.dtype)
            for i, unique_id in enumerate(unique_global_ids):
                mask = merged_global_ids == unique_id
                final_predictions[i] = merged_predictions[mask][0]
                final_true_labels[i] = merged_true_labels[mask][0]
        else:
            print(f'[全图评估] 无重复节点，直接聚合（分区已使用 eval_mask，性能最优）')
            final_predictions = merged_predictions
            final_true_labels = merged_true_labels
        correct_predictions = (final_predictions == final_true_labels).sum().item()
        total_predictions = len(final_predictions)
        global_accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0.0
        eval_time = time.perf_counter() - start_time
        print(f'[全图评估] 架构 {arch_id} 全图评估完成:')
        print(f'  - 总验证节点数: {total_predictions}')
        print(f'  - 正确预测数: {correct_predictions}')
        print(f'  - 全图准确率: {global_accuracy:.4f}')
        print(f'  - 全图评估耗时: {eval_time:.4f} 秒')
        import gc
        try:
            should_delete_final_predictions = 'final_predictions' in locals() and final_predictions is not merged_predictions
            should_delete_final_labels = 'final_true_labels' in locals() and final_true_labels is not merged_true_labels
            del all_predictions, all_global_ids, all_true_labels
            del merged_predictions, merged_global_ids, merged_true_labels
            if should_delete_final_predictions:
                del final_predictions
            if should_delete_final_labels:
                del final_true_labels
            gc.collect()
            print(f'[全图评估] 已清理中间tensor，释放内存')
        except Exception as e:
            print(f'[全图评估] 内存清理警告: {e}')
        return global_accuracy

    def _fallback_evaluate(self, arch_id: str) -> float:
        print(f'[传统评估] 架构 {arch_id} 使用传统评估方式')
        return 0.0

    def get_statistics(self) -> Dict[str, Any]:
        return {'total_tasks_executed': self.total_tasks_executed, 'total_training_time': self.total_training_time, 'failed_tasks': self.failed_tasks, 'success_rate': (self.total_tasks_executed - self.failed_tasks) / max(1, self.total_tasks_executed), 'average_task_time': self.total_training_time / max(1, self.total_tasks_executed), 'scheduler_type': self.scheduler.get_scheduler_type(), 'num_workers': self.num_workers}

    def validate_setup(self) -> Dict[str, Any]:
        issues = []
        if not self.workers:
            issues.append('没有可用的Worker')
        if not self.scheduler.is_ready():
            issues.append('调度器未准备就绪')
        scheduler_workers = getattr(self.scheduler, 'num_workers', 0)
        if scheduler_workers != self.num_workers:
            issues.append(f'Worker数量不一致: 编排器({self.num_workers}) vs 调度器({scheduler_workers})')
        return {'valid': len(issues) == 0, 'issues': issues, 'num_workers': self.num_workers, 'scheduler_ready': self.scheduler.is_ready(), 'scheduler_type': self.scheduler.get_scheduler_type()}