import torch
import ray
import time
from copy import deepcopy
from typing import List, Dict, Any, Tuple, Optional
from data.partition import PartitionGraph
from trainer.worker import TrainerWorker
from core.balance_controller import BalanceController
from core.load_estimator import MloadPredictor
from core.balancing import assign_to_buckets, get_bucket_assignment_summary

@ray.remote(num_gpus=0)
class Dispatcher:

    def __init__(self, worker_refs: Optional[List[ray.ObjectRef]]=None, train_config: Optional[Dict[str, Any]]=None, balance_controller: Optional[BalanceController]=None, load_predictor: Optional[MloadPredictor]=None, enable_balancing: bool=True):
        self.workers = worker_refs or []
        self.num_workers = len(self.workers)
        self.config = train_config or {}
        self.balance_controller = balance_controller
        self.load_predictor = load_predictor
        self.enable_balancing = enable_balancing
        self.model_pool: List[Dict] = []
        self.model_states: Dict[str, Dict] = {}
        self.bucket_assignments: List[List[Dict]] = []
        self.bucket_loads: List[float] = []
        self.bucket_rotation_order: List[int] = []
        self.current_bucket_index: int = 0
        self.use_bucket_mode: bool = False
        print(f'Dispatcher initialized for model rotation with {self.num_workers} workers.')
        print(f'Load balancing enabled: {self.enable_balancing}')
        if self.enable_balancing:
            print(f'Balance controller available: {self.balance_controller is not None}')
            print(f'Load predictor available: {self.load_predictor is not None}')

    def warmup_balancing(self, batch_size: Optional[int]=None, min_architectures: Optional[int]=None) -> Dict[str, Any]:
        result: Dict[str, Any] = {'ok': True, 'reason': '', 'steps': {'generate': 0.0, 'assign': 0.0, 'submit': 0.0, 'predictor_forward': 0.0}, 'batch_size': 0, 'min_architectures': 0}
        if not self.enable_balancing or self.balance_controller is None:
            result['ok'] = False
            result['reason'] = 'balancing_disabled_or_no_controller'
            print('WARMUP: 负载均衡未启用或BalanceController未设置，跳过预热')
            return result
        original_model_pool = list(self.model_pool)
        original_model_state_keys = set(self.model_states.keys())
        original_bucket_assignments = list(self.bucket_assignments)
        original_bucket_loads = list(self.bucket_loads)
        original_use_bucket_mode = self.use_bucket_mode
        try:
            bs = batch_size if isinstance(batch_size, int) and batch_size > 0 else max(2, self.num_workers)
            ma = min_architectures if isinstance(min_architectures, int) and min_architectures > 0 else max(1, self.num_workers)
            if hasattr(self.balance_controller, 'batch_size'):
                self.balance_controller.batch_size = bs
            if hasattr(self.balance_controller, 'min_architectures'):
                self.balance_controller.min_architectures = ma
            result['batch_size'] = bs
            result['min_architectures'] = ma
            print(f'WARMUP: 开始预热（batch_size={bs}, min_architectures={ma}）...')
            t0 = time.time()
            balanced_architectures = self.balance_controller.generate_balanced_architectures()
            t1 = time.time()
            result['steps']['generate'] = t1 - t0
            architectures_to_submit = []
            for arch in balanced_architectures:
                architectures_to_submit.append({'arch_id': arch['arch_id'], 'arch_spec': arch['arch_spec'], 'load': arch['load']})
            t2 = time.time()
            warm_buckets, warm_bucket_loads = assign_to_buckets(architectures_to_submit, max(1, self.num_workers))
            t3 = time.time()
            result['steps']['assign'] = t3 - t2
            t4 = time.time()
            self.submit_architectures(architectures_to_submit)
            t5 = time.time()
            result['steps']['submit'] = t5 - t4
            pred_forward_cost = 0.0
            if self.load_predictor is not None:
                dummy_sample = {'architecture': [{'activation': 'relu', 'attention_type': 'gcn', 'aggregation_type': 'mean', 'attention_heads': 1, 'hidden_units': 64}], 'graph': {'vertices': 100, 'edges': 500, 'avg_degree': 5.0, 'diameter': 10}, 'device': {'gpu_cores': 2048, 'core_frequency': 1.0, 'fp_capability': 6.0}}
                t6 = time.time()
                try:
                    _ = self.load_predictor.predict(dummy_sample)
                except Exception as _e:
                    pass
                pred_forward_cost = time.time() - t6
            result['steps']['predictor_forward'] = pred_forward_cost
            print(f"WARMUP: 完成 生成={result['steps']['generate']:.4f}s, 分配={result['steps']['assign']:.4f}s, 提交={result['steps']['submit']:.4f}s, 预测器前向={result['steps']['predictor_forward']:.4f}s")
        except Exception as e:
            result['ok'] = False
            result['reason'] = str(e)
            print(f'WARMUP: 预热失败: {e}')
        finally:
            try:
                new_keys = set(self.model_states.keys()) - original_model_state_keys
                for k in new_keys:
                    self.model_states.pop(k, None)
                self.model_pool = original_model_pool
                self.bucket_assignments = original_bucket_assignments
                self.bucket_loads = original_bucket_loads
                self.use_bucket_mode = original_use_bucket_mode
            except Exception:
                pass
        return result

    def get_architectures(self, format_type: str='detailed', sort_by: str='best_acc') -> Dict[str, Any]:
        if format_type == 'states_only':
            return self.model_states
        if format_type == 'pool_only':
            return {'model_pool': self.model_pool}
        result = {'total_architectures': len(self.model_states), 'active_models_in_pool': len(self.model_pool), 'num_workers': self.num_workers, 'architectures': {}}
        arch_items = list(self.model_states.items())
        if sort_by == 'best_acc':
            arch_items.sort(key=lambda x: x[1]['best_acc'], reverse=True)
        elif sort_by == 'arch_id':
            arch_items.sort(key=lambda x: x[0])
        if format_type == 'summary':
            result['architectures'] = {arch_id: {'best_accuracy': info['best_acc'], 'has_trained_state': info['state'] is not None, 'is_in_pool': any((model['arch_id'] == arch_id for model in self.model_pool))} for arch_id, info in arch_items}
            result['statistics'] = {'best_overall_accuracy': max([info['best_acc'] for info in self.model_states.values()], default=0.0), 'average_accuracy': sum([info['best_acc'] for info in self.model_states.values()]) / len(self.model_states) if self.model_states else 0.0, 'trained_models': sum((1 for info in self.model_states.values() if info['state'] is not None)), 'untrained_models': sum((1 for info in self.model_states.values() if info['state'] is None))}
        else:
            result['architectures'] = {arch_id: {'architecture_spec': info['spec'], 'best_accuracy': info['best_acc'], 'has_trained_state': info['state'] is not None, 'state_dict_size': len(info['state']) if info['state'] is not None else 0, 'is_in_pool': any((model['arch_id'] == arch_id for model in self.model_pool)), 'pool_position': next((i for i, model in enumerate(self.model_pool) if model['arch_id'] == arch_id), -1)} for arch_id, info in arch_items}
            accuracies = [info['best_acc'] for info in self.model_states.values()]
            result['statistics'] = {'best_overall_accuracy': max(accuracies, default=0.0), 'worst_accuracy': min(accuracies, default=0.0), 'average_accuracy': sum(accuracies) / len(accuracies) if accuracies else 0.0, 'accuracy_std': torch.std(torch.tensor(accuracies)).item() if len(accuracies) > 1 else 0.0, 'trained_models': sum((1 for info in self.model_states.values() if info['state'] is not None)), 'untrained_models': sum((1 for info in self.model_states.values() if info['state'] is None))}
            result['model_pool_info'] = {'current_rotation_order': [model['arch_id'] for model in self.model_pool], 'next_to_train': self.model_pool[0]['arch_id'] if self.model_pool else None}
            result['training_config'] = self.config.copy() if self.config else {}
        return result

    def submit_architectures(self, architectures: List[Dict]):
        for arch in architectures:
            if arch['arch_id'] not in self.model_states:
                self.model_pool.append(arch)
                self.model_states[arch['arch_id']] = {'spec': arch['arch_spec'], 'state': None, 'best_acc': 0.0, 'load': arch.get('load', 0.0)}
        while len(self.model_pool) > 0 and len(self.model_pool) < self.num_workers:
            self.model_pool.extend(self.model_pool)
        self.model_pool = self.model_pool[:self.num_workers]

    def submit_balanced_architectures(self, batch_size: int=50, min_architectures: int=10) -> int:
        if not self.enable_balancing or self.balance_controller is None:
            print('警告: 负载均衡未启用或BalanceController未设置')
            raise ValueError(f'警告: 负载均衡未启用或BalanceController未设置')
        if hasattr(self.balance_controller, 'batch_size'):
            self.balance_controller.batch_size = batch_size
            print(f'设置批量生成数量: {batch_size}')
        if hasattr(self.balance_controller, 'min_architectures'):
            self.balance_controller.min_architectures = min_architectures
            print(f'设置最小架构数量: {min_architectures}')
        print(f'\n=== 使用改进的负载均衡策略生成架构 ===')
        print(f'批量生成数量: {batch_size}')
        print(f'最小架构数量: {min_architectures}')
        balanced_architectures = self.balance_controller.generate_balanced_architectures()
        architectures_to_submit = []
        for arch in balanced_architectures:
            arch_dict = {'arch_id': arch['arch_id'], 'arch_spec': arch['arch_spec'], 'load': arch['load']}
            architectures_to_submit.append(arch_dict)
        self.submit_architectures(architectures_to_submit)
        print(f'已提交 {len(architectures_to_submit)} 个负载均衡的架构')
        return len(architectures_to_submit)

    def submit_balanced_architectures_with_buckets(self, batch_size: int=20, min_architectures: int=8) -> Dict[str, Any]:
        if not self.enable_balancing or self.balance_controller is None:
            print('警告: 负载均衡未启用或BalanceController未设置')
            raise ValueError(f'警告: 负载均衡未启用或BalanceController未设置')
        if hasattr(self.balance_controller, 'batch_size'):
            self.balance_controller.batch_size = batch_size
            print(f'[{self.submit_balanced_architectures_with_buckets.__name__}] 调试信息--设置批量生成数量: {batch_size}')
        if hasattr(self.balance_controller, 'min_architectures'):
            self.balance_controller.min_architectures = min_architectures
            print(f'[{self.submit_balanced_architectures_with_buckets.__name__}] 调试信息--设置最小架构数量: {min_architectures}')
        print(f'\n=== 使用改进的负载均衡策略生成架构（桶模式） ===')
        print(f'[{self.submit_balanced_architectures_with_buckets.__name__}] 调试信息--批量生成数量: {batch_size}')
        print(f'[{self.submit_balanced_architectures_with_buckets.__name__}] 调试信息--最小架构数量: {min_architectures}')
        balanced_architectures = self.balance_controller.generate_balanced_architectures()
        print(f'[{self.submit_balanced_architectures_with_buckets.__name__}] 调试信息--生成负载均衡的架构的数量: {len(balanced_architectures)}')
        architectures_to_submit = []
        for arch in balanced_architectures:
            arch_dict = {'arch_id': arch['arch_id'], 'arch_spec': arch['arch_spec'], 'load': arch['load']}
            architectures_to_submit.append(arch_dict)
        self.bucket_assignments, self.bucket_loads = assign_to_buckets(architectures_to_submit, self.num_workers)
        self.bucket_rotation_order = list(range(self.num_workers))
        self.current_bucket_index = 0
        self.use_bucket_mode = True
        self.submit_architectures(architectures_to_submit)
        bucket_summary = get_bucket_assignment_summary(self.bucket_assignments, self.bucket_loads)
        print(f'已提交 {len(architectures_to_submit)} 个负载均衡的架构到 {self.num_workers} 个桶')
        print(f'桶分配摘要: {bucket_summary}')
        return {'architectures_submitted': len(architectures_to_submit), 'bucket_assignments': self.bucket_assignments, 'bucket_loads': self.bucket_loads, 'bucket_summary': bucket_summary}

    def set_fixed_architectures_for_training(self, bucket_assignments: List[List[Dict]]):
        print(f'直接设置固定架构到 {len(bucket_assignments)} 个桶')
        if not bucket_assignments:
            return {'error': '桶分配不能为空'}
        self.model_pool = []
        self.model_states = {}
        self.bucket_assignments = bucket_assignments
        self.bucket_loads = []
        for bucket in bucket_assignments:
            bucket_load = sum((arch.get('load', 0.0) for arch in bucket))
            self.bucket_loads.append(bucket_load)
        self.bucket_rotation_order = list(range(len(bucket_assignments)))
        self.current_bucket_index = 0
        self.use_bucket_mode = True
        all_architectures = []
        for bucket in bucket_assignments:
            all_architectures.extend(bucket)
        self.submit_architectures(all_architectures)
        bucket_summary = get_bucket_assignment_summary(self.bucket_assignments, self.bucket_loads)
        print(f'已设置 {len(all_architectures)} 个固定架构到 {len(bucket_assignments)} 个桶')
        print(f'桶分配摘要: {bucket_summary}')
        return {'architectures_set': len(all_architectures), 'bucket_assignments': self.bucket_assignments, 'bucket_loads': self.bucket_loads, 'bucket_summary': bucket_summary}

    def run_bucket_based_training_rounds(self, num_complete_rounds: int, epochs_per_bucket: int=5):
        if not self.use_bucket_mode:
            print('警告: 未启用桶模式，请先使用 submit_balanced_architectures_with_buckets 提交架构')
            raise ValueError('未启用桶模式，请先使用 submit_balanced_architectures_with_buckets 提交架构')
            return
        if not self.bucket_assignments:
            print('警告: 没有桶分配信息')
            raise ValueError('没有桶分配信息')
            return
        print(f'\n=== 开始基于桶的训练轮转 ===')
        print(f'完整训练轮数: {num_complete_rounds}')
        print(f'每桶训练epochs: {epochs_per_bucket}')
        print(f'桶分配信息: {[len(bucket) for bucket in self.bucket_assignments]}')
        for complete_round in range(1, num_complete_rounds + 1):
            print(f"\n{'=' * 60}")
            print(f'🚀 开始第 {complete_round}/{num_complete_rounds} 轮完整训练')
            print(f"{'=' * 60}")
            for bucket_rotation_step in range(self.num_workers):
                print(f'\n--- 桶轮转步骤 {bucket_rotation_step + 1}/{self.num_workers} ---')
                current_bucket_assignment = self._get_current_bucket_assignment(bucket_rotation_step)
                futures = []
                for worker_id, bucket_id in enumerate(current_bucket_assignment):
                    bucket = self.bucket_assignments[bucket_id]
                    if not bucket:
                        print(f'  Worker {worker_id}: 桶 {bucket_id} 为空，跳过')
                        continue
                    print(f'  Worker {worker_id}: 开始训练桶 {bucket_id} 中的 {len(bucket)} 个模型')
                    for arch in bucket:
                        arch_id = arch['arch_id']
                        model_info = self.model_states[arch_id]
                        print(f"    -worker{worker_id} 训练模型 '{arch_id}' (负载: {arch['load']:.4f})")
                        future = self.workers[worker_id].train_model_on_partition.remote(arch_spec=model_info['spec'], state_dict=model_info['state'], config={**self.config, 'epochs_per_rotation': epochs_per_bucket})
                        futures.append((arch_id, future))
                print(f'  等待 {len(futures)} 个训练任务完成...')
                for arch_id, future in futures:
                    try:
                        val_acc, updated_state, _, _ = ray.get(future)
                        self.model_states[arch_id]['state'] = updated_state
                        if val_acc > self.model_states[arch_id]['best_acc']:
                            self.model_states[arch_id]['best_acc'] = val_acc
                            print(f"    🎉 '{arch_id}' 新最佳准确率: {val_acc:.4f}")
                    except Exception as e:
                        print(f"    ❌ '{arch_id}' 训练失败: {e}")
                        raise ValueError(f'架构 {arch_id} 训练失败: {e}')
                print(f'  桶轮转步骤 {bucket_rotation_step + 1} 完成')
            print(f'\n✅ 第 {complete_round}/{num_complete_rounds} 轮完整训练完成！')
        print(f"\n{'=' * 60}")
        print(f'🎉 所有 {num_complete_rounds} 轮基于桶的训练已完成！')
        print(f"{'=' * 60}")

    def retrain_architectures_in_buckets(self, architectures: List[Dict], num_complete_rounds: int, epochs_per_bucket: int=5, max_batch_size: int=8) -> Dict[str, Any]:
        if not architectures:
            return {'final_accuracies': {}, 'batches': []}
        original_model_states = dict(self.model_states)
        original_model_pool = list(self.model_pool)
        original_bucket_assignments = list(self.bucket_assignments)
        original_bucket_loads = list(self.bucket_loads)
        original_bucket_rotation_order = list(self.bucket_rotation_order)
        original_current_bucket_index = int(self.current_bucket_index)
        original_use_bucket_mode = bool(self.use_bucket_mode)
        final_accuracies: Dict[str, float] = {}
        aggregated_updates: Dict[str, Dict[str, Any]] = {}
        batch_infos: List[Dict[str, Any]] = []
        try:
            total = len(architectures)
            batch_size = max(1, int(max_batch_size))
            for start in range(0, total, batch_size):
                batch = architectures[start:start + batch_size]
                batch_ids = [a['arch_id'] for a in batch]
                batch_idx = start // batch_size + 1
                total_batches = (total - 1) // batch_size + 1
                print(f'\n=== Top-K 重训练 - 批次 {batch_idx} / {total_batches} ===')
                print(f'批次大小: {len(batch)}，将使用桶模式在 {self.num_workers} 个worker上训练')
                normalized_batch: List[Dict[str, Any]] = []
                for arch in batch:
                    if 'load' not in arch:
                        normalized_batch.append({**arch, 'load': 1.0})
                        raise ValueError(f"架构 {arch['arch_id']} 缺少负载信息")
                    else:
                        normalized_batch.append(arch)
                buckets, _bucket_loads = assign_to_buckets(normalized_batch, max(1, self.num_workers))
                set_ret = self.set_fixed_architectures_for_training(buckets)
                print(f"批次设置完成：{set_ret.get('architectures_set', 0)} 个架构，桶摘要: {set_ret.get('bucket_summary', {})}")
                self.run_bucket_based_training_rounds(num_complete_rounds=num_complete_rounds, epochs_per_bucket=epochs_per_bucket)
                for arch in normalized_batch:
                    aid = arch['arch_id']
                    acc = 0.0
                    state_entry = None
                    if aid in self.model_states:
                        state_entry = self.model_states[aid]
                        acc = float(state_entry.get('best_acc', 0.0))
                    final_accuracies[aid] = acc
                    if state_entry is not None:
                        aggregated_updates[aid] = {'spec': state_entry.get('spec', arch.get('arch_spec')), 'state': state_entry.get('state', None), 'best_acc': acc, 'load': state_entry.get('load', arch.get('load', 0.0))}
                batch_infos.append({'batch_arch_ids': batch_ids, 'epochs_per_bucket': epochs_per_bucket, 'num_complete_rounds': num_complete_rounds, 'num_workers': self.num_workers})
        finally:
            self.model_states = original_model_states
            self.model_pool = original_model_pool
            self.bucket_assignments = original_bucket_assignments
            self.bucket_loads = original_bucket_loads
            self.bucket_rotation_order = original_bucket_rotation_order
            self.current_bucket_index = original_current_bucket_index
            self.use_bucket_mode = original_use_bucket_mode
            for aid, entry in aggregated_updates.items():
                if aid in self.model_states:
                    self.model_states[aid]['state'] = entry.get('state', self.model_states[aid].get('state'))
                    self.model_states[aid]['best_acc'] = max(float(self.model_states[aid].get('best_acc', 0.0)), float(entry.get('best_acc', 0.0)))
                    if self.model_states[aid].get('spec') is None and entry.get('spec') is not None:
                        self.model_states[aid]['spec'] = entry['spec']
                    if 'load' not in self.model_states[aid]:
                        self.model_states[aid]['load'] = entry.get('load', 0.0)
                else:
                    self.model_states[aid] = {'spec': entry.get('spec'), 'state': entry.get('state'), 'best_acc': float(entry.get('best_acc', 0.0)), 'load': float(entry.get('load', 0.0))}
        return {'final_accuracies': final_accuracies, 'batches': batch_infos}

    def _get_current_bucket_assignment(self, rotation_step: int) -> List[int]:
        assignment = []
        for bucket_id in range(self.num_workers):
            _ = (bucket_id + rotation_step) % self.num_workers
            assignment.append(bucket_id)
        return assignment

    def get_bucket_status(self) -> Dict[str, Any]:
        if not self.use_bucket_mode:
            return {'error': '未启用桶模式'}
        bucket_summary = get_bucket_assignment_summary(self.bucket_assignments, self.bucket_loads)
        for bucket_detail in bucket_summary['bucket_details']:
            bucket_id = bucket_detail['bucket_id']
            bucket = self.bucket_assignments[bucket_id]
            bucket_detail['model_status'] = []
            for arch in bucket:
                arch_id = arch['arch_id']
                if arch_id in self.model_states:
                    model_info = self.model_states[arch_id]
                    bucket_detail['model_status'].append({'arch_id': arch_id, 'load': arch['load'], 'best_accuracy': model_info['best_acc'], 'has_trained_state': model_info['state'] is not None})
        return {'bucket_mode_enabled': self.use_bucket_mode, 'bucket_summary': bucket_summary, 'current_rotation_index': self.current_bucket_index}

    def run_training_rounds(self, num_complete_rounds: int):
        if not self.model_pool:
            print('No models in the pool to train.')
            return
        total_rotation_rounds = num_complete_rounds * self.num_workers
        print(f'\n=== 开始 {num_complete_rounds} 轮完整训练 ===')
        print(f'每轮完整训练包含 {self.num_workers} 次轮转')
        print(f'总计将执行 {total_rotation_rounds} 次底层轮转')
        for complete_round in range(1, num_complete_rounds + 1):
            print(f"\n{'=' * 60}")
            print(f'🚀 开始第 {complete_round}/{num_complete_rounds} 轮完整训练')
            print(f"{'=' * 60}")
            for rotation_step in range(self.num_workers):
                rotation_round = (complete_round - 1) * self.num_workers + rotation_step + 1
                print(f'\n--- 底层轮转 {rotation_round}/{total_rotation_rounds} (完整轮次 {complete_round}, 步骤 {rotation_step + 1}/{self.num_workers}) ---')
                futures = []
                for i, worker in enumerate(self.workers):
                    model_to_train = self.model_pool[i]
                    arch_id = model_to_train['arch_id']
                    model_info = self.model_states[arch_id]
                    print(f"  - Worker {i} will train model '{arch_id}' on partition {i}.")
                    future = worker.train_model_on_partition.remote(arch_spec=model_info['spec'], state_dict=model_info['state'], config=self.config)
                    futures.append((arch_id, future))
                for arch_id, future in futures:
                    val_acc, updated_state, correct_count, incorrect_count = ray.get(future)
                    self.model_states[arch_id]['state'] = updated_state
                    if val_acc > self.model_states[arch_id]['best_acc']:
                        self.model_states[arch_id]['best_acc'] = val_acc
                        print(f"  🎉 New best accuracy for '{arch_id}': {val_acc:.4f} (正确:{correct_count}, 错误:{incorrect_count})")
                if rotation_step < self.num_workers - 1:
                    self.model_pool.insert(0, self.model_pool.pop())
                    print(f"  - Model rotation complete. Next assignment: {self.model_pool[0]['arch_id']} -> Worker 0.")
            print(f'\n✅ 第 {complete_round}/{num_complete_rounds} 轮完整训练完成！')
            print(f'   所有模型都已在 {self.num_workers} 个数据分区上训练过一次')
            if complete_round < num_complete_rounds:
                print(f'   准备下一轮完整训练...')
        print(f"\n{'=' * 60}")
        final_epochs = num_complete_rounds * int(self.config.get('epochs_per_rotation', 5)) * int(self.config.get('steps_per_epoch', 1))
        print(f'🎉 所有 {num_complete_rounds} 轮完整训练已完成！')
        print(f'   每个模型总共训练了 {num_complete_rounds * self.num_workers * final_epochs} 次')
        print(f"   每个模型在每个数据分区上训练了 {int(self.config.get('epochs_per_rotation', 5)) * int(self.config.get('steps_per_epoch', 1))} 次")
        print(f"{'=' * 60}")
if __name__ == '__main__':
    import sys
    import os
    import ray
    import torch
    import pprint
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.append(project_root)
    from data.partition import PartitionGraph
    from trainer.worker import TrainerWorker
    print('===== 运行 Dispatcher 独立集成测试 =====')
    DATASET = 'Cora'
    NUM_WORKERS = int(torch.cuda.device_count()) if torch.cuda.is_available() else 2
    MAX_HOPS = 3
    TRAINING_ROUNDS = 10
    EPOCHS_PER_ROTATION = 5
    LEARNING_RATE = 0.01
    if NUM_WORKERS == 0:
        print('警告: 未检测到GPU将使用2个CPU Worker进行模拟')
        NUM_WORKERS = 2
    architectures_to_test = [{'arch_id': 'GCN-2-layer', 'arch_spec': {'layers': [{'type': 'gcn'}] * 2}}, {'arch_id': 'GCN-3-layer', 'arch_spec': {'layers': [{'type': 'gcn'}] * 3}}]
    try:
        print(f'\n[步骤 1] 初始化 Ray ({NUM_WORKERS} 个 Worker)...')
        if torch.cuda.is_available():
            ray.init(num_gpus=NUM_WORKERS, ignore_reinit_error=True)
        else:
            ray.init(num_cpus=NUM_WORKERS, ignore_reinit_error=True)
        print(f'   -> Ray 初始化成功. 可用资源: {ray.available_resources()}')
        print(f"\n[步骤 2] 为 {NUM_WORKERS} 个 Worker 划分 '{DATASET}' 数据集...")
        partitions = PartitionGraph.get_partitions(dataset_name=DATASET, num_partitions=NUM_WORKERS, max_hops=MAX_HOPS, strategy='contextual_khop')
        print(f'   -> 数据集已划分为 {len(partitions)} 个部分')
        print(f'\n[步骤 3] 创建 {NUM_WORKERS} 个 TrainerWorker...')
        train_config = {'epochs_per_rotation': EPOCHS_PER_ROTATION, 'lr': LEARNING_RATE}
        workers = []
        for i in range(NUM_WORKERS):
            use_gpu = torch.cuda.is_available()
            worker_actor = TrainerWorker.options(num_gpus=1 if use_gpu else 0).remote(worker_id=i, gpu_id=i if use_gpu else -1, data_partition=partitions[i], config=train_config)
            workers.append(worker_actor)
        print(f'   -> {len(workers)} 个 Worker 创建成功')
        print('\n[步骤 4] 创建 Dispatcher...')
        dispatcher = Dispatcher.remote(worker_refs=workers, train_config=train_config)
        print('   -> Dispatcher 创建成功')
        print(f'\n[步骤 5] 提交 {len(architectures_to_test)} 个模型架构...')
        ray.get(dispatcher.submit_architectures.remote(architectures_to_test))
        print('   -> 模型架构提交成功')
        print(f'\n[步骤 6] 开始执行 {TRAINING_ROUNDS} 轮训练...')
        ray.get(dispatcher.run_training_rounds.remote(num_complete_rounds=TRAINING_ROUNDS))
        print('   -> 训练执行完毕')
        print('\n[步骤 7] 获取并展示最终评估结果...')
        final_results = ray.get(dispatcher.get_architectures.remote())
        print('\n' + '=' * 50)
        print('           最终模型性能评估结果')
        print('=' * 50)
        pprint.pprint(final_results)
        print('=' * 50)
    except Exception as e:
        print(f'\n❌ 测试过程中发生错误: {e}')
        import traceback
        traceback.print_exc()
    finally:
        print('\n[步骤 8] 关闭 Ray...')
        ray.shutdown()
        print('   -> Ray 已关闭测试结束')