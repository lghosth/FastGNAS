import ray
import warnings
from typing import List, Dict, Any, Optional
from datetime import datetime
from .scheduling import SchedulerInterface, BucketScheduler, RotationScheduler, DynamicBatchScheduler
from .registry import ModelRegistry
from .orchestration import TrainingOrchestrator
from .state import StateManager
from .balance_controller import BalanceController
from .load_estimator import MloadPredictor
try:
    from ..utils.logger import GraphNASLogger
except ImportError:

    class GraphNASLogger:

        @staticmethod
        def get_logger(name):

            class SimpleLogger:

                def info(self, msg):
                    print(f'[INFO] {msg}')

                def warning(self, msg):
                    print(f'[WARNING] {msg}')

                def error(self, msg):
                    print(f'[ERROR] {msg}')
            return SimpleLogger()

@ray.remote(num_gpus=0)
class Dispatcher:

    def __init__(self, worker_refs: List[ray.ObjectRef], train_config: Dict[str, Any], scheduler_type: str='bucket', balance_controller: Optional[BalanceController]=None, load_predictor: Optional[MloadPredictor]=None, enable_balancing: bool=True):
        self.workers = worker_refs
        self.num_workers = len(worker_refs)
        self.config = train_config
        self.scheduler_type = scheduler_type
        self.enable_balancing = enable_balancing
        self.balance_controller = balance_controller
        self.load_predictor = load_predictor
        state_config = train_config.get('state_manager', {})
        max_memory_states = state_config.get('max_memory_states', 1000)
        self.scheduler = self._create_scheduler(scheduler_type)
        self.registry = ModelRegistry()
        self.state_manager = StateManager(max_memory_states=max_memory_states)
        self.orchestrator = TrainingOrchestrator(self.workers, self.scheduler, self.state_manager, self.registry)
        self.is_initialized = False
        self.creation_time = datetime.now()
        self.logger = GraphNASLogger.get_logger('Dispatcher_v2')
        self.verbose = bool(self.config.get('verbose_runtime', True))
        self.enable_state_save = bool(self.config.get('enable_state_save', True))
        try:
            setattr(self.orchestrator, 'verbose', self.verbose)
            setattr(self.orchestrator, 'enable_state_save', self.enable_state_save)
        except Exception:
            pass
        self.logger.info(f'新版Dispatcher初始化完成:')
        self.logger.info(f'  - Worker数量: {self.num_workers}')
        self.logger.info(f'  - 调度器类型: {scheduler_type}')
        self.logger.info(f'  - 负载均衡: {enable_balancing}')
        self.logger.info(f'  - 创建时间: {self.creation_time}')
        self.logger.info(f'  - 冗余打印(verbos): {self.verbose}')
        self.logger.info(f'  - 模型状态保存开关：{self.enable_state_save}')

    def _create_scheduler(self, scheduler_type: str) -> SchedulerInterface:
        if scheduler_type == 'bucket':
            return BucketScheduler(self.num_workers, target_rotations=1)
        elif scheduler_type == 'rotation':
            return RotationScheduler(self.num_workers)
        elif scheduler_type in ('subtask', 'dynamic', 'dynamic_batch'):
            st_cfg = self.config.get('subtask_scheduler', {}) if isinstance(self.config, dict) else {}
            epochs = int(self.config.get('epochs_per_rotation', 1)) if isinstance(self.config, dict) else 5
            steps_per_epoch = int(self.config.get('steps_per_epoch', 1)) if isinstance(self.config, dict) else 1
            load_var_th = float(st_cfg.get('load_variance_threshold', 0.3))
            num_rounds = int(self.config.get('num_eval_rounds', 5))
            return DynamicBatchScheduler(num_workers=self.num_workers, epochs_per_rotation=epochs, steps_per_epoch=steps_per_epoch, load_variance_threshold=load_var_th, num_rounds=num_rounds, st_cfg=st_cfg)
        else:
            raise ValueError(f'不支持的调度器类型: {scheduler_type}')

    def submit_architectures(self, architectures: List[Dict[str, Any]], strategy: str='bucket') -> Dict[str, Any]:
        if not architectures:
            raise ValueError('架构列表不能为空')
        if self.verbose:
            print(f'\n=== 提交 {len(architectures)} 个架构 ===')
            print(f'策略: {strategy}')
        if strategy in ('subtask', 'dynamic', 'dynamic_batch'):
            missing = [a.get('arch_id', f'idx_{i}') for i, a in enumerate(architectures) if 'load' not in a or a.get('load') is None]
            if missing:
                raise ValueError('子任务调度策略需要每个架构提供负载(load)。缺失项: ' + ', '.join(map(str, missing)) + '。请使用 BalanceController 生成带负载的架构，或切换到 rotation/bucket 策略。')
        try:
            self.scheduler.validate_architectures(architectures)
        except ValueError as e:
            raise ValueError(f'架构验证失败: {e}')
        for arch in architectures:
            self.registry.register(arch_id=arch['arch_id'], arch_spec=arch['arch_spec'], load=arch.get('load', 1.0), metadata={'submission_time': datetime.now().isoformat()})
        valid_strategies = ['rotation', 'bucket', 'subtask', 'dynamic', 'dynamic_batch']
        if strategy not in valid_strategies:
            raise ValueError(f'无效的调度策略: {strategy}，必须是以下之一: {valid_strategies}')
        if self.verbose:
            print(f'使用调度策略: {strategy}')
        current_type = self.scheduler.get_scheduler_type()
        need_switch = False
        if strategy == 'bucket' and 'Bucket' not in current_type or (strategy == 'rotation' and 'Rotation' not in current_type) or (strategy in ('subtask', 'dynamic', 'dynamic_batch') and 'Dynamic' not in current_type):
            need_switch = True
        if need_switch:
            if self.verbose:
                print(f'切换调度器: {current_type} -> {strategy}')
            self.scheduler = self._create_scheduler(strategy)
            self.orchestrator.scheduler = self.scheduler
        scheduler_type_name = self.scheduler.get_scheduler_type()
        if 'Rotation' in scheduler_type_name or 'Bucket' in scheduler_type_name or 'Dynamic' in scheduler_type_name:
            self.scheduler.initialize(architectures, train_config=self.config)
        else:
            self.scheduler.initialize(architectures)
        self.is_initialized = True
        if self.verbose:
            print(f'架构提交完成: {len(architectures)} 个架构已注册并初始化调度器')
        return {'submitted': len(architectures), 'strategy': strategy, 'scheduler_type': self.scheduler.get_scheduler_type(), 'registry_total': len(self.registry.get_all_entries())}

    def run_training(self, num_rounds: int=1, epochs_per_round: int=1, enable_global_evaluation: bool=False) -> Dict[str, Any]:
        if not self.is_initialized:
            raise ValueError('请先提交架构进行初始化')
        scheduler_type = self.scheduler.get_scheduler_type()
        actual_rounds = num_rounds
        if 'Rotation' in scheduler_type:
            if self.verbose:
                self.logger.info(f'\n=== 开始训练（Rotation模式 - 固定训练量）===')
                self.logger.info(f'训练量配置: num_rounds={num_rounds}, epochs_per_round={epochs_per_round}')
                self.logger.info(f'总batch数已在调度器初始化时计算并分配')
                self.logger.info(f'将执行完整轮转（每个架构在所有worker上都训练一遍）')
            actual_rounds = 9999
        elif 'Bucket' in scheduler_type:
            actual_rounds = self.num_workers
            if self.verbose:
                self.logger.info(f'\n=== 开始训练（Bucket模式 - 固定训练量）===')
                self.logger.info(f'训练量配置: num_rounds={num_rounds}, epochs_per_round={epochs_per_round}')
                self.logger.info(f'总batch数已在调度器初始化时计算并分配')
                self.logger.info(f'完整轮转次数: 1（每个架构在所有worker上训练一遍）')
                self.logger.info(f'Worker数量: {self.num_workers}')
                self.logger.info(f'实际调度步骤: {actual_rounds}（一次完整轮转）')
        elif 'Dynamic' in scheduler_type or 'Subtask' in scheduler_type:
            if self.verbose:
                self.logger.info(f'\n=== 开始训练（Subtask模式 - 固定训练量+动态负载均衡）===')
                self.logger.info(f'训练量配置: num_rounds={num_rounds}, epochs_per_round={epochs_per_round}')
                self.logger.info(f'总batch数已在调度器初始化时计算')
                self.logger.info(f'通过动态batch分配实现细粒度负载均衡')
                self.logger.info(f'训练将持续到所有分区完成其配额')
            actual_rounds = 9999
        elif self.verbose:
            self.logger.info(f'\n=== 开始训练 ===')
            self.logger.info(f'轮数: {num_rounds}')
        if self.verbose:
            self.logger.info(f'每轮epochs配置: {epochs_per_round}')
            self.logger.info(f"全图评估: {('启用' if enable_global_evaluation else '禁用')}")
        training_results = self.orchestrator.run_multiple_rounds(num_rounds=actual_rounds, epochs_per_task=epochs_per_round)
        final_accuracies = training_results['final_accuracies']
        if enable_global_evaluation:
            self.logger.info(f'执行全图评估')
            global_accuracies = {}
            for arch_id in final_accuracies.keys():
                try:
                    global_accuracy = self.orchestrator.global_evaluate(arch_id=arch_id, enable_distributed_eval=True)
                    global_accuracies[arch_id] = global_accuracy
                    if self.verbose:
                        print(f'架构 {arch_id}: 局部评估={final_accuracies[arch_id]:.4f}, 全图评估={global_accuracy:.4f}, 差异={abs(global_accuracy - final_accuracies[arch_id]):.4f}')
                except Exception as e:
                    if self.verbose:
                        print(f'架构 {arch_id} 全图评估失败: {e}')
                    raise RuntimeError('global_evaluation_failed')
                    global_accuracies[arch_id] = final_accuracies[arch_id]
            final_accuracies = global_accuracies
            import gc
            gc.collect()
            if self.verbose:
                print(f'=== 全图评估完成，已清理临时对象 ===\n')
        for arch_id, accuracy in final_accuracies.items():
            self.registry.update_performance(arch_id=arch_id, accuracy=accuracy, epochs=epochs_per_round * num_rounds)
        timing_stats = training_results.get('statistics', {}).get('timing_stats', {})
        if not timing_stats:
            timing_stats = {'model_migration': 0.0, 'batch_preparation': 0.0, 'forward': 0.0, 'backward': 0.0, 'optimizer_step': 0.0}
        return {'rounds_completed': training_results['statistics']['completed_rounds'], 'final_accuracies': final_accuracies, 'statistics': training_results['statistics'], 'timing_stats': timing_stats, 'registry_summary': self.registry.get_summary(), 'global_evaluation_enabled': enable_global_evaluation}

    def get_results(self, format: str='summary') -> Dict[str, Any]:
        if format == 'summary':
            return {'registry_summary': self.registry.get_summary(), 'scheduler_progress': self.scheduler.get_progress(), 'orchestrator_stats': self.orchestrator.get_statistics(), 'state_manager_stats': self.state_manager.get_statistics()}
        elif format == 'top_k':
            top_models = self.registry.get_top_k(k=10)
            return {'top_models': [{'arch_id': model.arch_id, 'best_accuracy': model.best_accuracy, 'training_epochs': model.training_epochs, 'load': model.load} for model in top_models]}
        else:
            return {'all_entries': {arch_id: entry.to_dict() for arch_id, entry in self.registry.get_all_entries().items()}, 'scheduler_progress': self.scheduler.get_progress(), 'orchestrator_stats': self.orchestrator.get_statistics(), 'state_manager_stats': self.state_manager.get_statistics()}

    def get_system_status(self) -> Dict[str, Any]:
        validation = self.orchestrator.validate_setup()
        return {'dispatcher_version': '2.0_refactored', 'creation_time': self.creation_time.isoformat(), 'is_initialized': self.is_initialized, 'num_workers': self.num_workers, 'components': {'scheduler': {'type': self.scheduler.get_scheduler_type(), 'ready': self.scheduler.is_ready(), 'progress': self.scheduler.get_progress()}, 'registry': {'total_models': len(self.registry.get_all_entries()), 'summary': self.registry.get_summary()}, 'orchestrator': {'validation': validation, 'statistics': self.orchestrator.get_statistics()}, 'state_manager': {'statistics': self.state_manager.get_statistics()}}}

    def submit_balanced_architectures_with_buckets(self, batch_size: int=20, min_architectures: int=8) -> Dict[str, Any]:
        warnings.warn("submit_balanced_architectures_with_buckets已弃用，请使用submit_architectures(strategy='bucket')", DeprecationWarning, stacklevel=2)
        if not self.enable_balancing or not self.balance_controller:
            return {'error': '负载均衡未启用或BalanceController未设置'}
        balanced_architectures = self.balance_controller.generate_balanced_architectures()
        result = self.submit_architectures(balanced_architectures, strategy='bucket')
        if 'error' in result:
            return result
        bucket_assignments = []
        bucket_loads = []
        if hasattr(self.scheduler, 'buckets'):
            bucket_assignments = self.scheduler.buckets
            bucket_loads = self.scheduler.bucket_loads
        return {'architectures_submitted': result['submitted'], 'bucket_assignments': bucket_assignments, 'bucket_loads': bucket_loads, 'bucket_summary': f'已分配到{self.num_workers}个桶'}

    def run_bucket_based_training_rounds(self, num_complete_rounds: int, epochs_per_bucket: int=5) -> None:
        warnings.warn('run_bucket_based_training_rounds已弃用，请使用run_training()', DeprecationWarning, stacklevel=2)
        self.run_training(num_rounds=num_complete_rounds * self.num_workers, epochs_per_round=epochs_per_bucket)

    def get_architectures(self, format_type: str='detailed', sort_by: str='best_acc') -> Dict[str, Any]:
        warnings.warn('get_architectures已弃用，请使用get_results()', DeprecationWarning, stacklevel=2)
        if format_type == 'summary':
            results = self.get_results('summary')
            return {'architectures': {entry.arch_id: {'best_accuracy': entry.best_accuracy, 'has_trained_state': True, 'is_in_pool': True} for entry in self.registry.get_all_entries().values()}, 'statistics': results['registry_summary']}
        else:
            return self.get_results('detailed')

    def warmup_system(self, batch_size: Optional[int]=None, min_architectures: Optional[int]=None) -> Dict[str, Any]:
        import time
        if self.verbose:
            print(f'\n=== 系统预热开始 ===')
        result = {'ok': False, 'reason': '', 'steps': {'architecture_generation': 0.0, 'scheduler_initialization': 0.0, 'worker_warmup': 0.0, 'predictor_warmup': 0.0, 'total_warmup': 0.0}, 'warmup_config': {'batch_size': batch_size, 'min_architectures': min_architectures, 'num_workers': self.num_workers}}
        warmup_batch_size = batch_size if isinstance(batch_size, int) and batch_size > 0 else max(2, self.num_workers)
        warmup_min_archs = min_architectures if isinstance(min_architectures, int) and min_architectures > 0 else max(1, self.num_workers)
        result['warmup_config']['effective_batch_size'] = warmup_batch_size
        result['warmup_config']['effective_min_architectures'] = warmup_min_archs
        total_start_time = time.time()
        try:
            t1_start = time.time()
            warmup_architectures = []
            if self.enable_balancing and self.balance_controller:
                try:
                    if self.verbose:
                        print('  -> 使用负载均衡控制器生成预热架构...')
                    warmup_architectures = self.balance_controller.generate_balanced_architectures()
                    if self.verbose:
                        print(f'  -> 生成了 {len(warmup_architectures)} 个负载均衡架构')
                except Exception as e:
                    if self.verbose:
                        print(f'  -> 负载均衡生成失败，使用默认架构: {e}')
            if not warmup_architectures:
                if self.verbose:
                    print('  -> 创建默认预热架构...')
                for i in range(warmup_min_archs):
                    warmup_arch = {'arch_id': f'warmup_arch_{i:03d}', 'arch_spec': {'layers': [{'type': 'gcn', 'hidden_dim': 64, 'act': 'relu'}, {'type': 'gcn', 'hidden_dim': 32, 'act': 'relu'}]}, 'load': 1.0}
                    warmup_architectures.append(warmup_arch)
                if self.verbose:
                    print(f'  -> 创建了 {len(warmup_architectures)} 个默认架构')
            t1_end = time.time()
            result['steps']['architecture_generation'] = t1_end - t1_start
            t2_start = time.time()
            if self.verbose:
                print('  -> 初始化调度器...')
            temp_submission = self.submit_architectures(architectures=warmup_architectures, strategy=self.scheduler_type)
            if self.verbose:
                print(f"  -> 调度器初始化完成: {temp_submission['scheduler_type']}")
            t2_end = time.time()
            result['steps']['scheduler_initialization'] = t2_end - t2_start
            t3_start = time.time()
            if self.verbose:
                print('  -> 预热所有Worker...')
            warmup_training_result = self.run_training(num_rounds=1, epochs_per_round=1, enable_global_evaluation=False)
            worker_warmup_success = len(warmup_training_result.get('final_accuracies', {})) > 0
            if self.verbose:
                print(f"  -> Worker预热{('成功' if worker_warmup_success else '失败')}: 训练了 {len(warmup_training_result.get('final_accuracies', {}))} 个架构")
            t3_end = time.time()
            result['steps']['worker_warmup'] = t3_end - t3_start
            t4_start = time.time()
            predictor_warmup_success = True
            if self.load_predictor is not None:
                if self.verbose:
                    print('  -> 预热负载预测器...')
                try:
                    dummy_sample = {'architecture': [{'activation': 'relu', 'attention_type': 'gcn', 'aggregation_type': 'mean', 'attention_heads': 1, 'hidden_units': 64}, {'activation': 'relu', 'attention_type': 'gcn', 'aggregation_type': 'mean', 'attention_heads': 1, 'hidden_units': 32}], 'graph': {'vertices': 2708, 'edges': 5429, 'avg_degree': 3.9, 'diameter': 10}, 'device': {'gpu_cores': 2048, 'core_frequency': 1.0, 'fp_capability': 6.0}}
                    _ = self.load_predictor.predict(dummy_sample)
                    if self.verbose:
                        print('  -> 负载预测器预热成功')
                except Exception as e:
                    if self.verbose:
                        print(f'  -> 负载预测器预热失败（不影响主流程）: {e}')
                    predictor_warmup_success = False
            elif self.verbose:
                print('  -> 无负载预测器，跳过预热')
            t4_end = time.time()
            result['steps']['predictor_warmup'] = t4_end - t4_start
            if self.verbose:
                print('  -> 清理预热状态...')
            self.reset()
            total_end_time = time.time()
            result['steps']['total_warmup'] = total_end_time - total_start_time
            result['ok'] = True
            result['reason'] = f"预热成功: 架构生成({result['steps']['architecture_generation']:.3f}s), " + f"调度器初始化({result['steps']['scheduler_initialization']:.3f}s), " + f"Worker预热({result['steps']['worker_warmup']:.3f}s), " + f"预测器预热({result['steps']['predictor_warmup']:.3f}s)"
            if self.verbose:
                print(f'=== 系统预热完成 ===')
                print(f"总耗时: {result['steps']['total_warmup']:.3f}s")
                print(f"详细耗时: {result['reason']}")
        except Exception as e:
            result['ok'] = False
            result['reason'] = f'预热失败: {str(e)}'
            result['steps']['total_warmup'] = time.time() - total_start_time
            if self.verbose:
                print(f'=== 系统预热失败 ===')
                print(f"失败原因: {result['reason']}")
                print(f"已耗时: {result['steps']['total_warmup']:.3f}s")
            try:
                self.reset()
            except:
                pass
        return result

    def reset(self) -> None:
        self.scheduler.reset()
        self.orchestrator.reset_scheduler()
        self.registry.clear()
        self.state_manager.clear_cache()
        self.is_initialized = False
        if self.verbose:
            print('Dispatcher已重置')

    def cleanup_checkpoints(self, keep_latest: int=3, max_age_days: int=7) -> int:
        print('纯内存StateManager: 无需清理文件')
        return 0

    def get_checkpoint_info(self, arch_id: str) -> List[Dict[str, Any]]:
        if self.state_manager.has_checkpoint(arch_id):
            return [{'arch_id': arch_id, 'exists': True, 'type': 'memory_cache'}]
        else:
            return []