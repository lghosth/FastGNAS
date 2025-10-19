import copy
import itertools
import math
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
import ray
try:
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
    from scipy.optimize import minimize
    from scipy.stats import norm
    BAYESIAN_AVAILABLE = True
except ImportError:
    BAYESIAN_AVAILABLE = False
from baseline.components.backends.ca_backends import CABackends
from baseline.gnas_ca_rl.run import run_single_architecture_experiment

class HyperparameterExplorer:
    PARAM_BOUNDS = {'hidden_dim': (8, 512), 'learning_rate': (0.001, 0.2), 'dropout': (0.1, 0.9), 'weight_decay': (0.0001, 0.1)}
    HIDDEN_DIM_CHOICES = [8, 16, 32, 64, 128, 256, 512]
    LEARNING_RATE_CHOICES = [0.001, 0.005, 0.01, 0.05, 0.1, 0.2]
    DROPOUT_CHOICES = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9]
    WEIGHT_DECAY_CHOICES = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1]

    def __init__(self, base_config: Dict[str, Any], logger, workers: Optional[List[ray.actor.ActorHandle]]=None, ca_backend: Optional[CABackends]=None, max_epochs: int=30, early_stop_patience: int=5, optimization_strategy: str='bayesian', n_initial_points: int=10, n_calls: int=50, acquisition_func: str='EI') -> None:
        self.workers = workers or []
        self.ca_backend = ca_backend
        if not self.workers and self.ca_backend is None:
            raise ValueError('HyperparameterExplorer 需要提供 TrainerWorkers 或 CA 后端之一')
        self.base_config = copy.deepcopy(base_config)
        self.logger = logger
        self.max_epochs = max_epochs
        self.base_config['num_eval_rounds'] = max_epochs
        self.early_stop_patience = early_stop_patience
        self.optimization_strategy = optimization_strategy.lower()
        self.n_initial_points = n_initial_points
        self.n_calls = n_calls
        self.acquisition_func = acquisition_func
        self.X_observed = []
        self.y_observed = []
        self.gp_model = None
        if self.optimization_strategy == 'bayesian' and (not BAYESIAN_AVAILABLE):
            self.logger.logger.warning('scikit-learn或scipy未安装，回退到网格搜索策略')
            self.optimization_strategy = 'grid'
        self._active_model_info: Optional[Dict[str, Any]] = None
        self._ca_experiment_index = 0

    def run(self, model_info: Dict[str, Any]) -> Dict[str, Any]:
        self._active_model_info = model_info
        arch_spec = model_info['arch_spec']
        try:
            if self.optimization_strategy == 'bayesian':
                return self._run_bayesian_optimization(arch_spec, model_info)
            elif self.optimization_strategy == 'random':
                return self._run_random_search(arch_spec, model_info)
            else:
                return self._run_grid_search(arch_spec, model_info)
        finally:
            self._active_model_info = None

    def _run_bayesian_optimization(self, arch_spec: Dict[str, Any], model_info: Dict[str, Any]) -> Dict[str, Any]:
        self.logger.logger.info(f'开始贝叶斯超参优化，初始探索{self.n_initial_points}点，总计{self.n_calls}次评估')
        best_result = {'accuracy': -1.0, 'params': None, 'details': None}
        self.logger.logger.info(f'阶段1: 随机探索{self.n_initial_points}个初始点...')
        for i in range(self.n_initial_points):
            params = self._sample_random_params()
            accuracy, per_worker = self._evaluate_params(arch_spec, params)
            self.X_observed.append(self._params_to_array(params))
            self.y_observed.append(accuracy)
            self.logger.logger.info(f'  初始点{i + 1}/{self.n_initial_points}: {params} -> 准确率: {accuracy:.4f}')
            if accuracy > best_result['accuracy']:
                best_result['accuracy'] = accuracy
                best_result['params'] = params
                best_result['details'] = {'per_worker_accuracy': per_worker}
        remaining_calls = self.n_calls - self.n_initial_points
        self.logger.logger.info(f'阶段2: 贝叶斯优化{remaining_calls}次迭代...')
        no_improvement_count = 0
        early_stopped = False
        for i in range(remaining_calls):
            self._train_gp_model()
            next_params = self._optimize_acquisition_function()
            accuracy, per_worker = self._evaluate_params(arch_spec, next_params)
            self.X_observed.append(self._params_to_array(next_params))
            self.y_observed.append(accuracy)
            improvement_threshold = 0.0001
            if accuracy > best_result['accuracy'] + improvement_threshold:
                improvement = accuracy - best_result['accuracy']
                best_result['accuracy'] = accuracy
                best_result['params'] = next_params
                best_result['details'] = {'per_worker_accuracy': per_worker}
                no_improvement_count = 0
                self.logger.logger.info(f'  贝叶斯迭代{i + 1}/{remaining_calls}: {next_params} -> 准确率: {accuracy:.4f} ✨ [提升 {improvement:.4f}]')
            else:
                no_improvement_count += 1
                self.logger.logger.info(f'  贝叶斯迭代{i + 1}/{remaining_calls}: {next_params} -> 准确率: {accuracy:.4f} [无改进 {no_improvement_count}/{self.early_stop_patience}]')
            if no_improvement_count >= self.early_stop_patience:
                self.logger.logger.info(f"🛑 早停触发: 连续 {self.early_stop_patience} 次迭代无改进，当前最佳精度: {best_result['accuracy']:.4f}")
                early_stopped = True
                break
        total_evals = len(self.X_observed)
        stop_reason = '早停' if early_stopped else '完成全部迭代'
        self.logger.logger.info(f"贝叶斯优化{stop_reason}，共评估{total_evals}组参数，最佳精度: {best_result['accuracy']:.4f}")
        return best_result

    def _run_random_search(self, arch_spec: Dict[str, Any], model_info: Dict[str, Any]) -> Dict[str, Any]:
        self.logger.logger.info(f'开始随机搜索，共{self.n_calls}次评估...')
        best_result = {'accuracy': -1.0, 'params': None, 'details': None}
        no_improvement_count = 0
        early_stopped = False
        for i in range(self.n_calls):
            params = self._sample_random_params()
            accuracy, per_worker = self._evaluate_params(arch_spec, params)
            improvement_threshold = 0.0001
            if accuracy > best_result['accuracy'] + improvement_threshold:
                improvement = accuracy - best_result['accuracy']
                best_result['accuracy'] = accuracy
                best_result['params'] = params
                best_result['details'] = {'per_worker_accuracy': per_worker}
                no_improvement_count = 0
                self.logger.logger.info(f'  随机搜索{i + 1}/{self.n_calls}: {params} -> 准确率: {accuracy:.4f} ✨ [提升 {improvement:.4f}]')
            else:
                no_improvement_count += 1
                self.logger.logger.info(f'  随机搜索{i + 1}/{self.n_calls}: {params} -> 准确率: {accuracy:.4f} [无改进 {no_improvement_count}/{self.early_stop_patience}]')
            if no_improvement_count >= self.early_stop_patience:
                self.logger.logger.info(f"🛑 早停触发: 连续 {self.early_stop_patience} 次迭代无改进，当前最佳精度: {best_result['accuracy']:.4f}")
                early_stopped = True
                break
        stop_reason = '早停' if early_stopped else '完成全部迭代'
        self.logger.logger.info(f"随机搜索{stop_reason}，最佳精度: {best_result['accuracy']:.4f}")
        return best_result

    def _run_grid_search(self, arch_spec: Dict[str, Any], model_info: Dict[str, Any]) -> Dict[str, Any]:
        HIDDEN_DIMS = [8, 16, 32, 64, 128, 256, 512]
        LEARNING_RATES = [0.001, 0.005, 0.01, 0.05, 0.1, 0.2]
        DROPOUTS = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9]
        WEIGHT_DECAYS = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1]
        best_result = {'accuracy': -1.0, 'params': None, 'details': None}
        combinations = list(itertools.product(HIDDEN_DIMS, LEARNING_RATES, DROPOUTS, WEIGHT_DECAYS))
        self.logger.logger.info(f"开始网格搜索，共 {len(combinations)} 组组合用于模型 {model_info.get('model_id', 'unknown')}")
        no_improvement_count = 0
        early_stopped = False
        for idx, (hidden_dim, lr, dropout, weight_decay) in enumerate(combinations, start=1):
            if idx > self.n_calls:
                break
            params = {'hidden_dim': hidden_dim, 'learning_rate': lr, 'dropout': dropout, 'weight_decay': weight_decay}
            accuracy, per_worker = self._evaluate_params(arch_spec, params)
            improvement_threshold = 0.0001
            if accuracy > best_result['accuracy'] + improvement_threshold:
                improvement = accuracy - best_result['accuracy']
                best_result['accuracy'] = accuracy
                best_result['params'] = params
                best_result['details'] = {'per_worker_accuracy': per_worker}
                no_improvement_count = 0
                self.logger.logger.info(f'[Grid {idx}/{min(len(combinations), self.n_calls)}] {params} -> 准确率: {accuracy:.4f} ✨ [提升 {improvement:.4f}]')
            else:
                no_improvement_count += 1
                self.logger.logger.info(f'[Grid {idx}/{min(len(combinations), self.n_calls)}] {params} -> 准确率: {accuracy:.4f} [无改进 {no_improvement_count}/{self.early_stop_patience}]')
            if no_improvement_count >= self.early_stop_patience:
                self.logger.logger.info(f"🛑 早停触发: 连续 {self.early_stop_patience} 次迭代无改进，当前最佳精度: {best_result['accuracy']:.4f}")
                early_stopped = True
                break
        stop_reason = '早停' if early_stopped else '完成全部迭代'
        self.logger.logger.info(f"网格搜索{stop_reason}，最佳精度 {best_result['accuracy']:.4f}，参数组合: {best_result['params']}")
        return best_result

    def _sample_random_params(self) -> Dict[str, Any]:
        hidden_dim = np.random.choice(self.HIDDEN_DIM_CHOICES)
        learning_rate = float(np.random.choice(self.LEARNING_RATE_CHOICES))
        dropout = float(np.random.choice(self.DROPOUT_CHOICES))
        weight_decay = float(np.random.choice(self.WEIGHT_DECAY_CHOICES))
        return {'hidden_dim': int(hidden_dim), 'learning_rate': learning_rate, 'dropout': dropout, 'weight_decay': weight_decay}

    def _params_to_array(self, params: Dict[str, Any]) -> np.ndarray:
        return np.array([math.log(params['hidden_dim']), math.log(params['learning_rate']), params['dropout'], math.log(params['weight_decay'])])

    def _array_to_params(self, x: np.ndarray) -> Dict[str, Any]:
        hidden_dim_continuous = math.exp(x[0])
        hidden_dim = min(self.HIDDEN_DIM_CHOICES, key=lambda h: abs(h - hidden_dim_continuous))
        lr_continuous = math.exp(x[1])
        learning_rate = min(self.LEARNING_RATE_CHOICES, key=lambda v: abs(math.log(v) - math.log(lr_continuous)))
        dropout_continuous = np.clip(x[2], *self.PARAM_BOUNDS['dropout'])
        dropout = min(self.DROPOUT_CHOICES, key=lambda v: abs(v - dropout_continuous))
        weight_decay_continuous = math.exp(x[3])
        weight_decay = min(self.WEIGHT_DECAY_CHOICES, key=lambda v: abs(math.log(v) - math.log(weight_decay_continuous)))
        return {'hidden_dim': int(hidden_dim), 'learning_rate': float(learning_rate), 'dropout': float(dropout), 'weight_decay': float(weight_decay)}

    def _train_gp_model(self):
        if not BAYESIAN_AVAILABLE or len(self.X_observed) < 2:
            return
        X = np.array(self.X_observed)
        y = np.array(self.y_observed)
        kernel = C(1.0, (0.001, 1000.0)) * RBF(length_scale=1.0, length_scale_bounds=(0.01, 100.0))
        self.gp_model = GaussianProcessRegressor(kernel=kernel, alpha=1e-06, n_restarts_optimizer=10, normalize_y=True)
        self.gp_model.fit(X, y)

    def _optimize_acquisition_function(self) -> Dict[str, Any]:
        if not BAYESIAN_AVAILABLE or self.gp_model is None:
            return self._sample_random_params()

        def acquisition(x):
            x = x.reshape(1, -1)
            mu, sigma = self.gp_model.predict(x, return_std=True)
            if self.acquisition_func == 'EI':
                mu_best = max(self.y_observed)
                with np.errstate(divide='warn'):
                    imp = mu - mu_best
                    Z = imp / sigma
                    ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
                    ei[sigma == 0.0] = 0.0
                return -ei[0]
            elif self.acquisition_func == 'UCB':
                beta = 2.0
                return -(mu[0] + beta * sigma[0])
            else:
                mu_best = max(self.y_observed)
                with np.errstate(divide='warn'):
                    Z = (mu - mu_best) / sigma
                    pi = norm.cdf(Z)
                return -pi[0]
        best_x = None
        best_val = float('inf')
        bounds = [(math.log(self.HIDDEN_DIM_CHOICES[0]), math.log(self.HIDDEN_DIM_CHOICES[-1])), (math.log(self.PARAM_BOUNDS['learning_rate'][0]), math.log(self.PARAM_BOUNDS['learning_rate'][1])), self.PARAM_BOUNDS['dropout'], (math.log(self.PARAM_BOUNDS['weight_decay'][0]), math.log(self.PARAM_BOUNDS['weight_decay'][1]))]
        for _ in range(10):
            x0 = self._params_to_array(self._sample_random_params())
            try:
                res = minimize(acquisition, x0, bounds=bounds, method='L-BFGS-B')
                if res.fun < best_val:
                    best_val = res.fun
                    best_x = res.x
            except:
                continue
        if best_x is None:
            return self._sample_random_params()
        return self._array_to_params(best_x)

    def _evaluate_params(self, arch_spec: Dict[str, Any], params: Dict[str, Any]) -> Tuple[float, List[float]]:
        spec_variant = self._with_hidden_dim(arch_spec, params['hidden_dim'])
        if self.ca_backend is not None:
            schedule = self._build_ca_schedule()
            return self._train_with_ca(spec_variant, schedule, params)
        config = self._build_worker_config(params['learning_rate'], params['dropout'], params['weight_decay'])
        return self._train_on_workers(spec_variant, config)

    def _build_worker_config(self, lr: float, dropout: float, weight_decay: float) -> Dict[str, Any]:
        cfg = copy.deepcopy(self.base_config)
        cfg['lr'] = lr
        cfg['dropout'] = dropout
        cfg['weight_decay'] = weight_decay
        cfg['early_stop_patience'] = self.early_stop_patience
        cfg.setdefault('num_eval_rounds', min(5, self.max_epochs))
        return cfg

    def _build_ca_schedule(self) -> Dict[str, Any]:
        epochs_per_round = int(self.base_config.get('epochs_per_rotation', 2))
        num_rounds = int(self.base_config.get('num_eval_rounds', 5))
        num_rounds = min(num_rounds, self.max_epochs)
        return {'training': {'epochs_per_round': epochs_per_round, 'num_rounds': num_rounds, 'batch_size': self.base_config.get('batch_size'), 'steps_per_epoch': self.base_config.get('steps_per_epoch')}}

    def _with_hidden_dim(self, arch_spec: Dict[str, Any], hidden_dim: int) -> Dict[str, Any]:
        spec = copy.deepcopy(arch_spec)
        if 'layers' in spec:
            layers = spec['layers']
            if not isinstance(layers, list):
                return spec
            for idx, layer in enumerate(layers):
                if idx < len(layers) - 1:
                    layer['hidden_dim'] = hidden_dim
                else:
                    layer.setdefault('hidden_dim', hidden_dim)
        else:
            self.logger.logger.warning("HyperparameterExplorer: 当前架构不包含'layers'字段，无法调整hidden_dim")
        return spec

    def _train_on_workers(self, arch_spec: Dict[str, Any], config: Dict[str, Any]) -> Tuple[float, List[float]]:
        if not self.workers:
            raise RuntimeError('未提供可用的 TrainerWorker 实例')
        futures = []
        for worker in self.workers:
            futures.append(worker.train_model_on_partition.remote(arch_spec=copy.deepcopy(arch_spec), state_dict={}, config=copy.deepcopy(config)))
        results = ray.get(futures)
        accuracies: List[float] = []
        global_correct = 0.0
        global_total = 0.0
        for res in results:
            if not res:
                continue
            acc = float(res[0])
            accuracies.append(acc)
            if len(res) >= 4:
                correct_count = res[2]
                incorrect_count = res[3]
                if isinstance(correct_count, (int, float)) and isinstance(incorrect_count, (int, float)):
                    global_correct += float(correct_count)
                    global_total += float(correct_count + incorrect_count)
        if global_total > 0:
            global_accuracy = global_correct / global_total
        else:
            global_accuracy = sum(accuracies) / len(accuracies) if accuracies else 0.0
        return (global_accuracy, accuracies)

    def _train_with_ca(self, arch_spec: Dict[str, Any], training_schedule: Dict[str, Any], params: Dict[str, Any]) -> Tuple[float, List[float]]:
        if self.ca_backend is None:
            raise RuntimeError('未提供 CA 后端，无法执行CA训练')
        if not self.ca_backend.workers:
            raise RuntimeError('CA 后端尚未初始化，请先调用 setup_distributed_training')
        dropout = float(np.clip(params['dropout'], *self.PARAM_BOUNDS['dropout']))
        learning_rate = float(params['learning_rate'])
        weight_decay = float(params['weight_decay'])
        hyper_updates = {'learning_rate': learning_rate, 'lr': learning_rate, 'weight_decay': weight_decay, 'dropout': dropout}
        self.ca_backend.update_training_config(hyper_updates)
        self._ca_experiment_index += 1
        model_id = 'model'
        if self._active_model_info:
            model_id = self._active_model_info.get('model_id', model_id)
        experiment_name = f'{model_id}_hp_{self._ca_experiment_index:03d}'
        result = run_single_architecture_experiment(ca_backends=self.ca_backend, arch_spec=arch_spec, arch_name=experiment_name, config=training_schedule, logger=self.logger)
        best_accuracy = float(result.get('best_accuracy', 0.0))
        per_worker = []
        training_history = result.get('training_history', [])
        if training_history:
            last_evaluation = training_history[-1].get('evaluation', {})
            worker_stats = last_evaluation.get('worker_stats', []) or []
            per_worker = [float(stat.get('accuracy', 0.0)) for stat in worker_stats if stat]
        if not per_worker:
            per_worker = [best_accuracy]
        return (best_accuracy, per_worker)