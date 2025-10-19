import time
from typing import List, Dict, Any, Tuple, Optional
from core.balancing import evaluate_balance
from core.controller import ControllerManager

class BalanceController:

    def __init__(self, controller_manager: ControllerManager, load_predictor: Optional[Any]=None, batch_size: int=None, min_architectures: int=None, min_samples: int=3, max_samples: int=10, tolerance: float=0.1, num_workers: int=2, graph_info: Optional[Dict[str, Any]]=None, device_info: Optional[Dict[str, Any]]=None):
        self.batch_size = batch_size or max_samples
        self.min_architectures = min_architectures or min_samples
        self.min_samples = min_samples
        self.max_samples = max_samples
        self.tolerance = tolerance
        self.num_workers = num_workers
        self.controller_manager = controller_manager
        self.load_predictor = load_predictor
        self.graph_info = graph_info or {'vertices': 2708, 'edges': 5429, 'avg_degree': 3.9, 'diameter': 10}
        self.device_info = device_info or {'gpu_cores': 3584, 'core_frequency': 1.328, 'fp_capability': 6.0}
        print(f'BalanceController initialized:')
        print(f'  - Batch size: {self.batch_size}')
        print(f'  - Min architectures: {self.min_architectures}')
        print(f'  - Tolerance: {self.tolerance}')
        print(f'  - Workers: {self.num_workers}')
        print(f'  - Load predictor available: {self.load_predictor is not None}')
        print(f'  - (Compatibility: min_samples={self.min_samples}, max_samples={self.max_samples})')

    def generate_balanced_architectures(self) -> List[Dict[str, Any]]:
        print(f'\n=== Starting batch generation of load-balanced architectures ===')
        start_time = time.time()
        all_architectures = self._generate_batch_architectures()
        end_time = time.time()
        print(f'Batch architecture generation: {end_time - start_time:.4f} seconds')
        start_time = time.time()
        architectures_with_load = self._predict_loads_for_all(all_architectures)
        end_time = time.time()
        print(f'Load prediction time: {end_time - start_time:.4f} seconds')
        start_time = time.time()
        best_assignment = self._find_optimal_assignment(architectures_with_load)
        end_time = time.time()
        print(f'Sequential optimal allocation search: {end_time - start_time:.4f} seconds')
        final_is_balanced, final_load_diff = evaluate_balance(best_assignment, self.num_workers, self.tolerance)
        print(f'\n=== Architecture generation completed ===')
        print(f'  Final architecture count: {len(best_assignment)}')
        print(f"  Load balance status: {('Balanced' if final_is_balanced else 'Unbalanced')}")
        print(f'  Load difference: {final_load_diff:.4f}')
        print(f'  Tolerance: {self.tolerance}')
        print(f'\n  Architecture load details:')
        for i, arch in enumerate(best_assignment):
            print(f"    Architecture {i + 1} ({arch['arch_id']}): Load = {arch['load']:.4f}")
        return best_assignment

    def _generate_batch_architectures(self) -> List[Dict[str, Any]]:
        print(f'步骤1：批量生成 {self.batch_size} 个架构...')
        try:
            raw_architectures = self.controller_manager.sample_architectures(self.batch_size)
            print(f'  成功生成 {len(raw_architectures)} 个架构')
            return raw_architectures
        except Exception as e:
            print(f'  警告: 批量生成架构失败: {e}')
            raise ValueError(f'批量生成架构失败: {e}')

    def _generate_architectures_with_load(self, num_samples: int) -> List[Dict[str, Any]]:
        raw_architectures = self.controller_manager.sample_architectures(num_samples)
        architectures_with_load = []
        for arch in raw_architectures:
            try:
                load_input = self._convert_architecture_to_load_input(arch['arch_spec'])
                if self.load_predictor:
                    predicted_load = self.load_predictor.predict(load_input)
                else:
                    import random
                    predicted_load = random.uniform(10, 50)
                architectures_with_load.append({'arch_id': arch['arch_id'], 'arch_spec': arch['arch_spec'], 'load': predicted_load})
                print(f"    架构 {arch['arch_id']} 预测负载: {predicted_load:.4f}")
            except Exception as e:
                print(f"    警告: 架构 {arch['arch_id']} 负载预测失败: {e}")
                default_load = 50.0
                architectures_with_load.append({'arch_id': arch['arch_id'], 'arch_spec': arch['arch_spec'], 'load': default_load})
        return architectures_with_load

    def _convert_architecture_to_load_input(self, arch_spec: Dict[str, Any]) -> Dict[str, Any]:
        architecture_layers = []
        if 'layers' in arch_spec:
            for layer in arch_spec['layers']:
                layer_info = {'activation': layer.get('activation', 'relu'), 'attention_type': self._map_encoding_to_attention_type(layer.get('encoding', 'gcn')), 'aggregation_type': layer.get('aggregation', 'mean'), 'attention_heads': layer.get('num_heads', 1), 'hidden_units': layer.get('hidden_dim', 64)}
                architecture_layers.append(layer_info)
        elif 'cell_spec' in arch_spec:
            cell_spec = arch_spec['cell_spec']
            num_cells = arch_spec.get('num_cells_to_stack', 2)
            for _ in range(num_cells):
                main_op = 'gcn'
                if 'connections' in cell_spec and cell_spec['connections']:
                    main_op = cell_spec['connections'][0].get('op', 'gcn')
                layer_info = {'activation': cell_spec.get('final_act', 'relu'), 'attention_type': self._map_cell_op_to_attention_type(main_op), 'aggregation_type': cell_spec.get('readout_op', 'add'), 'attention_heads': self._extract_heads_from_cell_op(main_op), 'hidden_units': 64}
                architecture_layers.append(layer_info)
        else:
            print(f'警告: 未知的架构格式，使用默认配置')
            architecture_layers = [{'activation': 'relu', 'attention_type': 'gcn', 'aggregation_type': 'mean', 'attention_heads': 1, 'hidden_units': 64}]
        return {'architecture': architecture_layers, 'graph': self.graph_info, 'device': self.device_info}

    def _map_encoding_to_attention_type(self, encoding: str) -> str:
        mapping = {'gcn': 'gcn', 'gat': 'gat', 'sym-gat': 'gat', 'sage': 'gcn', 'cheb': 'gcn', 'arma': 'gcn', 'sgc': 'gcn', 'const': 'none', 'cos': 'none', 'linear': 'none', 'gene-linear': 'none'}
        return mapping.get(encoding, 'gcn')

    def _map_cell_op_to_attention_type(self, cell_op: str) -> str:
        if 'gat' in cell_op:
            return 'gat'
        elif cell_op in ['gcn', 'chebnet', 'mean_sage', 'arma', 'sgc']:
            return 'gcn'
        else:
            return 'none'

    def _extract_heads_from_cell_op(self, cell_op: str) -> int:
        if 'gat_1_head' in cell_op:
            return 1
        elif 'gat_2_head' in cell_op:
            return 2
        elif 'gat_4_head' in cell_op:
            return 4
        elif 'gat_6_head' in cell_op:
            return 6
        elif 'gat_8_head' in cell_op:
            return 8
        else:
            return 1

    def update_graph_info(self, graph_info: Dict[str, Any]):
        self.graph_info.update(graph_info)
        print(f'图信息已更新: {self.graph_info}')

    def update_device_info(self, device_info: Dict[str, Any]):
        self.device_info.update(device_info)
        print(f'设备信息已更新: {self.device_info}')

    def _predict_loads_for_all(self, architectures: List[Dict]) -> List[Dict]:
        print(f'步骤2：预测 {len(architectures)} 个架构的负载...')
        architectures_with_load = []
        for arch in architectures:
            try:
                load_input = self._convert_architecture_to_load_input(arch['arch_spec'])
                if self.load_predictor:
                    predicted_load = self.load_predictor.predict(load_input)
                else:
                    import random
                    predicted_load = random.uniform(10, 50)
                architectures_with_load.append({'arch_id': arch['arch_id'], 'arch_spec': arch['arch_spec'], 'load': predicted_load})
                print(f"    架构 {arch['arch_id']}: 负载 = {predicted_load:.4f}")
            except Exception as e:
                print(f"    警告: 架构 {arch['arch_id']} 负载预测失败: {e}")
                default_load = 25.0
                architectures_with_load.append({'arch_id': arch['arch_id'], 'arch_spec': arch['arch_spec'], 'load': default_load})
        print(f'  成功预测 {len(architectures_with_load)} 个架构的负载')
        return architectures_with_load

    def _find_optimal_assignment(self, architectures: List[Dict]) -> List[Dict]:
        best_combination = None
        best_load_diff = float('inf')
        print(f'步骤3：开始顺序评估 {len(architectures)} 个架构...')
        print(f'  要求最少 {self.min_architectures} 个架构')
        if len(architectures) < self.min_architectures:
            print(f'  ⚠️ 可用架构数量 ({len(architectures)}) 少于最小要求 ({self.min_architectures})')
            print(f'  → 返回所有可用架构')
            return architectures
        for num_arch in range(self.min_architectures, len(architectures) + 1):
            current_combination = architectures[:num_arch]
            is_balanced, load_diff = evaluate_balance(current_combination, self.num_workers, self.tolerance)
            print(f'  评估前 {num_arch} 个架构:')
            print(f'    - 负载差异: {load_diff:.4f}')
            print(f'    - 是否均衡: {is_balanced}')
            if is_balanced:
                print(f'  ✅ 找到满足容忍度的分配：使用前 {num_arch} 个架构')
                return current_combination
            if load_diff < best_load_diff:
                best_load_diff = load_diff
                best_combination = current_combination
                print(f'    📊 更新最佳组合：负载差异 {load_diff:.4f}')
        if best_combination is None or len(best_combination) < self.min_architectures:
            print(f'  ⚠️ 最佳组合数量不足，强制返回前 {self.min_architectures} 个架构')
            best_combination = architectures[:self.min_architectures]
            _, best_load_diff = evaluate_balance(best_combination, self.num_workers, self.tolerance)
        print(f'  ⚠️ 未找到完美匹配，使用最佳分配：前 {len(best_combination)} 个架构，负载差异 {best_load_diff:.4f}')
        return best_combination
if __name__ == '__main__':
    print('===== BalanceController 独立测试 =====')
    try:
        from core.controller import ControllerManager
        controller_manager = ControllerManager(search_strategy='layer_wise', num_layers=2, learning_rate=0.001)
        load_predictor = None
        if SimpleMloadPredictor:
            try:
                load_predictor = SimpleMloadPredictor()
                print('✅ 成功加载简化版负载预测器')
            except Exception as e:
                print(f'⚠️  简化版预测器加载失败: {e}')
        balance_controller = BalanceController(controller_manager=controller_manager, load_predictor=load_predictor, min_samples=3, max_samples=8, tolerance=5.0, num_workers=2)
        print('\n--- 测试架构生成和负载均衡评估 ---')
        balanced_architectures = balance_controller.generate_balanced_architectures()
        print(f'\n最终生成了 {len(balanced_architectures)} 个架构:')
        for i, arch in enumerate(balanced_architectures):
            print(f"  {i + 1}. {arch['arch_id']}: 负载={arch['load']:.4f}")
        print('\n===== 测试完成 =====')
    except Exception as e:
        print(f'❌ 测试失败: {e}')
        import traceback
        traceback.print_exc()