# -*- coding: utf-8 -*-
"""
负载均衡控制器模块

该模块实现基于负载预测的均衡调度策略，包括：
1. 动态架构生成
2. 负载评估和均衡判断
3. 桶算法负载分配
"""

from typing import List, Dict, Any, Tuple, Optional
from core.balancing import evaluate_balance
from core.controller import ControllerManager

# 尝试导入torch版本的预测器，如果失败则使用简化版
try:
    import torch
    from core.load_estimator import MloadPredictor, Config
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    # 导入简化版预测器
    try:
        from generate_load_model import SimpleMloadPredictor
    except ImportError:
        SimpleMloadPredictor = None

class BalanceController:
    """
    负载均衡控制器，负责根据预测负载动态生成GNN架构并评估负载均衡。
    
    该类实现了文献中描述的探索式任务生成策略：
    1. 生成最小数量的GNN架构
    2. 使用负载估算器预测各架构的计算负载
    3. 评估负载均衡条件
    4. 如果不满足条件则继续生成，直到满足条件或达到最大数量
    """
    
    def __init__(self, 
                 controller_manager: ControllerManager,
                 load_predictor: Optional[Any] = None,  # 支持不同类型的预测器
                 min_samples: int = 3,
                 max_samples: int = 10,
                 tolerance: float = 0.1,
                 num_workers: int = 2,
                 graph_info: Optional[Dict[str, Any]] = None,
                 device_info: Optional[Dict[str, Any]] = None):
        """
        初始化负载均衡控制器
        
        Args:
            controller_manager: 现有的控制器管理器
            load_predictor: 负载预测器
            min_samples: 最小GNN架构数量 (NGmin)
            max_samples: 最大GNN架构数量 (NGmax)
            tolerance: 负载不均衡容忍度 (NGT)
            num_workers: 工作节点数量
            graph_info: 图数据信息 (vertices, edges, avg_degree, diameter)
            device_info: 硬件信息 (gpu_cores, core_frequency, fp_capability)
        """
        self.controller_manager = controller_manager
        self.load_predictor = load_predictor
        self.min_samples = min_samples
        self.max_samples = max_samples
        self.tolerance = tolerance
        self.num_workers = num_workers
        
        # 设置默认的图和设备信息
        self.graph_info = graph_info or {
            "vertices": 2708,  # Cora数据集默认值
            "edges": 5429,
            "avg_degree": 3.9,
            "diameter": 10
        }
        
        self.device_info = device_info or {
            "gpu_cores": 3584,     # Tesla P100默认值
            "core_frequency": 1.328,
            "fp_capability": 6.0
        }
        
        print(f"BalanceController initialized:")
        print(f"  - Min samples: {self.min_samples}")
        print(f"  - Max samples: {self.max_samples}")
        print(f"  - Tolerance: {self.tolerance}")
        print(f"  - Workers: {self.num_workers}")
        print(f"  - Load predictor available: {self.load_predictor is not None}")
        print(f"  - Torch available: {TORCH_AVAILABLE}")
        
        # 如果没有提供预测器，尝试加载简化版
        if self.load_predictor is None and not TORCH_AVAILABLE and SimpleMloadPredictor:
            try:
                self.load_predictor = SimpleMloadPredictor()
                print(f"  - Auto-loaded simplified predictor")
            except Exception as e:
                print(f"  - Could not load simplified predictor: {e}")
    
    def generate_balanced_architectures(self) -> List[Dict[str, Any]]:
        """
        生成负载均衡的GNN架构集合
        
        Returns:
            List[Dict]: 架构列表，每个dict包含 'arch_id', 'arch_spec', 'load'
        """
        print(f"\n=== 开始生成负载均衡的架构集合 ===")
        
        # 1. 生成最小数量的架构
        print(f"步骤 1: 生成最少 {self.min_samples} 个架构...")
        architectures = self._generate_architectures_with_load(self.min_samples)
        
        current_count = len(architectures)
        print(f"  已生成 {current_count} 个架构")
        
        # 2. 如果没有负载预测器，直接返回
        if self.load_predictor is None:
            print("  警告: 未提供负载预测器，无法进行负载均衡评估")
            return architectures
        
        # 3. 循环检查均衡条件
        best_balance_result = None
        best_architectures = architectures.copy()
        
        while current_count < self.max_samples:
            # 评估当前架构集合的负载均衡
            is_balanced, load_difference = evaluate_balance(architectures, self.num_workers, self.tolerance)
            
            print(f"  当前 {current_count} 个架构的负载均衡评估:")
            print(f"    - 是否均衡: {is_balanced}")
            print(f"    - 负载差异: {load_difference:.4f}")
            print(f"    - 容忍度: {self.tolerance}")
            
            # 记录最佳均衡结果
            if best_balance_result is None or load_difference < best_balance_result[1]:
                best_balance_result = (is_balanced, load_difference)
                best_architectures = architectures.copy()
            
            # 如果满足均衡条件，提前结束
            if is_balanced:
                print(f"  ✅ 已达到负载均衡条件，停止生成")
                break
            
            # 生成一个新架构
            print(f"  负载不均衡，继续生成第 {current_count + 1} 个架构...")
            new_architectures = self._generate_architectures_with_load(1)
            architectures.extend(new_architectures)
            current_count = len(architectures)
        
        # 4. 如果达到最大数量仍未均衡，使用最佳结果
        if current_count >= self.max_samples:
            print(f"  已达到最大架构数量 {self.max_samples}")
            final_is_balanced, final_load_diff = evaluate_balance(architectures, self.num_workers, self.tolerance)
            
            if best_balance_result and final_load_diff > best_balance_result[1]:
                print(f"  使用之前的最佳均衡结果 (负载差异: {best_balance_result[1]:.4f})")
                architectures = best_architectures
            else:
                print(f"  使用当前结果 (负载差异: {final_load_diff:.4f})")
        
        # 5. 输出最终结果
        final_is_balanced, final_load_diff = evaluate_balance(architectures, self.num_workers, self.tolerance)
        print(f"\n=== 架构生成完成 ===")
        print(f"  最终架构数量: {len(architectures)}")
        print(f"  负载均衡状态: {'均衡' if final_is_balanced else '不均衡'}")
        print(f"  负载差异: {final_load_diff:.4f}")
        print(f"  容忍度: {self.tolerance}")
        
        # 打印每个架构的负载信息
        print(f"\n  各架构负载详情:")
        for i, arch in enumerate(architectures):
            print(f"    架构{i+1} ({arch['arch_id']}): 负载 = {arch['load']:.4f}")
        
        return architectures
    
    def _generate_architectures_with_load(self, num_samples: int) -> List[Dict[str, Any]]:
        """
        生成指定数量的架构并预测其负载
        
        Args:
            num_samples: 要生成的架构数量
            
        Returns:
            List[Dict]: 包含负载信息的架构列表
        """
        # 1. 使用控制器管理器生成架构
        raw_architectures = self.controller_manager.sample_architectures(num_samples)
        
        # 2. 为每个架构预测负载
        architectures_with_load = []
        for arch in raw_architectures:
            try:
                # 转换架构格式为负载预测器可接受的格式
                load_input = self._convert_architecture_to_load_input(arch['arch_spec'])
                
                # 预测负载
                if self.load_predictor:
                    predicted_load = self.load_predictor.predict(load_input)
                else:
                    # 如果没有预测器，使用简单的估算方法
                    import random
                    predicted_load = random.uniform(10, 50)
                
                architectures_with_load.append({
                    'arch_id': arch['arch_id'],
                    'arch_spec': arch['arch_spec'],
                    'load': predicted_load
                })
                
                print(f"    架构 {arch['arch_id']} 预测负载: {predicted_load:.4f}")
                
            except Exception as e:
                print(f"    警告: 架构 {arch['arch_id']} 负载预测失败: {e}")
                # 使用默认负载
                default_load = 50.0
                architectures_with_load.append({
                    'arch_id': arch['arch_id'],
                    'arch_spec': arch['arch_spec'],
                    'load': default_load
                })
        
        return architectures_with_load
    
    def _convert_architecture_to_load_input(self, arch_spec: Dict[str, Any]) -> Dict[str, Any]:
        """
        将GNN架构规范转换为负载预测器可接受的输入格式
        
        Args:
            arch_spec: GNN架构规范
            
        Returns:
            Dict: 负载预测器输入格式
        """
        architecture_layers = []
        
        if 'layers' in arch_spec:
            # Layer-wise 策略
            for layer in arch_spec['layers']:
                # 映射架构字段到负载预测器期望的字段名
                layer_info = {
                    'activation': layer.get('activation', 'relu'),
                    'attention_type': self._map_encoding_to_attention_type(layer.get('encoding', 'gcn')),
                    'aggregation_type': layer.get('aggregation', 'mean'),
                    'attention_heads': layer.get('num_heads', 1),
                    'hidden_units': layer.get('hidden_dim', 64)
                }
                architecture_layers.append(layer_info)
        
        elif 'cell_spec' in arch_spec:
            # Cell-based 策略 - 转换为等效的layer-wise表示
            cell_spec = arch_spec['cell_spec']
            num_cells = arch_spec.get('num_cells_to_stack', 2)
            
            # 为每个cell创建一个等效的层
            for _ in range(num_cells):
                # 使用cell中最主要的操作类型
                main_op = 'gcn'  # 默认值
                if 'connections' in cell_spec and cell_spec['connections']:
                    main_op = cell_spec['connections'][0].get('op', 'gcn')
                
                layer_info = {
                    'activation': cell_spec.get('final_act', 'relu'),
                    'attention_type': self._map_cell_op_to_attention_type(main_op),
                    'aggregation_type': cell_spec.get('readout_op', 'add'),
                    'attention_heads': self._extract_heads_from_cell_op(main_op),
                    'hidden_units': 64  # 默认隐藏单元数
                }
                architecture_layers.append(layer_info)
        
        else:
            # 未知格式，创建默认层
            print(f"警告: 未知的架构格式，使用默认配置")
            architecture_layers = [{
                'activation': 'relu',
                'attention_type': 'gcn',
                'aggregation_type': 'mean',
                'attention_heads': 1,
                'hidden_units': 64
            }]
        
        return {
            'architecture': architecture_layers,
            'graph': self.graph_info,
            'device': self.device_info
        }
    
    def _map_encoding_to_attention_type(self, encoding: str) -> str:
        """将layer-wise编码映射到负载预测器的attention_type"""
        mapping = {
            'gcn': 'gcn',
            'gat': 'gat',
            'sym-gat': 'gat',
            'sage': 'gcn',  # GraphSAGE映射为gcn类型
            'cheb': 'gcn',
            'arma': 'gcn',
            'sgc': 'gcn',
            'const': 'none',
            'cos': 'none',
            'linear': 'none',
            'gene-linear': 'none'
        }
        return mapping.get(encoding, 'gcn')
    
    def _map_cell_op_to_attention_type(self, cell_op: str) -> str:
        """将cell-based操作映射到负载预测器的attention_type"""
        if 'gat' in cell_op:
            return 'gat'
        elif cell_op in ['gcn', 'chebnet', 'mean_sage', 'arma', 'sgc']:
            return 'gcn'
        else:
            return 'none'
    
    def _extract_heads_from_cell_op(self, cell_op: str) -> int:
        """从cell操作中提取注意力头数"""
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
            return 1  # 默认值
    
    def update_graph_info(self, graph_info: Dict[str, Any]):
        """更新图数据信息"""
        self.graph_info.update(graph_info)
        print(f"图信息已更新: {self.graph_info}")
    
    def update_device_info(self, device_info: Dict[str, Any]):
        """更新设备信息"""
        self.device_info.update(device_info)
        print(f"设备信息已更新: {self.device_info}")


# ===================================================================================
# | 测试模块
# ===================================================================================
if __name__ == '__main__':
    print("===== BalanceController 独立测试 =====")
    
    try:
        # 1. 初始化控制器管理器
        from core.controller import ControllerManager
        controller_manager = ControllerManager(
            search_strategy='layer_wise',
            num_layers=2,
            learning_rate=0.001
        )
        
        # 2. 尝试加载简化版预测器
        load_predictor = None
        if SimpleMloadPredictor:
            try:
                load_predictor = SimpleMloadPredictor()
                print("✅ 成功加载简化版负载预测器")
            except Exception as e:
                print(f"⚠️  简化版预测器加载失败: {e}")
        
        # 3. 创建负载均衡控制器
        balance_controller = BalanceController(
            controller_manager=controller_manager,
            load_predictor=load_predictor,
            min_samples=3,
            max_samples=8,
            tolerance=5.0,  # 放宽容忍度以便测试
            num_workers=2
        )
        
        # 4. 测试架构生成
        print("\n--- 测试架构生成和负载均衡评估 ---")
        balanced_architectures = balance_controller.generate_balanced_architectures()
        
        # 5. 输出结果
        print(f"\n最终生成了 {len(balanced_architectures)} 个架构:")
        for i, arch in enumerate(balanced_architectures):
            print(f"  {i+1}. {arch['arch_id']}: 负载={arch['load']:.4f}")
        
        print("\n===== 测试完成 =====")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
