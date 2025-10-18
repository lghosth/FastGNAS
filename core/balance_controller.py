# -*- coding: utf-8 -*-
"""
Load Balancing Controller Module

Implements a load-balancing scheduling strategy based on load prediction, including:
1. Dynamic architecture generation
2. Load evaluation and balancing assessment
3. Bucket-based load allocation
"""

from typing import List, Dict, Any, Tuple, Optional
from core.balancing import evaluate_balance
from core.controller import ControllerManager

# Try importing torch-based predictor, fallback to a simplified version
try:
    import torch
    from core.load_estimator import MloadPredictor, Config
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    # Import simplified predictor
    try:
        from generate_load_model import SimpleMloadPredictor
    except ImportError:
        SimpleMloadPredictor = None

class BalanceController:
    """
    Manages dynamic GNN architecture generation and load balancing based on predicted load.
    
    Implements an exploratory task generation strategy:
    1. Generate a minimum number of GNN architectures.
    2. Predict the computational load for each architecture using a load estimator.
    3. Evaluate if the load balancing condition is met.
    4. If not, continue generating until the condition is met or a maximum number is reached.
    """
    
    def __init__(self, 
                 controller_manager: ControllerManager,
                 load_predictor: Optional[Any] = None,  # Supports different predictor types
                 min_samples: int = 3,
                 max_samples: int = 10,
                 tolerance: float = 0.1,
                 num_workers: int = 2,
                 graph_info: Optional[Dict[str, Any]] = None,
                 device_info: Optional[Dict[str, Any]] = None):
        """
        Initializes the Balance Controller.
        
        Args:
            controller_manager: An existing controller manager instance.
            load_predictor: The load predictor.
            min_samples: Minimum number of GNN architectures (NGmin).
            max_samples: Maximum number of GNN architectures (NGmax).
            tolerance: Load imbalance tolerance (NGT).
            num_workers: Number of worker nodes.
            graph_info: Graph information (vertices, edges, etc.).
            device_info: Hardware information (gpu_cores, etc.).
        """
        self.controller_manager = controller_manager
        self.load_predictor = load_predictor
        self.min_samples = min_samples
        self.max_samples = max_samples
        self.tolerance = tolerance
        self.num_workers = num_workers
        
        # Set default graph and device info
        self.graph_info = graph_info or {
            "vertices": 2708,  # Default for Cora
            "edges": 5429,
            "avg_degree": 3.9,
            "diameter": 10
        }
        
        self.device_info = device_info or {
            "gpu_cores": 3584,     # Default for Tesla P100
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
        
        # Fallback to simplified predictor if none provided
        if self.load_predictor is None and not TORCH_AVAILABLE and SimpleMloadPredictor:
            try:
                self.load_predictor = SimpleMloadPredictor()
                print(f"  - Auto-loaded simplified predictor")
            except Exception as e:
                print(f"  - Could not load simplified predictor: {e}")
    
    def generate_balanced_architectures(self) -> List[Dict[str, Any]]:
        """
        Generates a set of load-balanced GNN architectures.
        
        Returns:
            List[Dict]: A list of architectures, each with 'arch_id', 'arch_spec', and 'load'.
        """
        print(f"\n=== Starting balanced architecture generation ===")
        
        print(f"Step 1: Generating at least {self.min_samples} architectures...")
        architectures = self._generate_architectures_with_load(self.min_samples)
        
        current_count = len(architectures)
        print(f"  Generated {current_count} architectures.")
        
        if self.load_predictor is None:
            print("  Warning: No load predictor provided. Skipping balancing evaluation.")
            return architectures

        best_balance_result = None
        best_architectures = architectures.copy()
        
        while current_count < self.max_samples:
            # Evaluate load balance of the current set
            is_balanced, load_difference = evaluate_balance(architectures, self.num_workers, self.tolerance)
            
            print(f"  Balance evaluation for {current_count} architectures:")
            print(f"    - Balanced: {is_balanced}")
            print(f"    - Load difference: {load_difference:.4f}")
            print(f"    - Tolerance: {self.tolerance}")
            
            # Keep track of the best result
            if best_balance_result is None or load_difference < best_balance_result[1]:
                best_balance_result = (is_balanced, load_difference)
                best_architectures = architectures.copy()
            
            # Exit early if balanced
            if is_balanced:
                print(f"  Load balance condition met. Stopping generation.")
                break
            
            # Generate one more architecture
            print(f"  Not balanced. Generating architecture {current_count + 1}...")
            new_architectures = self._generate_architectures_with_load(1)
            architectures.extend(new_architectures)
            current_count = len(architectures)
        

        if current_count >= self.max_samples:
            print(f"  Reached max samples ({self.max_samples}).")
            final_is_balanced, final_load_diff = evaluate_balance(architectures, self.num_workers, self.tolerance)
            
            if best_balance_result and final_load_diff > best_balance_result[1]:
                print(f"  Using previous best result (load diff: {best_balance_result[1]:.4f}).")
                architectures = best_architectures
            else:
                print(f"  Using current result (load diff: {final_load_diff:.4f}).")
        

        final_is_balanced, final_load_diff = evaluate_balance(architectures, self.num_workers, self.tolerance)
        print(f"\n=== Architecture generation complete ===")
        print(f"  Final number of architectures: {len(architectures)}")
        print(f"  Balancing status: {'Balanced' if final_is_balanced else 'Not Balanced'}")
        print(f"  Load difference: {final_load_diff:.4f}")
        print(f"  Tolerance: {self.tolerance}")
        
        # Print load details
        print(f"\n  Load details per architecture:")
        for i, arch in enumerate(architectures):
            print(f"    Arch {i+1} ({arch['arch_id']}): Load = {arch['load']:.4f}")
        
        return architectures
    
    def _generate_architectures_with_load(self, num_samples: int) -> List[Dict[str, Any]]:
        """
        Generates architectures and predicts their load.
        """

        raw_architectures = self.controller_manager.sample_architectures(num_samples)
        
        architectures_with_load = []
        for arch in raw_architectures:
            try:
                # Convert arch spec to the format required by the load predictor
                load_input = self._convert_architecture_to_load_input(arch['arch_spec'])
                
                # Predict load
                if self.load_predictor:
                    predicted_load = self.load_predictor.predict(load_input)
                else:
                    # Use a simple estimation if no predictor is available
                    import random
                    predicted_load = random.uniform(10, 50)
                
                architectures_with_load.append({
                    'arch_id': arch['arch_id'],
                    'arch_spec': arch['arch_spec'],
                    'load': predicted_load
                })
                
                print(f"    Arch {arch['arch_id']} predicted load: {predicted_load:.4f}")
                
            except Exception as e:
                print(f"    Warning: Load prediction failed for arch {arch['arch_id']}: {e}")
                # Use a default load on failure
                default_load = 50.0
                architectures_with_load.append({
                    'arch_id': arch['arch_id'],
                    'arch_spec': arch['arch_spec'],
                    'load': default_load
                })
        
        return architectures_with_load
    
    def _convert_architecture_to_load_input(self, arch_spec: Dict[str, Any]) -> Dict[str, Any]:
        """
        Converts a GNN arch spec to the input format for the load predictor.
        """
        architecture_layers = []
        
        if 'layers' in arch_spec:
            # Layer-wise strategy
            for layer in arch_spec['layers']:
                layer_info = {
                    'activation': layer.get('activation', 'relu'),
                    'attention_type': self._map_encoding_to_attention_type(layer.get('encoding', 'gcn')),
                    'aggregation_type': layer.get('aggregation', 'mean'),
                    'attention_heads': layer.get('num_heads', 1),
                    'hidden_units': layer.get('hidden_dim', 64)
                }
                architecture_layers.append(layer_info)
        
        elif 'cell_spec' in arch_spec:
            # Cell-based strategy - convert to equivalent layer-wise representation
            cell_spec = arch_spec['cell_spec']
            num_cells = arch_spec.get('num_cells_to_stack', 2)
            
            for _ in range(num_cells):
                # Use the main operation type from the cell
                main_op = 'gcn'  # Default
                if 'connections' in cell_spec and cell_spec['connections']:
                    main_op = cell_spec['connections'][0].get('op', 'gcn')
                
                layer_info = {
                    'activation': cell_spec.get('final_act', 'relu'),
                    'attention_type': self._map_cell_op_to_attention_type(main_op),
                    'aggregation_type': cell_spec.get('readout_op', 'add'),
                    'attention_heads': self._extract_heads_from_cell_op(main_op),
                    'hidden_units': 64  # Default hidden units
                }
                architecture_layers.append(layer_info)
        
        else:
            # Unknown format, create a default layer
            print(f"Warning: Unknown architecture format, using default.")
            architecture_layers = [{
                'activation': 'relu', 'attention_type': 'gcn',
                'aggregation_type': 'mean', 'attention_heads': 1, 'hidden_units': 64
            }]
        
        return {
            'architecture': architecture_layers,
            'graph': self.graph_info,
            'device': self.device_info
        }
    
    def _map_encoding_to_attention_type(self, encoding: str) -> str:
        """Maps layer-wise encoding to predictor's attention_type."""
        mapping = {
            'gcn': 'gcn', 'gat': 'gat', 'sym-gat': 'gat', 'sage': 'gcn', 
            'cheb': 'gcn', 'arma': 'gcn', 'sgc': 'gcn', 'const': 'none', 
            'cos': 'none', 'linear': 'none', 'gene-linear': 'none'
        }
        return mapping.get(encoding, 'gcn')
    
    def _map_cell_op_to_attention_type(self, cell_op: str) -> str:
        """Maps cell-based op to predictor's attention_type."""
        if 'gat' in cell_op:
            return 'gat'
        elif cell_op in ['gcn', 'chebnet', 'mean_sage', 'arma', 'sgc']:
            return 'gcn'
        else:
            return 'none'
    
    def _extract_heads_from_cell_op(self, cell_op: str) -> int:
        """Extracts number of attention heads from cell op name."""
        if 'gat_1_head' in cell_op: return 1
        if 'gat_2_head' in cell_op: return 2
        if 'gat_4_head' in cell_op: return 4
        if 'gat_6_head' in cell_op: return 6
        if 'gat_8_head' in cell_op: return 8
        return 1  # Default
    
    def update_graph_info(self, graph_info: Dict[str, Any]):
        """Updates graph information."""
        self.graph_info.update(graph_info)
        print(f"Graph info updated: {self.graph_info}")
    
    def update_device_info(self, device_info: Dict[str, Any]):
        """Updates device information."""
        self.device_info.update(device_info)
        print(f"Device info updated: {self.device_info}")



if __name__ == '__main__':
    print("===== Independent Test for BalanceController =====")
    
    try:

        from core.controller import ControllerManager
        controller_manager = ControllerManager(
            search_strategy='layer_wise',
            num_layers=2,
            learning_rate=0.001
        )
        

        load_predictor = None
        if SimpleMloadPredictor:
            try:
                load_predictor = SimpleMloadPredictor()
                print("Successfully loaded simplified predictor.")
            except Exception as e:
                print(f"Warning: Failed to load simplified predictor: {e}")
        

        balance_controller = BalanceController(
            controller_manager=controller_manager,
            load_predictor=load_predictor,
            min_samples=3,
            max_samples=8,
            tolerance=5.0,  # Relax tolerance for testing
            num_workers=2
        )
        

        print("\n--- Testing architecture generation and balancing ---")
        balanced_architectures = balance_controller.generate_balanced_architectures()
        

        print(f"\nGenerated {len(balanced_architectures)} architectures:")
        for i, arch in enumerate(balanced_architectures):
            print(f"  {i+1}. {arch['arch_id']}: Load={arch['load']:.4f}")
        
        print("\n===== Test complete =====")
        
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
