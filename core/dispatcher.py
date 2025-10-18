# graphnas_dist/core/dispatcher.py (modified)

import torch
import ray
from typing import List, Dict, Any, Tuple, Optional
import sys
import os
import pprint

# Project module imports
from data.partition import partition_training_graph
from trainer.worker import TrainerWorker
from core.balance_controller import BalanceController
from core.load_estimator import MloadPredictor

@ray.remote(num_gpus=0)  
class Dispatcher:
    """
    Orchestrates the "data partition + model rotation" workflow.
    """
    
    def __init__(self, 
                 worker_refs: Optional[List[ray.ObjectRef]] = None,
                 train_config: Optional[Dict[str, Any]] = None,
                 balance_controller: Optional[BalanceController] = None,
                 load_predictor: Optional[MloadPredictor] = None,
                 enable_balancing: bool = False): # training config
                 
        self.workers = worker_refs or []
        self.num_workers = len(self.workers)
        self.config = train_config or {}
        
        # Load balancing related
        self.balance_controller = balance_controller
        self.load_predictor = load_predictor
        self.enable_balancing = enable_balancing
        
        # model_pool stores all architectures to be trained
        self.model_pool: List[Dict] = []
        # model_states stores the latest parameters and best performance for each model
        # Format: {arch_id: {'spec': ..., 'state': ..., 'best_acc': ..., 'load': ...}}
        self.model_states: Dict[str, Dict] = {}
        
        print(f"Dispatcher initialized for model rotation with {self.num_workers} workers.")
        print(f"Load balancing enabled: {self.enable_balancing}")
        if self.enable_balancing:
            print(f"Balance controller available: {self.balance_controller is not None}")
            print(f"Load predictor available: {self.load_predictor is not None}")

    def get_architectures(self, format_type: str = 'detailed', sort_by: str = 'best_acc') -> Dict[str, Any]:
        """
        Returns the current status of all architectures.
        
        Args:
            format_type (str): Return format type ('detailed', 'summary', 'states_only', 'pool_only').
            sort_by (str): Sort order ('best_acc', 'arch_id', 'none').
        
        Returns:
            Dict[str, Any]: A dictionary containing architecture status information.
        """
        
        if format_type == 'states_only':
            return self.model_states
        
        if format_type == 'pool_only':
            return {'model_pool': self.model_pool}
        
        # Create basic information
        result = {
            'total_architectures': len(self.model_states),
            'active_models_in_pool': len(self.model_pool),
            'num_workers': self.num_workers,
            'architectures': {}
        }
        
        # Get and sort the list of architectures
        arch_items = list(self.model_states.items())
        
        if sort_by == 'best_acc':
            arch_items.sort(key=lambda x: x[1]['best_acc'], reverse=True)
        elif sort_by == 'arch_id':
            arch_items.sort(key=lambda x: x[0])
        
        # Generate output based on format type
        if format_type == 'summary':
            result['architectures'] = {
                arch_id: {
                    'best_accuracy': info['best_acc'],
                    'has_trained_state': info['state'] is not None,
                    'is_in_pool': any(model['arch_id'] == arch_id for model in self.model_pool)
                }
                for arch_id, info in arch_items
            }
            # Add statistics
            result['statistics'] = {
                'best_overall_accuracy': max([info['best_acc'] for info in self.model_states.values()], default=0.0),
                'average_accuracy': sum([info['best_acc'] for info in self.model_states.values()]) / len(self.model_states) if self.model_states else 0.0,
                'trained_models': sum(1 for info in self.model_states.values() if info['state'] is not None),
                'untrained_models': sum(1 for info in self.model_states.values() if info['state'] is None)
            }
        else:  # 'detailed' format (default)
            result['architectures'] = {
                arch_id: {
                    'architecture_spec': info['spec'],
                    'best_accuracy': info['best_acc'],
                    'has_trained_state': info['state'] is not None,
                    'state_dict_size': len(info['state']) if info['state'] is not None else 0,
                    'is_in_pool': any(model['arch_id'] == arch_id for model in self.model_pool),
                    'pool_position': next((i for i, model in enumerate(self.model_pool) if model['arch_id'] == arch_id), -1)
                }
                for arch_id, info in arch_items
            }
            # Add detailed statistics
            accuracies = [info['best_acc'] for info in self.model_states.values()]
            result['statistics'] = {
                'best_overall_accuracy': max(accuracies, default=0.0),
                'worst_accuracy': min(accuracies, default=0.0),
                'average_accuracy': sum(accuracies) / len(accuracies) if accuracies else 0.0,
                'accuracy_std': torch.std(torch.tensor(accuracies)).item() if len(accuracies) > 1 else 0.0,
                'trained_models': sum(1 for info in self.model_states.values() if info['state'] is not None),
                'untrained_models': sum(1 for info in self.model_states.values() if info['state'] is None)
            }
            # Add model pool information
            result['model_pool_info'] = {
                'current_rotation_order': [model['arch_id'] for model in self.model_pool],
                'next_to_train': self.model_pool[0]['arch_id'] if self.model_pool else None
            }
            # Add configuration information
            result['training_config'] = self.config.copy() if self.config else {}
            
        return result

    def submit_architectures(self, architectures: List[Dict]):
        """API: Submit a batch of architectures to train (each with 'arch_id', 'arch_spec')."""
        for arch in architectures:
            if arch['arch_id'] not in self.model_states:
                self.model_pool.append(arch)
                self.model_states[arch['arch_id']] = {
                    'spec': arch['arch_spec'],
                    'state': None, # initial state is None
                    'best_acc': 0.0,
                    'load': arch.get('load', 0.0)  # supports load info
                }
        # Expand the number of models to match the number of workers for rotation
        while 0 < len(self.model_pool) < self.num_workers:
            self.model_pool.extend(self.model_pool)
        self.model_pool = self.model_pool[:self.num_workers]
    
    def submit_balanced_architectures(self) -> int:
        """
        Generate and submit architectures using the load balancing strategy.
        
        Returns:
            int: The number of architectures actually generated.
        """
        if not self.enable_balancing or self.balance_controller is None:
            print("Warning: Load balancing is disabled or BalanceController is not set")
            return 0
        
        print("\n=== Generating architectures with load balancing strategy ===")
        
        # Generate balanced architectures using BalanceController
        balanced_architectures = self.balance_controller.generate_balanced_architectures()
        
        # Convert format and submit
        architectures_to_submit = [
            {'arch_id': arch['arch_id'], 'arch_spec': arch['arch_spec'], 'load': arch['load']}
            for arch in balanced_architectures
        ]
        
        self.submit_architectures(architectures_to_submit)
        
        print(f"Submitted {len(architectures_to_submit)} balanced architectures")
        return len(architectures_to_submit)

    def run_training_rounds(self, num_complete_rounds: int):
        """
        Executes a specified number of complete training rounds.
        A complete round means each model is trained once on every data partition.
        
        Args:
            num_complete_rounds (int): Number of complete training rounds.
        """
        if not self.model_pool:
            print("No models in the pool to train.")
            return

        # Calculate total underlying rotation rounds
        total_rotation_rounds = num_complete_rounds * self.num_workers
        
        print(f"\n=== Starting {num_complete_rounds} complete training rounds ===")
        print(f"Each complete round consists of {self.num_workers} rotations")
        print(f"A total of {total_rotation_rounds} underlying rotations will be executed")
        
        # Track the completion of each full round
        for complete_round in range(1, num_complete_rounds + 1):
            print(f"\n{'='*60}")
            print(f"Starting complete round {complete_round}/{num_complete_rounds}")
            print(f"{'='*60}")
            
            # In one complete round, num_workers rotations are needed
            for rotation_step in range(self.num_workers):
                rotation_round = (complete_round - 1) * self.num_workers + rotation_step + 1
                print(f"\n--- Rotation {rotation_round}/{total_rotation_rounds} (Complete Round {complete_round}, Step {rotation_step + 1}/{self.num_workers}) ---")
                

                futures = []
                for i, worker in enumerate(self.workers):
                    # Determine the model to train based on the current rotation position
                    model_to_train = self.model_pool[i]
                    arch_id = model_to_train['arch_id']
                    model_info = self.model_states[arch_id]
                    
                    print(f"  - Worker {i} will train model '{arch_id}' on partition {i}.")
                    
                    future = worker.train_model_on_partition.remote(
                        arch_spec=model_info['spec'],
                        state_dict=model_info['state'],
                        config=self.config
                    )
                    futures.append((arch_id, future))
                

                for arch_id, future in futures:
                    val_acc, updated_state = ray.get(future)
                    self.model_states[arch_id]['state'] = updated_state
                    if val_acc > self.model_states[arch_id]['best_acc']:
                        self.model_states[arch_id]['best_acc'] = val_acc
                        print(f"  New best accuracy for '{arch_id}': {val_acc:.4f}")


                if rotation_step < self.num_workers - 1:
                    self.model_pool.insert(0, self.model_pool.pop())
                    print(f"  - Model rotation complete. Next assignment: {self.model_pool[0]['arch_id']} -> Worker 0.")
            
            print(f"\nComplete round {complete_round}/{num_complete_rounds} finished!")
            print(f"   All models have been trained once on all {self.num_workers} data partitions.")
            
            if complete_round < num_complete_rounds:
                print(f"   Preparing for the next complete round...")
        
        print(f"\n{'='*60}")
        final_epochs = num_complete_rounds * int(self.config.get('epochs_per_rotation', 5)) * int(self.config.get('steps_per_epoch', 1))
        print(f"All {num_complete_rounds} complete training rounds are finished!")
        print(f"   Each model was trained a total of {num_complete_rounds * self.num_workers * final_epochs} times.")
        print(f"   Each model was trained {int(self.config.get('epochs_per_rotation', 5)) * int(self.config.get('steps_per_epoch', 1))} times on each data partition.")
        print(f"{'='*60}")

if __name__ == '__main__':
    # Add project root to Python path to ensure correct imports
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.append(project_root)

    print("===== Running Dispatcher Integration Test =====")

    # --- Test Configuration ---
    DATASET = 'Cora'
    # Auto-detect GPUs, or simulate 2 CPU workers
    NUM_WORKERS = int(torch.cuda.device_count()) if torch.cuda.is_available() else 2
    MAX_HOPS = 3  # Must be >= max layers in any model
    TRAINING_ROUNDS = 10 
    EPOCHS_PER_ROTATION = 5
    LEARNING_RATE = 0.01

    if NUM_WORKERS == 0:
        print("Warning: No GPU detected. Simulating with 2 CPU workers.")
        NUM_WORKERS = 2

    # Define two model architectures for testing
    architectures_to_test = [
        {'arch_id': 'GCN-2-layer', 'arch_spec': {'layers': [{'type': 'gcn'}] * 2}},
        {'arch_id': 'GCN-3-layer', 'arch_spec': {'layers': [{'type': 'gcn'}] * 3}},
    ]

    try:

        print(f"\n[Step 1] Initializing Ray with {NUM_WORKERS} Workers...")
        if torch.cuda.is_available():
            ray.init(num_gpus=NUM_WORKERS, ignore_reinit_error=True)
        else:
            ray.init(num_cpus=NUM_WORKERS, ignore_reinit_error=True)
        print(f"   -> Ray initialized successfully. Available resources: {ray.available_resources()}")


        print(f"\n[Step 2] Partitioning '{DATASET}' for {NUM_WORKERS} workers...")
        partitions = partition_training_graph(DATASET, NUM_WORKERS, MAX_HOPS)
        print(f"   -> Dataset partitioned into {len(partitions)} parts.")


        print(f"\n[Step 3] Creating {NUM_WORKERS} TrainerWorkers...")
        workers = [
            TrainerWorker.options(num_gpus=1 if torch.cuda.is_available() else 0).remote(
                worker_id=i,
                gpu_id=i if torch.cuda.is_available() else -1, # CPU worker uses -1
                data_partition=partitions[i]
            ) for i in range(NUM_WORKERS)
        ]
        print(f"   -> {len(workers)} workers created successfully.")


        print("\n[Step 4] Creating Dispatcher...")
        train_config = {'epochs_per_rotation': EPOCHS_PER_ROTATION, 'lr': LEARNING_RATE}
        dispatcher = Dispatcher.remote(worker_refs=workers, train_config=train_config)
        print("   -> Dispatcher created successfully.")


        print(f"\n[Step 5] Submitting {len(architectures_to_test)} model architectures...")
        ray.get(dispatcher.submit_architectures.remote(architectures_to_test))
        print("   -> Model architectures submitted successfully.")


        print(f"\n[Step 6] Starting {TRAINING_ROUNDS} training rounds...")
        ray.get(dispatcher.run_training_rounds.remote(num_complete_rounds=TRAINING_ROUNDS))
        print("   -> Training execution finished.")


        print("\n[Step 7] Fetching and displaying final evaluation results...")
        final_results = ray.get(dispatcher.get_architectures.remote())

        print("\n" + "="*50)
        print("           Final Model Performance Evaluation")
        print("="*50)
        pprint.pprint(final_results)
        print("="*50)

    except Exception as e:
        print(f"\nAn error occurred during testing: {e}")
        import traceback
        traceback.print_exc()

    finally:

        print("\n[Step 8] Shutting down Ray...")
        ray.shutdown()
        print("   -> Ray has been shut down. Test finished.")
