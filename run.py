# file: run.py

import os
import sys
import yaml
import torch
import ray
import argparse
import pprint
from tqdm import tqdm
from datetime import datetime

# Add project root to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)

from core.controller import ControllerManager
from core.dispatcher import Dispatcher
from trainer.worker import TrainerWorker
from data.data_utils import get_dataset
from data.partition import partition_training_graph
from utils.logger import GraphNASLogger
from core.balancing import evaluate_balance
from core.load_estimator import MloadPredictor, Config
from core.balance_controller import BalanceController

def format_architecture_for_display(arch_spec: dict, search_strategy: str) -> str:
    """
    Format architecture spec for display.
    """
    lines = []
    lines.append("GNN Architecture Details:")
    lines.append("─" * 50)
    
    if search_strategy == "layer_wise":
        lines.append("Architecture Type: Layer-wise Search")
        if 'layers' in arch_spec:
            lines.append(f"Total Layers: {len(arch_spec['layers'])}")
            lines.append("")
            
            for i, layer in enumerate(arch_spec['layers']):
                layer_num = i + 1
                lines.append(f"Layer {layer_num}:")
                
                encoding = layer.get('encoding', 'unknown')
                gnn_type_map = {
                    'gcn': 'GCN', 'gat': 'GAT', 'sym-gat': 'Symmetric GAT',
                    'sage': 'GraphSAGE', 'cheb': 'ChebNet', 'arma': 'ARMA',
                    'sgc': 'SGC', 'const': 'Constant', 'cos': 'Cosine',
                    'linear': 'Linear', 'gene-linear': 'Generalized Linear'
                }
                gnn_name = gnn_type_map.get(encoding, f"Unknown ({encoding})")
                lines.append(f"  Model Type: {gnn_name}")
                
                if 'hidden_dim' in layer:
                    lines.append(f"  Hidden Dim: {layer['hidden_dim']}")
                if 'num_heads' in layer and layer.get('num_heads', 1) > 1:
                    lines.append(f"  Attention Heads: {layer['num_heads']}")
                if 'activation' in layer:
                    act_map = {
                        'relu': 'ReLU', 'tanh': 'Tanh', 'sigmoid': 'Sigmoid',
                        'elu': 'ELU', 'leaky_relu': 'LeakyReLU', 'none': 'None',
                        'linear': 'Linear', 'softplus': 'Softplus', 'relu6': 'ReLU6'
                    }
                    act_name = act_map.get(layer['activation'], layer['activation'])
                    lines.append(f"  Activation: {act_name}")
                if 'aggregation' in layer:
                    agg_map = {'mean': 'Mean', 'sum': 'Sum', 'max': 'Max', 'mlp': 'MLP'}
                    agg_name = agg_map.get(layer['aggregation'], layer['aggregation'])
                    lines.append(f"  Aggregation: {agg_name}")
                lines.append("")
                
    elif search_strategy == "cell_based":
        lines.append("Architecture Type: Cell-based Search")
        if 'cell_spec' in arch_spec:
            cell_spec = arch_spec['cell_spec']
            num_cells = arch_spec.get('num_cells_to_stack', 2)
            lines.append(f"Stacked Cells: {num_cells}")
            lines.append(f"Intermediate Nodes: {cell_spec.get('intermediate_nodes', 'unknown')}")
            lines.append("")
            
            if 'connections' in cell_spec:
                lines.append("Cell Connections:")
                for i, conn in enumerate(cell_spec['connections']):
                    op = conn.get('op', 'unknown')
                    from_node = conn.get('from_node', 'unknown')
                    
                    op_type_map = {
                        'gcn': 'GCN', 'gat_1_head': 'GAT (1 head)',
                        'gat_2_head': 'GAT (2 heads)', 'gat_4_head': 'GAT (4 heads)',
                        'gat_6_head': 'GAT (6 heads)', 'gat_8_head': 'GAT (8 heads)',
                        'mean_sage': 'GraphSAGE (Mean)', 'chebnet': 'ChebNet',
                        'arma': 'ARMA', 'sgc': 'SGC', 'identity': 'Identity', 'zero': 'Zero'
                    }
                    op_name = op_type_map.get(op, f"Unknown ({op})")
                    lines.append(f"  Connection {i+1}: Node {from_node} -> {op_name}")
            
            if 'readout_op' in cell_spec:
                readout_map = {'add': 'Add', 'concat': 'Concat', 'multiply': 'Multiply'}
                readout_name = readout_map.get(cell_spec['readout_op'], cell_spec['readout_op'])
                lines.append(f"  Readout Op: {readout_name}")
            
            if 'final_act' in cell_spec:
                act_map = {
                    'relu': 'ReLU', 'tanh': 'Tanh', 'sigmoid': 'Sigmoid', 'elu': 'ELU',
                    'leaky_relu': 'LeakyReLU', 'none': 'None', 'linear': 'Linear',
                    'softplus': 'Softplus', 'relu6': 'ReLU6'
                }
                act_name = act_map.get(cell_spec['final_act'], cell_spec['final_act'])
                lines.append(f"  Final Activation: {act_name}")
    else:
        lines.append(f"Architecture Type: Unknown ({search_strategy})")
        lines.append("Original arch spec:")
        lines.append(pprint.pformat(arch_spec, width=60))
    
    lines.append("─" * 50)
    return "\n".join(lines)

def load_config(config_path: str) -> dict:
    """
    Load YAML config. Create default if not exists.
    """
    default_config = {
        'nas': {
            'num_search_rounds': 200, 'samples_per_round': 5, 'gnn_layers': 3,
            'controller_lr': 0.001, 'controller_baseline_decay': 0.9,
        },
        'training': {
            'dataset': 'Cora', 'lr': 0.01, 'dropout': 0.6,
            'epochs_per_rotation': 10, 'batch_size': 64, 'steps_per_epoch': 50,
        }
    }

    if not os.path.exists(config_path):
        print(f"Warning: Config '{config_path}' not found. Creating default.")
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        with open(config_path, 'w') as f:
            yaml.dump(default_config, f, sort_keys=False)
        return default_config

    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f) or {}
        # Deep merge with default
        for key, value in default_config.items():
            if key not in config:
                config[key] = value
            elif isinstance(value, dict):
                config[key] = {**value, **config[key]}
        print(f"Successfully loaded and merged config from '{config_path}'.")
        return config
    except Exception as e:
        print(f"Error: Failed to load config: {e}. Using default.")
        return default_config

def main():
    """
    Main execution function for GraphNAS search.
    """
    parser = argparse.ArgumentParser(description="GraphNAS - Distributed GNN Architecture Search")
    parser.add_argument('--config', type=str, default='configs/train.yaml', help='Path to config file')
    args = parser.parse_args()

    run_timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    log_dir = os.path.join("logs", "logings", f"run_{run_timestamp}")
    logger = GraphNASLogger(log_dir=log_dir, name="main_process")
    logger.logger.info("GraphNAS search process started...")

    config = load_config(args.config)
    nas_config = config['nas']
    train_config = config['training']
    logger.logger.info(f"Loaded config: \n{yaml.dump(config, indent=2)}")

    logger.logger.info("Initializing Ray...")
    num_gpus = int(torch.cuda.device_count())
    if num_gpus == 0:
        logger.logger.warning("No GPU detected. Using CPU for simulation.")
        ray.init(num_cpus=2, ignore_reinit_error=True)
    else:
        ray.init(num_gpus=num_gpus, ignore_reinit_error=True)
    logger.logger.info(f"-> Ray initialized. Resources: {ray.available_resources()}")

    logger.logger.info("Preparing and partitioning dataset...")
    dataset_name = train_config['dataset']
    num_workers = int(ray.available_resources().get("GPU", 2))
    max_hops = nas_config['gnn_layers'] 
    
    partitions = partition_training_graph(dataset_name, num_workers, max_hops)
    logger.logger.info(f"-> Dataset '{dataset_name}' partitioned into {len(partitions)} parts.")

    logger.logger.info("Creating backend components (Workers & Dispatcher)...")
    workers = []
    for i in range(num_workers):
        use_gpu = torch.cuda.is_available()
        worker_actor = TrainerWorker.options(num_gpus=1 if use_gpu else 0).remote(
            worker_id=i, gpu_id=i if use_gpu else -1, data_partition=partitions[i]
        )
        workers.append(worker_actor)
    
    logger.logger.info(f"-> {len(workers)} Workers created.")

    logger.logger.info("Initializing RL controller...")
    controller = ControllerManager(
        search_strategy=nas_config['search_strategy'],
        num_layers=nas_config['gnn_layers'],
        num_intermediate_nodes=nas_config['num_intermediate_nodes'],
        learning_rate=nas_config['controller_lr'],
        baseline_decay=nas_config['controller_baseline_decay'],
        hidden_size=nas_config.get('controller_hidden_size', 100),
        tanh_constant=nas_config.get('tanh_constant', 2.5),
        temperature=nas_config.get('temperature', 5.0),
        entropy_weight=nas_config.get('entropy_weight', 0.0001),
        weight_init_range=nas_config.get('weight_init_range', 0.1)
    )
    logger.logger.info("-> Controller initialized.")

    load_predictor = None
    balance_controller = None
    enable_balancing = nas_config.get('balancing_strategy', {}).get('enable', False)
    
    if enable_balancing:
        logger.logger.info("Initializing balancing components...")
        
        model_file_path = "mload_checkpoint.pth"
        if os.path.exists(model_file_path):
            load_predictor = MloadPredictor.load_from_file(model_file_path)
            logger.logger.info(f"Load predictor loaded: {model_file_path}")
        else:
            logger.logger.warning(f"Load predictor not found: {model_file_path}")
            logger.logger.warning("Using simulated load for testing.")
        
        balance_strategy_config = nas_config.get('balancing_strategy', {})
        balance_controller = BalanceController(
            controller_manager=controller,
            load_predictor=load_predictor,
            min_samples=balance_strategy_config.get('min_samples', 3),
            max_samples=balance_strategy_config.get('max_samples', 10),
            tolerance=balance_strategy_config.get('tolerance', 0.1),
            num_workers=num_workers,
            graph_info={
                "vertices": 2708, "edges": 5429,
                "avg_degree": 3.9, "diameter": 10
            }
        )
        logger.logger.info("Balance controller initialized.")

    logger.logger.info(f"Starting {nas_config['num_search_rounds']} rounds of architecture search...")
    
    all_candidate_models = {}
    model_counter = 0
    
    for round_num in tqdm(range(1, nas_config['num_search_rounds'] + 1), desc="GraphNAS Search"):
        logger.logger.info(f"--- Round {round_num}/{nas_config['num_search_rounds']} ---")
        
        if enable_balancing and balance_controller:
            logger.logger.info("Generating archs with balancing strategy...")
            dispatcher = Dispatcher.remote(
                worker_refs=workers, train_config=train_config,
                balance_controller=balance_controller,
                load_predictor=load_predictor, enable_balancing=True
            )
            num_generated = ray.get(dispatcher.submit_balanced_architectures.remote())
            logger.logger.info(f"Balancing strategy generated {num_generated} archs.")
            
            arch_results = ray.get(dispatcher.get_architectures.remote(format_type='pool_only'))
            architectures_to_eval = arch_results.get('model_pool', []) if isinstance(arch_results, dict) else []
        else:
            architectures_to_eval = controller.sample_architectures(
                num_samples=nas_config['samples_per_round']
            )
            logger.logger.info(f"Sampled {len(architectures_to_eval)} new archs.")
            
            dispatcher = Dispatcher.remote(worker_refs=workers, train_config=train_config)
            ray.get(dispatcher.submit_architectures.remote(architectures_to_eval))
        
        current_round_models = {}
        for arch in architectures_to_eval:
            model_counter += 1
            model_id = f"model_{model_counter:04d}"
            current_round_models[arch['arch_id']] = model_id
            all_candidate_models[model_id] = {
                'arch_spec': arch['arch_spec'], 'best_accuracy': 0.0,
                'round': round_num, 'original_arch_id': arch['arch_id']
            }
        
        ray.get(dispatcher.run_training_rounds.remote(num_complete_rounds=1))
        
        results_summary = ray.get(dispatcher.get_architectures.remote(format_type='summary'))
        eval_rewards = {}
        if isinstance(results_summary, dict) and hasattr(results_summary, 'get'):
            architectures_dict = results_summary.get('architectures', {})
            if isinstance(architectures_dict, dict):
                eval_rewards = {
                    arch_id: data.get('best_accuracy', 0.0) 
                    for arch_id, data in architectures_dict.items() if isinstance(data, dict)
                }
        else:
            logger.logger.warning(f"Failed to get valid eval results: {results_summary}")
        
        for arch_id, accuracy in eval_rewards.items():
            if arch_id in current_round_models:
                model_id = current_round_models[arch_id]
                all_candidate_models[model_id]['best_accuracy'] = accuracy
                logger.logger.info(f"Model {model_id} (Round {round_num}) Acc: {accuracy:.4f}")
        
        if eval_rewards:
            logger.log_metrics(eval_rewards, step=round_num, prefix='nas_rewards/')
        
        controller.train_step(rewards=eval_rewards)
        
        ray.kill(dispatcher)

    logger.logger.info("Architecture search complete!")

    logger.logger.info("Selecting Top K candidates for retraining...")
    
    sorted_models = sorted(all_candidate_models.items(), key=lambda x: x[1]['best_accuracy'], reverse=True)
    
    logger.logger.info(f"Total evaluated models: {len(all_candidate_models)}")
    logger.logger.info("=== Top 10 Models ===")
    for i, (model_id, model_info) in enumerate(sorted_models[:10]):
        logger.logger.info(f"Rank {i+1}: {model_id} (Round {model_info['round']}) - Acc: {model_info['best_accuracy']:.4f}")
    
    final_candidates_count = nas_config.get('final_candidates', 5)
    top_k_models = sorted_models[:final_candidates_count]
    
    logger.logger.info(f"=== Top {final_candidates_count} for retraining ===")
    for i, (model_id, model_info) in enumerate(top_k_models):
        logger.logger.info(f"Candidate {i+1}: {model_id} - Initial Acc: {model_info['best_accuracy']:.4f}")
    
    final_retrain_epochs = nas_config.get('final_retrain_epochs', 20)
    logger.logger.info(f"Retraining Top {final_candidates_count} for {final_retrain_epochs} epochs...")
    
    final_train_config = train_config.copy()
    final_train_config['epochs_per_rotation'] = 5
    final_train_config['steps_per_epoch'] = 5
    
    final_results = {}
    
    for i, (model_id, model_info) in enumerate(top_k_models):
        logger.logger.info(f"--- Retraining {i+1}/{final_candidates_count}: {model_id} ---")
        
        final_arch = {'arch_id': f"{model_id}_final", 'arch_spec': model_info['arch_spec']}
        
        final_dispatcher = Dispatcher.remote(worker_refs=workers, train_config=final_train_config)
        ray.get(final_dispatcher.submit_architectures.remote([final_arch]))
        
        ray.get(final_dispatcher.run_training_rounds.remote(num_complete_rounds=5))
        
        final_summary = ray.get(final_dispatcher.get_architectures.remote(format_type='summary'))
        final_accuracy = 0.0
        if isinstance(final_summary, dict):
            arch_dict = final_summary.get('architectures', {})
            if isinstance(arch_dict, dict) and f"{model_id}_final" in arch_dict:
                final_accuracy = arch_dict[f"{model_id}_final"].get('best_accuracy', 0.0)
        
        final_results[model_id] = {
            'model_info': model_info, 'initial_accuracy': model_info['best_accuracy'],
            'final_accuracy': final_accuracy,
            'improvement': final_accuracy - model_info['best_accuracy']
        }
        
        logger.logger.info(f"Model {model_id} retraining complete:")
        logger.logger.info(f"  Initial Acc: {model_info['best_accuracy']:.4f}")
        logger.logger.info(f"  Final Acc: {final_accuracy:.4f}")
        logger.logger.info(f"  Improvement: {final_accuracy - model_info['best_accuracy']:.4f}")
        
        ray.kill(final_dispatcher)
    
    logger.logger.info("Final results...")
    final_results_header = "="*25 + " GraphNAS Final Results " + "="*25
    logger.logger.info(final_results_header)
    
    best_final_model = max(final_results.items(), key=lambda x: x[1]['final_accuracy'])
    best_model_id, best_model_result = best_final_model
    
    logger.logger.info(f"Dataset: {dataset_name}")
    logger.logger.info(f"Search Strategy: {nas_config['search_strategy']}")
    logger.logger.info(f"Total Search Rounds: {nas_config['num_search_rounds']}")
    logger.logger.info(f"Total Evaluated Models: {len(all_candidate_models)}")
    logger.logger.info(f"Final Candidates: {final_candidates_count}")
    
    logger.logger.info(f"\nBest Model: {best_model_id}")
    logger.logger.info(f"  Source Round: {best_model_result['model_info']['round']}")
    logger.logger.info(f"  Initial Acc: {best_model_result['initial_accuracy']:.4f}")
    logger.logger.info(f"  Final Acc: {best_model_result['final_accuracy']:.4f}")
    logger.logger.info(f"  Improvement: {best_model_result['improvement']:.4f}")
    
    formatted_arch = format_architecture_for_display(
        best_model_result['model_info']['arch_spec'], nas_config['search_strategy']
    )
    logger.logger.info(f"\nBest GNN Architecture:\n{formatted_arch}")
    logger.logger.info(f"Best validation accuracy: {best_model_result['final_accuracy']:.4f}")
    
    logger.logger.info(f"\n=== All Final Candidates ===")
    sorted_final = sorted(final_results.items(), key=lambda x: x[1]['final_accuracy'], reverse=True)
    for i, (model_id, result) in enumerate(sorted_final):
        logger.logger.info(f"Rank {i+1}: {model_id}")
        logger.logger.info(f"  Final Acc: {result['final_accuracy']:.4f}")
        logger.logger.info(f"  Initial Acc: {result['initial_accuracy']:.4f}")
        logger.logger.info(f"  Improvement: {result['improvement']:.4f}")
    
    logger.logger.info("="*len(final_results_header))

    csv_path = logger.save_metrics_to_csv()
    logger.logger.info(f"All metrics saved to: {csv_path}")

    logger.logger.info("Shutting down Ray...")
    ray.shutdown()
    logger.logger.info("-> Process finished.")

if __name__ == "__main__":
    main()
