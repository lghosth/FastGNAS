# File: core/controller.py

import torch
import torch.nn as nn
from torch.distributions import Categorical
import numpy as np
from typing import List, Dict, Any, Tuple


class SearchSpace:
    """
    Defines the search space for GNN architectures.
    This version is updated based on Tables 1, 2, and 3 from the GraphNAS++ paper.
    Supports both 'layer_wise' and 'cell_based' search strategies.
    """
    def __init__(self, strategy: str = 'layer_wise'):
        self.strategy = strategy

        if self.strategy == 'layer_wise':

            self.encoding_ops = ['const', 'gcn', 'gat', 'sym-gat', 'cos', 'linear', 'gene-linear']
            self.aggregation_ops = ['sum', 'mean', 'max', 'mlp']
            self.activations = ['sigmoid', 'tanh', 'relu', 'none', 'softplus', 'leaky_relu', 'relu6', 'elu']
            self.num_heads = [1, 2, 4, 6, 8, 16]
            self.hidden_dims = [8, 16, 32, 64, 128, 256, 512]
            

            self._create_maps(['encoding_ops', 'aggregation_ops', 'activations', 'num_heads', 'hidden_dims'])

        elif self.strategy == 'cell_based':

            self.cell_operations = [
                'identity', 'zero', 'gat_8_head', 'gat_6_head', 'gat_4_head', 
                'gat_2_head', 'gat_1_head', 'gcn', 'chebnet', 'mean_sage', 'arma', 'sgc'
            ]
            self.readout_operations = ['add', 'concat'] # Paper also mentions 'multiply', simplified here
            self.activations = ['sigmoid', 'tanh', 'relu', 'none', 'softplus', 'leaky_relu', 'relu6', 'elu']

            self._create_maps(['cell_operations', 'readout_operations', 'activations'])
        else:
            raise ValueError(f"Unknown search strategy: '{self.strategy}'. Options: 'layer_wise' or 'cell_based'.")

    def _create_maps(self, names: List[str]):
        """Helper to create value-to-index maps for given attributes."""
        for name in names:
            values = getattr(self, name)
            setattr(self, f"{name}_map", {val: i for i, val in enumerate(values)})

    def get_len(self, key: str) -> int:
        """Returns the number of options for a given dimension."""
        return len(getattr(self, key)) if hasattr(self, key) else 0



class GNNController(nn.Module):
    """
    An LSTM-based policy network that sequentially generates GNN architecture descriptions.
    """
    def __init__(self, search_space: SearchSpace, num_layers: int = 2,
                 hidden_size: int = 100, num_intermediate_nodes: int = 4,
                 tanh_constant: float = 2.5,
                 temperature: float = 5.0,
                 weight_init_range: float = 0.1):
        super(GNNController, self).__init__()
        
        self.search_space = search_space
        self.strategy = self.search_space.strategy
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.num_intermediate_nodes = num_intermediate_nodes
        self.tanh_constant = tanh_constant
        self.temperature = temperature

        self.input_embedding = nn.Embedding(1, self.hidden_size)
        self.lstm = nn.LSTMCell(self.hidden_size, self.hidden_size)
        
        # Dynamically create classifiers based on strategy
        if self.strategy == 'layer_wise':
            self.encoding_classifier = nn.Linear(self.hidden_size, self.search_space.get_len('encoding_ops'))
            self.aggregation_classifier = nn.Linear(self.hidden_size, self.search_space.get_len('aggregation_ops'))
            self.activation_classifier = nn.Linear(self.hidden_size, self.search_space.get_len('activations'))
            self.heads_classifier = nn.Linear(self.hidden_size, self.search_space.get_len('num_heads'))
            self.dims_classifier = nn.Linear(self.hidden_size, self.search_space.get_len('hidden_dims'))
        
        elif self.strategy == 'cell_based':
            self.op_classifier = nn.Linear(self.hidden_size, self.search_space.get_len('cell_operations'))
            self.readout_classifier = nn.Linear(self.hidden_size, self.search_space.get_len('readout_operations'))
            self.activation_classifier = nn.Linear(self.hidden_size, self.search_space.get_len('activations'))
            self.prev_node_classifiers = nn.ModuleList([
                nn.Linear(self.hidden_size, 2 + i) for i in range(self.num_intermediate_nodes)
            ])
            
        self.h_0 = nn.Parameter(torch.zeros(1, self.hidden_size), requires_grad=False)
        self.c_0 = nn.Parameter(torch.zeros(1, self.hidden_size), requires_grad=False)

        self._init_weights(weight_init_range)

    def _init_weights(self, weight_init_range: float):
        """Initializes controller weights with a uniform distribution."""
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.uniform_(param, -weight_init_range, weight_init_range)
            elif 'bias' in name:
                nn.init.zeros_(param)

    def _sample_action(self, logits: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Samples an action from logits and returns the action, its log prob, and entropy."""
        scaled_logits = logits
        if self.temperature > 0:
            scaled_logits = logits / self.temperature
            scaled_logits = self.tanh_constant * torch.tanh(scaled_logits)
        
        dist = Categorical(logits=scaled_logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return action, log_prob, entropy

    def _forward_layer_wise(self) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
        inputs = self.input_embedding.weight
        h, c = self.h_0, self.c_0
        all_log_probs, all_actions, all_entropies = [], [], []

        for _ in range(self.num_layers):
            h, c = self.lstm(inputs, (h, c))
            classifiers = [
                self.encoding_classifier, self.aggregation_classifier, self.activation_classifier,
                self.heads_classifier, self.dims_classifier
            ]
            for classifier in classifiers:
                logits = classifier(h)
                action, log_prob, entropy = self._sample_action(logits)
                all_actions.append(action)
                all_log_probs.append(log_prob)
                all_entropies.append(entropy)
            inputs = h
        return all_log_probs, all_actions, all_entropies

    def _forward_cell_based(self) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
        inputs = self.input_embedding.weight
        h, c = self.h_0, self.c_0
        all_log_probs, all_actions, all_entropies = [], [], []

        for i in range(self.num_intermediate_nodes):
            h, c = self.lstm(inputs, (h, c))
            
            prev_node_logits = self.prev_node_classifiers[i](h)
            prev_node_action, prev_node_log_prob, prev_node_entropy = self._sample_action(prev_node_logits)
            all_actions.extend([prev_node_action])
            all_log_probs.extend([prev_node_log_prob])
            all_entropies.extend([prev_node_entropy])

            op_logits = self.op_classifier(h)
            op_action, op_log_prob, op_entropy = self._sample_action(op_logits)
            all_actions.extend([op_action])
            all_log_probs.extend([op_log_prob])
            all_entropies.extend([op_entropy])
            inputs = h
        
        h, c = self.lstm(inputs, (h, c))
        for classifier in [self.readout_classifier, self.activation_classifier]:
            logits = classifier(h)
            action, log_prob, entropy = self._sample_action(logits)
            all_actions.append(action)
            all_log_probs.append(log_prob)
            all_entropies.append(entropy)
        
        return all_log_probs, all_actions, all_entropies

    def forward(self) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
        if self.strategy == 'layer_wise':
            return self._forward_layer_wise()
        elif self.strategy == 'cell_based':
            return self._forward_cell_based()
        else:
            raise ValueError(f"Unknown search strategy: '{self.strategy}'")


class ControllerManager:
    """
    Manages all operations of the RL controller.
    """
    def __init__(self, search_strategy: str = 'layer_wise', num_layers: int = 2, 
                 num_intermediate_nodes: int = 4,
                 learning_rate: float = 0.00035,
                 baseline_decay: float = 0.9,
                 hidden_size: int = 100,
                 tanh_constant: float = 2.5,
                 temperature: float = 5.0,
                 entropy_weight: float = 0.0001,
                 weight_init_range: float = 0.1):
        
        print(f"Initializing ControllerManager with strategy: '{search_strategy}'")
        self.strategy = search_strategy
        self.search_space = SearchSpace(strategy=self.strategy)
        self.entropy_weight = entropy_weight
        print(f"Entropy weight set to: {self.entropy_weight}")

        self.controller = GNNController(
            self.search_space, 
            num_layers=num_layers,
            hidden_size=hidden_size,
            tanh_constant=tanh_constant,
            temperature=temperature,
            weight_init_range=weight_init_range,
            num_intermediate_nodes=num_intermediate_nodes
        )
        self.optimizer = torch.optim.Adam(self.controller.parameters(), lr=learning_rate)
        
        self.baseline = 0.0
        self.baseline_decay = baseline_decay
        self.best_arch_spec = None
        self.best_arch_accuracy = 0.0
        self.history = []

    def sample_architectures(self, num_samples: int = 1) -> List[Dict[str, Any]]:
        sampled_specs = []
        for i in range(num_samples):
            log_probs, actions, entropies = self.controller()
            spec = self._actions_to_spec(actions)
            unique_id = f"arch_{len(self.history)}_{i}"
            self.history.append({
                'id': unique_id, 'spec': spec,
                'log_probs': torch.stack(log_probs),
                'actions': actions,
                'entropies': torch.stack(entropies)
            })
            sampled_specs.append({'arch_id': unique_id, 'arch_spec': spec})
        return sampled_specs

    def train_step(self, rewards: Dict[str, float]):
        self.optimizer.zero_grad()
        total_loss = 0
        
        if not rewards:
            self.history = []
            return

        current_avg_reward = np.mean(list(rewards.values()))
        self.baseline = self.baseline_decay * self.baseline + (1 - self.baseline_decay) * current_avg_reward
        
        for sample in self.history:
            arch_id = sample['id']
            if arch_id in rewards:
                reward = rewards[arch_id]
                advantage = reward - self.baseline
                
                # Loss includes policy gradient and entropy bonus
                policy_loss = -sample['log_probs'].sum() * advantage
                entropy_bonus = sample['entropies'].sum()

                loss = policy_loss - self.entropy_weight * entropy_bonus
                total_loss += loss

                if reward > self.best_arch_accuracy:
                    self.best_arch_accuracy = reward
                    self.best_arch_spec = sample['spec']
                    print(f"New best architecture found: {arch_id}, Accuracy: {reward:.4f}")

        if self.history and total_loss != 0:
            (total_loss / len(self.history)).backward() # Average loss
            self.optimizer.step()
        self.history = []

    def _actions_to_spec(self, actions: List[torch.Tensor]) -> Dict[str, Any]:
        """Converts a list of action indices to a human-readable architecture spec dict."""
        if self.strategy == 'layer_wise':
            spec_layers, action_idx = [], 0
            for _ in range(self.controller.num_layers):
                spec_layers.append({
                    'encoding': self.search_space.encoding_ops[actions[action_idx].item()],
                    'aggregation': self.search_space.aggregation_ops[actions[action_idx + 1].item()],
                    'activation': self.search_space.activations[actions[action_idx + 2].item()],
                    'num_heads': self.search_space.num_heads[actions[action_idx + 3].item()],
                    'hidden_dim': self.search_space.hidden_dims[actions[action_idx + 4].item()]
                })
                action_idx += 5
            return {'num_layers': self.controller.num_layers, 'layers': spec_layers}

        elif self.strategy == 'cell_based':
            cell_connections, action_idx = [], 0
            for _ in range(self.controller.num_intermediate_nodes):
                cell_connections.append({
                    'from_node': actions[action_idx].item(),
                    'op': self.search_space.cell_operations[actions[action_idx + 1].item()]
                })
                action_idx += 2
            
            readout_op = self.search_space.readout_operations[actions[action_idx].item()]
            final_act = self.search_space.activations[actions[action_idx + 1].item()]

            return {
                'num_cells_to_stack': self.controller.num_layers,
                'cell_spec': {
                    'intermediate_nodes': self.controller.num_intermediate_nodes,
                    'connections': cell_connections,
                    'readout_op': readout_op,
                    'final_act': final_act
                }
            }
        else:
            raise ValueError(f"Unknown search strategy: '{self.strategy}'")


if __name__ == '__main__':

    print("===== Independent Test: 'layer_wise' Strategy =====")
    manager_layer = ControllerManager(search_strategy='layer_wise', num_layers=2)
    
    print("\n[Step 1] Sampling 1 'layer_wise' architecture...")
    architectures_layer = manager_layer.sample_architectures(num_samples=1)
    print("  - Sampling successful:")
    for arch in architectures_layer:
        print(f"    - ID: {arch['arch_id']}")
        print(f"    - Spec: {arch['arch_spec']}")
    
    print("\n[Step 2] Simulating reward and training...")
    mock_rewards_layer = {architectures_layer[0]['arch_id']: 0.9}
    manager_layer.train_step(mock_rewards_layer)
    print("  - Train step completed.")
    print(f"  - Best recorded accuracy: {manager_layer.best_arch_accuracy}")

    print("\n===== 'layer_wise' test complete. =====\n")


    print("===== Independent Test: 'cell_based' Strategy =====")
    manager_cell = ControllerManager(search_strategy='cell_based', num_layers=3, num_intermediate_nodes=4)
    
    print("\n[Step 1] Sampling 1 'cell_based' architecture...")
    architectures_cell = manager_cell.sample_architectures(num_samples=1)
    print("  - Sampling successful:")
    for arch in architectures_cell:
        print(f"    - ID: {arch['arch_id']}")
        print(f"    - Spec: {arch['arch_spec']}")
    
    print("\n[Step 2] Simulating reward and training...")
    mock_rewards_cell = {architectures_cell[0]['arch_id']: 0.92}
    manager_cell.train_step(mock_rewards_cell)
    print("  - Train step completed.")
    print(f"  - Best recorded accuracy: {manager_cell.best_arch_accuracy}")
    
    print("\n===== 'cell_based' test complete. =====")
