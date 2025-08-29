# 文件: core/controller.py

import torch
import torch.nn as nn
from torch.distributions import Categorical
import numpy as np
from typing import List, Dict, Any, Tuple

# +---------------------------------------------------------------------------------+
# | 模块 1: 定义GNN架构的搜索空间 (已根据GraphNAS++文献更新)                        |
# +---------------------------------------------------------------------------------+

class SearchSpace:
    """
    定义GNN架构的搜索空间。
    该版本已根据 GraphNAS++ 文献中的 Table 1, 2, 3 进行更新。
    支持 'layer_wise' 和 'cell_based' 两种搜索策略。
    """
    def __init__(self, strategy: str = 'layer_wise'):
        self.strategy = strategy

        if self.strategy == 'layer_wise':
            # ----- 策略1: 逐层搜索空间 (对应文献 Section 4.1) -----
            # 编码函数 (Encd), 对应文献 Table 2
            self.encoding_ops = ['const', 'gcn', 'gat', 'sym-gat', 'cos', 'linear', 'gene-linear']
            # 聚合函数 (Aggr), 对应文献 Table 1
            self.aggregation_ops = ['sum', 'mean', 'max', 'mlp']
            # 激活函数 (σ), 对应文献 Table 1
            self.activations = ['sigmoid', 'tanh', 'relu', 'none', 'softplus', 'leaky_relu', 'relu6', 'elu']
            # 多头数量 (k), 对应文献 Table 1
            self.num_heads = [1, 2, 4, 6, 8, 16]
            # 输出维度/隐藏层大小 (d), 对应文献 Table 1
            self.hidden_dims = [8, 16, 32, 64, 128, 256, 512]
            
            # 创建从名称/值到整数索引的映射
            self._create_maps(['encoding_ops', 'aggregation_ops', 'activations', 'num_heads', 'hidden_dims'])

        elif self.strategy == 'cell_based':
            # ----- 策略2: 基于单元的搜索空间 (对应文献 Section 4.3) -----
            # 图卷积算子, 对应文献 Table 3 中的12种算子
            self.cell_operations = [
                'identity', 'zero', 'gat_8_head', 'gat_6_head', 'gat_4_head', 
                'gat_2_head', 'gat_1_head', 'gcn', 'chebnet', 'mean_sage', 'arma', 'sgc'
            ]
            # 用于组合中间状态的Readout算子
            self.readout_operations = ['add', 'concat'] # 论文中还提到 multiply, 此处简化
            # 激活函数
            self.activations = ['sigmoid', 'tanh', 'relu', 'none', 'softplus', 'leaky_relu', 'relu6', 'elu']

            self._create_maps(['cell_operations', 'readout_operations', 'activations'])
        else:
            raise ValueError(f"未知的搜索策略: '{self.strategy}'。可用选项为 'layer_wise' 或 'cell_based'。")

    def _create_maps(self, names: List[str]):
        """辅助函数，为给定的属性列表创建值到索引的映射。"""
        for name in names:
            values = getattr(self, name)
            setattr(self, f"{name}_map", {val: i for i, val in enumerate(values)})

    def get_len(self, key: str) -> int:
        """返回给定维度选项的数量。"""
        if hasattr(self, key):
            return len(getattr(self, key))
        return 0

# +---------------------------------------------------------------------------------+
# | 模块 2: 基于LSTM的强化学习控制器（策略网络） (已更新以支持新搜索空间)           |
# +---------------------------------------------------------------------------------+

class GNNController(nn.Module):
    """
    一个基于LSTM的策略网络，它顺序地生成GNN架构的描述。
    已更新以处理文献中定义的更复杂的 layer-wise 和 cell-based 搜索空间。
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
        # 保存采样参数
        self.tanh_constant = tanh_constant
        self.temperature = temperature

        self.input_embedding = nn.Embedding(1, self.hidden_size)
        self.lstm = nn.LSTMCell(self.hidden_size, self.hidden_size)
        
        # --- 根据策略动态创建分类器 ---
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

        # 添加权重初始化
        self._init_weights(weight_init_range)

    def _init_weights(self, weight_init_range: float):
        """初始化控制器权重为均匀分布 Uniform(-init_range, init_range)"""
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.uniform_(param, -weight_init_range, weight_init_range)
            elif 'bias' in name:
                nn.init.zeros_(param)

    def _sample_action(self, logits: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """从logits中采样一个动作并返回动作、其对数概率以及策略的熵"""
        if self.temperature > 0:
            scaled_logits = logits / self.temperature
            scaled_logits = self.tanh_constant * torch.tanh(scaled_logits)
        else:
            scaled_logits = logits
        
        dist = Categorical(logits=scaled_logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return action, log_prob, entropy

        # scaled_logits = self.tanh_constant * torch.tanh(logits / self.temperature)
        # dist = Categorical(logits=scaled_logits)
        # action = dist.sample()
        # log_prob = dist.log_prob(action)
        # return action, log_prob

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
            all_actions.append(prev_node_action)
            all_log_probs.append(prev_node_log_prob)
            all_entropies.append(prev_node_entropy)

            op_logits = self.op_classifier(h)
            op_action, op_log_prob, op_entropy = self._sample_action(op_logits)
            all_actions.append(op_action)
            all_log_probs.append(op_log_prob)
            all_entropies.append(op_entropy)
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
            raise ValueError(f"未知的搜索策略: '{self.strategy}'")

# +---------------------------------------------------------------------------------+
# | 模块 3: 控制器管理器 (已更新以处理新策略和超参数)                               |
# +---------------------------------------------------------------------------------+
class ControllerManager:
    """
    管理RL控制器的所有操作。
    已更新以接受 'search_strategy' 超参数并处理两种策略的架构生成。
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
        
        print(f"初始化 ControllerManager, 策略: '{search_strategy}'")
        self.strategy = search_strategy
        self.search_space = SearchSpace(strategy=self.strategy)
        self.entropy_weight = entropy_weight
        print(f"熵奖励权重设置为: {self.entropy_weight}")

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
        
        # 检查是否有奖励，避免在空字典上调用np.mean
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
                
                # 完善损失函数，加入熵奖励项
                # 目标是最大化 (奖励 + 熵)，等同于最小化 -(log_prob * advantage + entropy_term)
                policy_loss = -sample['log_probs'].sum() * advantage
                entropy_bonus = sample['entropies'].sum()

                loss = policy_loss - self.entropy_weight * entropy_bonus
                # loss = -sample['log_probs'].sum() * advantage
                total_loss += loss

                if reward > self.best_arch_accuracy:
                    self.best_arch_accuracy = reward
                    self.best_arch_spec = sample['spec']
                    print(f"✨ 发现新的最佳架构: {arch_id}, 准确率: {reward:.4f}")

        if self.history and total_loss != 0:
            (total_loss / len(self.history)).backward() # 除以历史样本数，避免梯度爆炸
            self.optimizer.step()
        self.history = []

    def _actions_to_spec(self, actions: List[torch.Tensor]) -> Dict[str, Any]:
        """将动作索引列表转换为人类可读的架构规范字典。"""
        if self.strategy == 'layer_wise':
            spec_layers, action_idx = [], 0
            for _ in range(self.controller.num_layers):
                enc_idx = actions[action_idx].item()
                agg_idx = actions[action_idx + 1].item()
                act_idx = actions[action_idx + 2].item()
                head_idx = actions[action_idx + 3].item()
                dim_idx = actions[action_idx + 4].item()
                action_idx += 5
                
                spec_layers.append({
                    'encoding': self.search_space.encoding_ops[int(enc_idx)],
                    'aggregation': self.search_space.aggregation_ops[int(agg_idx)],
                    'activation': self.search_space.activations[int(act_idx)],
                    'num_heads': self.search_space.num_heads[int(head_idx)],
                    'hidden_dim': self.search_space.hidden_dims[int(dim_idx)]
                })
            return {'num_layers': self.controller.num_layers, 'layers': spec_layers}

        elif self.strategy == 'cell_based':
            cell_connections, action_idx = [], 0
            for _ in range(self.controller.num_intermediate_nodes):
                prev_node_idx = actions[action_idx].item()
                op_idx = actions[action_idx + 1].item()
                action_idx += 2
                cell_connections.append({
                    'from_node': prev_node_idx,
                    'op': self.search_space.cell_operations[int(op_idx)]
                })
            
            readout_idx = actions[action_idx].item()
            act_idx = actions[action_idx + 1].item()

            return {
                'num_cells_to_stack': self.controller.num_layers,
                'cell_spec': {
                    'intermediate_nodes': self.controller.num_intermediate_nodes,
                    'connections': cell_connections,
                    'readout_op': self.search_space.readout_operations[int(readout_idx)],
                    'final_act': self.search_space.activations[int(act_idx)]
                }
            }
        else:
            raise ValueError(f"未知的搜索策略: '{self.strategy}'")

# ===================================================================================
# | 独立测试模块: 验证两种策略的功能。                                              |
# ===================================================================================
if __name__ == '__main__':
    # ----- 测试 1: Layer-wise 策略 -----
    print("===== 独立测试: 'layer_wise' 策略 =====")
    manager_layer = ControllerManager(search_strategy='layer_wise', num_layers=2)
    
    print("\n[步骤 1] 采样 1 个 'layer_wise' 架构...")
    architectures_layer = manager_layer.sample_architectures(num_samples=1)
    print("  ✅ 采样成功:")
    for arch in architectures_layer:
        print(f"    - ID: {arch['arch_id']}")
        print(f"    - Spec: {arch['arch_spec']}")
    
    print("\n[步骤 2] 模拟奖励和训练...")
    mock_rewards_layer = {architectures_layer[0]['arch_id']: 0.9}
    manager_layer.train_step(mock_rewards_layer)
    print("  ✅ 训练步骤完成。")
    print(f"  ✅ 记录的最佳准确率: {manager_layer.best_arch_accuracy}")

    print("\n===== 'layer_wise' 测试完成。 =====\n")

    # ----- 测试 2: Cell-based 策略 -----
    print("===== 独立测试: 'cell_based' 策略 =====")
    manager_cell = ControllerManager(search_strategy='cell_based', num_layers=3, num_intermediate_nodes=4)
    
    print("\n[步骤 1] 采样 1 个 'cell_based' 架构...")
    architectures_cell = manager_cell.sample_architectures(num_samples=1)
    print("  ✅ 采样成功:")
    for arch in architectures_cell:
        print(f"    - ID: {arch['arch_id']}")
        print(f"    - Spec: {arch['arch_spec']}")
    
    print("\n[步骤 2] 模拟奖励和训练...")
    mock_rewards_cell = {architectures_cell[0]['arch_id']: 0.92}
    manager_cell.train_step(mock_rewards_cell)
    print("  ✅ 训练步骤完成。")
    print(f"  ✅ 记录的最佳准确率: {manager_cell.best_arch_accuracy}")
    
    print("\n===== 'cell_based' 测试完成。 =====")
