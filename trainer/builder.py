import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, List, Optional
from torch_geometric.nn.conv import GCNConv, SAGEConv, GATConv, ChebConv, ARMAConv, SGConv

def act_map(act_name: str):
    if act_name == 'linear' or act_name == 'none':
        return lambda x: x
    elif act_name == 'elu':
        return F.elu
    elif act_name == 'sigmoid':
        return torch.sigmoid
    elif act_name == 'tanh':
        return torch.tanh
    elif act_name == 'relu':
        return F.relu
    elif act_name == 'relu6':
        return F.relu6
    elif act_name == 'softplus':
        return F.softplus
    elif act_name == 'leaky_relu':
        return F.leaky_relu
    else:
        raise ValueError(f"不支持的激活函数: '{act_name}'")

def gnn_map(gnn_name: str, in_dim: int, out_dim: int, heads: int=1) -> nn.Module:
    if gnn_name == 'gcn':
        return GCNConv(in_dim, out_dim)
    elif gnn_name == 'sage':
        return SAGEConv(in_dim, out_dim)
    elif gnn_name == 'gat':
        return GATConv(in_dim, out_dim, heads=heads)
    elif gnn_name == 'cheb':
        return ChebConv(in_dim, out_dim, K=2)
    elif gnn_name == 'arma':
        return ARMAConv(in_dim, out_dim)
    elif gnn_name == 'sg':
        return SGConv(in_dim, out_dim)
    else:
        raise ValueError(f"不支持的GNN层类型: '{gnn_name}'")

class DynamicGNN(nn.Module):

    def __init__(self, arch_spec: Dict[str, Any], input_dim: int, output_dim: int, dropout: float=0.5):
        super(DynamicGNN, self).__init__()
        self.arch_spec = arch_spec
        self.dropout = dropout
        self.layers = nn.ModuleList()
        self.activations = []
        in_channels = input_dim
        for i, layer_info in enumerate(arch_spec['layers']):
            layer_type = layer_info.get('type', 'gcn')
            hidden_dim = layer_info.get('hidden_dim', 64)
            out_channels = output_dim if i == len(arch_spec['layers']) - 1 else hidden_dim
            heads = layer_info.get('heads', 1)
            self.layers.append(gnn_map(layer_type, in_channels, out_channels, heads))
            act_name = layer_info.get('act', 'relu')
            self.activations.append(act_map(act_name))
            in_channels = out_channels * heads if layer_type == 'gat' else out_channels

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        for i, (layer, act) in enumerate(zip(self.layers, self.activations)):
            if i < len(self.layers) - 1:
                x = F.dropout(x, p=self.dropout, training=self.training)
            x = layer(x, edge_index)
            x = act(x)
        return x

class ModelBuilder:

    @staticmethod
    def build_model(arch_spec: Dict[str, Any], input_dim: int, output_dim: int, config: Optional[Dict[str, Any]]=None) -> nn.Module:
        dropout_rate = config.get('dropout', 0.5) if config else 0.5
        if 'layers' in arch_spec:
            normalized_spec = ModelBuilder._normalize_layer_spec(arch_spec)
        elif 'cell_spec' in arch_spec:
            normalized_spec = ModelBuilder._convert_cell_to_layer_spec(arch_spec, input_dim, output_dim)
        else:
            raise ValueError(f"未知的架构格式: {arch_spec.keys()}. 期望 'layers' 或 'cell_spec'")
        return DynamicGNN(normalized_spec, input_dim, output_dim, dropout=dropout_rate)

    @staticmethod
    def _normalize_layer_spec(arch_spec: Dict[str, Any]) -> Dict[str, Any]:
        normalized_layers = []
        for layer_info in arch_spec['layers']:
            encoding_to_type_map = {'gcn': 'gcn', 'gat': 'gat', 'sym-gat': 'gat', 'const': 'gcn', 'cos': 'gcn', 'linear': 'gcn', 'gene-linear': 'gcn'}
            layer_type = encoding_to_type_map.get(layer_info.get('encoding', 'gcn'), 'gcn')
            normalized_layer = {'type': layer_type, 'hidden_dim': layer_info.get('hidden_dim', 64), 'act': layer_info.get('activation', 'relu'), 'heads': layer_info.get('num_heads', 1)}
            normalized_layers.append(normalized_layer)
        return {'layers': normalized_layers}

    @staticmethod
    def _convert_cell_to_layer_spec(cell_arch: Dict[str, Any], input_dim: int, output_dim: int) -> Dict[str, Any]:
        cell_spec = cell_arch['cell_spec']
        num_cells = cell_arch.get('num_cells_to_stack', 2)
        op_to_layer_map = {'gcn': 'gcn', 'gat_1_head': 'gat', 'gat_2_head': 'gat', 'gat_4_head': 'gat', 'gat_6_head': 'gat', 'gat_8_head': 'gat', 'mean_sage': 'sage', 'chebnet': 'cheb', 'arma': 'arma', 'sgc': 'sg', 'identity': 'gcn', 'zero': 'gcn'}

        def get_heads_from_op(op_name: str) -> int:
            if 'gat_' in op_name and '_head' in op_name:
                return int(op_name.split('_')[1])
            return 1
        layers = []
        connections = cell_spec.get('connections', [])
        for cell_idx in range(num_cells):
            if connections and cell_idx < len(connections):
                op_name = connections[cell_idx]['op']
                layer_type = op_to_layer_map.get(op_name, 'gcn')
                heads = get_heads_from_op(op_name)
            else:
                layer_type = 'gcn'
                heads = 1
            if cell_idx == 0:
                hidden_dim = 128
            elif cell_idx == num_cells - 1:
                hidden_dim = output_dim
            else:
                hidden_dim = 64
            if cell_idx == num_cells - 1:
                activation = 'linear'
            else:
                activation = cell_spec.get('final_act', 'relu')
            layer_info = {'type': layer_type, 'hidden_dim': hidden_dim, 'act': activation, 'heads': heads}
            layers.append(layer_info)
        return {'layers': layers}
if __name__ == '__main__':
    print('===== 开始独立测试 trainer/builder.py =====')
    INPUT_DIM = 1433
    OUTPUT_DIM = 7
    TRAIN_CONFIG = {'dropout': 0.6}
    print('\n[测试用例 1] 构建一个2层GCN模型...')
    gcn_spec = {'layers': [{'type': 'gcn', 'hidden_dim': 128, 'act': 'relu'}, {'type': 'gcn', 'hidden_dim': 999, 'act': 'linear'}]}
    try:
        gcn_model = ModelBuilder.build_model(gcn_spec, INPUT_DIM, OUTPUT_DIM, TRAIN_CONFIG)
        print('  ✅ 模型构建成功!')
        print(gcn_model)
        assert isinstance(gcn_model.layers[0], GCNConv)
        assert gcn_model.layers[0].out_channels == 128
        assert isinstance(gcn_model.layers[1], GCNConv)
        assert gcn_model.layers[1].out_channels == OUTPUT_DIM
        print('  ✅ 模型结构和维度验证通过')
    except Exception as e:
        print(f'  ❌ 测试失败: {e}')
    print('\n[测试用例 2] 构建一个GCN -> GAT的混合模型...')
    mixed_spec = {'layers': [{'type': 'gcn', 'hidden_dim': 256, 'act': 'relu'}, {'type': 'gat', 'hidden_dim': 64, 'heads': 4, 'act': 'elu'}, {'type': 'sage', 'hidden_dim': 999, 'act': 'linear'}]}
    try:
        mixed_model = ModelBuilder.build_model(mixed_spec, INPUT_DIM, OUTPUT_DIM, TRAIN_CONFIG)
        print('  ✅ 模型构建成功!')
        print(mixed_model)
        assert isinstance(mixed_model.layers[0], GCNConv)
        assert mixed_model.layers[0].out_channels == 256
        assert isinstance(mixed_model.layers[1], GATConv)
        assert mixed_model.layers[1].out_channels == 64
        assert mixed_model.layers[1].heads == 4
        assert isinstance(mixed_model.layers[2], SAGEConv)
        assert mixed_model.layers[2].in_channels == 256
        assert mixed_model.layers[2].out_channels == OUTPUT_DIM
        print('  ✅ 模型结构和维度验证通过')
    except Exception as e:
        print(f'  ❌ 测试失败: {e}')
    print('\n===== trainer/builder.py 独立测试完成 =====')