# file: trainer/builder.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, List, Optional
from torch_geometric.nn.conv import GCNConv, SAGEConv, GATConv, ChebConv, ARMAConv, SGConv

# +---------------------------------------------------------------------------------+
# | 功能 1: 借鉴自 graphnas/micro_search_space.py 的核心映射函数                       |
# | 这些函数将字符串名称动态转换为实际的PyG层或激活函数。                              |
# +---------------------------------------------------------------------------------+

def act_map(act_name: str):
    """根据字符串名称返回对应的激活函数。"""
    if act_name == "linear" or act_name == "none":
        return lambda x: x
    elif act_name == "elu":
        return F.elu
    elif act_name == "sigmoid":
        return torch.sigmoid
    elif act_name == "tanh":
        return torch.tanh
    elif act_name == "relu":
        return F.relu
    elif act_name == "relu6":
        return F.relu6
    elif act_name == "softplus":
        return F.softplus
    elif act_name == "leaky_relu":
        return F.leaky_relu
    else:
        raise ValueError(f"不支持的激活函数: '{act_name}'")

def gnn_map(gnn_name: str, in_dim: int, out_dim: int, heads: int = 1) -> nn.Module:
    """根据字符串名称返回对应的PyG图卷积层实例。"""
    if gnn_name == "gcn":
        return GCNConv(in_dim, out_dim)
    elif gnn_name == "sage":
        return SAGEConv(in_dim, out_dim)
    elif gnn_name == "gat":
        # 对于GAT，heads参数是必须的
        return GATConv(in_dim, out_dim, heads=heads)
    elif gnn_name == "cheb":
        return ChebConv(in_dim, out_dim, K=2) # K可以作为参数传入
    elif gnn_name == "arma":
        return ARMAConv(in_dim, out_dim)
    elif gnn_name == "sg":
        return SGConv(in_dim, out_dim)
    else:
        raise ValueError(f"不支持的GNN层类型: '{gnn_name}'")


# +---------------------------------------------------------------------------------+
# | 功能 2: 动态GNN模型定义                                                          |
# | 这个类会根据传入的 `arch_spec` 动态构建网络结构。                                |
# +---------------------------------------------------------------------------------+

class DynamicGNN(nn.Module):
    """
    一个可以根据架构描述（arch_spec）动态构建的GNN模型。
    """
    def __init__(self, arch_spec: Dict[str, Any], input_dim: int, output_dim: int, dropout: float = 0.5):
        super(DynamicGNN, self).__init__()
        
        self.arch_spec = arch_spec
        self.dropout = dropout
        
        self.layers = nn.ModuleList()
        self.activations = []
        
        in_channels = input_dim
        
        # 解析 'layers' 列表来构建网络
        for i, layer_info in enumerate(arch_spec['layers']):
            layer_type = layer_info.get('type', 'gcn')
            hidden_dim = layer_info.get('hidden_dim', 64)
            
            # 确定输出维度
            # 如果是最后一层，输出维度固定为类别数
            # 否则，使用该层定义的hidden_dim
            out_channels = output_dim if i == len(arch_spec['layers']) - 1 else hidden_dim
            
            # 获取GAT所需的头数，如果不是GAT则默认为1
            heads = layer_info.get('heads', 1)

            # 使用 gnn_map 创建图层
            self.layers.append(gnn_map(layer_type, in_channels, out_channels, heads))
            
            # 获取激活函数
            act_name = layer_info.get('act', 'relu')
            self.activations.append(act_map(act_name))
            
            # 更新下一层的输入维度
            # 如果是GAT层，输出维度是 out_channels * heads
            in_channels = out_channels * heads if layer_type == 'gat' else out_channels

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        for i, (layer, act) in enumerate(zip(self.layers, self.activations)):
            # 对非最后一层的输出应用dropout
            if i < len(self.layers) - 1:
                x = F.dropout(x, p=self.dropout, training=self.training)
            
            x = layer(x, edge_index)
            x = act(x)
            
        # 最后一层输出原始logits，由损失函数处理
        return x

# +---------------------------------------------------------------------------------+
# | 功能 3: 模型构建器工厂 (主入口)                                                    |
# | 这是外部模块（如TrainerWorker）唯一需要调用的部分。                               |
# +---------------------------------------------------------------------------------+

class ModelBuilder:
    """
    GNN模型构建工厂，根据架构描述动态生成GNN模型。
    """
    
    @staticmethod
    def build_model(arch_spec: Dict[str, Any], 
                    input_dim: int, 
                    output_dim: int,
                    config: Optional[Dict[str, Any]] = None) -> nn.Module:
        """
        根据架构描述构建GNN模型。

        Args:
            arch_spec (Dict[str, Any]): 架构描述字典。
                layer_wise示例: {'layers': [{'encoding':'gcn', 'hidden_dim': 64, 'activation': 'relu'},
                                          {'encoding':'gat', 'num_heads': 8, 'hidden_dim': 32, 'activation': 'elu'}]}
                cell_based示例: {'num_cells_to_stack': 2, 'cell_spec': {...}}
            input_dim (int): 输入特征维度。
            output_dim (int): 输出类别数。
            config (Dict, optional): 包含额外配置的字典, 如 'dropout'。

        Returns:
            nn.Module: 构建好的PyTorch模型实例。
        """
        dropout_rate = config.get('dropout', 0.5) if config else 0.5
        
        # 检测架构类型并转换为统一格式
        if 'layers' in arch_spec:
            # layer_wise格式，需要转换字段名
            normalized_spec = ModelBuilder._normalize_layer_spec(arch_spec)
        elif 'cell_spec' in arch_spec:
            # cell_based格式，需要转换为layer_wise格式
            normalized_spec = ModelBuilder._convert_cell_to_layer_spec(arch_spec, input_dim, output_dim)
        else:
            raise ValueError(f"未知的架构格式: {arch_spec.keys()}. 期望 'layers' 或 'cell_spec'")
        
        return DynamicGNN(normalized_spec, input_dim, output_dim, dropout=dropout_rate)
    
    @staticmethod
    def _normalize_layer_spec(arch_spec: Dict[str, Any]) -> Dict[str, Any]:
        """
        将layer_wise架构的字段名转换为DynamicGNN期望的格式
        """
        normalized_layers = []
        for layer_info in arch_spec['layers']:
            # 映射encoding到type
            encoding_to_type_map = {
                'gcn': 'gcn',
                'gat': 'gat',
                'sym-gat': 'gat',
                'const': 'gcn',
                'cos': 'gcn',
                'linear': 'gcn',
                'gene-linear': 'gcn'
            }
            
            layer_type = encoding_to_type_map.get(layer_info.get('encoding', 'gcn'), 'gcn')
            
            normalized_layer = {
                'type': layer_type,
                'hidden_dim': layer_info.get('hidden_dim', 64),
                'act': layer_info.get('activation', 'relu'),
                'heads': layer_info.get('num_heads', 1)
            }
            normalized_layers.append(normalized_layer)
        
        return {'layers': normalized_layers}
    
    @staticmethod
    def _convert_cell_to_layer_spec(cell_arch: Dict[str, Any], input_dim: int, output_dim: int) -> Dict[str, Any]:
        """
        将cell_based架构转换为layer_wise架构格式
        """
        cell_spec = cell_arch['cell_spec']
        num_cells = cell_arch.get('num_cells_to_stack', 2)
        
        # 从cell操作映射到layer类型
        op_to_layer_map = {
            'gcn': 'gcn',
            'gat_1_head': 'gat', 'gat_2_head': 'gat', 'gat_4_head': 'gat', 
            'gat_6_head': 'gat', 'gat_8_head': 'gat',
            'mean_sage': 'sage',
            'chebnet': 'cheb',
            'arma': 'arma',
            'sgc': 'sg',
            'identity': 'gcn',  # 使用gcn作为identity的替代
            'zero': 'gcn'       # 使用gcn作为zero的替代
        }
        
        # 从头数操作中提取头数
        def get_heads_from_op(op_name: str) -> int:
            if 'gat_' in op_name and '_head' in op_name:
                return int(op_name.split('_')[1])
            return 1
        
        layers = []
        connections = cell_spec.get('connections', [])
        
        # 为每个cell生成层
        for cell_idx in range(num_cells):
            # 选择第一个连接的操作作为该层的类型
            if connections and cell_idx < len(connections):
                op_name = connections[cell_idx]['op']
                layer_type = op_to_layer_map.get(op_name, 'gcn')
                heads = get_heads_from_op(op_name)
            else:
                layer_type = 'gcn'
                heads = 1
            
            # 设置隐藏维度（第一层较大，后续层较小）
            if cell_idx == 0:
                hidden_dim = 128
            elif cell_idx == num_cells - 1:
                hidden_dim = output_dim  # 最后一层直接输出
            else:
                hidden_dim = 64
            
            # 设置激活函数
            if cell_idx == num_cells - 1:
                activation = 'linear'  # 最后一层不使用激活函数
            else:
                activation = cell_spec.get('final_act', 'relu')
            
            layer_info = {
                'type': layer_type,
                'hidden_dim': hidden_dim,
                'act': activation,
                'heads': heads
            }
            layers.append(layer_info)
        
        return {'layers': layers}

# ===================================================================================
# | 独立的测试模块，用于验证该文件功能的正确性                                           |
# ===================================================================================
if __name__ == '__main__':
    print("===== 开始独立测试 trainer/builder.py =====")
    
    # 模拟Cora数据集的维度
    INPUT_DIM = 1433
    OUTPUT_DIM = 7
    TRAIN_CONFIG = {'dropout': 0.6}

    # --- 测试用例 1: 构建一个简单的两层GCN ---
    print("\n[测试用例 1] 构建一个2层GCN模型...")
    gcn_spec = {
        'layers': [
            {'type': 'gcn', 'hidden_dim': 128, 'act': 'relu'},
            # 第二层（最后一层）的hidden_dim会被忽略，直接使用OUTPUT_DIM
            {'type': 'gcn', 'hidden_dim': 999, 'act': 'linear'} 
        ]
    }
    try:
        gcn_model = ModelBuilder.build_model(gcn_spec, INPUT_DIM, OUTPUT_DIM, TRAIN_CONFIG)
        print("  ✅ 模型构建成功!")
        print(gcn_model)
        assert isinstance(gcn_model.layers[0], GCNConv)
        assert gcn_model.layers[0].out_channels == 128
        assert isinstance(gcn_model.layers[1], GCNConv)
        assert gcn_model.layers[1].out_channels == OUTPUT_DIM
        print("  ✅ 模型结构和维度验证通过。")
    except Exception as e:
        print(f"  ❌ 测试失败: {e}")

    # --- 测试用例 2: 构建一个混合了GCN和GAT的复杂模型 ---
    print("\n[测试用例 2] 构建一个GCN -> GAT的混合模型...")
    mixed_spec = {
        'layers': [
            {'type': 'gcn', 'hidden_dim': 256, 'act': 'relu'},
            # GAT层将有 256 * 4 的输出维度
            {'type': 'gat', 'hidden_dim': 64, 'heads': 4, 'act': 'elu'}, 
             # 最后一层的输入维度应为 64 * 4 = 256
            {'type': 'sage', 'hidden_dim': 999, 'act': 'linear'}
        ]
    }
    try:
        mixed_model = ModelBuilder.build_model(mixed_spec, INPUT_DIM, OUTPUT_DIM, TRAIN_CONFIG)
        print("  ✅ 模型构建成功!")
        print(mixed_model)
        assert isinstance(mixed_model.layers[0], GCNConv)
        assert mixed_model.layers[0].out_channels == 256
        assert isinstance(mixed_model.layers[1], GATConv)
        assert mixed_model.layers[1].out_channels == 64
        assert mixed_model.layers[1].heads == 4
        assert isinstance(mixed_model.layers[2], SAGEConv)
        assert mixed_model.layers[2].in_channels == 256 # 验证输入维度是否正确衔接
        assert mixed_model.layers[2].out_channels == OUTPUT_DIM
        print("  ✅ 模型结构和维度验证通过。")

    except Exception as e:
        print(f"  ❌ 测试失败: {e}")

    print("\n===== trainer/builder.py 独立测试完成 =====")