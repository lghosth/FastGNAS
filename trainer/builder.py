# file: trainer/builder.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, List, Optional
from torch_geometric.nn.conv import GCNConv, SAGEConv, GATConv, ChebConv, ARMAConv, SGConv

# +---------------------------------------------------------------------------------+
# | Section 1: Core mapping functions from graphnas/micro_search_space.py           |
# | These functions dynamically convert string names to PyG layers or activations.  |
# +---------------------------------------------------------------------------------+

def act_map(act_name: str):
    """Returns an activation function based on its name."""
    if act_name in ("linear", "none"):
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
        raise ValueError(f"Unsupported activation: '{act_name}'")

def gnn_map(gnn_name: str, in_dim: int, out_dim: int, heads: int = 1) -> nn.Module:
    """Returns a PyG graph convolution layer instance based on its name."""
    if gnn_name == "gcn":
        return GCNConv(in_dim, out_dim)
    elif gnn_name == "sage":
        return SAGEConv(in_dim, out_dim)
    elif gnn_name == "gat":
        return GATConv(in_dim, out_dim, heads=heads)
    elif gnn_name == "cheb":
        return ChebConv(in_dim, out_dim, K=2) # K can be parameterized
    elif gnn_name == "arma":
        return ARMAConv(in_dim, out_dim)
    elif gnn_name == "sg":
        return SGConv(in_dim, out_dim)
    else:
        raise ValueError(f"Unsupported GNN layer: '{gnn_name}'")


# +---------------------------------------------------------------------------------+
# | Section 2: Dynamic GNN Model Definition                                         |
# | This class dynamically builds a network based on the provided `arch_spec`.      |
# +---------------------------------------------------------------------------------+

class DynamicGNN(nn.Module):
    """
    A GNN model that is dynamically built based on an architecture specification.
    """
    def __init__(self, arch_spec: Dict[str, Any], input_dim: int, output_dim: int, dropout: float = 0.5):
        super(DynamicGNN, self).__init__()
        
        self.arch_spec = arch_spec
        self.dropout = dropout
        
        self.layers = nn.ModuleList()
        self.activations = []
        
        in_channels = input_dim
        
        # Build network from the 'layers' list
        for i, layer_info in enumerate(arch_spec['layers']):
            layer_type = layer_info.get('type', 'gcn')
            hidden_dim = layer_info.get('hidden_dim', 64)
            
            # Determine output dimension
            # Final layer outputs to class number, otherwise to hidden_dim
            out_channels = output_dim if i == len(arch_spec['layers']) - 1 else hidden_dim
            
            # Get number of heads for GAT, default to 1
            heads = layer_info.get('heads', 1)

            # Create layer using gnn_map
            self.layers.append(gnn_map(layer_type, in_channels, out_channels, heads))
            
            # Get activation function
            act_name = layer_info.get('act', 'relu')
            self.activations.append(act_map(act_name))
            
            # Update input dimension for the next layer
            # GAT output is out_channels * heads
            in_channels = out_channels * heads if layer_type == 'gat' else out_channels

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        for i, (layer, act) in enumerate(zip(self.layers, self.activations)):
            # Apply dropout to non-final layers
            if i < len(self.layers) - 1:
                x = F.dropout(x, p=self.dropout, training=self.training)
            
            x = layer(x, edge_index)
            x = act(x)
            
        # Final layer returns raw logits for the loss function
        return x

# +---------------------------------------------------------------------------------+
# | Section 3: Model Builder Factory (Main Entry Point)                             |
# | This is the only part that external modules (like TrainerWorker) should call.   |
# +---------------------------------------------------------------------------------+

class ModelBuilder:
    """
    A factory for dynamically building GNN models from architecture specifications.
    """
    
    @staticmethod
    def build_model(arch_spec: Dict[str, Any], 
                    input_dim: int, 
                    output_dim: int,
                    config: Optional[Dict[str, Any]] = None) -> nn.Module:
        """
        Builds a GNN model from an architecture specification.

        Args:
            arch_spec (Dict[str, Any]): The architecture specification.
            input_dim (int): Input feature dimension.
            output_dim (int): Number of output classes.
            config (Dict, optional): Additional config like 'dropout'.

        Returns:
            nn.Module: The constructed PyTorch model.
        """
        dropout_rate = config.get('dropout', 0.5) if config else 0.5
        
        # Detect and normalize architecture format
        if 'layers' in arch_spec:
            # Layer-wise format, requires field name normalization
            normalized_spec = ModelBuilder._normalize_layer_spec(arch_spec)
        elif 'cell_spec' in arch_spec:
            # Cell-based format, convert to layer-wise
            normalized_spec = ModelBuilder._convert_cell_to_layer_spec(arch_spec, input_dim, output_dim)
        else:
            raise ValueError(f"Unknown arch format: {arch_spec.keys()}. Expected 'layers' or 'cell_spec'")
        
        return DynamicGNN(normalized_spec, input_dim, output_dim, dropout=dropout_rate)
    
    @staticmethod
    def _normalize_layer_spec(arch_spec: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normalizes layer-wise spec field names for DynamicGNN.
        """
        normalized_layers = []
        for layer_info in arch_spec['layers']:
            # Map 'encoding' to 'type'
            encoding_to_type_map = {
                'gcn': 'gcn', 'gat': 'gat', 'sym-gat': 'gat', 'const': 'gcn',
                'cos': 'gcn', 'linear': 'gcn', 'gene-linear': 'gcn'
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
        Converts a cell-based architecture to a layer-wise format.
        """
        cell_spec = cell_arch['cell_spec']
        num_cells = cell_arch.get('num_cells_to_stack', 2)
        
        # Map cell op to layer type
        op_to_layer_map = {
            'gcn': 'gcn', 'gat_1_head': 'gat', 'gat_2_head': 'gat', 'gat_4_head': 'gat', 
            'gat_6_head': 'gat', 'gat_8_head': 'gat', 'mean_sage': 'sage',
            'chebnet': 'cheb', 'arma': 'arma', 'sgc': 'sg',
            'identity': 'gcn',  # Use GCN as a substitute for identity
            'zero': 'gcn'       # Use GCN as a substitute for zero
        }
        
        # Extract heads from op name
        def get_heads_from_op(op_name: str) -> int:
            if 'gat_' in op_name and '_head' in op_name:
                return int(op_name.split('_')[1])
            return 1
        
        layers = []
        connections = cell_spec.get('connections', [])
        
        # Generate a layer for each cell
        for cell_idx in range(num_cells):
            # Use the first connection's op as the layer type
            if connections and cell_idx < len(connections):
                op_name = connections[cell_idx]['op']
                layer_type = op_to_layer_map.get(op_name, 'gcn')
                heads = get_heads_from_op(op_name)
            else:
                layer_type = 'gcn'
                heads = 1
            
            # Set hidden dimension
            if cell_idx == 0:
                hidden_dim = 128
            elif cell_idx == num_cells - 1:
                hidden_dim = output_dim
            else:
                hidden_dim = 64
            
            # Set activation
            activation = 'linear' if cell_idx == num_cells - 1 else cell_spec.get('final_act', 'relu')
            
            layer_info = {
                'type': layer_type, 'hidden_dim': hidden_dim,
                'act': activation, 'heads': heads
            }
            layers.append(layer_info)
        
        return {'layers': layers}

# ===================================================================================
# | Independent Test Module                                                         |
# ===================================================================================
if __name__ == '__main__':
    print("===== Running Independent Test for trainer/builder.py =====")
    
    # Mock dimensions for Cora dataset
    INPUT_DIM = 1433
    OUTPUT_DIM = 7
    TRAIN_CONFIG = {'dropout': 0.6}

    # --- Test Case 1: Build a simple 2-layer GCN ---
    print("\n[Test Case 1] Building a 2-layer GCN model...")
    gcn_spec = {
        'layers': [
            {'type': 'gcn', 'hidden_dim': 128, 'act': 'relu'},
            {'type': 'gcn', 'hidden_dim': 999, 'act': 'linear'} 
        ]
    }
    try:
        gcn_model = ModelBuilder.build_model(gcn_spec, INPUT_DIM, OUTPUT_DIM, TRAIN_CONFIG)
        print("  - Model built successfully!")
        print(gcn_model)
        assert isinstance(gcn_model.layers[0], GCNConv)
        assert gcn_model.layers[0].out_channels == 128
        assert isinstance(gcn_model.layers[1], GCNConv)
        assert gcn_model.layers[1].out_channels == OUTPUT_DIM
        print("  - Structure and dimensions verified.")
    except Exception as e:
        print(f"  - Test failed: {e}")

    # --- Test Case 2: Build a mixed GCN and GAT model ---
    print("\n[Test Case 2] Building a GCN -> GAT -> SAGE model...")
    mixed_spec = {
        'layers': [
            {'type': 'gcn', 'hidden_dim': 256, 'act': 'relu'},
            {'type': 'gat', 'hidden_dim': 64, 'heads': 4, 'act': 'elu'}, 
            {'type': 'sage', 'hidden_dim': 999, 'act': 'linear'}
        ]
    }
    try:
        mixed_model = ModelBuilder.build_model(mixed_spec, INPUT_DIM, OUTPUT_DIM, TRAIN_CONFIG)
        print("  - Model built successfully!")
        print(mixed_model)
        assert isinstance(mixed_model.layers[0], GCNConv)
        assert mixed_model.layers[0].out_channels == 256
        assert isinstance(mixed_model.layers[1], GATConv)
        assert mixed_model.layers[1].out_channels == 64
        assert mixed_model.layers[1].heads == 4
        assert isinstance(mixed_model.layers[2], SAGEConv)
        assert mixed_model.layers[2].in_channels == 256 # Verify correct input dimension
        assert mixed_model.layers[2].out_channels == OUTPUT_DIM
        print("  - Structure and dimensions verified.")

    except Exception as e:
        print(f"  - Test failed: {e}")

    print("\n===== Independent test for trainer/builder.py complete. =====")
