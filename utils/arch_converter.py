from typing import Dict, Any, List

class ArchitectureConverter:
    GNN_TYPE_MAP = {'GCNConv': 'gcn', 'GATConv': 'gat', 'GENConv': 'gen', 'SGConv': 'sgc', 'GraphConv': 'graph', 'linear': 'linear'}
    ACTIVATION_MAP = {'Relu': 'relu', 'relu': 'relu', 'Elu': 'elu', 'elu': 'elu', 'linear': 'linear', 'Softplus': 'softplus', 'sigmoid': 'sigmoid', 'tanh': 'tanh', 'relu6': 'relu6', 'leaky_relu': 'leaky_relu'}
    REVERSE_ACTIVATION_MAP = {'relu': 'Relu', 'elu': 'Elu', 'linear': 'linear', 'softplus': 'Softplus', 'sigmoid': 'sigmoid', 'tanh': 'tanh', 'relu6': 'relu6', 'leaky_relu': 'leaky_relu'}
    NORM_MAP = {'GraphNorm': 'graph_norm', 'InstanceNorm': 'instance_norm', 'BatchNorm': 'batch_norm', False: None, 'False': None, None: None}

    @staticmethod
    def is_dgnap_format(arch_spec: Dict[str, Any]) -> bool:
        required_keys = {'gnnConv1', 'gnnConv2'}
        return required_keys.issubset(set(arch_spec.keys()))

    @staticmethod
    def is_layer_wise_format(arch_spec: Dict[str, Any]) -> bool:
        return 'layers' in arch_spec

    @staticmethod
    def is_cell_based_format(arch_spec: Dict[str, Any]) -> bool:
        return 'cell_spec' in arch_spec and 'num_cells_to_stack' in arch_spec

    @classmethod
    def dgnap_to_layer_wise(cls, dgnap_spec: Dict[str, Any]) -> Dict[str, Any]:
        layers = []
        hidden_dim = dgnap_spec.get('hidden_channels', 64)
        dropout = dgnap_spec.get('dropout', 0.5)
        layer_count = 0
        for i in range(1, 10):
            if f'gnnConv{i}' in dgnap_spec:
                layer_count = i
            else:
                break
        if layer_count == 0:
            layer_count = 2
        for i in range(1, layer_count + 1):
            gnn_type = dgnap_spec.get(f'gnnConv{i}', 'GCNConv')
            activation = dgnap_spec.get(f'activation{i}', 'Relu')
            normalize = dgnap_spec.get(f'normalize{i}', False)
            aggregation = dgnap_spec.get(f'aggregation{i}', 'add')
            layer_type = cls.GNN_TYPE_MAP.get(gnn_type, 'gcn')
            act = cls.ACTIVATION_MAP.get(activation, 'relu')
            norm = cls.NORM_MAP.get(normalize, None)
            layer = {'type': layer_type, 'hidden_dim': hidden_dim, 'act': act}
            if norm:
                layer['norm'] = norm
            if layer_type == 'gat':
                layer['heads'] = 4
            if aggregation and aggregation != 'add':
                layer['aggr'] = aggregation
            layers.append(layer)
        return {'layers': layers, 'dropout': dropout}

    @classmethod
    def layer_wise_to_dgnap(cls, layer_wise_spec: Dict[str, Any]) -> Dict[str, Any]:
        if not cls.is_layer_wise_format(layer_wise_spec):
            raise ValueError('输入不是layer-wise格式的架构规范')
        layers = layer_wise_spec.get('layers', [])
        if not layers:
            raise ValueError('layer-wise格式中缺少layers字段或layers为空')
        reverse_gnn_map = {v: k for k, v in cls.GNN_TYPE_MAP.items()}
        reverse_activation_map = cls.REVERSE_ACTIVATION_MAP
        reverse_norm_map = {v: k for k, v in cls.NORM_MAP.items()}
        dgnap_spec = {}
        dropout = layer_wise_spec.get('dropout', 0.5)
        dgnap_spec['dropout'] = dropout
        dgnap_spec['lr'] = 0.01
        dgnap_spec['weight_decay'] = 0.0005
        dgnap_spec['optimizer'] = 'adam'
        dgnap_spec['criterion'] = 'CrossEntropyLoss'
        dgnap_spec['pooling'] = 'global_add_pool'
        for i, layer in enumerate(layers[:2], 1):
            layer_type = layer.get('type', 'gcn')
            gnn_conv = reverse_gnn_map.get(layer_type, 'GCNConv')
            dgnap_spec[f'gnnConv{i}'] = gnn_conv
            activation = layer.get('act', 'relu')
            activation_name = reverse_activation_map.get(activation, 'Relu')
            dgnap_spec[f'activation{i}'] = activation_name
            norm = layer.get('norm')
            if norm is None:
                normalize = False
            else:
                normalize = reverse_norm_map.get(norm, False)
            dgnap_spec[f'normalize{i}'] = normalize
            aggregation = layer.get('aggr', 'add')
            dgnap_spec[f'aggregation{i}'] = aggregation
            hidden_dim = layer.get('hidden_dim', 64)
            dgnap_spec['hidden_channels'] = hidden_dim
        if len(layers) == 1:
            dgnap_spec['gnnConv2'] = 'GCNConv'
            dgnap_spec['activation2'] = 'Relu'
            dgnap_spec['normalize2'] = False
            dgnap_spec['aggregation2'] = 'add'
        return dgnap_spec

    @classmethod
    def convert(cls, arch_spec: Dict[str, Any]) -> Dict[str, Any]:
        if cls.is_layer_wise_format(arch_spec) or cls.is_cell_based_format(arch_spec):
            return arch_spec
        if cls.is_dgnap_format(arch_spec):
            return cls.dgnap_to_layer_wise(arch_spec)
        return cls.dgnap_to_layer_wise(arch_spec)

    @classmethod
    def convert_to_dgnap(cls, arch_spec: Dict[str, Any]) -> Dict[str, Any]:
        if cls.is_dgnap_format(arch_spec):
            return arch_spec
        if cls.is_layer_wise_format(arch_spec):
            return cls.layer_wise_to_dgnap(arch_spec)
        if cls.is_cell_based_format(arch_spec):
            raise ValueError('暂不支持cell-based格式到DGNAP格式的转换')
        return arch_spec

def convert_architecture(arch_spec: Dict[str, Any]) -> Dict[str, Any]:
    return ArchitectureConverter.convert(arch_spec)

def convert_architecture_to_dgnap(arch_spec: Dict[str, Any]) -> Dict[str, Any]:
    return ArchitectureConverter.convert_to_dgnap(arch_spec)
if __name__ == '__main__':
    dgnap_arch = {'gnnConv1': 'GCNConv', 'gnnConv2': 'GATConv', 'activation1': 'Relu', 'activation2': 'Elu', 'normalize1': 'BatchNorm', 'normalize2': False, 'aggregation1': 'add', 'aggregation2': 'mean', 'hidden_channels': 64, 'dropout': 0.5, 'lr': 0.001, 'optimizer': 'adam'}
    converter = ArchitectureConverter()
    layer_wise = converter.convert(dgnap_arch)
    print('DGNAP格式:')
    print(dgnap_arch)
    print('\n转换为Layer-wise格式:')
    print(layer_wise)
    assert 'layers' in layer_wise
    assert len(layer_wise['layers']) == 2
    assert layer_wise['layers'][0]['type'] == 'gcn'
    assert layer_wise['layers'][1]['type'] == 'gat'
    print('\n✓ DGNAP -> Layer-wise 转换测试通过!')
    print('\n' + '=' * 50)
    print('测试反向转换: Layer-wise -> DGNAP')
    print('=' * 50)
    dgnap_back = converter.layer_wise_to_dgnap(layer_wise)
    print('Layer-wise格式:')
    print(layer_wise)
    print('\n转换为DGNAP格式:')
    print(dgnap_back)
    assert 'gnnConv1' in dgnap_back
    assert 'gnnConv2' in dgnap_back
    assert 'activation1' in dgnap_back
    assert 'activation2' in dgnap_back
    assert 'hidden_channels' in dgnap_back
    assert 'dropout' in dgnap_back
    assert dgnap_back['gnnConv1'] == 'GCNConv'
    assert dgnap_back['gnnConv2'] == 'GATConv'
    assert dgnap_back['activation1'] == 'Relu'
    assert dgnap_back['activation2'] == 'Elu'
    assert dgnap_back['hidden_channels'] == 64
    assert dgnap_back['dropout'] == 0.5
    print('\n✓ Layer-wise -> DGNAP 反向转换测试通过!')
    print('\n' + '=' * 50)
    print('测试往返转换一致性')
    print('=' * 50)
    layer_wise_2 = converter.dgnap_to_layer_wise(dgnap_arch)
    dgnap_2 = converter.layer_wise_to_dgnap(layer_wise_2)
    print('原始DGNAP:')
    print(dgnap_arch)
    print('\n往返转换后的DGNAP:')
    print(dgnap_2)
    key_fields = ['gnnConv1', 'gnnConv2', 'activation1', 'activation2', 'hidden_channels', 'dropout']
    for field in key_fields:
        if field in dgnap_arch and field in dgnap_2:
            assert dgnap_arch[field] == dgnap_2[field], f'字段 {field} 不一致: {dgnap_arch[field]} vs {dgnap_2[field]}'
    print('\n✓ 往返转换一致性测试通过!')