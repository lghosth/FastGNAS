import os
import yaml
import pprint
from typing import Union, List, Dict, Any, Optional
from pathlib import Path

def parse_fan_out(fan_out_config: Union[int, List[int], str]) -> List[int]:
    if isinstance(fan_out_config, int):
        return [fan_out_config]
    if isinstance(fan_out_config, list):
        if not fan_out_config:
            raise ValueError('fan_out 列表不能为空')
        if all((isinstance(x, int) for x in fan_out_config)):
            return list(fan_out_config)
        else:
            raise ValueError(f'fan_out列表中包含非整数元素: {fan_out_config}')
    if isinstance(fan_out_config, str):
        fan_out_str = fan_out_config.strip()
        if fan_out_str.lstrip('-').isdigit():
            return [int(fan_out_str)]
        if fan_out_str.startswith('[') and fan_out_str.endswith(']'):
            fan_out_str = fan_out_str[1:-1]
        try:
            elements = [int(x.strip()) for x in fan_out_str.split(',') if x.strip()]
            if not elements:
                raise ValueError('fan_out 字符串解析后为空列表')
            return elements
        except ValueError as e:
            raise ValueError(f"无法解析fan_out字符串 '{fan_out_config}': {e}")
    raise TypeError(f'fan_out类型不支持: {type(fan_out_config)}, 值为 {fan_out_config}')

def normalize_fan_out_for_layers(fan_out: Union[int, List[int]], num_layers: int) -> List[int]:
    if isinstance(fan_out, int):
        return [fan_out] * num_layers
    if isinstance(fan_out, list):
        if len(fan_out) == num_layers:
            return fan_out
        elif len(fan_out) > num_layers:
            return fan_out[:num_layers]
        else:
            return fan_out + [fan_out[-1]] * (num_layers - len(fan_out))
    raise TypeError(f'fan_out必须是int或List[int]，当前类型: {type(fan_out)}')

def format_architecture_for_display(arch_spec: dict, search_strategy: str) -> str:
    lines = []
    lines.append('🏗️  GNN架构详情:')
    lines.append('─' * 50)
    if search_strategy == 'layer_wise':
        lines.append('📋 架构类型: Layer-wise搜索策略')
        if 'layers' in arch_spec:
            lines.append(f"📊 总层数: {len(arch_spec['layers'])}")
            lines.append('')
            for i, layer in enumerate(arch_spec['layers']):
                layer_num = i + 1
                lines.append(f'🔸 第 {layer_num} 层:')
                encoding = layer.get('encoding', 'unknown')
                gnn_type_map = {'gcn': 'GCN (图卷积网络)', 'gat': 'GAT (图注意力网络)', 'sym-gat': 'Symmetric GAT', 'sage': 'GraphSAGE', 'cheb': 'ChebNet (切比雪夫网络)', 'arma': 'ARMA', 'sgc': 'SGC (简化图卷积)', 'const': 'Constant', 'cos': 'Cosine', 'linear': 'Linear', 'gene-linear': 'Generalized Linear'}
                gnn_name = gnn_type_map.get(encoding, f'Unknown ({encoding})')
                lines.append(f'   🧠 模型类型: {gnn_name}')
                if 'hidden_dim' in layer:
                    lines.append(f"   📐 隐藏维度: {layer['hidden_dim']}")
                if 'num_heads' in layer and layer.get('num_heads', 1) > 1:
                    lines.append(f"   👁️  注意力头数: {layer['num_heads']}")
                if 'activation' in layer:
                    act_map = {'relu': 'ReLU', 'tanh': 'Tanh', 'sigmoid': 'Sigmoid', 'elu': 'ELU', 'leaky_relu': 'LeakyReLU', 'none': '无激活', 'linear': '线性', 'softplus': 'Softplus', 'relu6': 'ReLU6'}
                    act_name = act_map.get(layer['activation'], layer['activation'])
                    lines.append(f'   ⚡ 激活函数: {act_name}')
                if 'aggregation' in layer:
                    agg_map = {'mean': '平均聚合', 'sum': '求和聚合', 'max': '最大聚合', 'mlp': 'MLP聚合'}
                    agg_name = agg_map.get(layer['aggregation'], layer['aggregation'])
                    lines.append(f'   🔄 聚合方式: {agg_name}')
                lines.append('')
    elif search_strategy == 'cell_based':
        lines.append('📋 架构类型: Cell-based搜索策略')
        if 'cell_spec' in arch_spec:
            cell_spec = arch_spec['cell_spec']
            num_cells = arch_spec.get('num_cells_to_stack', 2)
            lines.append(f'📊 堆叠单元数: {num_cells}')
            lines.append(f"🔗 中间节点数: {cell_spec.get('intermediate_nodes', 'unknown')}")
            lines.append('')
            if 'connections' in cell_spec:
                lines.append('🔸 单元内连接:')
                for i, conn in enumerate(cell_spec['connections']):
                    op = conn.get('op', 'unknown')
                    from_node = conn.get('from_node', 'unknown')
                    op_type_map = {'gcn': 'GCN (图卷积网络)', 'gat_1_head': 'GAT (1头)', 'gat_2_head': 'GAT (2头)', 'gat_4_head': 'GAT (4头)', 'gat_6_head': 'GAT (6头)', 'gat_8_head': 'GAT (8头)', 'mean_sage': 'GraphSAGE (平均聚合)', 'chebnet': 'ChebNet (切比雪夫网络)', 'arma': 'ARMA', 'sgc': 'SGC (简化图卷积)', 'identity': 'Identity (恒等映射)', 'zero': 'Zero (零操作)'}
                    op_name = op_type_map.get(op, f'Unknown ({op})')
                    lines.append(f'   连接 {i + 1}: 节点{from_node} → {op_name}')
            if 'readout_op' in cell_spec:
                readout_map = {'add': '加法', 'concat': '拼接', 'multiply': '乘法'}
                readout_name = readout_map.get(cell_spec['readout_op'], cell_spec['readout_op'])
                lines.append(f'   🔄 读出操作: {readout_name}')
            if 'final_act' in cell_spec:
                act_map = {'relu': 'ReLU', 'tanh': 'Tanh', 'sigmoid': 'Sigmoid', 'elu': 'ELU', 'leaky_relu': 'LeakyReLU', 'none': '无激活', 'linear': '线性', 'softplus': 'Softplus', 'relu6': 'ReLU6'}
                act_name = act_map.get(cell_spec['final_act'], cell_spec['final_act'])
                lines.append(f'   ⚡ 最终激活: {act_name}')
    else:
        lines.append(f'📋 架构类型: 未知策略 ({search_strategy})')
        lines.append('原始架构规范:')
        lines.append(pprint.pformat(arch_spec, width=60))
    lines.append('─' * 50)
    return '\n'.join(lines)

def load_config(config_path: str) -> dict:
    default_config = {'nas': {'num_search_rounds': 200, 'samples_per_round': 5, 'gnn_layers': 3, 'controller_lr': 0.001, 'controller_baseline_decay': 0.9}, 'training': {'dataset': 'Cora', 'lr': 0.01, 'dropout': 0.6, 'epochs_per_rotation': 10, 'batch_size': 64, 'steps_per_epoch': 50}}
    if not os.path.exists(config_path):
        print(f"警告: 配置文件 '{config_path}' 不存在将创建并使用默认配置")
        raise FileNotFoundError(f'配置文件不存在: {config_path}')
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f) or {}
        for key, value in default_config.items():
            if key not in config:
                config[key] = value
            elif isinstance(value, dict):
                config[key] = {**value, **config[key]}
        print(f"成功从 '{config_path}' 加载并合并配置")
        return config
    except Exception as e:
        print(f'错误: 加载配置文件失败: {e}将使用默认配置')
        raise e

def load_dataset_config(dataset_name: str, dataset_config_path: str='experiment/dataset.yaml') -> Dict[str, Any]:
    config_path = Path(dataset_config_path)
    if not config_path.exists():
        raise FileNotFoundError(f'数据集配置文件不存在: {config_path}')
    with open(config_path, 'r', encoding='utf-8') as f:
        all_configs = yaml.safe_load(f)
    if dataset_name not in all_configs:
        special_sections = {'defaults', 'dataset_groups', 'device_configs', 'experiment_templates'}
        available = [k for k in all_configs.keys() if k not in special_sections]
        raise ValueError(f"数据集 '{dataset_name}' 不存在。可用数据集: {available}")
    return all_configs[dataset_name]

def merge_dataset_config_into_train_config(train_config: Dict[str, Any], dataset_name: str, dataset_config_path: str='experiment/dataset.yaml', device_type: Optional[str]=None) -> Dict[str, Any]:
    dataset_cfg = load_dataset_config(dataset_name, dataset_config_path)
    if device_type:
        config_path = Path(dataset_config_path)
        with open(config_path, 'r', encoding='utf-8') as f:
            all_configs = yaml.safe_load(f)
        device_configs = all_configs.get('device_configs', {})
        if device_type in device_configs:
            device_cfg = device_configs[device_type]
            batch_multiplier = device_cfg.get('recommended_batch_multiplier', 1.0)
            if 'train_batch_size' in dataset_cfg:
                dataset_cfg['train_batch_size'] = int(dataset_cfg['train_batch_size'] * batch_multiplier)
            if 'eval_batch_size' in dataset_cfg:
                dataset_cfg['eval_batch_size'] = int(dataset_cfg['eval_batch_size'] * batch_multiplier)
    key_mapping = {'train_batch_size': 'batch_size', 'eval_batch_size': 'evaluation.batch_size', 'fan_out': 'fan_out', 'steps_per_epoch': 'steps_per_epoch', 'partition_strategy': 'train_partition_strategy', 'num_classes': 'num_classes'}
    for src_key, dst_key in key_mapping.items():
        if src_key not in dataset_cfg:
            continue
        value = dataset_cfg[src_key]
        if '.' in dst_key:
            keys = dst_key.split('.')
            current = train_config
            for key in keys[:-1]:
                if key not in current:
                    current[key] = {}
                current = current[key]
            current[keys[-1]] = value
        else:
            train_config[dst_key] = value
    train_config['dataset'] = dataset_cfg['name']
    return train_config

def load_experiment_config(base_config_path: str, dataset_name: str, device_type: str='a6000', dataset_config_path: str='experiment/dataset.yaml') -> Dict[str, Any]:
    config = load_config(base_config_path)
    if 'training' in config:
        merge_dataset_config_into_train_config(config['training'], dataset_name, dataset_config_path, device_type)
    return config

def list_available_datasets(dataset_config_path: str='experiment/dataset.yaml') -> List[str]:
    config_path = Path(dataset_config_path)
    if not config_path.exists():
        return []
    with open(config_path, 'r', encoding='utf-8') as f:
        all_configs = yaml.safe_load(f)
    special_sections = {'defaults', 'dataset_groups', 'device_configs', 'experiment_templates'}
    datasets = [k for k in all_configs.keys() if k not in special_sections]
    return datasets

def get_dataset_group(group_name: str, dataset_config_path: str='experiment/dataset.yaml') -> List[str]:
    config_path = Path(dataset_config_path)
    with open(config_path, 'r', encoding='utf-8') as f:
        all_configs = yaml.safe_load(f)
    dataset_groups = all_configs.get('dataset_groups', {})
    if group_name not in dataset_groups:
        raise ValueError(f"数据集组 '{group_name}' 不存在。可用组: {list(dataset_groups.keys())}")
    return dataset_groups[group_name]