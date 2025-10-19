import torch
import torch.nn as nn
import numpy as np
import json
import os
import yaml
import ray
from datetime import datetime
from typing import List, Dict, Any, Tuple, Optional

class Config:
    L_MAX = 5
    ACTIVATION_TYPES = ['relu', 'sigmoid', 'tanh', 'none']
    ATTENTION_TYPES = ['gat', 'gcn', 'none']
    AGGREGATION_TYPES = ['sum', 'mean', 'max', 'mlp']
    ARCH_NUMERICAL_FEATURES = ['attention_heads', 'hidden_units']
    GRAPH_NUMERICAL_FEATURES = ['vertices', 'edges', 'avg_degree', 'diameter']
    DEVICE_NUMERICAL_FEATURES = ['gpu_cores', 'core_frequency', 'fp_capability']

    @staticmethod
    def get_category_maps() -> Dict[str, Dict[str, int]]:
        return {'activation': {name: i for i, name in enumerate(Config.ACTIVATION_TYPES)}, 'attention': {name: i for i, name in enumerate(Config.ATTENTION_TYPES)}, 'aggregation': {name: i for i, name in enumerate(Config.AGGREGATION_TYPES)}}

class DataPreprocessor:

    def __init__(self, config: Config):
        self.config = config
        self.category_maps = config.get_category_maps()
        self.normalization_params = {}

    def _one_hot_encode(self, value: str, category_type: str) -> List[float]:
        num_classes = len(self.category_maps[category_type])
        encoding = [0.0] * num_classes
        if value in self.category_maps[category_type]:
            encoding[self.category_maps[category_type][value]] = 1.0
        return encoding

    def _normalize(self, value: float, feature_name: str) -> float:
        if feature_name not in self.normalization_params:
            return value
        min_val, max_val = self.normalization_params[feature_name]
        if max_val == min_val:
            return 0.5
        return (value - min_val) / (max_val - min_val)

    def fit(self, samples: List[Dict[str, Any]]):
        print('Fitting preprocessor on training data...')
        numerical_data = {key: [] for key in self.config.ARCH_NUMERICAL_FEATURES + self.config.GRAPH_NUMERICAL_FEATURES + self.config.DEVICE_NUMERICAL_FEATURES}
        for sample in samples:
            for layer in sample['architecture']:
                for key in self.config.ARCH_NUMERICAL_FEATURES:
                    numerical_data[key].append(layer.get(key, 0))
            for key in self.config.GRAPH_NUMERICAL_FEATURES:
                numerical_data[key].append(sample['graph'].get(key, 0))
            for key in self.config.DEVICE_NUMERICAL_FEATURES:
                numerical_data[key].append(sample['device'].get(key, 0))
        for key, values in numerical_data.items():
            if values:
                self.normalization_params[key] = (min(values), max(values))
        print('Preprocessor fitted.')

    def transform_single(self, sample: Dict[str, Any]) -> torch.Tensor:
        x_arc_list = []
        num_layers = len(sample['architecture'])
        for i in range(self.config.L_MAX):
            if i < num_layers:
                layer = sample['architecture'][i]
                act_enc = self._one_hot_encode(layer['activation'], 'activation')
                att_enc = self._one_hot_encode(layer['attention_type'], 'attention')
                agg_enc = self._one_hot_encode(layer['aggregation_type'], 'aggregation')
                heads_norm = self._normalize(layer['attention_heads'], 'attention_heads')
                hidden_norm = self._normalize(layer['hidden_units'], 'hidden_units')
                x_arc_list.extend(act_enc + att_enc + agg_enc + [heads_norm, hidden_norm])
            else:
                layer_feature_dim = len(self.config.ACTIVATION_TYPES) + len(self.config.ATTENTION_TYPES) + len(self.config.AGGREGATION_TYPES) + len(self.config.ARCH_NUMERICAL_FEATURES)
                x_arc_list.extend([0.0] * layer_feature_dim)
        x_arc = torch.tensor(x_arc_list, dtype=torch.float32)
        x_gra_list = [self._normalize(sample['graph'][key], key) for key in self.config.GRAPH_NUMERICAL_FEATURES]
        x_gra = torch.tensor(x_gra_list, dtype=torch.float32)
        x_dev_list = [self._normalize(sample['device'][key], key) for key in self.config.DEVICE_NUMERICAL_FEATURES]
        x_dev = torch.tensor(x_dev_list, dtype=torch.float32)
        return torch.cat([x_arc, x_gra, x_dev])

    def fit_transform(self, samples: List[Dict[str, Any]], labels: List[float]) -> Tuple[torch.Tensor, torch.Tensor]:
        self.fit(samples)
        transformed_samples = [self.transform_single(s) for s in samples]
        X = torch.stack(transformed_samples)
        y = torch.tensor(labels, dtype=torch.float32).view(-1, 1)
        return (X, y)

    def get_state(self) -> Dict[str, Any]:
        return {'normalization_params': self.normalization_params}

    def set_state(self, state: Dict[str, Any]):
        self.normalization_params = state['normalization_params']

class MloadModel(nn.Module):

    def __init__(self, input_dim: int):
        super(MloadModel, self).__init__()
        self.input_dim = input_dim
        self.network = nn.Sequential(nn.Linear(input_dim, 256), nn.BatchNorm1d(256), nn.GELU(), nn.Dropout(p=0.2), nn.Linear(256, 128), nn.BatchNorm1d(128), nn.GELU(), nn.Dropout(p=0.2), nn.Linear(128, 64), nn.BatchNorm1d(64), nn.GELU(), nn.Dropout(p=0.2), nn.Linear(64, 1), nn.Softplus())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

class MloadPredictor:

    def __init__(self, model: MloadModel, preprocessor: DataPreprocessor):
        self._model = model
        self._preprocessor = preprocessor
        self._model.eval()

    @classmethod
    def load_from_file(cls, filepath: str, config: Config=Config()):
        try:
            checkpoint = torch.load(filepath, weights_only=True)
            input_dim = checkpoint['input_dim']
            model = MloadModel(input_dim)
            model.load_state_dict(checkpoint['model_state_dict'])
            preprocessor = DataPreprocessor(config)
            preprocessor.set_state(checkpoint['preprocessor_state'])
            print(f'Predictor loaded successfully from {filepath}')
            return cls(model, preprocessor)
        except FileNotFoundError:
            print(f'Error: Checkpoint file not found at {filepath}')
            return None

    def predict(self, sample: Dict[str, Any]) -> float:
        input_tensor = self._preprocessor.transform_single(sample)
        input_tensor = input_tensor.unsqueeze(0)
        with torch.no_grad():
            predicted_value = self._model(input_tensor)
        return predicted_value.item()

class MloadDataCollector:

    def __init__(self, dataset_dir: str='dataset/Mload_dataset', device_type: str='a6000', collector_type: str='auto', priority_order: Optional[List[str]]=None, data_collection_cfg: Optional[Dict[str, Any]]=None):
        self.dataset_dir = dataset_dir
        self.output_file = os.path.join(self.dataset_dir, 'test.yaml')
        self.device_type = device_type.lower()
        self.collector_type = collector_type.lower()
        self.priority_order = priority_order or ['partition', 'dataset']
        os.makedirs(self.dataset_dir, exist_ok=True)
        self.data_samples: List[Dict[str, Any]] = []
        self.device_info = self._get_predefined_device_info()
        self._validate_config()

    def _validate_config(self):
        valid_types = ['partition', 'dataset', 'auto', 'both']
        if self.collector_type not in valid_types:
            raise ValueError(f'无效的collector_type: {self.collector_type}. 支持的类型: {valid_types}')
        valid_priorities = ['partition', 'dataset']
        for p in self.priority_order:
            if p not in valid_priorities:
                raise ValueError(f'无效的priority: {p}. 支持的优先级: {valid_priorities}')
        print(f'数据收集器配置: type={self.collector_type}, priority={self.priority_order}')

    def _get_predefined_device_info(self) -> Dict[str, Any]:
        p100_device_info = {'gpu_cores': 3584, 'core_frequency': 1.328, 'fp_capability': 6.0}
        a6000_device_info = {'gpu_cores': 10752, 'core_frequency': 2.1, 'fp_capability': 8.6}
        return a6000_device_info if self.device_type == 'a6000' else p100_device_info

    def _map_encoding_to_attention(self, encoding: str) -> str:
        mapping = {'gcn': 'gcn', 'gat': 'gat', 'sym-gat': 'gat', 'sage': 'gcn', 'cheb': 'gcn', 'arma': 'gcn', 'sgc': 'gcn', 'const': 'none', 'cos': 'none', 'linear': 'none', 'gene-linear': 'none'}
        return mapping.get(encoding, 'gcn')

    def _map_cell_op_to_attention(self, cell_op: str) -> str:
        if 'gat' in cell_op:
            return 'gat'
        elif cell_op in ['gcn', 'chebnet', 'mean_sage', 'arma', 'sgc']:
            return 'gcn'
        else:
            return 'none'

    def _extract_heads_from_cell_op(self, cell_op: str) -> int:
        head_mapping = {'gat_1_head': 1, 'gat_2_head': 2, 'gat_4_head': 4, 'gat_6_head': 6, 'gat_8_head': 8}
        return head_mapping.get(cell_op, 1)

    def _convert_arch_to_mload_format(self, arch_spec: Dict[str, Any]) -> List[Dict[str, Any]]:
        architecture_layers: List[Dict[str, Any]] = []
        if 'layers' in arch_spec:
            for layer in arch_spec['layers']:
                architecture_layers.append({'activation': layer.get('activation', 'relu'), 'attention_type': self._map_encoding_to_attention(layer.get('encoding', 'gcn')), 'aggregation_type': layer.get('aggregation', 'mean'), 'attention_heads': layer.get('num_heads', 1), 'hidden_units': layer.get('hidden_dim', 64)})
        elif 'cell_spec' in arch_spec:
            cell_spec = arch_spec['cell_spec']
            num_cells = arch_spec.get('num_cells_to_stack', 2)
            for _ in range(num_cells):
                main_op = 'gcn'
                if 'connections' in cell_spec and cell_spec['connections']:
                    main_op = cell_spec['connections'][0].get('op', 'gcn')
                architecture_layers.append({'activation': cell_spec.get('final_act', 'relu'), 'attention_type': self._map_cell_op_to_attention(main_op), 'aggregation_type': cell_spec.get('readout_op', 'add'), 'attention_heads': self._extract_heads_from_cell_op(main_op), 'hidden_units': 64})
        else:
            architecture_layers.append({'activation': 'relu', 'attention_type': 'gcn', 'aggregation_type': 'mean', 'attention_heads': 1, 'hidden_units': 64})
        return architecture_layers

    def add_training_record(self, arch_spec: Dict[str, Any], graph_info: Dict[str, Any], training_time: float, config: Dict[str, Any]) -> Dict[str, Any]:
        sample: Dict[str, Any] = {'architecture': self._convert_arch_to_mload_format(arch_spec), 'graph': {'vertices': int(graph_info.get('vertices', 0)), 'edges': int(graph_info.get('edges', 0)), 'avg_degree': float(graph_info.get('avg_degree', 0.0)), 'diameter': int(graph_info.get('diameter', 0))}, 'device': dict(self.device_info), 'training_time': float(training_time), 'timestamp': datetime.now().isoformat(), 'config': {'epochs_per_rotation': int(config.get('epochs_per_rotation', 5)), 'batch_size': int(config.get('batch_size', 64)), 'lr': float(config.get('lr', 0.01))}}
        self.data_samples.append(sample)
        return sample

    def _read_existing(self) -> Dict[str, Any]:
        if not os.path.exists(self.output_file):
            return {'samples': [], 'labels': []}
        try:
            with open(self.output_file, 'r') as f:
                content = f.read().strip()
                if not content:
                    return {'samples': [], 'labels': []}
                data = yaml.safe_load(content)
                if not isinstance(data, dict):
                    return {'samples': [], 'labels': []}
                data.setdefault('samples', [])
                data.setdefault('labels', [])
                return data
        except Exception:
            return {'samples': [], 'labels': []}

    def save_dataset(self, filename: str='test.yaml') -> None:
        if not self.data_samples:
            return
        self.output_file = os.path.join(self.dataset_dir, filename)
        existing = self._read_existing()
        existing_samples = existing.get('samples', [])
        existing_labels = existing.get('labels', [])
        existing_samples.extend(self.data_samples)
        existing_labels.extend([s.get('training_time', 0.0) for s in self.data_samples])
        merged = {'samples': existing_samples, 'labels': existing_labels}
        with open(self.output_file, 'w') as f:
            yaml.dump(merged, f, sort_keys=False, allow_unicode=True)
        self.data_samples.clear()

    def _is_valid_stats(self, stats: Dict[str, Any]) -> bool:
        required_keys = ['vertices', 'edges', 'avg_degree', 'diameter']
        return all((k in stats for k in required_keys))

    def _normalize_stats(self, stats: Dict[str, Any]) -> Dict[str, Any]:
        return {'vertices': int(stats['vertices']), 'edges': int(stats['edges']), 'avg_degree': float(stats['avg_degree']), 'diameter': int(stats['diameter'])}

    def _get_default_stats(self) -> Dict[str, Any]:
        return {'vertices': 2708, 'edges': 5278, 'avg_degree': 3.9, 'diameter': 19}

    def _compute_partition_stats_on_demand(self, data_partition: Any) -> Dict[str, Any]:
        try:
            x = getattr(data_partition, 'x', None)
            edge_index = getattr(data_partition, 'edge_index', None)
            num_nodes = int(x.shape[0]) if isinstance(x, torch.Tensor) else 0
            num_edges = int(edge_index.shape[1]) if isinstance(edge_index, torch.Tensor) else 0
            avg_degree = float(num_edges / num_nodes) if num_nodes > 0 else 0.0
            if num_nodes < 1000:
                diameter = 8
            elif num_nodes < 10000:
                diameter = 15
            else:
                diameter = 25
            return {'vertices': num_nodes, 'edges': num_edges, 'avg_degree': avg_degree, 'diameter': diameter}
        except Exception:
            return self._get_default_stats()

    def _extract_partition_stats(self, data_partition: Any) -> Dict[str, Any]:
        cached = getattr(data_partition, 'graph_stats', None)
        if isinstance(cached, dict) and self._is_valid_stats(cached):
            return self._normalize_stats(cached)
        return self._compute_partition_stats_on_demand(data_partition)

    def _extract_dataset_stats(self, data_partition: Any) -> Dict[str, Any]:
        ds_cached = getattr(data_partition, 'dataset_graph_stats', None)
        if isinstance(ds_cached, dict) and self._is_valid_stats(ds_cached):
            return self._normalize_stats(ds_cached)
        print(f"--》原本打算记录数据集数据，但是没有在data_partition中找到相关'dataset_graph_stats'字段")
        return self._get_default_stats()

    def _extract_both_stats(self, data_partition: Any) -> Dict[str, Any]:
        partition_stats = self._extract_partition_stats(data_partition)
        dataset_stats = self._extract_dataset_stats(data_partition)
        return {'partition_info': partition_stats, 'dataset_info': dataset_stats, 'vertices': partition_stats['vertices'], 'edges': partition_stats['edges'], 'avg_degree': partition_stats['avg_degree'], 'diameter': partition_stats['diameter']}

    def _extract_auto_stats(self, data_partition: Any) -> Dict[str, Any]:
        for priority in self.priority_order:
            try:
                if priority == 'partition':
                    result = self._extract_partition_stats(data_partition)
                    if result != self._get_default_stats():
                        return result
                elif priority == 'dataset':
                    result = self._extract_dataset_stats(data_partition)
                    if result != self._get_default_stats():
                        return result
            except Exception:
                continue
        return self._get_default_stats()

    def extract_graph_info(self, data_partition: Any) -> Dict[str, Any]:
        if self.collector_type == 'partition':
            return self._extract_partition_stats(data_partition)
        elif self.collector_type == 'dataset':
            return self._extract_dataset_stats(data_partition)
        elif self.collector_type == 'both':
            return self._extract_both_stats(data_partition)
        elif self.collector_type == 'auto':
            return self._extract_auto_stats(data_partition)
        return self._extract_auto_stats(data_partition)

@ray.remote(num_gpus=0)
class DataCollectorActor:

    def __init__(self, dataset_dir: str='dataset/Mload_dataset', filename: str='test.yaml', flush_every: int=1, data_collection_cfg: Optional[Dict[str, Any]]=None):
        self.dataset_dir = dataset_dir
        self.filename = filename
        self.flush_every = max(int(flush_every), 1)
        os.makedirs(self.dataset_dir, exist_ok=True)
        self.output_file = os.path.join(self.dataset_dir, self.filename)
        self.data_collection_cfg = data_collection_cfg
        self.time_window_mode = data_collection_cfg.get('time_window_mode', 'task') if data_collection_cfg else 'task'
        self._data: Dict[str, Any] = self._safe_read_existing()
        self._pending = 0

    def add_record(self, sample: Dict[str, Any]) -> int:
        if not isinstance(sample, dict):
            return len(self._data.get('samples', []))
        self._data.setdefault('samples', []).append(sample)
        self._data.setdefault('labels', []).append(float(sample.get('training_time', 0.0)))
        self._pending += 1
        if self._pending >= self.flush_every:
            self._flush()
        return len(self._data['samples'])

    def add_records(self, samples: List[Dict[str, Any]]) -> int:
        for s in samples or []:
            self._data.setdefault('samples', []).append(s)
            self._data.setdefault('labels', []).append(float(s.get('training_time', 0.0)))
            self._pending += 1
        if self._pending >= self.flush_every:
            self._flush()
        return len(self._data.get('samples', []))

    def force_flush(self) -> None:
        self._flush()

    def _safe_read_existing(self) -> Dict[str, Any]:
        if not os.path.exists(self.output_file):
            return {'samples': [], 'labels': []}
        try:
            with open(self.output_file, 'r') as f:
                content = f.read().strip()
                if not content:
                    return {'samples': [], 'labels': []}
                data = yaml.safe_load(content)
                if not isinstance(data, dict):
                    return {'samples': [], 'labels': []}
                data.setdefault('samples', [])
                data.setdefault('labels', [])
                return data
        except Exception:
            return {'samples': [], 'labels': []}

    def _flush(self) -> None:
        tmp_path = self.output_file + '.tmp'
        try:
            with open(tmp_path, 'w') as f:
                yaml.dump(self._data, f, sort_keys=False, allow_unicode=True)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_path, self.output_file)
            self._pending = 0
        except Exception:
            pass

def train_model(model: MloadModel, X_train: torch.Tensor, y_train: torch.Tensor, epochs: int=200, learning_rate: float=0.001):
    print(f'\nStarting training for {epochs} epochs...')
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    for epoch in range(epochs):
        model.train()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 20 == 0:
            print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.6f}')
    print('Training finished.')

def save_checkpoint(model: MloadModel, preprocessor: DataPreprocessor, filepath: str):
    checkpoint = {'input_dim': model.input_dim, 'model_state_dict': model.state_dict(), 'preprocessor_state': preprocessor.get_state()}
    torch.save(checkpoint, filepath)
    print(f'Model and preprocessor saved to {filepath}')
if __name__ == '__main__':
    p100_device_info = {'gpu_cores': 3584, 'core_frequency': 1.328, 'fp_capability': 6.0}
    a6000_device_info = {'gpu_cores': 10752, 'core_frequency': 2.1, 'fp_capability': 8.6}
    sample_data = [{'architecture': [{'activation': 'relu', 'attention_type': 'gat', 'aggregation_type': 'sum', 'attention_heads': 8, 'hidden_units': 128}, {'activation': 'relu', 'attention_type': 'gat', 'aggregation_type': 'sum', 'attention_heads': 8, 'hidden_units': 64}], 'graph': {'vertices': 1000, 'edges': 5000, 'avg_degree': 5.0, 'diameter': 10}, 'device': p100_device_info}, {'architecture': [{'activation': 'tanh', 'attention_type': 'gcn', 'aggregation_type': 'mean', 'attention_heads': 1, 'hidden_units': 256}], 'graph': {'vertices': 50000, 'edges': 200000, 'avg_degree': 4.0, 'diameter': 25}, 'device': p100_device_info}, {'architecture': [{'activation': 'relu', 'attention_type': 'none', 'aggregation_type': 'max', 'attention_heads': 1, 'hidden_units': 64}, {'activation': 'relu', 'attention_type': 'none', 'aggregation_type': 'max', 'attention_heads': 1, 'hidden_units': 64}, {'activation': 'relu', 'attention_type': 'none', 'aggregation_type': 'max', 'attention_heads': 1, 'hidden_units': 32}], 'graph': {'vertices': 2000, 'edges': 8000, 'avg_degree': 4.0, 'diameter': 12}, 'device': p100_device_info}, {'architecture': [{'activation': 'sigmoid', 'attention_type': 'gat', 'aggregation_type': 'mlp', 'attention_heads': 16, 'hidden_units': 512}], 'graph': {'vertices': 100000, 'edges': 1000000, 'avg_degree': 10.0, 'diameter': 15}, 'device': p100_device_info}, {'architecture': [{'activation': 'relu', 'attention_type': 'gcn', 'aggregation_type': 'mean', 'attention_heads': 1, 'hidden_units': 128}, {'activation': 'relu', 'attention_type': 'gcn', 'aggregation_type': 'mean', 'attention_heads': 1, 'hidden_units': 128}, {'activation': 'relu', 'attention_type': 'gcn', 'aggregation_type': 'mean', 'attention_heads': 1, 'hidden_units': 128}, {'activation': 'relu', 'attention_type': 'gcn', 'aggregation_type': 'mean', 'attention_heads': 1, 'hidden_units': 128}], 'graph': {'vertices': 80000, 'edges': 400000, 'avg_degree': 5.0, 'diameter': 30}, 'device': p100_device_info}]
    labels = [15.5, 80.2, 25.0, 250.8, 120.3]
    config_instance = Config()
    preprocessor = DataPreprocessor(config_instance)
    X_train_tensor, y_train_tensor = preprocessor.fit_transform(sample_data, labels)
    input_dimension = X_train_tensor.shape[1]
    model = MloadModel(input_dim=input_dimension)
    train_model(model, X_train_tensor, y_train_tensor, epochs=500)
    MODEL_FILE_PATH = 'mload_checkpoint.pth'
    save_checkpoint(model, preprocessor, MODEL_FILE_PATH)
    print('\n' + '=' * 50)
    print('DEMONSTRATING PREDICTION INTERFACE')
    print('=' * 50 + '\n')
    predictor = MloadPredictor.load_from_file(MODEL_FILE_PATH)
    if predictor:
        new_sample = {'architecture': [{'activation': 'relu', 'attention_type': 'gat', 'aggregation_type': 'mean', 'attention_heads': 4, 'hidden_units': 200}, {'activation': 'tanh', 'attention_type': 'none', 'aggregation_type': 'sum', 'attention_heads': 1, 'hidden_units': 50}], 'graph': {'vertices': 30000, 'edges': 150000, 'avg_degree': 5.0, 'diameter': 20}, 'device': {'gpu_cores': 5120, 'core_frequency': 1.8, 'fp_capability': 7.5}}
        predicted_runtime = predictor.predict(new_sample)
        print(f'New Sample Configuration:\n{json.dumps(new_sample, indent=2)}')
        print(f'Predicted Runtime: {predicted_runtime:.2f} seconds')
        another_sample = sample_data[0]
        predicted_runtime_2 = predictor.predict(another_sample)
        print(f'\nPredicting on a known sample (first training sample):')
        print(f'Predicted Runtime: {predicted_runtime_2:.2f} seconds (Actual was: {labels[0]})')