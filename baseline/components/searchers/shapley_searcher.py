from __future__ import annotations
import math
import os
import random
import sys
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, MutableMapping, Optional, Sequence, Set, Tuple, Union
import numpy as np
try:
    import shap
except Exception as exc:
    shap = None
    _shap_import_error = exc
else:
    _shap_import_error = None
try:
    from sklearn.neural_network import MLPRegressor
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import train_test_split
except Exception as exc:
    MLPRegressor = None
    RandomForestRegressor = None
    _sklearn_import_error = exc
else:
    _sklearn_import_error = None
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except Exception:
    torch = None
    nn = None
    F = None
try:
    from torch_geometric.data import Data
    from torch_geometric.loader import DataLoader
    from torch_geometric.nn import GATConv, GCNConv, global_mean_pool, global_max_pool
except Exception as exc:
    Data = None
    DataLoader = None
    GCNConv = None
    global_mean_pool = None
    _pyg_import_error = exc
else:
    _pyg_import_error = None
from .dgnap_shapley_adapter import ReferenceShapleyConfig, compute_reference_shapley, remap_feature_names
from .dgnap_search_space_adapter import SparseSpaceSpec, build_sparse_search_space
DEFAULT_SEARCH_SPACE: Dict[str, Tuple[Any, ...]] = {'gnnConv1': ('GCNConv', 'GENConv', 'SGConv', 'linear', 'GraphConv', 'GATConv'), 'gnnConv2': ('GCNConv', 'GENConv', 'SGConv', 'linear', 'GraphConv', 'GATConv'), 'aggregation1': ('add', 'max', 'mean'), 'aggregation2': ('add', 'max', 'mean'), 'normalize1': (False, 'GraphNorm', 'InstanceNorm', 'BatchNorm'), 'normalize2': (False, 'GraphNorm', 'InstanceNorm', 'BatchNorm'), 'activation1': ('Relu', 'Elu', 'linear', 'Softplus', 'sigmoid', 'tanh', 'relu6', 'leaky_relu'), 'activation2': ('Relu', 'Elu', 'linear', 'Softplus', 'sigmoid', 'tanh', 'relu6', 'leaky_relu'), 'hidden_channels': (16, 32, 64, 128, 256), 'dropout': (0.0, 0.2, 0.4, 0.6), 'lr': (0.01, 0.001, 0.005, 0.0001), 'weight_decay': (0, 0.001, 0.0005), 'optimizer': ('adam', 'adamW'), 'criterion': ('CrossEntropyLoss', 'fn_loss'), 'pooling': ('global_add_pool', 'global_max_pool', 'global_mean_pool')}
DEFAULT_CONFIG: Dict[str, Any] = {'random_seed': None, 'min_training_samples': 10, 'pruning_threshold': 0.5, 'pruning_iterations': 1, 'pruning_threshold_schedule': None, 'pruning_retrain_each_round': True, 'pruning_stop_when_no_change': True, 'top_k': 5, 'max_candidates': 10000, 'shap_background_size': 50, 'shap_background_method': 'kmeans', 'shap_sample_size': 200, 'component_normalization': True, 'multi_source_keys': ['reward'], 'self_distillation_steps': 0, 'self_distillation_sample_size': 0, 'predictor_type': 'gnn', 'use_tree_explainer': True, 'use_dgnap_pruning': True, 'tree_predictor_params': {'n_estimators': 1000, 'max_depth': 4, 'random_state': None}, 'predictor_params': {'hidden_layer_sizes': (128, 64), 'max_iter': 200, 'random_state': None}, 'gnn_hidden_dim': 64, 'gnn_layers': 2, 'gnn_dropout': 0.1, 'gnn_learning_rate': 0.001, 'gnn_weight_decay': 0.0, 'gnn_epochs': 200, 'gnn_batch_size': 16, 'gnn_device': 'auto', 'gnn_attention_heads': 4, 'gnn_ff_hidden_dim': 128, 'gnn_architecture': 'graphnap', 'reference_shapley': None}

@dataclass
class ArchitectureRecord:
    arch_id: str
    arch_spec: Dict[str, Any]
    encoding: np.ndarray
    reward: float
    metadata: Dict[str, Any] = field(default_factory=dict)
DEFAULT_DAG_EDGES: Tuple[Tuple[str, str], ...] = (('gnnConv1', 'normalize1'), ('aggregation1', 'gnnConv1'), ('hidden_channels', 'gnnConv1'), ('normalize1', 'activation1'), ('activation1', 'gnnConv2'), ('gnnConv2', 'normalize2'), ('aggregation2', 'gnnConv2'), ('hidden_channels', 'gnnConv2'), ('normalize2', 'activation2'), ('activation2', 'pooling'), ('lr', 'optimizer'), ('weight_decay', 'optimizer'), ('dropout', 'pooling'), ('pooling', 'criterion'), ('criterion', 'optimizer'), ('optimizer', 'gnnConv1'))

class SearchSpaceManager:

    def __init__(self, space: Optional[MutableMapping[str, Sequence[Any]]]=None, dag_edges: Optional[Sequence[Tuple[str, str]]]=None, encoding_strategy: str='onehot', sparse_spec: Optional['SparseSpaceSpec']=None) -> None:
        self._original_space = self._normalize_space(space or DEFAULT_SEARCH_SPACE)
        self._components: List[str] = list(self._original_space.keys())
        self._component_index: Dict[str, int] = {comp: idx for idx, comp in enumerate(self._components)}
        self._pruned_space: Optional[Dict[str, Tuple[Any, ...]]] = None
        self._component_choices: Dict[str, List[Any]] = {comp: list(choices) for comp, choices in self._original_space.items()}
        self._component_cardinality: Dict[str, int] = {comp: len(choices) for comp, choices in self._component_choices.items()}
        self._dag_edges: Tuple[Tuple[str, str], ...] = tuple(dag_edges or DEFAULT_DAG_EDGES)
        self._node_slices: Dict[str, Tuple[int, int]] = {}
        self._edge_slices: Dict[Tuple[str, str], Tuple[int, int]] = {}
        self._node_feature_names: Dict[str, List[str]] = {}
        self._edge_feature_names: Dict[Tuple[str, str], List[List[str]]] = {}
        self._edges_by_src: Dict[str, List[str]] = {comp: [] for comp in self._components}
        self._edges_by_dst: Dict[str, List[str]] = {comp: [] for comp in self._components}
        self._feature_names: List[str] = []
        self._edge_type_ids: List[int] = []
        self._edge_type_count: int = 0
        self._edge_index_array: np.ndarray = np.zeros((2, 0), dtype=np.int64)
        self._encoding_strategy = str(encoding_strategy).lower()
        self._sparse_spec = sparse_spec
        self._option_decoder: Dict[str, Dict[int, Any]] = {comp: {} for comp in self._components}
        self._component_embeddings: Dict[str, Dict[Any, np.ndarray]] = {comp: {} for comp in self._components}
        self._component_embedding_dim: Dict[str, int] = {}
        self._option_feature_names: Dict[str, Dict[Any, List[str]]] = {comp: {} for comp in self._components}
        self._global_option_decoder: Dict[int, Tuple[str, Any]] = sparse_spec.option_decoder if sparse_spec is not None else {}
        if self._encoding_strategy == 'sparse':
            if self._sparse_spec is None:
                raise ValueError("encoding_strategy 'sparse' requires sparse_spec")
            self._build_sparse_feature_layout()
        else:
            self._encoding_strategy = 'onehot'
            self._build_feature_layout()

    @staticmethod
    def _normalize_space(space: MutableMapping[str, Sequence[Any]]) -> Dict[str, Tuple[Any, ...]]:
        return {component: tuple(values) for component, values in space.items()}

    def _build_feature_layout(self) -> None:
        index_cursor = 0
        for comp in self._components:
            choices = self._component_choices[comp]
            names: List[str] = []
            for choice in choices:
                feature_name = f'node::{comp}::{choice}'
                names.append(feature_name)
                self._feature_names.append(feature_name)
            start = index_cursor
            end = index_cursor + len(choices)
            self._node_slices[comp] = (start, end)
            self._node_feature_names[comp] = names
            index_cursor = end
        edge_sources: List[int] = []
        edge_targets: List[int] = []
        edge_type_ids: List[int] = []
        for src, dst in self._dag_edges:
            if src not in self._component_choices or dst not in self._component_choices:
                continue
            self._edges_by_src[src].append(dst)
            self._edges_by_dst[dst].append(src)
            src_choices = self._component_choices[src]
            dst_choices = self._component_choices[dst]
            src_idx = self._component_index[src]
            dst_idx = self._component_index[dst]
            type_id = self._edge_type_count
            self._edge_type_count += 1
            edge_sources.extend([src_idx, dst_idx])
            edge_targets.extend([dst_idx, src_idx])
            edge_type_ids.extend([type_id, type_id])
            matrix_names: List[List[str]] = []
            for s_choice in src_choices:
                row: List[str] = []
                for d_choice in dst_choices:
                    name = f'edge::{src}::{s_choice}->{dst}::{d_choice}'
                    row.append(name)
                    self._feature_names.append(name)
                matrix_names.append(row)
            start = index_cursor
            end = index_cursor + len(src_choices) * len(dst_choices)
            self._edge_slices[src, dst] = (start, end)
            self._edge_feature_names[src, dst] = matrix_names
            index_cursor = end
        self._feature_dim = index_cursor
        if edge_sources:
            self._edge_index_array = np.vstack([np.array(edge_sources, dtype=np.int64), np.array(edge_targets, dtype=np.int64)])
            self._edge_type_ids = edge_type_ids
        else:
            self._edge_index_array = np.zeros((2, 0), dtype=np.int64)
            self._edge_type_ids = []

    def _build_sparse_feature_layout(self) -> None:
        index_cursor = 0
        spec = self._sparse_spec
        assert spec is not None
        for comp in self._components:
            if comp not in spec.component_embeddings:
                raise ValueError(f"Sparse spec missing component '{comp}'")
            comp_map = spec.component_embeddings[comp]
            feature_names: List[str] = []
            embedding_dim: Optional[int] = None
            for choice in self._component_choices[comp]:
                if choice not in comp_map:
                    raise ValueError(f"Sparse spec missing choice '{choice}' for component '{comp}'")
                code, encoding = comp_map[choice]
                encoding_array = np.array(encoding, dtype=float)
                if embedding_dim is None:
                    embedding_dim = len(encoding_array)
                    feature_names = [f'node::{comp}::dim{idx}' for idx in range(embedding_dim)]
                    self._feature_names.extend(feature_names)
                elif len(encoding_array) != embedding_dim:
                    raise ValueError(f"Inconsistent embedding length for component '{comp}': expected {embedding_dim}, got {len(encoding_array)}")
                self._component_embeddings[comp][choice] = encoding_array
                self._option_decoder[comp][code] = choice
                self._option_feature_names[comp][choice] = list(feature_names)
            if embedding_dim is None:
                embedding_dim = 0
            self._component_embedding_dim[comp] = embedding_dim
            self._node_feature_names[comp] = list(feature_names)
            self._node_slices[comp] = (index_cursor, index_cursor + embedding_dim)
            index_cursor += embedding_dim
        self._feature_dim = index_cursor
        self._edge_slices.clear()
        self._edge_feature_names.clear()
        self._edge_index_array = np.zeros((2, 0), dtype=np.int64)
        self._edge_type_ids = []
        self._edge_type_count = 0

    @property
    def feature_names(self) -> List[str]:
        return self._feature_names

    def current_space(self) -> Dict[str, Tuple[Any, ...]]:
        return self._pruned_space or self._original_space

    def original_space(self) -> Dict[str, Tuple[Any, ...]]:
        return self._original_space

    def is_pruned(self) -> bool:
        return self._pruned_space is not None

    def reset_pruned_space(self) -> None:
        self._pruned_space = None

    def set_pruned_space(self, pruned: Dict[str, Sequence[Any]]) -> None:
        self._pruned_space = self._normalize_space(pruned)

    def space_size(self, space: Optional[Dict[str, Tuple[Any, ...]]]=None) -> int:
        target = space or self.current_space()
        size = 1
        for choices in target.values():
            size *= len(choices)
        return size

    def components(self) -> List[str]:
        return list(self._components)

    def component_choices(self, component: str) -> List[Any]:
        current = self.current_space()
        if component in current:
            return list(current[component])
        return list(self._component_choices[component])

    def component_cardinality(self, component: str) -> int:
        return self._component_cardinality[component]

    def choice_index(self, component: str, choice: Any) -> int:
        return self._component_choices[component].index(choice)

    def node_feature_names(self, component: str) -> List[str]:
        return self._node_feature_names[component]

    def node_feature_name(self, component: str, choice: Any) -> str:
        if self._encoding_strategy == 'sparse':
            names = self._option_feature_names[component].get(choice)
            if not names:
                raise KeyError(f"Missing feature names for component '{component}' choice '{choice}'")
            return names[0]
        idx = self.choice_index(component, choice)
        return self._node_feature_names[component][idx]

    def node_option_feature_names(self, component: str, choice: Any) -> List[str]:
        if self._encoding_strategy == 'sparse':
            names = self._option_feature_names[component].get(choice)
            if names is None:
                raise KeyError(f"Missing feature names for component '{component}' choice '{choice}'")
            return list(names)
        return [self.node_feature_name(component, choice)]

    def node_embedding(self, component: str, choice: Any) -> np.ndarray:
        if self._encoding_strategy == 'sparse':
            embedding = self._component_embeddings[component].get(choice)
            if embedding is None:
                raise KeyError(f"Missing embedding for component '{component}' choice '{choice}'")
            return embedding
        vec = np.zeros(len(self._component_choices[component]), dtype=float)
        vec[self.choice_index(component, choice)] = 1.0
        return vec

    def edges_from(self, component: str) -> List[str]:
        return list(self._edges_by_src.get(component, []))

    def edges_to(self, component: str) -> List[str]:
        return list(self._edges_by_dst.get(component, []))

    def edge_row_feature_names(self, src: str, dst: str, src_choice: Any) -> List[str]:
        matrix = self._edge_feature_names.get((src, dst), [])
        src_idx = self.choice_index(src, src_choice)
        if src_idx < len(matrix):
            return matrix[src_idx]
        return []

    def edge_column_feature_names(self, src: str, dst: str, dst_choice: Any) -> List[str]:
        matrix = self._edge_feature_names.get((src, dst), [])
        dst_idx = self.choice_index(dst, dst_choice)
        names: List[str] = []
        for row in matrix:
            if dst_idx < len(row):
                names.append(row[dst_idx])
        return names

    def edge_feature_name(self, src: str, dst: str, src_choice: Any, dst_choice: Any) -> Optional[str]:
        matrix = self._edge_feature_names.get((src, dst))
        if matrix is None:
            return None
        src_idx = self.choice_index(src, src_choice)
        dst_idx = self.choice_index(dst, dst_choice)
        if src_idx >= len(matrix):
            return None
        row = matrix[src_idx]
        if dst_idx >= len(row):
            return None
        return row[dst_idx]

    def edge_option_feature_names(self, src: str, dst: str, src_choice: Any) -> List[str]:
        if self._encoding_strategy == 'sparse':
            return []
        matrix = self._edge_feature_names.get((src, dst))
        if matrix is None:
            return []
        src_idx = self.choice_index(src, src_choice)
        if src_idx >= len(matrix):
            return []
        return list(matrix[src_idx])

    def edge_option_feature_names_to(self, src: str, dst: str, dst_choice: Any) -> List[str]:
        if self._encoding_strategy == 'sparse':
            return []
        matrix = self._edge_feature_names.get((src, dst))
        if matrix is None:
            return []
        dst_idx = self.choice_index(dst, dst_choice)
        names: List[str] = []
        for row in matrix:
            if dst_idx < len(row):
                names.append(row[dst_idx])
        return names

    def encoding_strategy(self) -> str:
        return self._encoding_strategy

    def node_option_score(self, component: str, choice: Any, option_scores: Dict[str, float]) -> float:
        if self._encoding_strategy == 'sparse':
            names = self._node_feature_names[component]
            embedding = self.node_embedding(component, choice)
            return float(sum((option_scores.get(name, 0.0) * float(embedding[idx]) for idx, name in enumerate(names))))
        name = self.node_feature_name(component, choice)
        return float(option_scores.get(name, 0.0))

    def edge_option_score_from(self, component: str, choice: Any, option_scores: Dict[str, float]) -> float:
        if self._encoding_strategy == 'sparse':
            return 0.0
        score = 0.0
        for dst in self.edges_from(component):
            for feature_name in self.edge_option_feature_names(component, dst, choice):
                score += option_scores.get(feature_name, 0.0)
        return float(score)

    def edge_option_score_to(self, component: str, choice: Any, option_scores: Dict[str, float]) -> float:
        if self._encoding_strategy == 'sparse':
            return 0.0
        score = 0.0
        for src in self.edges_to(component):
            for feature_name in self.edge_option_feature_names_to(src, component, choice):
                score += option_scores.get(feature_name, 0.0)
        return float(score)

    def edge_index_tensor(self) -> torch.Tensor:
        if torch is None:
            raise RuntimeError('PyTorch is required to build graph tensors')
        return torch.tensor(self._edge_index_array, dtype=torch.long)

    def edge_type_tensor(self) -> torch.Tensor:
        if torch is None:
            raise RuntimeError('PyTorch is required to build graph tensors')
        return torch.tensor(self._edge_type_ids, dtype=torch.long)

    def edge_type_count(self) -> int:
        return self._edge_type_count

    def decode(self, encoding: np.ndarray) -> Dict[str, Any]:
        if encoding.ndim != 1:
            raise ValueError('encoding must be a 1-D vector')
        spec: Dict[str, Any] = {}
        for comp in self._components:
            start, end = self._node_slices[comp]
            slice_vec = encoding[start:end]
            if self._encoding_strategy == 'sparse':
                best_choice = None
                best_distance = float('inf')
                for choice, embedding in self._component_embeddings[comp].items():
                    distance = float(np.linalg.norm(slice_vec - embedding))
                    if distance < best_distance:
                        best_distance = distance
                        best_choice = choice
                if best_choice is None:
                    raise RuntimeError(f"Unable to decode component '{comp}' from sparse encoding")
                spec[comp] = best_choice
            else:
                if slice_vec.sum() <= 0:
                    idx = 0
                else:
                    idx = int(np.argmax(slice_vec))
                idx = min(idx, len(self._component_choices[comp]) - 1)
                spec[comp] = self._component_choices[comp][idx]
        return spec

    def encode(self, arch_spec: Dict[str, Any]) -> np.ndarray:
        vector = np.zeros(self._feature_dim, dtype=float)
        for comp in self._components:
            if comp not in arch_spec:
                raise KeyError(f"Missing component '{comp}' in architecture spec")
            choice = arch_spec[comp]
            if choice not in self._component_choices[comp]:
                raise ValueError(f"Invalid choice '{choice}' for component '{comp}'")
            start, end = self._node_slices[comp]
            if self._encoding_strategy == 'sparse':
                embedding = self._component_embeddings[comp].get(choice)
                if embedding is None:
                    raise KeyError(f"Missing embedding for component '{comp}' choice '{choice}'")
                vector[start:end] = embedding
            else:
                local_index = self.choice_index(comp, choice)
                vector[start + local_index] = 1.0
        if self._encoding_strategy != 'sparse':
            for (src, dst), (start, end) in self._edge_slices.items():
                src_start, src_end = self._node_slices[src]
                dst_start, dst_end = self._node_slices[dst]
                src_vector = vector[src_start:src_end]
                dst_vector = vector[dst_start:dst_end]
                vector[start:end] = np.outer(src_vector, dst_vector).reshape(-1)
        return vector

    def sample_architecture(self, space: Optional[Dict[str, Tuple[Any, ...]]]=None) -> Dict[str, Any]:
        target_space = space or self.current_space()
        arch_spec: Dict[str, Any] = {}
        for component, choices in target_space.items():
            arch_spec[component] = random.choice(choices)
        return arch_spec

    def enumerate_architectures(self, space: Optional[Dict[str, Tuple[Any, ...]]]=None) -> Iterable[Dict[str, Any]]:
        from itertools import product
        target_space = space or self.current_space()
        components = list(target_space.keys())
        for combination in product(*[target_space[comp] for comp in components]):
            yield dict(zip(components, combination))

    def sample_unique_architectures(self, sample_size: int, space: Optional[Dict[str, Tuple[Any, ...]]]=None, max_attempts: Optional[int]=None) -> List[Dict[str, Any]]:
        if sample_size <= 0:
            return []
        target_space = space or self.current_space()
        total_size = self.space_size(target_space)
        if total_size <= sample_size:
            return list(self.enumerate_architectures(target_space))
        seen: Set[Tuple[Tuple[str, Any], ...]] = set()
        samples: List[Dict[str, Any]] = []
        attempts_limit = max_attempts or max(sample_size * 10, sample_size + 5)
        attempts = 0
        while len(samples) < sample_size and attempts < attempts_limit:
            attempts += 1
            spec = self.sample_architecture(target_space)
            key = tuple(sorted(spec.items()))
            if key in seen:
                continue
            seen.add(key)
            samples.append(spec)
        if len(samples) < sample_size:
            for spec in self.enumerate_architectures(target_space):
                key = tuple(sorted(spec.items()))
                if key in seen:
                    continue
                samples.append(spec)
                seen.add(key)
                if len(samples) >= sample_size:
                    break
        return samples

    def build_graph_data(self, arch_spec: Dict[str, Any], reward: Optional[float]=None, meta_features: Optional[Sequence[float]]=None) -> 'Data':
        if torch is None or Data is None:
            raise RuntimeError('PyTorch Geometric is required for the GNN predictor')
        choice_indices: List[int] = []
        component_ids: List[int] = []
        for comp in self._components:
            if comp not in arch_spec:
                raise KeyError(f"Missing component '{comp}' in architecture spec")
            choice = arch_spec[comp]
            idx = self.choice_index(comp, choice)
            choice_indices.append(idx)
            component_ids.append(self._component_index[comp])
        edge_index = self.edge_index_tensor().clone()
        edge_type = self.edge_type_tensor().clone()
        data = Data()
        data.choice_index = torch.tensor(choice_indices, dtype=torch.long)
        data.component_id = torch.tensor(component_ids, dtype=torch.long)
        data.edge_index = edge_index
        data.edge_type = edge_type
        data.num_nodes = len(choice_indices)
        if reward is not None:
            data.y = torch.tensor([reward], dtype=torch.float32)
        if meta_features is not None:
            if isinstance(meta_features, torch.Tensor):
                data.meta_features = meta_features.float()
            else:
                data.meta_features = torch.tensor(meta_features, dtype=torch.float32)
        return data

class SimpleGNNArchitecturePredictor(nn.Module):

    def __init__(self, component_cardinalities: Sequence[int], hidden_dim: int=64, num_layers: int=2, dropout: float=0.1, meta_dim: int=0) -> None:
        if nn is None or GCNConv is None:
            raise RuntimeError('PyTorch Geometric is required for the GNN predictor')
        super().__init__()
        self.component_cardinalities = list(component_cardinalities)
        self.embed_dim = hidden_dim
        self.dropout = dropout
        self.meta_dim = meta_dim
        self.embeddings = nn.ModuleList([nn.Embedding(cardinality, hidden_dim) for cardinality in self.component_cardinalities])
        convs: List[GCNConv] = []
        if num_layers <= 0:
            num_layers = 1
        convs.append(GCNConv(hidden_dim, hidden_dim))
        for _ in range(1, num_layers):
            convs.append(GCNConv(hidden_dim, hidden_dim))
        self.convs = nn.ModuleList(convs)
        output_input_dim = hidden_dim * 2 + meta_dim
        self.output = nn.Sequential(nn.Linear(output_input_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim, 1))

    def _embed_nodes(self, choice_index: torch.Tensor, component_id: torch.Tensor) -> torch.Tensor:
        device = choice_index.device
        x = torch.zeros(choice_index.size(0), self.embed_dim, device=device)
        for comp_id, embedding in enumerate(self.embeddings):
            mask = component_id == comp_id
            if mask.any():
                x[mask] = embedding(choice_index[mask])
        return x

    def forward(self, data: 'Data', return_details: bool=False):
        edge_index = data.edge_index
        batch = getattr(data, 'batch', None)
        if batch is None:
            batch = torch.zeros(data.num_nodes, dtype=torch.long, device=edge_index.device)
        choice_index = data.choice_index
        component_id = data.component_id
        x = self._embed_nodes(choice_index, component_id)
        for idx, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            x = F.relu(x)
            if idx < len(self.convs) - 1:
                x = F.dropout(x, p=self.dropout, training=self.training)
        mean_pool = global_mean_pool(x, batch)
        max_pool = global_max_pool(x, batch)
        graph_rep = torch.cat([mean_pool, max_pool], dim=1)
        if self.meta_dim > 0:
            meta = getattr(data, 'meta_features', None)
            if meta is None:
                meta = torch.zeros((graph_rep.size(0), self.meta_dim), device=graph_rep.device)
            else:
                meta = meta.to(graph_rep.device)
                if meta.dim() == 1:
                    meta = meta.unsqueeze(0)
            graph_rep = torch.cat([graph_rep, meta], dim=1)
        out = self.output(graph_rep).squeeze(-1)
        if return_details:
            details = {'node_embeddings': x.detach(), 'attentions': [], 'batch': batch.detach(), 'edge_index': edge_index.detach(), 'edge_type': getattr(data, 'edge_type', None)}
            return (out, details)
        return out

class GraphNAPPredictor(nn.Module):

    def __init__(self, component_cardinalities: Sequence[int], edge_type_count: int, hidden_dim: int=64, num_layers: int=3, heads: int=4, dropout: float=0.1, ff_hidden_dim: int=128, meta_dim: int=0) -> None:
        if nn is None or GATConv is None:
            raise RuntimeError('PyTorch Geometric is required for the GNN predictor')
        super().__init__()
        self.component_cardinalities = list(component_cardinalities)
        self.embed_dim = hidden_dim
        self.dropout = dropout
        self.heads = max(1, heads)
        self.meta_dim = meta_dim
        self.embeddings = nn.ModuleList([nn.Embedding(cardinality, hidden_dim) for cardinality in self.component_cardinalities])
        self.edge_embeddings = nn.Embedding(max(1, edge_type_count), hidden_dim)
        convs: List[GATConv] = []
        feed_forwards: List[nn.Sequential] = []
        if num_layers <= 0:
            num_layers = 1
        for _ in range(num_layers):
            convs.append(GATConv(in_channels=hidden_dim, out_channels=hidden_dim, heads=self.heads, concat=False, edge_dim=hidden_dim, dropout=dropout, add_self_loops=False))
            feed_forwards.append(nn.Sequential(nn.Linear(hidden_dim, ff_hidden_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(ff_hidden_dim, hidden_dim)))
        self.convs = nn.ModuleList(convs)
        self.feed_forwards = nn.ModuleList(feed_forwards)
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])
        self.output = nn.Sequential(nn.Linear(hidden_dim * 2 + meta_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim, 1))

    def _embed_nodes(self, choice_index: torch.Tensor, component_id: torch.Tensor) -> torch.Tensor:
        device = choice_index.device
        x = torch.zeros(choice_index.size(0), self.embed_dim, device=device)
        for comp_id, embedding in enumerate(self.embeddings):
            mask = component_id == comp_id
            if mask.any():
                x[mask] = embedding(choice_index[mask])
        return x

    def forward(self, data: 'Data', return_details: bool=False):
        edge_index = data.edge_index
        edge_type = getattr(data, 'edge_type', None)
        batch = getattr(data, 'batch', None)
        if batch is None:
            batch = torch.zeros(data.num_nodes, dtype=torch.long, device=edge_index.device)
        choice_index = data.choice_index
        component_id = data.component_id
        x = self._embed_nodes(choice_index, component_id)
        if edge_index.numel() == 0:
            mean_pool = global_mean_pool(x, batch)
            max_pool = global_max_pool(x, batch)
            graph_rep = torch.cat([mean_pool, max_pool], dim=1)
            if self.meta_dim > 0:
                meta = getattr(data, 'meta_features', None)
                if meta is None:
                    meta = torch.zeros((graph_rep.size(0), self.meta_dim), device=graph_rep.device)
                else:
                    meta = meta.to(graph_rep.device)
                    if meta.dim() == 1:
                        meta = meta.unsqueeze(0)
                graph_rep = torch.cat([graph_rep, meta], dim=1)
            out = self.output(graph_rep).squeeze(-1)
            if return_details:
                details = {'node_embeddings': x.detach(), 'attentions': [], 'batch': batch.detach(), 'edge_index': edge_index.detach(), 'edge_type': edge_type.detach() if isinstance(edge_type, torch.Tensor) else edge_type}
                return (out, details)
            return out
        if edge_type is None:
            edge_attr = None
        else:
            edge_attr = self.edge_embeddings(edge_type)
        attention_details: List[Tuple[torch.Tensor, torch.Tensor]] = []
        for conv, ff, norm in zip(self.convs, self.feed_forwards, self.norms):
            residual = x
            if return_details:
                x, (edge_index, att_weight) = conv(x, edge_index, edge_attr=edge_attr, return_attention_weights=True)
                attention_details.append((edge_index.detach(), att_weight.detach()))
            else:
                x = conv(x, edge_index, edge_attr=edge_attr)
            x = F.elu(x)
            x = norm(residual + x)
            ff_out = ff(x)
            x = x + ff_out
            x = F.dropout(x, p=self.dropout, training=self.training)
        mean_pool = global_mean_pool(x, batch)
        max_pool = global_max_pool(x, batch)
        graph_rep = torch.cat([mean_pool, max_pool], dim=1)
        if self.meta_dim > 0:
            meta = getattr(data, 'meta_features', None)
            if meta is None:
                meta = torch.zeros((graph_rep.size(0), self.meta_dim), device=graph_rep.device)
            else:
                meta = meta.to(graph_rep.device)
                if meta.dim() == 1:
                    meta = meta.unsqueeze(0)
            graph_rep = torch.cat([graph_rep, meta], dim=1)
        out = self.output(graph_rep).squeeze(-1)
        if return_details:
            details = {'node_embeddings': x.detach(), 'attentions': [(ei.cpu(), aw.cpu()) for ei, aw in attention_details], 'batch': batch.detach(), 'edge_index': edge_index.detach(), 'edge_type': edge_type.detach() if isinstance(edge_type, torch.Tensor) else edge_type}
            return (out, details)
        return out
GNNArchitecturePredictor = Union[SimpleGNNArchitecturePredictor, GraphNAPPredictor]

class GNNPredictorWrapper:

    def __init__(self, model: nn.Module, space_manager: SearchSpaceManager, device: torch.device, meta_keys: Sequence[str], default_meta: Sequence[float], meta_builder: Callable[[Dict[str, Any], Optional[Dict[str, Any]], Optional[float]], List[float]]) -> None:
        self.model = model
        self.space_manager = space_manager
        self.device = device
        self.meta_keys = list(meta_keys)
        self.default_meta_vector = torch.tensor(default_meta, dtype=torch.float32)
        self.meta_builder = meta_builder

    def predict(self, X: np.ndarray) -> np.ndarray:
        if X.ndim == 1:
            X = X.reshape(1, -1)
        preds: List[float] = []
        self.model.eval()
        with torch.no_grad():
            for row in X:
                arch_spec = self.space_manager.decode(row)
                meta_vec = self.meta_builder(arch_spec, None, None) if self.meta_builder else self.default_meta_vector.tolist()
                if not isinstance(meta_vec, torch.Tensor):
                    meta_vec = torch.tensor(meta_vec, dtype=torch.float32)
                data = self.space_manager.build_graph_data(arch_spec, meta_features=meta_vec)
                data = data.to(self.device)
                pred = self.model(data)
                preds.append(float(pred.cpu().item()))
        return np.array(preds, dtype=float)

class ShapleySearcher:

    def __init__(self, config: Optional[Dict[str, Any]]=None, search_space: Optional[MutableMapping[str, Sequence[Any]]]=None, predictor_factory: Optional[Any]=None) -> None:
        if shap is None:
            raise ImportError("ShapleySearcher requires the 'shap' package" + (f': {_shap_import_error}' if _shap_import_error else ''))
        self.config: Dict[str, Any] = {**DEFAULT_CONFIG, **(config or {})}
        predictor_type = getattr(self, '_predictor_type', str(self.config.get('predictor_type', 'tree')).lower())
        self._predictor_type = predictor_type
        if predictor_type in {'tree', 'mlp'} and RandomForestRegressor is None and (MLPRegressor is None):
            raise ImportError("predictor_type 'tree' or 'mlp' requires scikit-learn" + (f': {_sklearn_import_error}' if _sklearn_import_error else ''))
        if predictor_type == 'gnn' and (torch is None or Data is None or GCNConv is None):
            missing = []
            if torch is None:
                missing.append('PyTorch')
            if Data is None or GCNConv is None:
                missing.append('torch_geometric')
            raise ImportError("predictor_type 'gnn' requires " + ', '.join(missing))
        if predictor_type == 'gnn':
            self.config['use_tree_explainer'] = False
        self._predictor_factory = predictor_factory or self._default_predictor_factory
        encoding_mode = str(self.config.get('encoding_strategy', 'onehot')).lower()
        base_space_mapping: MutableMapping[str, Sequence[Any]]
        if search_space is None:
            base_space_mapping = dict(DEFAULT_SEARCH_SPACE)
        else:
            base_space_mapping = dict(search_space)
        sparse_spec: Optional[SparseSpaceSpec] = None
        encoding_strategy = 'onehot'
        if encoding_mode.startswith('sparse'):
            use_reference = encoding_mode in {'sparse_reference', 'sparse-ref', 'reference_sparse'}
            sparse_spec = build_sparse_search_space(base_space_mapping, use_reference=use_reference)
            encoding_strategy = 'sparse'
        self.space_manager = SearchSpaceManager(base_space_mapping, encoding_strategy=encoding_strategy, sparse_spec=sparse_spec)
        self._arch_counter = 0
        self._records: List[ArchitectureRecord] = []
        self._predictor: Optional[Any] = None
        self._best_record: Optional[ArchitectureRecord] = None
        self._ranked_candidates: List[Dict[str, Any]] = []
        self._pruning_history: List[Dict[str, Any]] = []
        self._multi_source_keys: List[str] = list(self.config.get('multi_source_keys', ['reward']))
        self._meta_default_vector: List[float] = [0.0] * len(self._multi_source_keys)
        self._known_spec_keys: Set[Tuple[Any, ...]] = set()
        self._seed_rng()

    def _seed_rng(self) -> None:
        seed = self.config.get('random_seed')
        if seed is None:
            return
        random.seed(seed)
        np.random.seed(seed)
        if torch is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

    def _default_predictor_factory(self) -> Any:
        predictor_type = getattr(self, '_predictor_type', str(self.config.get('predictor_type', 'tree')).lower())
        if predictor_type == 'tree':
            params = dict(self.config.get('tree_predictor_params', {}))
            if params.get('random_state') is None and self.config.get('random_seed') is not None:
                params['random_state'] = self.config['random_seed']
            if RandomForestRegressor is None:
                raise ImportError("RandomForestRegressor is unavailable; install scikit-learn or set predictor_type to 'mlp' or 'gnn'")
            return RandomForestRegressor(**params)
        if predictor_type == 'mlp':
            params = dict(self.config.get('predictor_params', {}))
            if params.get('random_state') is None and self.config.get('random_seed') is not None:
                params['random_state'] = self.config['random_seed']
            if MLPRegressor is None:
                raise ImportError("MLPRegressor is unavailable; install scikit-learn or set predictor_type to 'tree' or 'gnn'")
            return MLPRegressor(**params)
        if predictor_type == 'gnn':
            return None
        raise ValueError(f'Unsupported predictor_type: {predictor_type}')

    def sample_architectures(self, num_samples: int, stage: str='auto') -> List[Dict[str, Any]]:
        if num_samples <= 0:
            return []
        if stage not in {'auto', 'initial', 'refined'}:
            raise ValueError(f'Unknown stage: {stage}')
        use_pruned = stage == 'refined' or (stage == 'auto' and self.space_manager.is_pruned())
        space = self.space_manager.current_space() if use_pruned else self.space_manager.original_space()
        samples: List[Dict[str, Any]] = []
        for _ in range(num_samples):
            arch_spec = self.space_manager.sample_architecture(space)
            self._arch_counter += 1
            arch_id = f'shap_arch_{self._arch_counter:05d}'
            samples.append({'arch_id': arch_id, 'arch_spec': arch_spec})
        return samples

    def register_evaluation(self, arch_id: str, arch_spec: Dict[str, Any], reward: float, metadata: Optional[Dict[str, Any]]=None) -> None:
        encoding = self.space_manager.encode(arch_spec)
        meta = dict(metadata or {})
        meta.setdefault('reward', float(reward))
        record = ArchitectureRecord(arch_id=arch_id, arch_spec=arch_spec, encoding=encoding, reward=float(reward), metadata=meta)
        self._records.append(record)
        spec_key = self._spec_to_key(arch_spec)
        self._known_spec_keys.add(spec_key)
        if self._best_record is None or record.reward > self._best_record.reward:
            self._best_record = record
        self._update_meta_defaults()

    def train_predictor(self) -> Any:
        if len(self._records) < int(self.config.get('min_training_samples', 1)):
            raise RuntimeError('Not enough samples to train the performance predictor')
        predictor = self._fit_predictor()
        self._predictor = predictor
        self._maybe_self_distill()
        return self._predictor

    def _require_predictor(self) -> Any:
        if self._predictor is None:
            raise RuntimeError('Call train_predictor() before pruning or ranking')
        return self._predictor

    def _fit_predictor(self) -> Any:
        predictor_type = getattr(self, '_predictor_type', str(self.config.get('predictor_type', 'tree')).lower())
        if predictor_type == 'gnn':
            return self._train_gnn_predictor()
        predictor = self._predictor_factory()
        if predictor is None:
            raise RuntimeError('Predictor factory returned None; check configuration')
        X = np.stack([r.encoding for r in self._records])
        y = np.array([r.reward for r in self._records], dtype=float)
        predictor.fit(X, y)
        return predictor

    def _maybe_self_distill(self) -> None:
        steps = int(self.config.get('self_distillation_steps', 0))
        sample_size = int(self.config.get('self_distillation_sample_size', 0))
        if steps <= 0 or sample_size <= 0 or self._predictor is None:
            return
        min_samples = int(self.config.get('min_training_samples', 1))
        for step_idx in range(steps):
            new_records = self._self_distill_step(step_idx, sample_size)
            if not new_records:
                break
            if len(self._records) < min_samples:
                break
            self._predictor = self._fit_predictor()

    def _self_distill_step(self, step_idx: int, sample_size: int) -> List[str]:
        predictor = self._predictor
        if predictor is None:
            return []
        samples = self.sample_architectures(sample_size, stage='auto')
        new_ids: List[str] = []
        for idx, sample in enumerate(samples):
            arch_spec = sample['arch_spec']
            spec_key = self._spec_to_key(arch_spec)
            if spec_key in self._known_spec_keys:
                continue
            encoding = self.space_manager.encode(arch_spec).reshape(1, -1)
            try:
                predicted = float(predictor.predict(encoding)[0])
            except Exception:
                continue
            metadata = {'pseudo': True, 'distill_step': step_idx, 'teacher_prediction': predicted}
            arch_id = f"{sample['arch_id']}_distill{step_idx}_{idx}"
            self.register_evaluation(arch_id, arch_spec, predicted, metadata)
            new_ids.append(arch_id)
        return new_ids

    def _train_gnn_predictor(self) -> GNNPredictorWrapper:
        if torch is None or Data is None or DataLoader is None:
            raise RuntimeError("PyTorch Geometric is required for predictor_type='gnn'")
        device_cfg = str(self.config.get('gnn_device', 'auto')).lower()
        if device_cfg == 'auto':
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            device = torch.device(device_cfg)
        hidden_dim = int(self.config.get('gnn_hidden_dim', 64))
        num_layers = int(self.config.get('gnn_layers', 2))
        dropout = float(self.config.get('gnn_dropout', 0.1))
        lr = float(self.config.get('gnn_learning_rate', 0.001))
        weight_decay = float(self.config.get('gnn_weight_decay', 0.0))
        epochs = int(self.config.get('gnn_epochs', 200))
        batch_size = int(self.config.get('gnn_batch_size', 16))
        heads = int(self.config.get('gnn_attention_heads', 4))
        ff_hidden_dim = int(self.config.get('gnn_ff_hidden_dim', 128))
        gnn_arch = str(self.config.get('gnn_architecture', 'graphnap')).lower()
        component_cardinalities = [self.space_manager.component_cardinality(comp) for comp in self.space_manager.components()]
        edge_type_count = max(1, self.space_manager.edge_type_count())
        if gnn_arch == 'graphnap':
            model = GraphNAPPredictor(component_cardinalities=component_cardinalities, edge_type_count=edge_type_count, hidden_dim=hidden_dim, num_layers=num_layers, heads=heads, dropout=dropout, ff_hidden_dim=ff_hidden_dim, meta_dim=len(self._multi_source_keys)).to(device)
        else:
            model = SimpleGNNArchitecturePredictor(component_cardinalities=component_cardinalities, hidden_dim=hidden_dim, num_layers=num_layers, dropout=dropout, meta_dim=len(self._multi_source_keys)).to(device)
        dataset = []
        for record in self._records:
            meta_vec = self._meta_vector_from_record(record)
            data = self.space_manager.build_graph_data(record.arch_spec, record.reward, meta_vec)
            dataset.append(data)
        if not dataset:
            raise RuntimeError('No records available to train GNN predictor')

        def custom_collate_fn(batch):
            if len(batch) == 1:
                return batch[0]
            data = batch[0]
            node_counts = [data.num_nodes for data in batch]
            total_nodes = sum(node_counts)
            new_choice_index = torch.zeros(total_nodes, dtype=torch.long)
            new_component_id = torch.zeros(total_nodes, dtype=torch.long)
            start_idx = 0
            for i, data in enumerate(batch):
                end_idx = start_idx + data.num_nodes
                new_choice_index[start_idx:end_idx] = data.choice_index
                new_component_id[start_idx:end_idx] = data.component_id
                start_idx = end_idx
            new_data = Data()
            new_data.choice_index = new_choice_index
            new_data.component_id = new_component_id
            new_data.num_nodes = total_nodes
            if hasattr(data, 'edge_index') and data.edge_index.numel() > 0:
                edge_indices = []
                start_idx = 0
                for i, data in enumerate(batch):
                    if data.edge_index.numel() > 0:
                        adjusted_edge_index = data.edge_index + start_idx
                        edge_indices.append(adjusted_edge_index)
                    start_idx += data.num_nodes
                if edge_indices:
                    new_data.edge_index = torch.cat(edge_indices, dim=1)
                else:
                    new_data.edge_index = torch.zeros((2, 0), dtype=torch.long)
            else:
                new_data.edge_index = torch.zeros((2, 0), dtype=torch.long)
            if hasattr(data, 'edge_type') and data.edge_type.numel() > 0:
                edge_types = []
                for data in batch:
                    if data.edge_type.numel() > 0:
                        edge_types.append(data.edge_type)
                if edge_types:
                    new_data.edge_type = torch.cat(edge_types, dim=0)
                else:
                    new_data.edge_type = torch.zeros(0, dtype=torch.long)
            else:
                new_data.edge_type = torch.zeros(0, dtype=torch.long)
            if hasattr(data, 'y') and data.y is not None:
                y_values = []
                for data in batch:
                    if data.y is not None:
                        y_values.append(data.y)
                if y_values:
                    new_data.y = torch.cat(y_values, dim=0)
                else:
                    new_data.y = torch.zeros(0, dtype=torch.float32)
            else:
                new_data.y = torch.zeros(0, dtype=torch.float32)
            if hasattr(data, 'meta_features') and data.meta_features is not None:
                meta_features = []
                for data in batch:
                    if data.meta_features is not None:
                        meta_features.append(data.meta_features)
                if meta_features:
                    new_data.meta_features = torch.cat(meta_features, dim=0)
                else:
                    new_data.meta_features = torch.zeros(0, dtype=torch.float32)
            else:
                new_data.meta_features = torch.zeros(0, dtype=torch.float32)
            new_data.batch = torch.zeros(total_nodes, dtype=torch.long)
            start_idx = 0
            for i, data in enumerate(batch):
                end_idx = start_idx + data.num_nodes
                new_data.batch[start_idx:end_idx] = i
                start_idx = end_idx
            return new_data
        loader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=custom_collate_fn)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        model.train()
        for _ in range(max(1, epochs)):
            for batch in loader:
                batch = batch.to(device)
                optimizer.zero_grad()
                preds = model(batch)
                loss = F.mse_loss(preds, batch.y.view(-1))
                loss.backward()
                optimizer.step()
        model.eval()
        return GNNPredictorWrapper(model, self.space_manager, device, self._multi_source_keys, self._meta_default_vector, self._meta_vector_from_spec)

    def _stage_threshold(self, stage_index: int) -> float:
        schedule = self.config.get('pruning_threshold_schedule')
        if isinstance(schedule, (list, tuple)) and schedule:
            idx = min(stage_index, len(schedule) - 1)
            return float(schedule[idx])
        if isinstance(schedule, dict) and schedule:
            value = schedule.get(stage_index)
            if value is not None:
                return float(value)
        if isinstance(schedule, str):
            schedule_lower = schedule.lower()
            base = float(self.config.get('pruning_threshold', 0.5))
            iterations = max(1, int(self.config.get('pruning_iterations', 1)))
            if schedule_lower == 'linear' and iterations > 1:
                factor = max(0.0, 1.0 - stage_index / (iterations - 1))
                return float(base * factor)
            if schedule_lower == 'geometric':
                return float(base * 0.5 ** stage_index)
        return float(self.config.get('pruning_threshold', 0.5))

    def _filter_records_to_current_space(self) -> None:
        current_space = self.space_manager.current_space()
        allowed = {component: set(choices) for component, choices in current_space.items()}
        filtered: List[ArchitectureRecord] = []
        for record in self._records:
            if all((record.arch_spec.get(component) in allowed[component] for component in allowed)):
                filtered.append(record)
        if len(filtered) != len(self._records):
            self._records = filtered
            self._reset_best_record()

    def _reset_best_record(self) -> None:
        if self._records:
            self._best_record = max(self._records, key=lambda r: r.reward)
        else:
            self._best_record = None
        self._known_spec_keys = {self._spec_to_key(rec.arch_spec) for rec in self._records}
        self._update_meta_defaults()

    def _spec_to_key(self, arch_spec: Dict[str, Any]) -> Tuple[Tuple[str, Any], ...]:
        return tuple(sorted(arch_spec.items()))

    def _meta_vector_from_spec(self, arch_spec: Dict[str, Any], metadata: Optional[Dict[str, Any]]=None, reward: Optional[float]=None) -> List[float]:
        vector: List[float] = []
        metadata = metadata or {}
        for key in self._multi_source_keys:
            if key == 'reward':
                value = reward if reward is not None else metadata.get('reward', 0.0)
            else:
                value = metadata.get(key, 0.0)
            vector.append(float(value))
        if not vector:
            vector = [0.0]
        return vector

    def _meta_vector_from_record(self, record: ArchitectureRecord) -> List[float]:
        return self._meta_vector_from_spec(record.arch_spec, record.metadata, record.reward)

    def _update_meta_defaults(self) -> None:
        if not self._records:
            self._meta_default_vector = [0.0] * len(self._multi_source_keys)
            return
        vectors = np.array([self._meta_vector_from_record(rec) for rec in self._records], dtype=float)
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)
        self._meta_default_vector = vectors.mean(axis=0).tolist()

    def _apply_pruning_stage(self, stage_index: int, option_scores: Dict[str, float], threshold: float) -> Dict[str, Any]:
        before_space = self.space_manager.current_space()
        before_sets = {comp: set(choices) for comp, choices in before_space.items()}
        if self.config.get('use_dgnap_pruning', True):
            self._prune_negative_contributions(option_scores)
        else:
            self._prune_with_threshold(option_scores, threshold)
        after_space = self.space_manager.current_space()
        after_sets = {comp: set(choices) for comp, choices in after_space.items()}
        removed: Dict[str, List[Any]] = {}
        for comp, before in before_sets.items():
            diff = before - after_sets.get(comp, set())
            if diff:
                removed[comp] = sorted(diff)
        changed = any(removed.values())
        self._filter_records_to_current_space()

        def safe_sort(items):
            try:
                return sorted(items)
            except TypeError:
                type_groups = {}
                for item in items:
                    item_type = type(item).__name__
                    if item_type not in type_groups:
                        type_groups[item_type] = []
                    type_groups[item_type].append(item)
                sorted_items = []
                for type_name in sorted(type_groups.keys()):
                    sorted_items.extend(sorted(type_groups[type_name]))
                return sorted_items
        stage_info = {'stage': stage_index, 'threshold': threshold, 'removed': removed, 'remaining_space': {comp: safe_sort(list(after_sets.get(comp, set()))) for comp in after_sets}, 'num_records': len(self._records), 'changed': changed}
        return stage_info

    def prune_search_space(self) -> Dict[str, Tuple[Any, ...]]:
        if not self._records:
            raise RuntimeError('No evaluated architectures available for pruning')
        iterations = max(1, int(self.config.get('pruning_iterations', 1)))
        retrain_each_round = bool(self.config.get('pruning_retrain_each_round', True))
        stop_when_no_change = bool(self.config.get('pruning_stop_when_no_change', True))
        min_samples = int(self.config.get('min_training_samples', 1))
        stage_history: List[Dict[str, Any]] = []
        for stage_idx in range(iterations):
            predictor = self._require_predictor()
            option_scores = self._compute_option_importance(predictor)
            threshold = self._stage_threshold(stage_idx)
            stage_info = self._apply_pruning_stage(stage_idx, option_scores, threshold)
            stage_history.append(stage_info)
            if stop_when_no_change and (not stage_info['changed']):
                break
            if stage_idx < iterations - 1:
                if retrain_each_round:
                    if len(self._records) >= min_samples:
                        self._predictor = self._fit_predictor()
                    else:
                        break
        self._pruning_history.extend(stage_history)
        return self.space_manager.current_space()

    def _prune_negative_contributions(self, option_scores: Dict[str, float]) -> Dict[str, Tuple[Any, ...]]:
        pruned_space: Dict[str, List[Any]] = {}
        for component, choices in self.space_manager.current_space().items():
            ranked = self._aggregate_component_scores(component, option_scores)
            if not ranked:
                pruned_space[component] = list(choices)
                continue
            kept_choices = [choice for choice, score in ranked if score > 0]
            if not kept_choices:
                kept_choices = [ranked[0][0]] if ranked else list(choices)
            pruned_space[component] = kept_choices
        self.space_manager.set_pruned_space(pruned_space)
        return self.space_manager.current_space()

    def _prune_with_threshold(self, option_scores: Dict[str, float], threshold: float) -> Dict[str, Tuple[Any, ...]]:
        pruned_space: Dict[str, List[Any]] = {}
        for component, choices in self.space_manager.current_space().items():
            ranked_choices = self._aggregate_component_scores(component, option_scores)
            if not ranked_choices:
                pruned_space[component] = list(choices)
                continue
            kept_choices: List[Any] = []
            if threshold > 1:
                keep_count = max(1, int(math.ceil(threshold)))
                kept_choices = [choice for choice, _ in ranked_choices[:keep_count]]
            else:
                total_importance = sum((max(score, 0.0) for _, score in ranked_choices))
                target = total_importance * threshold
                cumulative = 0.0
                for choice, score in ranked_choices:
                    kept_choices.append(choice)
                    cumulative += max(score, 0.0)
                    if target <= 0 or cumulative >= target:
                        break
                if not kept_choices:
                    kept_choices = [ranked_choices[0][0]]
            pruned_space[component] = kept_choices
        self.space_manager.set_pruned_space(pruned_space)
        return self.space_manager.current_space()

    def _compute_option_importance(self, predictor: Any) -> Dict[str, float]:
        reference_cfg = self.config.get('reference_shapley')
        if reference_cfg:
            return self._compute_reference_csv_importance(reference_cfg)
        X_full = np.stack([r.encoding for r in self._records])
        y_full = np.array([r.reward for r in self._records], dtype=float)
        use_tree = self.config.get('use_tree_explainer', True)
        predictor_type = getattr(self, '_predictor_type', 'tree')
        if use_tree and predictor_type != 'gnn' and hasattr(predictor, 'feature_importances_'):
            return self._compute_tree_importance(predictor, X_full, y_full)
        elif predictor_type == 'gnn':
            return self._compute_gnn_importance(predictor, X_full)
        else:
            return self._compute_kernel_importance(predictor, X_full)

    def _compute_reference_csv_importance(self, cfg: Any) -> Dict[str, float]:
        if not isinstance(cfg, dict):
            raise TypeError('reference_shapley configuration must be a dictionary')
        csv_path = cfg.get('csv_path')
        if not csv_path:
            raise ValueError("reference_shapley configuration requires 'csv_path'")
        adapter_cfg = ReferenceShapleyConfig(csv_path=csv_path, target_column=cfg.get('target_column', 'reward'), shap_type=str(cfg.get('shap_type', 'tree')), nsamples=cfg.get('nsamples'), random_seed=cfg.get('random_seed', self.config.get('random_seed')), kernel_background_size=cfg.get('kernel_background_size'))
        raw_importance = compute_reference_shapley(adapter_cfg)
        feature_mapping = cfg.get('feature_name_map')
        allowed = self.space_manager.feature_names
        option_scores = remap_feature_names(raw_importance, mapping=feature_mapping, allowed_features=allowed)
        if not option_scores:
            raise ValueError("Reference Shapley computation returned no matching features. Provide a 'feature_name_map' that maps CSV columns to search-space features.")
        return option_scores

    def _compute_tree_importance(self, predictor: Any, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        try:
            sample_size_cfg = int(self.config.get('shap_sample_size', len(X)))
            if sample_size_cfg < len(X) and len(X) > 1000:
                from sklearn.model_selection import train_test_split
                try:
                    X_sample, _, _, _ = train_test_split(X, y, train_size=sample_size_cfg, stratify=None, random_state=self.config.get('random_seed', 42))
                except Exception:
                    indices = np.random.RandomState(self.config.get('random_seed', 42)).choice(len(X), size=sample_size_cfg, replace=False)
                    X_sample = X[indices]
            else:
                X_sample = X
            if X_sample.size == 0:
                raise ValueError('样本数据为空')
            if np.any(np.isnan(X_sample)) or np.any(np.isinf(X_sample)):
                from sklearn.impute import SimpleImputer
                imputer = SimpleImputer(strategy='median')
                X_sample = imputer.fit_transform(X_sample)
            explainer = shap.TreeExplainer(predictor)
            shap_values = explainer(X_sample)
            if hasattr(shap_values, 'values'):
                importance_matrix = shap_values.values
            else:
                importance_matrix = shap_values
            importance_raw = np.mean(np.abs(importance_matrix), axis=0)
        except Exception as e:
            print(f'TreeExplainer failed: {e}, using feature_importances_')
            importance_raw = getattr(predictor, 'feature_importances_', np.zeros(X.shape[1]))
        option_names = self.space_manager.feature_names
        option_scores = dict(zip(option_names, importance_raw.tolist()))
        if self.config.get('component_normalization', True):
            for component in self.space_manager.components():
                node_names = self.space_manager.node_feature_names(component)
                total = sum((option_scores.get(name, 0.0) for name in node_names))
                if total > 0:
                    for name in node_names:
                        option_scores[name] = option_scores.get(name, 0.0) / total
        return option_scores

    def _compute_gnn_importance(self, predictor: Any, X_full: np.ndarray) -> Dict[str, float]:
        if torch is None or Data is None:
            raise RuntimeError('PyTorch Geometric is required for GNN importance computation')
        if not isinstance(predictor, GNNPredictorWrapper):
            raise ValueError('GNN predictor wrapper expected')
        option_scores = {name: 0.0 for name in self.space_manager.feature_names}
        components = self.space_manager.components()
        device = predictor.device
        for record in self._records:
            meta_vec = self._meta_vector_from_record(record)
            data = self.space_manager.build_graph_data(record.arch_spec, record.reward, meta_vec)
            data = data.to(device)
            output, details = predictor.model(data, return_details=True)
            node_emb = details.get('node_embeddings')
            batch = details.get('batch')
            if node_emb is None:
                continue
            node_emb = node_emb.detach().cpu()
            component_ids = data.component_id.cpu()
            node_scores = node_emb.abs().sum(dim=1)
            for idx, score in enumerate(node_scores):
                comp = components[int(component_ids[idx])]
                choice = record.arch_spec[comp]
                feature_name = self.space_manager.node_feature_name(comp, choice)
                option_scores[feature_name] = option_scores.get(feature_name, 0.0) + float(score.item())
            for edge_info in details.get('attentions', []):
                edge_index_tensor, att_weight = edge_info
                edge_index_cpu = edge_index_tensor.cpu()
                att = att_weight.cpu()
                if att.dim() > 1:
                    att = att.mean(dim=1)
                for src_idx, dst_idx, weight in zip(edge_index_cpu[0].tolist(), edge_index_cpu[1].tolist(), att.tolist()):
                    src_comp = components[int(component_ids[src_idx])]
                    dst_comp = components[int(component_ids[dst_idx])]
                    if dst_comp not in self.space_manager.edges_from(src_comp):
                        continue
                    src_choice = record.arch_spec[src_comp]
                    dst_choice = record.arch_spec[dst_comp]
                    feature_name = self.space_manager.edge_feature_name(src_comp, dst_comp, src_choice, dst_choice)
                    if feature_name is None:
                        continue
                    option_scores[feature_name] = option_scores.get(feature_name, 0.0) + abs(weight)
        if self.config.get('component_normalization', True):
            for component in self.space_manager.components():
                node_names = self.space_manager.node_feature_names(component)
                total = sum((option_scores.get(name, 0.0) for name in node_names))
                if total > 0:
                    for name in node_names:
                        option_scores[name] = option_scores.get(name, 0.0) / total
        return option_scores

    def _compute_kernel_importance(self, predictor: Any, X_full: np.ndarray) -> Dict[str, float]:
        sample_size_cfg = int(self.config.get('shap_sample_size', len(X_full)))
        sample_size = min(len(X_full), sample_size_cfg) or len(X_full)
        if sample_size < len(X_full):
            indices = np.random.choice(len(X_full), sample_size, replace=False)
            X = X_full[indices]
        else:
            X = X_full
        background_size_cfg = int(self.config.get('shap_background_size', 50))
        background_size = min(len(X_full), max(1, background_size_cfg))
        background_method = str(self.config.get('shap_background_method', 'kmeans')).lower()
        if background_method == 'kmeans' and hasattr(shap, 'kmeans') and (len(X_full) > 0):
            k = min(background_size, len(X_full))
            background = shap.kmeans(X_full, k)
        elif background_size < len(X_full):
            bg_indices = np.random.choice(len(X_full), background_size, replace=False)
            background = X_full[bg_indices]
        else:
            background = X_full
        explainer = shap.KernelExplainer(predictor.predict, background)
        shap_values = explainer.shap_values(X)
        if isinstance(shap_values, list):
            shap_values = shap_values[0]
        importance_raw = np.mean(np.abs(shap_values), axis=0)
        option_names = self.space_manager.feature_names
        option_scores = dict(zip(option_names, importance_raw.tolist()))
        if self.config.get('component_normalization', True):
            for component in self.space_manager.components():
                node_names = self.space_manager.node_feature_names(component)
                total = sum((option_scores.get(name, 0.0) for name in node_names))
                if total > 0:
                    for name in node_names:
                        option_scores[name] = option_scores.get(name, 0.0) / total
        return option_scores

    def _aggregate_component_scores(self, component: str, option_scores: Dict[str, float]) -> List[Tuple[Any, float]]:
        choices = self.space_manager.component_choices(component)
        ranked: List[Tuple[Any, float]] = []
        for choice in choices:
            score = self.space_manager.node_option_score(component, choice, option_scores)
            score += self.space_manager.edge_option_score_from(component, choice, option_scores)
            score += self.space_manager.edge_option_score_to(component, choice, option_scores)
            ranked.append((choice, float(score)))
        ranked.sort(key=lambda item: item[1], reverse=True)
        return ranked

    def rank_candidates(self, top_k: Optional[int]=None, max_candidates: Optional[int]=None) -> List[Dict[str, Any]]:
        predictor = self._require_predictor()
        if self.space_manager.space_size() == 0:
            raise RuntimeError('Current search space is empty')
        space = self.space_manager.current_space()
        if not space:
            raise RuntimeError('Current search space is empty')
        limit = top_k or int(self.config.get('top_k', 5))
        limit = max(1, limit)
        max_total = max_candidates or int(self.config.get('max_candidates', 10000))
        sample_budget = max(limit, max_total)
        total_space_size = self.space_manager.space_size(space)
        if total_space_size <= sample_budget:
            all_candidates = list(self.space_manager.enumerate_architectures(space))
        else:
            all_candidates = self.space_manager.sample_unique_architectures(sample_budget, space=space)
        encodings = np.stack([self.space_manager.encode(spec) for spec in all_candidates])
        predicted = predictor.predict(encodings)
        ranking = list(zip(all_candidates, predicted.tolist()))
        ranking.sort(key=lambda item: item[1], reverse=True)
        limited = ranking[:limit]
        ranked_payload: List[Dict[str, Any]] = []
        for spec, score in limited:
            self._arch_counter += 1
            arch_id = f'shap_candidate_{self._arch_counter:05d}'
            ranked_payload.append({'arch_id': arch_id, 'arch_spec': spec, 'predicted_reward': score})
        self._ranked_candidates = ranked_payload
        return ranked_payload

    def best_architecture(self) -> Optional[Tuple[Dict[str, Any], float]]:
        if self._best_record is None:
            return None
        return (self._best_record.arch_spec, self._best_record.reward)

    def last_ranked_candidates(self) -> List[Dict[str, Any]]:
        return list(self._ranked_candidates)

    def evaluated_architectures(self) -> List[ArchitectureRecord]:
        return list(self._records)

    def pruning_history(self) -> List[Dict[str, Any]]:
        return list(self._pruning_history)

    def reset(self) -> None:
        self._arch_counter = 0
        self._records.clear()
        self._predictor = None
        self._best_record = None
        self._ranked_candidates = []
        self._pruning_history.clear()
        self._known_spec_keys.clear()
        self._meta_default_vector = [0.0] * len(self._multi_source_keys)
        self.space_manager.reset_pruned_space()

class ShapleySearchManager:

    def __init__(self, config: Optional[Dict[str, Any]]=None, search_space: Optional[MutableMapping[str, Sequence[Any]]]=None) -> None:
        self.searcher = ShapleySearcher(config=config, search_space=search_space)

    def configure(self, **kwargs: Any) -> None:
        self.searcher.config.update(kwargs)

    def sample_architectures(self, num_samples: int, stage: str='auto') -> List[Dict[str, Any]]:
        return self.searcher.sample_architectures(num_samples, stage=stage)

    def register_evaluation(self, arch_id: str, arch_spec: Dict[str, Any], reward: float, metadata: Optional[Dict[str, Any]]=None) -> None:
        self.searcher.register_evaluation(arch_id, arch_spec, reward, metadata)

    def train_predictor(self) -> Any:
        return self.searcher.train_predictor()

    def prune_and_rank(self, run_pruning: bool=True, top_k: Optional[int]=None) -> List[Dict[str, Any]]:
        if run_pruning:
            self.searcher.prune_search_space()
        return self.searcher.rank_candidates(top_k=top_k)

    def best_architecture(self) -> Optional[Tuple[Dict[str, Any], float]]:
        return self.searcher.best_architecture()

    def pruning_history(self) -> List[Dict[str, Any]]:
        return self.searcher.pruning_history()

    def reset(self) -> None:
        self.searcher.reset()
__all__ = ['ShapleySearcher', 'ShapleySearchManager', 'SearchSpaceManager', 'ArchitectureRecord', 'SimpleGNNArchitecturePredictor', 'GraphNAPPredictor']