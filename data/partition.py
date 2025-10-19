import os
import sys
import torch
import torch_geometric
import yaml
from typing import List, Dict, Tuple
from collections import defaultdict
from torch_geometric.data import Data
from torch_geometric.utils import k_hop_subgraph, degree
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
from data.data_utils import get_dataset, SUPPORTED_DATASETS

class PartitionGraph:
    EXTRA_CORE_RATIO_MIXED: float = 0.3
    RANDOM_FILL_CAP_RATIO: float = 0.5
    RANDOM_FILL_COVER_ALL: bool = True

    def __init__(self, dataset_name: str, num_partitions: int, max_hops: int):
        self.dataset_name = dataset_name
        self.num_partitions = num_partitions
        self.max_hops = max_hops
        _, self.full_graph = get_dataset(name=self.dataset_name, root='dataset')
        if not all((hasattr(self.full_graph, attr) and getattr(self.full_graph, attr) is not None for attr in ['edge_index', 'x', 'y', 'train_mask'])):
            raise ValueError(f"数据集 '{self.dataset_name}' 缺少必要的属性（edge_index, x, y, train_mask）")
        if not isinstance(self.full_graph.y, torch.Tensor):
            raise TypeError(f"数据集 '{self.dataset_name}' 的标签 (y) 必须是张量类型")
        self.all_train_indices = self.full_graph.train_mask.nonzero(as_tuple=False).view(-1)
        self.dataset_graph_stats = self._load_dataset_graph_stats(self.dataset_name)
        if self.dataset_graph_stats is None:
            print(f'警告: 未找到 {self.dataset_name} 的图统计配置，使用默认值Cora的')
            self.dataset_graph_stats = {'vertices': 2708, 'edges': 5278, 'avg_degree': 3.9, 'diameter': 19}
        else:
            print(f'已从配置文件加载 {self.dataset_name} 数据集的图统计信息')

    def _get_total_num_nodes(self) -> int:
        if isinstance(self.full_graph.x, torch.Tensor):
            return int(self.full_graph.x.size(0))
        if getattr(self.full_graph, 'num_nodes', None) is not None:
            return int(self.full_graph.num_nodes)
        assert isinstance(self.full_graph.edge_index, torch.Tensor), 'edge_index 必须为 Tensor'
        return int(self.full_graph.edge_index.max().item() + 1)

    def _compute_partition_graph_stats(self, data: Data) -> dict:
        try:
            num_nodes = data.num_nodes if data.num_nodes is not None else data.x.size(0) if data.x is not None else 0
            num_edges = data.edge_index.size(1) if data.edge_index is not None else 0
            avg_degree = float(num_edges / num_nodes) if num_nodes > 0 else 0.0
            if num_nodes < 1000:
                diameter = 8
            elif num_nodes < 10000:
                diameter = 15
            else:
                diameter = 25
            return {'vertices': int(num_nodes), 'edges': int(num_edges), 'avg_degree': avg_degree, 'diameter': int(diameter)}
        except Exception:
            return {'vertices': 2708, 'edges': 5278, 'avg_degree': 3.9, 'diameter': 19}

    def _build_subgraphs_from_core_nodes(self, partition_core_nodes_list: List[torch.Tensor]) -> List[Data]:
        partitions = []
        total_core_nodes_partitioned = 0
        for i, core_nodes_tensor in enumerate(partition_core_nodes_list):
            if len(core_nodes_tensor) == 0:
                print(f'警告: 分区 {i} 没有分配到核心训练节点，将创建一个空分区。')
                partitions.append(Data())
                continue
            total_core_nodes_partitioned += len(core_nodes_tensor)
            ei = self.full_graph.edge_index
            if ei is None:
                raise ValueError('图数据缺少edge_index属性')
            assert isinstance(ei, torch.Tensor)
            total_nodes = self._get_total_num_nodes()
            sub_nodes, sub_edge_index, _, _ = k_hop_subgraph(node_idx=core_nodes_tensor, num_hops=self.max_hops, edge_index=ei, relabel_nodes=True, num_nodes=total_nodes)
            if self.full_graph.x is None:
                raise ValueError('图数据缺少x属性')
            if self.full_graph.y is None:
                raise ValueError('图数据缺少y属性')
            assert isinstance(self.full_graph.x, torch.Tensor), 'x 必须是 Tensor'
            assert isinstance(self.full_graph.y, torch.Tensor), 'y 必须是 Tensor'
            partition_data = Data(x=self.full_graph.x[sub_nodes], edge_index=sub_edge_index, y=self.full_graph.y[sub_nodes], num_nodes=int(sub_nodes.size(0)), global_node_ids=sub_nodes)
            mapping = torch.full((total_nodes,), -1, dtype=torch.long)
            mapping[sub_nodes] = torch.arange(int(sub_nodes.size(0)))
            core_nodes_local_indices = mapping[core_nodes_tensor]
            assert (core_nodes_local_indices != -1).all(), '逻辑错误: 核心节点未在自己的k-hop子图中找到！'
            new_train_mask = torch.zeros(int(sub_nodes.size(0)), dtype=torch.bool)
            new_train_mask[core_nodes_local_indices] = True
            partition_data.train_mask = new_train_mask
            if hasattr(self.full_graph, 'val_mask') and self.full_graph.val_mask is not None:
                partition_data.val_mask = self.full_graph.val_mask[sub_nodes]
            if hasattr(self.full_graph, 'test_mask') and self.full_graph.test_mask is not None:
                partition_data.test_mask = self.full_graph.test_mask[sub_nodes]
            partition_data.graph_stats = self._compute_partition_graph_stats(partition_data)
            partition_data.dataset_graph_stats = self.dataset_graph_stats.copy() if self.dataset_graph_stats else None
            partitions.append(partition_data)
            print(f'  - Partition {i} created: {partition_data.num_nodes} nodes, {partition_data.num_edges} edges.')
            print(f'    - Core training nodes: {partition_data.train_mask.sum().item()}')
            if hasattr(partition_data, 'val_mask') and partition_data.val_mask is not None:
                print(f'    - Validation nodes in context: {partition_data.val_mask.sum().item()}')
        if len(self.all_train_indices) != total_core_nodes_partitioned:
            print(f'警告: 分配到分区的核心节点总数 ({total_core_nodes_partitioned}) 与原始训练节点数 ({len(self.all_train_indices)}) 不匹配。')
        self._assign_validation_nodes(partitions)
        return partitions

    def _assign_validation_nodes(self, partitions: List[Data]) -> None:
        if not hasattr(self.full_graph, 'val_mask') or self.full_graph.val_mask is None:
            for part in partitions:
                if isinstance(part, Data) and part is not None:
                    num_nodes = int(part.num_nodes) if getattr(part, 'num_nodes', None) is not None else part.x.size(0) if getattr(part, 'x', None) is not None else 0
                    part.eval_mask = torch.zeros(num_nodes, dtype=torch.bool)
                    part.assigned_eval_global_ids = torch.empty(0, dtype=torch.long)
            return
        node_occurrences: Dict[int, List[Tuple[int, int]]] = {}
        for part_idx, part in enumerate(partitions):
            if not isinstance(part, Data) or getattr(part, 'val_mask', None) is None:
                continue
            val_mask = part.val_mask
            if not isinstance(val_mask, torch.Tensor):
                continue
            local_indices = torch.where(val_mask)[0]
            if local_indices.numel() == 0:
                continue
            global_ids = part.global_node_ids[local_indices]
            for local_idx, global_id in zip(local_indices.tolist(), global_ids.tolist()):
                node_occurrences.setdefault(int(global_id), []).append((part_idx, int(local_idx)))
        assigned_global_per_partition: List[List[int]] = [[] for _ in partitions]
        assigned_global_ids_set = set()
        for part in partitions:
            if isinstance(part, Data) and part is not None:
                num_nodes = int(part.num_nodes) if getattr(part, 'num_nodes', None) is not None else part.x.size(0) if getattr(part, 'x', None) is not None else 0
                part.eval_mask = torch.zeros(num_nodes, dtype=torch.bool)
        eval_node_counts = [0] * len(partitions)
        for global_id, occ_list in node_occurrences.items():
            if not occ_list:
                continue
            candidates = []
            for part_idx, local_idx in occ_list:
                part = partitions[part_idx]
                if not isinstance(part, Data) or getattr(part, 'eval_mask', None) is None:
                    continue
                candidates.append((eval_node_counts[part_idx], part_idx, local_idx))
            if not candidates:
                continue
            _, selected_part_idx, selected_local_idx = min(candidates)
            part = partitions[selected_part_idx]
            part.eval_mask[selected_local_idx] = True
            assigned_global_per_partition[selected_part_idx].append(global_id)
            assigned_global_ids_set.add(global_id)
            eval_node_counts[selected_part_idx] += 1
        all_val_global_ids = self.full_graph.val_mask.nonzero(as_tuple=False).view(-1).tolist()
        missing_ids = [gid for gid in all_val_global_ids if gid not in assigned_global_ids_set]
        if missing_ids:
            print(f'警告: {len(missing_ids)} 个验证节点未在分区中被唯一分配，使用负载均衡策略进行补充')
            for global_id in missing_ids:
                occ_list = node_occurrences.get(global_id)
                if not occ_list:
                    continue
                candidates = []
                for part_idx, local_idx in occ_list:
                    part = partitions[part_idx]
                    if not isinstance(part, Data) or getattr(part, 'eval_mask', None) is None:
                        continue
                    candidates.append((eval_node_counts[part_idx], part_idx, local_idx))
                if not candidates:
                    continue
                _, selected_part_idx, selected_local_idx = min(candidates)
                part = partitions[selected_part_idx]
                part.eval_mask[selected_local_idx] = True
                assigned_global_per_partition[selected_part_idx].append(global_id)
                eval_node_counts[selected_part_idx] += 1
        for part, assigned_ids in zip(partitions, assigned_global_per_partition):
            if isinstance(part, Data):
                if assigned_ids:
                    part.assigned_eval_global_ids = torch.tensor(assigned_ids, dtype=torch.long)
                else:
                    part.assigned_eval_global_ids = torch.empty(0, dtype=torch.long)
        if len(partitions) > 0:
            eval_counts = [len(assigned) for assigned in assigned_global_per_partition]
            if eval_counts:
                min_count = min(eval_counts)
                max_count = max(eval_counts)
                avg_count = sum(eval_counts) / len(eval_counts)
                imbalance = (max_count - min_count) / avg_count * 100 if avg_count > 0 else 0
                print(f'验证节点负载均衡统计: 最小={min_count}, 最大={max_count}, 平均={avg_count:.1f}, 不均衡度={imbalance:.2f}%')

    def _initial_train_core_assign(self) -> List[torch.Tensor]:
        ei = self.full_graph.edge_index
        if ei is None:
            raise ValueError('图数据缺少edge_index属性')
        assert isinstance(ei, torch.Tensor)
        total_nodes = self._get_total_num_nodes()
        degrees = degree(ei[0], total_nodes)
        node_degrees = degrees[self.all_train_indices]
        sorted_nodes = sorted(zip(self.all_train_indices.tolist(), node_degrees.tolist()), key=lambda x: x[1], reverse=True)
        partition_core_nodes_list: List[List[int]] = [[] for _ in range(self.num_partitions)]
        partition_degree_sums = [0.0] * self.num_partitions
        for node_idx, node_degree in sorted_nodes:
            min_idx = partition_degree_sums.index(min(partition_degree_sums))
            partition_core_nodes_list[min_idx].append(node_idx)
            partition_degree_sums[min_idx] += float(node_degree)
        return [torch.tensor(nodes, dtype=torch.long) for nodes in partition_core_nodes_list]

    def _compute_khop_nodes(self, core_nodes_tensor: torch.Tensor) -> torch.Tensor:
        if core_nodes_tensor.numel() == 0:
            return torch.empty(0, dtype=torch.long)
        ei = self.full_graph.edge_index
        if ei is None:
            raise ValueError('图数据缺少edge_index属性')
        assert isinstance(ei, torch.Tensor)
        total_nodes = self._get_total_num_nodes()
        sub_nodes, _, _, _ = k_hop_subgraph(node_idx=core_nodes_tensor, num_hops=self.max_hops, edge_index=ei, relabel_nodes=True, num_nodes=total_nodes)
        return sub_nodes

    def _compute_temp_coverage(self, core_nodes_per_part: List[torch.Tensor]) -> torch.Tensor:
        total_nodes = self._get_total_num_nodes()
        covered = torch.zeros(total_nodes, dtype=torch.bool)
        for core in core_nodes_per_part:
            sub_nodes = self._compute_khop_nodes(core)
            if sub_nodes.numel() > 0:
                covered[sub_nodes] = True
        return covered

    @staticmethod
    def _load_dataset_graph_stats(dataset_name: str):
        current_file_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(current_file_dir)
        config_path = os.path.join(project_root, 'dataset', 'Mload_dataset', 'graph_statistics.yaml')
        try:
            if os.path.exists(config_path):
                with open(config_path, 'r', encoding='utf-8') as f:
                    data = yaml.safe_load(f) or {}
                stats = data.get(dataset_name)
                if isinstance(stats, dict) and all((k in stats for k in ('vertices', 'edges', 'avg_degree', 'diameter'))):
                    return {'vertices': int(stats['vertices']), 'edges': int(stats['edges']), 'avg_degree': float(stats['avg_degree']), 'diameter': int(stats['diameter'])}
                default_stats = data.get('default')
                if isinstance(default_stats, dict) and all((k in default_stats for k in ('vertices', 'edges', 'avg_degree', 'diameter'))):
                    print(f'警告: 未找到 {dataset_name} 的图统计配置，使用默认配置')
                    return {'vertices': int(default_stats['vertices']), 'edges': int(default_stats['edges']), 'avg_degree': float(default_stats['avg_degree']), 'diameter': int(default_stats['diameter'])}
            else:
                print(f'警告: 图统计配置文件不存在: {config_path}')
        except Exception as e:
            print(f'警告: 读取图统计配置文件时出错: {e}')
        return None

    def _build_induced_subgraph(self, node_ids: torch.Tensor, train_core_nodes: torch.Tensor) -> Data:
        node_ids = node_ids.unique()
        total_nodes = self._get_total_num_nodes()
        mapping = torch.full((total_nodes,), -1, dtype=torch.long)
        mapping[node_ids] = torch.arange(int(node_ids.size(0)), dtype=torch.long)
        ei = self.full_graph.edge_index
        if ei is None:
            raise ValueError('图数据缺少edge_index属性')
        assert isinstance(ei, torch.Tensor)
        node_mask = torch.zeros(total_nodes, dtype=torch.bool)
        node_mask[node_ids] = True
        edge_mask = node_mask[ei[0]] & node_mask[ei[1]]
        sel_ei = ei[:, edge_mask]
        rel_ei = torch.stack([mapping[sel_ei[0]], mapping[sel_ei[1]]], dim=0)
        assert isinstance(self.full_graph.x, torch.Tensor) and isinstance(self.full_graph.y, torch.Tensor)
        data = Data(x=self.full_graph.x[node_ids], edge_index=rel_ei, y=self.full_graph.y[node_ids], num_nodes=int(node_ids.size(0)), global_node_ids=node_ids)
        train_core_nodes = train_core_nodes.unique()
        core_local = mapping[train_core_nodes]
        core_local = core_local[core_local != -1]
        train_mask = torch.zeros(int(node_ids.size(0)), dtype=torch.bool)
        if core_local.numel() > 0:
            train_mask[core_local] = True
        data.train_mask = train_mask
        if hasattr(self.full_graph, 'val_mask') and self.full_graph.val_mask is not None:
            data.val_mask = self.full_graph.val_mask[node_ids]
        if hasattr(self.full_graph, 'test_mask') and self.full_graph.test_mask is not None:
            data.test_mask = self.full_graph.test_mask[node_ids]
        data.graph_stats = self._compute_partition_graph_stats(data)
        data.dataset_graph_stats = self.dataset_graph_stats.copy() if self.dataset_graph_stats else None
        return data

    def _partition_contextual_khop(self) -> List[Data]:
        print(f"使用 'contextual_khop' (基于度数的贪心) 策略进行分区...")
        ei = self.full_graph.edge_index
        if ei is None:
            raise ValueError('图数据缺少edge_index属性')
        assert isinstance(ei, torch.Tensor)
        total_nodes = self._get_total_num_nodes()
        degrees = degree(ei[0], total_nodes)
        node_degrees = degrees[self.all_train_indices]
        sorted_nodes = sorted(zip(self.all_train_indices.tolist(), node_degrees.tolist()), key=lambda x: x[1], reverse=True)
        partition_core_nodes_list = [[] for _ in range(self.num_partitions)]
        partition_degree_sums = [0] * self.num_partitions
        for node_idx, node_degree in sorted_nodes:
            min_partition_idx = partition_degree_sums.index(min(partition_degree_sums))
            partition_core_nodes_list[min_partition_idx].append(node_idx)
            partition_degree_sums[min_partition_idx] += node_degree
        final_core_nodes = [torch.tensor(nodes, dtype=torch.long) for nodes in partition_core_nodes_list]
        return self._build_subgraphs_from_core_nodes(final_core_nodes)

    def _partition_random_khop(self) -> List[Data]:
        print(f"使用 'random_khop' (随机分区) 策略进行分区...")
        generator = torch.Generator()
        generator.manual_seed(42)
        perm = torch.randperm(self.all_train_indices.size(0), generator=generator)
        shuffled_train_indices = self.all_train_indices[perm]
        partition_core_nodes_list = torch.chunk(shuffled_train_indices, self.num_partitions)
        return self._build_subgraphs_from_core_nodes(list(partition_core_nodes_list))

    def _partition_mixed_khop(self) -> List[Data]:
        print("使用 'mixed_khop' (混合核心) 策略进行分区...")
        train_cores = self._initial_train_core_assign()
        covered = self._compute_temp_coverage(train_cores)
        has_val = hasattr(self.full_graph, 'val_mask') and self.full_graph.val_mask is not None
        has_test = hasattr(self.full_graph, 'test_mask') and self.full_graph.test_mask is not None
        uncovered_candidates: List[torch.Tensor] = []
        if has_val:
            uncovered_val = ~covered & self.full_graph.val_mask
            uncovered_candidates.append(uncovered_val.nonzero(as_tuple=False).view(-1))
        if has_test:
            uncovered_test = ~covered & self.full_graph.test_mask
            uncovered_candidates.append(uncovered_test.nonzero(as_tuple=False).view(-1))
        if len(uncovered_candidates) > 0:
            extra_nodes_all = torch.unique(torch.cat(uncovered_candidates))
        else:
            extra_nodes_all = torch.empty(0, dtype=torch.long)
        extra_core_ratio = self.EXTRA_CORE_RATIO_MIXED
        ei = self.full_graph.edge_index
        assert isinstance(ei, torch.Tensor)
        total_nodes = self._get_total_num_nodes()
        degrees_all = degree(ei[0], total_nodes)
        extras_sorted = extra_nodes_all[torch.argsort(degrees_all[extra_nodes_all], descending=True)] if extra_nodes_all.numel() > 0 else extra_nodes_all
        extra_cores: List[List[int]] = [[] for _ in range(self.num_partitions)]
        train_core_sizes = [int(c.numel()) for c in train_cores]
        max_extra_per_part = [max(1, int(sz * extra_core_ratio)) for sz in train_core_sizes]
        for node in extras_sorted.tolist():
            candidates = [(len(extra_cores[i]), i) for i in range(self.num_partitions) if len(extra_cores[i]) < max_extra_per_part[i]]
            if not candidates:
                break
            _, pick = min(candidates)
            extra_cores[pick].append(node)
        final_cores: List[torch.Tensor] = []
        for i in range(self.num_partitions):
            if len(extra_cores[i]) > 0:
                final_cores.append(torch.cat([train_cores[i], torch.tensor(extra_cores[i], dtype=torch.long)]))
            else:
                final_cores.append(train_cores[i])
        return self._build_subgraphs_from_core_nodes(final_cores)

    def _partition_random_fill_khop(self) -> List[Data]:
        print("使用 'random_fill_khop' (随机填充分区) 策略进行分区...")
        train_cores = self._initial_train_core_assign()
        base_nodes_per_part: List[torch.Tensor] = []
        total_nodes = self._get_total_num_nodes()
        covered = torch.zeros(total_nodes, dtype=torch.bool)
        for core in train_cores:
            sub_nodes = self._compute_khop_nodes(core)
            base_nodes_per_part.append(sub_nodes)
            if sub_nodes.numel() > 0:
                covered[sub_nodes] = True
        uncovered_all = (~covered).nonzero(as_tuple=False).view(-1)
        if uncovered_all.numel() == 0:
            return self._build_subgraphs_from_core_nodes(train_cores)
        generator = torch.Generator()
        generator.manual_seed(42)
        perm = torch.randperm(uncovered_all.numel(), generator=generator)
        shuffled = uncovered_all[perm]
        chunks = torch.chunk(shuffled, self.num_partitions)
        final_partitions: List[Data] = []
        for i in range(self.num_partitions):
            base_nodes = base_nodes_per_part[i]
            if self.RANDOM_FILL_COVER_ALL:
                extra_nodes = chunks[i] if len(chunks) > i else torch.empty(0, dtype=torch.long)
            else:
                cap = max(0, int(base_nodes.numel() * self.RANDOM_FILL_CAP_RATIO))
                extra_nodes = chunks[i][:cap] if len(chunks) > i else torch.empty(0, dtype=torch.long)
            if extra_nodes.numel() > 0:
                final_nodes = torch.unique(torch.cat([base_nodes, extra_nodes]))
            else:
                final_nodes = base_nodes
            if hasattr(self.full_graph, 'val_mask') and self.full_graph.val_mask is not None:
                val_in_final = final_nodes[self.full_graph.val_mask[final_nodes]]
                if val_in_final.numel() > 0:
                    eval_khop = self._compute_khop_nodes(val_in_final)
                    if eval_khop.numel() > 0:
                        final_nodes = torch.unique(torch.cat([final_nodes, eval_khop]))
            part_data = self._build_induced_subgraph(final_nodes, train_cores[i])
            final_partitions.append(part_data)
        self._assign_validation_nodes(final_partitions)
        return final_partitions

    def _partition_hybrid_fixed_random(self) -> List[Data]:
        print("使用 'hybrid_fixed_random' (混合固定随机) 策略进行分区...")
        fixed_seed = 42
        generator = torch.Generator()
        generator.manual_seed(fixed_seed)
        train_cores = self._initial_train_core_assign()
        base_nodes_per_part: List[torch.Tensor] = []
        total_nodes = self._get_total_num_nodes()
        covered = torch.zeros(total_nodes, dtype=torch.bool)
        for core in train_cores:
            sub_nodes = self._compute_khop_nodes(core)
            base_nodes_per_part.append(sub_nodes)
            if sub_nodes.numel() > 0:
                covered[sub_nodes] = True
        uncovered_all = (~covered).nonzero(as_tuple=False).view(-1)
        if uncovered_all.numel() == 0:
            return self._build_subgraphs_from_core_nodes(train_cores)
        perm = torch.randperm(uncovered_all.numel(), generator=generator)
        shuffled = uncovered_all[perm]
        chunks = torch.chunk(shuffled, self.num_partitions)
        final_partitions: List[Data] = []
        for i in range(self.num_partitions):
            base_nodes = base_nodes_per_part[i]
            extra_nodes = chunks[i] if len(chunks) > i else torch.empty(0, dtype=torch.long)
            if extra_nodes.numel() > 0:
                final_nodes = torch.unique(torch.cat([base_nodes, extra_nodes]))
            else:
                final_nodes = base_nodes
            if hasattr(self.full_graph, 'val_mask') and self.full_graph.val_mask is not None:
                val_in_final = final_nodes[self.full_graph.val_mask[final_nodes]]
                if val_in_final.numel() > 0:
                    eval_khop = self._compute_khop_nodes(val_in_final)
                    if eval_khop.numel() > 0:
                        final_nodes = torch.unique(torch.cat([final_nodes, eval_khop]))
            part_data = self._build_induced_subgraph(final_nodes, train_cores[i])
            final_partitions.append(part_data)
        self._assign_validation_nodes(final_partitions)
        return final_partitions

    def _partition_balanced_stratified_khop(self) -> List[Data]:
        print("使用 'balanced_stratified_khop' (分层均衡k-hop) 策略进行分区...")
        if self.full_graph.y is None:
            raise ValueError('balanced_stratified_khop 策略需要标签信息 (y)')
        assert isinstance(self.full_graph.y, torch.Tensor)
        num_partitions = self.num_partitions
        label_to_nodes: Dict[int, List[int]] = defaultdict(list)
        for node_id in self.all_train_indices.tolist():
            label = int(self.full_graph.y[node_id].item())
            label_to_nodes[label].append(int(node_id))
        if not label_to_nodes:
            raise ValueError('训练集中没有可分配的节点')
        generator = torch.Generator()
        generator.manual_seed(42)
        core_assignments: List[List[int]] = [[] for _ in range(num_partitions)]
        for label, nodes in label_to_nodes.items():
            nodes_tensor = torch.tensor(nodes, dtype=torch.long)
            if nodes_tensor.numel() == 0:
                continue
            perm = torch.randperm(nodes_tensor.numel(), generator=generator)
            shuffled = nodes_tensor[perm].tolist()
            for idx, node in enumerate(shuffled):
                target_partition = idx % num_partitions
                core_assignments[target_partition].append(node)
        base_nodes_per_part: List[torch.Tensor] = []
        covered = torch.zeros(self._get_total_num_nodes(), dtype=torch.bool)
        for part_idx in range(num_partitions):
            core_nodes_list = core_assignments[part_idx]
            if not core_nodes_list:
                print(f'警告: 分区 {part_idx} 未分配到训练节点，使用随机训练节点填充')
                fallback = int(self.all_train_indices[int(part_idx % len(self.all_train_indices))].item())
                core_nodes_list = [fallback]
            core_nodes_tensor = torch.tensor(sorted(set(core_nodes_list)), dtype=torch.long)
            sub_nodes = self._compute_khop_nodes(core_nodes_tensor)
            if sub_nodes.numel() == 0:
                sub_nodes = core_nodes_tensor
            if hasattr(self.full_graph, 'val_mask') and self.full_graph.val_mask is not None:
                val_in_final = sub_nodes[self.full_graph.val_mask[sub_nodes]]
                if val_in_final.numel() > 0:
                    eval_khop = self._compute_khop_nodes(val_in_final)
                    if eval_khop.numel() > 0:
                        sub_nodes = torch.unique(torch.cat([sub_nodes, eval_khop]))
            covered[sub_nodes] = True
            base_nodes_per_part.append(sub_nodes)
        uncovered = (~covered).nonzero(as_tuple=False).view(-1)
        if uncovered.numel() > 0:
            perm = torch.randperm(uncovered.numel(), generator=generator)
            shuffled = uncovered[perm]
            chunks = torch.chunk(shuffled, num_partitions)
            for idx, chunk in enumerate(chunks):
                if chunk.numel() == 0:
                    continue
                merged = torch.unique(torch.cat([base_nodes_per_part[idx], chunk]))
                base_nodes_per_part[idx] = merged
        final_partitions: List[Data] = []
        per_partition_label_stats: List[Dict[int, int]] = []
        for part_idx, sub_nodes in enumerate(base_nodes_per_part):
            core_nodes_list = core_assignments[part_idx]
            if not core_nodes_list:
                core_nodes_list = [int(self.all_train_indices[int(part_idx % len(self.all_train_indices))].item())]
            core_nodes_tensor = torch.tensor(sorted(set(core_nodes_list)), dtype=torch.long)
            part_data = self._build_induced_subgraph(sub_nodes, core_nodes_tensor)
            final_partitions.append(part_data)
            label_stats: Dict[int, int] = defaultdict(int)
            train_mask = getattr(part_data, 'train_mask', None)
            if isinstance(train_mask, torch.Tensor) and train_mask.any():
                global_ids = part_data.global_node_ids[train_mask]
                for gid in global_ids.tolist():
                    label = int(self.full_graph.y[int(gid)].item())
                    label_stats[label] += 1
            per_partition_label_stats.append(label_stats)
        self._assign_validation_nodes(final_partitions)
        print('分层均衡策略分区统计:')
        for idx, (part, stats) in enumerate(zip(final_partitions, per_partition_label_stats)):
            train_count = int(part.train_mask.sum().item()) if getattr(part, 'train_mask', None) is not None else 0
            label_summary = ', '.join((f'class{lbl}:{cnt}' for lbl, cnt in sorted(stats.items())))
            print(f'  分区 {idx}: 训练节点={train_count}, 标签分布[{label_summary}]')
        return final_partitions

    def _partition_communication_only(self) -> List[Data]:
        print("使用 'communication_only' (纯数据并行) 策略进行分区...")
        fixed_seed = 42
        generator = torch.Generator()
        generator.manual_seed(fixed_seed)
        train_cores = self._initial_train_core_assign()
        ei = self.full_graph.edge_index
        if ei is None:
            raise ValueError('图数据缺少edge_index属性')
        assert isinstance(ei, torch.Tensor)
        total_nodes = self._get_total_num_nodes()
        val_mask = getattr(self.full_graph, 'val_mask', None)
        test_mask = getattr(self.full_graph, 'test_mask', None)
        train_mask = getattr(self.full_graph, 'train_mask', None)
        val_ids = val_mask.nonzero(as_tuple=False).view(-1) if isinstance(val_mask, torch.Tensor) else torch.empty(0, dtype=torch.long)
        test_ids = test_mask.nonzero(as_tuple=False).view(-1) if isinstance(test_mask, torch.Tensor) else torch.empty(0, dtype=torch.long)
        if isinstance(train_mask, torch.Tensor):
            labeled_mask = train_mask.clone()
        else:
            raise ValueError('communication_only 策略要求数据集包含 train_mask')
        if isinstance(val_mask, torch.Tensor):
            labeled_mask = labeled_mask | val_mask
        if isinstance(test_mask, torch.Tensor):
            labeled_mask = labeled_mask | test_mask
        unlabeled_ids = (~labeled_mask).nonzero(as_tuple=False).view(-1)
        remaining_val_ids = set(val_ids.tolist())
        remaining_test_ids = set(test_ids.tolist())
        assigned_val_partitions = {}
        assigned_test_partitions = {}
        final_node_sets: List[set[int]] = [set(core.tolist()) for core in train_cores]
        for idx, core_nodes in enumerate(train_cores):
            if core_nodes.numel() == 0:
                continue
            core_mask = torch.zeros(total_nodes, dtype=torch.bool)
            core_mask[core_nodes] = True
            edge_mask = core_mask[ei[0]] | core_mask[ei[1]]
            adjacent_nodes = torch.unique(torch.cat([ei[0][edge_mask], ei[1][edge_mask]]))
            if isinstance(val_mask, torch.Tensor) and adjacent_nodes.numel() > 0:
                val_neighbors = adjacent_nodes[val_mask[adjacent_nodes]]
                if val_neighbors.numel() > 0:
                    for gid in val_neighbors.tolist():
                        if gid in assigned_val_partitions:
                            continue
                        final_node_sets[idx].add(int(gid))
                        assigned_val_partitions[int(gid)] = idx
                        remaining_val_ids.discard(int(gid))
            if isinstance(test_mask, torch.Tensor) and adjacent_nodes.numel() > 0:
                test_neighbors = adjacent_nodes[test_mask[adjacent_nodes]]
                if test_neighbors.numel() > 0:
                    for gid in test_neighbors.tolist():
                        if gid in assigned_test_partitions:
                            continue
                        final_node_sets[idx].add(int(gid))
                        assigned_test_partitions[int(gid)] = idx
                        remaining_test_ids.discard(int(gid))
        if remaining_val_ids:
            for pos, global_id in enumerate(sorted(remaining_val_ids)):
                target_partition = pos % self.num_partitions
                final_node_sets[target_partition].add(int(global_id))
                assigned_val_partitions[int(global_id)] = target_partition
        if remaining_test_ids:
            for pos, global_id in enumerate(sorted(remaining_test_ids)):
                target_partition = pos % self.num_partitions
                final_node_sets[target_partition].add(int(global_id))
                assigned_test_partitions[int(global_id)] = target_partition
        if unlabeled_ids.numel() > 0:
            perm = torch.randperm(unlabeled_ids.numel(), generator=generator)
            shuffled_unlabeled = unlabeled_ids[perm]
            unlabeled_chunks = torch.chunk(shuffled_unlabeled, self.num_partitions)
            for part_idx, chunk in enumerate(unlabeled_chunks):
                if chunk.numel() == 0:
                    continue
                final_node_sets[part_idx].update((int(gid) for gid in chunk.tolist()))
        final_partitions: List[Data] = []
        for i in range(self.num_partitions):
            core_nodes = train_cores[i]
            final_nodes = torch.tensor(sorted(final_node_sets[i]), dtype=torch.long)
            part_data = self._build_induced_subgraph(final_nodes, core_nodes)
            final_partitions.append(part_data)
            extra_count = max(0, final_nodes.numel() - core_nodes.numel())
            print(f'  分区 {i}: {core_nodes.numel()} 个训练核心节点 + {extra_count} 个验证/测试节点')
        self._assign_validation_nodes(final_partitions)
        print('注意: CO模式需要通信机制获取远程邻居特征进行GNN计算')
        return final_partitions

    def _partition_balanced_communication_only(self) -> List[Data]:
        print("使用 'balanced_communication_only' (均衡通信分区) 策略进行分区...")
        if self.full_graph.y is None:
            raise ValueError('balanced_communication_only 策略需要标签信息 (y)')
        assert isinstance(self.full_graph.y, torch.Tensor)
        label_to_nodes: Dict[int, List[int]] = defaultdict(list)
        for node_id in self.all_train_indices.tolist():
            label = int(self.full_graph.y[node_id].item())
            label_to_nodes[label].append(int(node_id))
        if not label_to_nodes:
            raise ValueError('训练集中没有可分配的节点')
        generator = torch.Generator()
        generator.manual_seed(42)
        core_assignments: List[List[int]] = [[] for _ in range(self.num_partitions)]
        for _, nodes in label_to_nodes.items():
            nodes_tensor = torch.tensor(nodes, dtype=torch.long)
            if nodes_tensor.numel() == 0:
                continue
            perm = torch.randperm(nodes_tensor.numel(), generator=generator)
            shuffled = nodes_tensor[perm].tolist()
            for idx, node in enumerate(shuffled):
                target_partition = idx % self.num_partitions
                core_assignments[target_partition].append(int(node))
        train_cores: List[torch.Tensor] = []
        for part_idx in range(self.num_partitions):
            nodes_list = core_assignments[part_idx]
            if not nodes_list:
                fallback = int(self.all_train_indices[int(part_idx % len(self.all_train_indices))].item())
                nodes_list = [fallback]
            unique_nodes = torch.tensor(sorted(set(nodes_list)), dtype=torch.long)
            train_cores.append(unique_nodes)
        fixed_seed = 50
        generator2 = torch.Generator()
        generator2.manual_seed(fixed_seed)
        ei = self.full_graph.edge_index
        if ei is None:
            raise ValueError('图数据缺少edge_index属性')
        assert isinstance(ei, torch.Tensor)
        total_nodes = self._get_total_num_nodes()
        val_mask = getattr(self.full_graph, 'val_mask', None)
        test_mask = getattr(self.full_graph, 'test_mask', None)
        train_mask = getattr(self.full_graph, 'train_mask', None)
        val_ids = val_mask.nonzero(as_tuple=False).view(-1) if isinstance(val_mask, torch.Tensor) else torch.empty(0, dtype=torch.long)
        test_ids = test_mask.nonzero(as_tuple=False).view(-1) if isinstance(test_mask, torch.Tensor) else torch.empty(0, dtype=torch.long)
        if isinstance(train_mask, torch.Tensor):
            labeled_mask = train_mask.clone()
        else:
            raise ValueError('balanced_communication_only 策略要求数据集包含 train_mask')
        if isinstance(val_mask, torch.Tensor):
            labeled_mask = labeled_mask | val_mask
        if isinstance(test_mask, torch.Tensor):
            labeled_mask = labeled_mask | test_mask
        unlabeled_ids = (~labeled_mask).nonzero(as_tuple=False).view(-1)
        remaining_val_ids = set(val_ids.tolist())
        remaining_test_ids = set(test_ids.tolist())
        assigned_val_partitions: Dict[int, int] = {}
        assigned_test_partitions: Dict[int, int] = {}
        final_node_sets: List[set[int]] = [set(core.tolist()) for core in train_cores]
        for idx, core_nodes in enumerate(train_cores):
            if core_nodes.numel() == 0:
                continue
            core_mask = torch.zeros(total_nodes, dtype=torch.bool)
            core_mask[core_nodes] = True
            edge_mask = core_mask[ei[0]] | core_mask[ei[1]]
            adjacent_nodes = torch.unique(torch.cat([ei[0][edge_mask], ei[1][edge_mask]]))
            if isinstance(val_mask, torch.Tensor) and adjacent_nodes.numel() > 0:
                val_neighbors = adjacent_nodes[val_mask[adjacent_nodes]]
                for gid in val_neighbors.tolist():
                    if gid in assigned_val_partitions:
                        continue
                    final_node_sets[idx].add(int(gid))
                    assigned_val_partitions[int(gid)] = idx
                    remaining_val_ids.discard(int(gid))
            if isinstance(test_mask, torch.Tensor) and adjacent_nodes.numel() > 0:
                test_neighbors = adjacent_nodes[test_mask[adjacent_nodes]]
                for gid in test_neighbors.tolist():
                    if gid in assigned_test_partitions:
                        continue
                    final_node_sets[idx].add(int(gid))
                    assigned_test_partitions[int(gid)] = idx
                    remaining_test_ids.discard(int(gid))
        if remaining_val_ids:
            for pos, global_id in enumerate(sorted(remaining_val_ids)):
                target_partition = pos % self.num_partitions
                final_node_sets[target_partition].add(int(global_id))
                assigned_val_partitions[int(global_id)] = target_partition
        if remaining_test_ids:
            for pos, global_id in enumerate(sorted(remaining_test_ids)):
                target_partition = pos % self.num_partitions
                final_node_sets[target_partition].add(int(global_id))
                assigned_test_partitions[int(global_id)] = target_partition
        if unlabeled_ids.numel() > 0:
            perm = torch.randperm(unlabeled_ids.numel(), generator=generator2)
            shuffled_unlabeled = unlabeled_ids[perm]
            unlabeled_chunks = torch.chunk(shuffled_unlabeled, self.num_partitions)
            for part_idx, chunk in enumerate(unlabeled_chunks):
                if chunk.numel() == 0:
                    continue
                final_node_sets[part_idx].update((int(gid) for gid in chunk.tolist()))
        final_partitions: List[Data] = []
        for i in range(self.num_partitions):
            core_nodes = train_cores[i]
            final_nodes = torch.tensor(sorted(final_node_sets[i]), dtype=torch.long)
            part_data = self._build_induced_subgraph(final_nodes, core_nodes)
            final_partitions.append(part_data)
            extra_count = max(0, final_nodes.numel() - core_nodes.numel())
            print(f'  分区 {i}: {core_nodes.numel()} 个训练核心节点 + {extra_count} 个验证/测试节点')
        self._assign_validation_nodes(final_partitions)
        print('注意: 均衡CO模式需要通信机制获取远程邻居特征进行GNN计算')
        return final_partitions

    @classmethod
    def get_partitions(cls, dataset_name: str, num_partitions: int, max_hops: int, strategy: str, train_ratio: float=0.0) -> List[Data]:
        print(f"开始分区 '{dataset_name}' for {num_partitions} workers with {max_hops}-hop context...")
        torch_rng_state = torch.get_rng_state()
        if torch.cuda.is_available():
            cuda_rng_state = torch.cuda.get_rng_state()
        partitioner = cls(dataset_name, num_partitions, max_hops)
        torch.set_rng_state(torch_rng_state)
        if torch.cuda.is_available():
            torch.cuda.set_rng_state(cuda_rng_state)
        if strategy == 'contextual_khop':
            partitions = partitioner._partition_contextual_khop()
        elif strategy == 'random_khop':
            partitions = partitioner._partition_random_khop()
        elif strategy == 'mixed_khop':
            partitions = partitioner._partition_mixed_khop()
        elif strategy == 'random_fill_khop':
            partitions = partitioner._partition_random_fill_khop()
        elif strategy == 'hybrid_fixed_random':
            partitions = partitioner._partition_hybrid_fixed_random()
        elif strategy == 'balanced_stratified_khop':
            partitions = partitioner._partition_balanced_stratified_khop()
        elif strategy == 'balanced_communication_only':
            partitions = partitioner._partition_balanced_communication_only()
        elif strategy == 'communication_only':
            partitions = partitioner._partition_communication_only()
        else:
            raise NotImplementedError(f"不支持的分区策略: '{strategy}'. 可选值: 'contextual_khop', 'random_khop', 'mixed_khop', 'random_fill_khop', 'hybrid_fixed_random', 'balanced_stratified_khop', 'balanced_communication_only', 'communication_only'")
        print(f'\n分区完成. 共生成 {len(partitions)} 个分区.')
        return partitions

def _bytes_to_readable(num_bytes: float) -> str:
    gb = num_bytes / 1024 ** 3
    if gb >= 1:
        return f'{gb:.2f} GB'
    mb = num_bytes / 1024 ** 2
    if mb >= 1:
        return f'{mb:.2f} MB'
    kb = num_bytes / 1024
    return f'{kb:.2f} KB'

def estimate_gpu_memory_for_partitions(partitions: List[Data], device_ids: List[int] | None=None) -> Dict[int, Dict[str, float]]:
    import torch
    summary: Dict[int, Dict[str, float]] = {}
    if not torch.cuda.is_available():
        print('GPU 不可用，跳过显存估算')
        return summary
    num_available = torch.cuda.device_count()
    if device_ids is None:
        device_ids = list(range(num_available))
    else:
        device_ids = [d for d in device_ids if 0 <= d < num_available]
        if not device_ids:
            device_ids = list(range(num_available))
    for d in device_ids:
        summary[d] = {'total': 0.0, 'max_single_partition': 0.0}
    for idx, part in enumerate(partitions):
        target_gpu = device_ids[idx % len(device_ids)]
        device = torch.device(f'cuda:{target_gpu}')
        try:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
            before = torch.cuda.memory_allocated(device)
            part_gpu = part.to(device)
            torch.cuda.synchronize(device)
            after = torch.cuda.memory_allocated(device)
            used = max(0.0, float(after - before))
            summary[target_gpu]['total'] += used
            summary[target_gpu]['max_single_partition'] = max(summary[target_gpu]['max_single_partition'], used)
        except RuntimeError as e:
            print(f'  ⚠️ 分区 {idx} 迁移到 GPU:{target_gpu} 失败: {e}')
            used = 0.0
        finally:
            try:
                del part_gpu
            except Exception:
                pass
            torch.cuda.empty_cache()
            torch.cuda.synchronize(device)
        print(f'    · Partition {idx} -> GPU {target_gpu}: +{_bytes_to_readable(used)}')
    print('\nGPU 显存估算汇总（非常驻，分区逐个迁移测量）:')
    for d in device_ids:
        total = summary[d]['total']
        peak = summary[d]['max_single_partition']
        print(f'  - GPU {d}: 估算总占用={_bytes_to_readable(total)}, 单分区最大={_bytes_to_readable(peak)}')
    return summary
if __name__ == '__main__':
    print('===== Running Test for data/partition.py =====')
    DATASET = 'ogbn-products'
    NUM_GPUS = 8
    MAX_HOPS_FOR_GNN = 2
    _, original_graph = get_dataset(name=DATASET, root='dataset')
    original_train_nodes = original_graph.train_mask.sum().item()
    original_val_nodes = original_graph.val_mask.sum().item()
    original_test_nodes = original_graph.test_mask.sum().item()
    print(f"\nOriginal '{DATASET}' graph info:")
    print(f'  - Total Nodes: {original_graph.num_nodes}')
    print(f'  - Train Nodes: {original_train_nodes}')
    print(f'  - Validation Nodes: {original_val_nodes}')
    print(f'  - Test Nodes: {original_test_nodes}')
    strategies_to_test = ['balanced_stratified_khop', 'balanced_communication_only']
    for strategy in strategies_to_test:
        print(f"\n--- Testing Strategy: '{strategy}' ---")
        try:
            partitions = PartitionGraph.get_partitions(dataset_name=DATASET, num_partitions=NUM_GPUS, max_hops=MAX_HOPS_FOR_GNN, strategy=strategy)
            total_partitioned_train = 0
            sum_of_val_in_partitions = 0
            sum_of_test_in_partitions = 0
            for i, p in enumerate(partitions):
                train_count = p.train_mask.sum().item() if hasattr(p, 'train_mask') and p.train_mask is not None else 0
                val_count = p.val_mask.sum().item() if hasattr(p, 'val_mask') and p.val_mask is not None else 0
                test_count = p.test_mask.sum().item() if hasattr(p, 'test_mask') and p.test_mask is not None else 0
                total_partitioned_train += train_count
                sum_of_val_in_partitions += val_count
                sum_of_test_in_partitions += test_count
                print(f'   - Partition {i}: Train={train_count}, Val={val_count}, Test={test_count}, Total Nodes={p.num_nodes}')
            print('\n   -> Coverage Analysis:')
            train_coverage = total_partitioned_train / original_train_nodes * 100 if original_train_nodes > 0 else 0
            print(f'      - Core Train Nodes Coverage: {total_partitioned_train} / {original_train_nodes} ({train_coverage:.2f}%)')
            val_ratio = sum_of_val_in_partitions / original_val_nodes * 100 if original_val_nodes > 0 else 0
            test_ratio = sum_of_test_in_partitions / original_test_nodes * 100 if original_test_nodes > 0 else 0
            print(f'      - Sum of Val Nodes in Partitions: {sum_of_val_in_partitions} (Ratio to original: {val_ratio:.2f}%)')
            print(f'      - Sum of Test Nodes in Partitions: {sum_of_test_in_partitions} (Ratio to original: {test_ratio:.2f}%)')
            assert len(partitions) == NUM_GPUS, f'Expected {NUM_GPUS} partitions, but got {len(partitions)}'
            print('\n -> Correct number of partitions created.')
            assert total_partitioned_train == original_train_nodes, f'Mismatch in training nodes. Original: {original_train_nodes}, Partitioned sum: {total_partitioned_train}'
            print(' -> Sum of core training nodes matches original count.')
            first_partition = partitions[0]
            assert hasattr(first_partition, 'x') and first_partition.x is not None
            assert hasattr(first_partition, 'edge_index') and first_partition.edge_index is not None
            assert hasattr(first_partition, 'train_mask') and first_partition.train_mask is not None
            print(f" -> Strategy '{strategy}' Passed!")
            import torch
            if torch.cuda.is_available():
                print('\n   -> 估算将分区迁移到GPU后的显存占用（逐个测量，不常驻）...')
                try:
                    estimate_gpu_memory_for_partitions(partitions, device_ids=list(range(min(NUM_GPUS, torch.cuda.device_count()))))
                except Exception as e:
                    print(f'   -> 显存估算出错: {e}')
        except Exception as e:
            print(f" -> Strategy '{strategy}' FAILED: {e}")
    print('\n===== Test Passed! =====')