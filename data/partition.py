# RingGNN/data/partition.py
# graphnas_dist/data/partition.py

import torch
from typing import List
from torch_geometric.data import Data
from torch_geometric.utils import k_hop_subgraph
from torch_geometric.utils import degree
from data.data_utils import get_dataset

def partition_training_graph(dataset_name: str, num_partitions: int, max_hops: int) -> List[Data]:
    """
    图分区：
    1.  仅将【训练节点】均分为N份，作为每个分区的“核心”
    2.  为每个分区的“核心”节点，从全局图中提取其k-hop邻域，构建一个完整的、自包含的训练子图

    Args:
        dataset_name (str): 数据集名称 (如 'Cora')
        num_partitions (int): 分区数量, 应等于GPU数量
        max_hops (int): 为每个分区包含的最大邻居跳数，应等于搜索空间中GNN的最大层数

    Returns:
        List[Data]: 每个元素是一个包含k-hop上下文的、可直接训练的子图分区
    """
    print(f"Partitioning '{dataset_name}' for {num_partitions} workers with {max_hops}-hop context...")
    
    # 1. 加载完整的全局图
    _, full_graph = get_dataset(name=dataset_name, root='dataset')
    
    # 确保图数据对象包含所有必要的属性
    if not hasattr(full_graph, 'edge_index') or full_graph.edge_index is None:
        raise ValueError(f"数据集 '{dataset_name}' 缺少边信息 (edge_index)")
    if not hasattr(full_graph, 'x') or full_graph.x is None:
        raise ValueError(f"数据集 '{dataset_name}' 缺少节点特征 (x)")
    if not hasattr(full_graph, 'y') or full_graph.y is None:
        raise ValueError(f"数据集 '{dataset_name}' 缺少节点标签 (y)")
    if not hasattr(full_graph, 'train_mask') or full_graph.train_mask is None:
        raise ValueError(f"数据集 '{dataset_name}' 缺少训练掩码 (train_mask)")
    
    # 确保 y 是张量而不是标量
    if not isinstance(full_graph.y, torch.Tensor):
        raise ValueError(f"数据集 '{dataset_name}' 的节点标签 (y) 不是张量类型")
    
    # 2. 核心步骤：只对训练节点进行分区
    #    从 train_mask 中获取所有训练节点的全局索引
    all_train_indices = full_graph.train_mask.nonzero(as_tuple=False).view(-1)
    sum_train_indices = len(all_train_indices)
    
    #    随机打乱训练节点，以确保分区是随机的，避免数据倾斜
    # perm = torch.randperm(all_train_indices.size(0))
    # shuffled_train_indices = all_train_indices[perm]
    shuffled_train_indices = all_train_indices # 不进行打乱

    # 将排序好的训练节点切分呈N块，保证每块分区度数之和尽可能接近
    # 统计每个训练节点的自身的度数
    degrees = degree(full_graph.edge_index[0], full_graph.num_nodes)
    node_degrees_list = degrees[all_train_indices]

    # 按照度数从大到小排序，但保留原始索引信息
    sorted_indices_with_degrees = list(zip(all_train_indices.tolist(), node_degrees_list.tolist()))
    sorted_indices_with_degrees.sort(key=lambda x: x[1], reverse=True)  # 按度数降序排序

    # 贪心算法：将节点分配给当前度数之和最小的分区
    partition_core_nodes_list = [[] for _ in range(num_partitions)]
    partition_degree_sums = [0] * num_partitions

    for node_idx, node_degree in sorted_indices_with_degrees:
        # 找到当前度数之和最小的分区
        min_partition_idx = partition_degree_sums.index(min(partition_degree_sums))
        partition_core_nodes_list[min_partition_idx].append(node_idx)
        partition_degree_sums[min_partition_idx] += node_degree

    # 将每个分区内的节点按原始索引排序
    for i in range(num_partitions):
        partition_core_nodes_list[i].sort()
        partition_core_nodes_list[i] = torch.tensor(partition_core_nodes_list[i], dtype=torch.long).tolist()


    # 将打乱后的训练节点切分为 N 块，每一块都是一个分区的“核心”
    # partition_core_nodes_list = torch.chunk(shuffled_train_indices, num_partitions)

    partitions = []
    total_core_nodes_partitioned = 0
    for i, core_nodes in enumerate(partition_core_nodes_list):
        total_core_nodes_partitioned += len(core_nodes)
        
        # 3. 核心步骤：为每个分区的“核心”节点，提取其k-hop邻域作为上下文
        #    这是整个分区逻辑中最关键、最昂贵的一步，在系统启动时一次性完成。
        core_nodes_tensor = torch.tensor(core_nodes, dtype=torch.long)
        sub_nodes, sub_edge_index, inv, _ = k_hop_subgraph(
            node_idx=core_nodes_tensor,      # 指定中心节点
            num_hops=max_hops,               # 指定扩展的跳数 k
            edge_index=full_graph.edge_index,# 使用全局图的边信息
            relabel_nodes=True,              # 关键！将子图中的节点ID重新从0开始编号
            num_nodes=full_graph.num_nodes   # 传入全局图节点数以提高效率
        )
        
        # 4. 构建一个新的、自包含的Data对象
        partition_data = Data()
        partition_data.num_nodes = sub_nodes.size(0)
        partition_data.x = full_graph.x[sub_nodes]
        partition_data.edge_index = sub_edge_index
        partition_data.y = full_graph.y[sub_nodes]
        
        # 5. 精确生成新的掩码 (使用更稳健的方法)
        
        #    a. 创建一个覆盖整个图的、从全局ID到本地ID的映射
        #       - 用-1填充，表示默认情况下节点不在子图中
        #       - 然后为子图中的节点填充它们的本地ID (0, 1, 2, ...)
        num_nodes_in_full_graph = full_graph.num_nodes
        if num_nodes_in_full_graph is None:
            num_nodes_in_full_graph = int(full_graph.edge_index.max().item() + 1)
            
        mapping = torch.full((num_nodes_in_full_graph,), -1, dtype=torch.long)
        mapping[sub_nodes] = torch.arange(sub_nodes.size(0))

        #    b. 新的train_mask：只包含当前分区的“核心”训练节点
        new_train_mask = torch.zeros(sub_nodes.size(0), dtype=torch.bool)
        
        #    c. 使用我们创建的全局映射来查找核心节点的本地ID
        core_nodes_local_indices = mapping[core_nodes_tensor]

        #    d. 添加断言进行验证：确保所有核心节点都被包含在了子图中
        #       如果某个核心节点的本地索引是-1，说明它不在子图中，这是一个逻辑错误。
        assert (core_nodes_local_indices != -1).all(), \
            "逻辑错误: 一个核心节点没有在它自己的k-hop子图中被找到！"
        
        new_train_mask[core_nodes_local_indices] = True
        partition_data.train_mask = new_train_mask
        
        #    e. 新的val/test_mask：包含落入此k-hop子图范围内的所有原始验证/测试节点
        partition_data.val_mask = full_graph.val_mask[sub_nodes]
        partition_data.test_mask = full_graph.test_mask[sub_nodes]

        partitions.append(partition_data)
        print(f"  - Partition {i} created for Worker-{i}:")
        print(f"    - Subgraph size: {partition_data.num_nodes} nodes, {partition_data.num_edges} edges.")
        print(f"    - Core training nodes in this partition: {partition_data.train_mask.sum().item()}")
        print(f"    - Validation nodes in this partition's context: {partition_data.val_mask.sum().item()}")
    
    if sum_train_indices != total_core_nodes_partitioned:
        print(f"Warning: Total core nodes assigned to partitions ({total_core_nodes_partitioned}) does not match the original training nodes ({sum_train_indices}).")
    
    print(f"\nPartitioning complete.")
    print(f"  - Total original training nodes: {all_train_indices.size(0)}")
    print(f"  - Total core nodes assigned to partitions: {total_core_nodes_partitioned}")
    
    return partitions

# ====================
#  独立的测试模块
# ====================
if __name__ == '__main__':
    print("===== Running Test for data/partition.py =====")
    
    DATASET = 'Cora'
    NUM_GPUS = 2
    MAX_HOPS_FOR_GNN = 2
    
    # 1. 加载原始数据集信息
    _, original_graph = get_dataset(name=DATASET, root='dataset')
    original_train_nodes = original_graph.train_mask.sum().item()
    print(f"\nOriginal '{DATASET}' graph has {original_train_nodes} training nodes.")
    
    # 2. 调用我们的分区函数
    partitions = partition_training_graph(
        dataset_name=DATASET, 
        num_partitions=NUM_GPUS, 
        max_hops=MAX_HOPS_FOR_GNN
    )
    
    # 3. 验证分区的正确性
    print("\n--- Verifying Partition Integrity ---")
    assert len(partitions) == NUM_GPUS, f"Expected {NUM_GPUS} partitions, but got {len(partitions)}"
    print("Correct number of partitions created.")
    
    total_partitioned_train_nodes = sum(p.train_mask.sum().item() for p in partitions)
    assert total_partitioned_train_nodes == original_train_nodes, \
        f"Mismatch in training nodes. Original: {original_train_nodes}, Partitioned sum: {total_partitioned_train_nodes}"
    print("Sum of core training nodes in all partitions matches the original count.")
    
    first_partition = partitions[0]
    assert hasattr(first_partition, 'x') and first_partition.x is not None
    assert hasattr(first_partition, 'edge_index') and first_partition.edge_index is not None
    assert hasattr(first_partition, 'train_mask') and first_partition.train_mask is not None
    print("Partitions are valid PyG Data objects with all necessary attributes.")
    print("\n===== Test Passed! =====")