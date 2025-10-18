# graphnas_dist/data/partition.py

import torch
from typing import List
from torch_geometric.data import Data
from torch_geometric.utils import k_hop_subgraph, degree
from data.data_utils import get_dataset

def partition_training_graph(dataset_name: str, num_partitions: int, max_hops: int) -> List[Data]:
    """
    Graph Partitioning:
    1.  Divide only the training nodes into N parts as the "core" of each partition.
    2.  For each core, extract its k-hop neighborhood from the global graph to build a self-contained subgraph.

    Args:
        dataset_name (str): Name of the dataset (e.g., 'Cora').
        num_partitions (int): Number of partitions, should equal the number of GPUs.
        max_hops (int): Max hops for k-hop neighborhood, should equal the max GNN layers.

    Returns:
        List[Data]: A list of trainable subgraphs, each with its k-hop context.
    """
    print(f"Partitioning '{dataset_name}' for {num_partitions} workers with {max_hops}-hop context...")
    

    _, full_graph = get_dataset(name=dataset_name, root='dataset')
    
    # Ensure necessary attributes exist
    if not hasattr(full_graph, 'edge_index') or full_graph.edge_index is None:
        raise ValueError(f"Dataset '{dataset_name}' is missing edge_index.")
    if not hasattr(full_graph, 'x') or full_graph.x is None:
        raise ValueError(f"Dataset '{dataset_name}' is missing node features (x).")
    if not hasattr(full_graph, 'y') or full_graph.y is None:
        raise ValueError(f"Dataset '{dataset_name}' is missing node labels (y).")
    if not hasattr(full_graph, 'train_mask') or full_graph.train_mask is None:
        raise ValueError(f"Dataset '{dataset_name}' is missing train_mask.")
    
    # Ensure y is a tensor
    if not isinstance(full_graph.y, torch.Tensor):
        raise ValueError(f"Node labels (y) for '{dataset_name}' are not a tensor.")
    

    all_train_indices = full_graph.train_mask.nonzero(as_tuple=False).view(-1)
    sum_train_indices = len(all_train_indices)
    
    # Greedy algorithm to balance partitions by the sum of node degrees
    degrees = degree(full_graph.edge_index[0], full_graph.num_nodes)
    node_degrees_list = degrees[all_train_indices]

    # Sort nodes by degree in descending order, keeping original indices
    sorted_indices_with_degrees = sorted(zip(all_train_indices.tolist(), node_degrees_list.tolist()), key=lambda x: x[1], reverse=True)

    # Assign each node to the partition with the current minimum sum of degrees
    partition_core_nodes_list = [[] for _ in range(num_partitions)]
    partition_degree_sums = [0] * num_partitions

    for node_idx, node_degree in sorted_indices_with_degrees:
        min_partition_idx = partition_degree_sums.index(min(partition_degree_sums))
        partition_core_nodes_list[min_partition_idx].append(node_idx)
        partition_degree_sums[min_partition_idx] += node_degree

    # Sort nodes within each partition by their original index
    for i in range(num_partitions):
        partition_core_nodes_list[i].sort()
        partition_core_nodes_list[i] = torch.tensor(partition_core_nodes_list[i], dtype=torch.long).tolist()

    partitions = []
    total_core_nodes_partitioned = 0
    for i, core_nodes in enumerate(partition_core_nodes_list):
        total_core_nodes_partitioned += len(core_nodes)
        

        core_nodes_tensor = torch.tensor(core_nodes, dtype=torch.long)
        sub_nodes, sub_edge_index, inv, _ = k_hop_subgraph(
            node_idx=core_nodes_tensor,
            num_hops=max_hops,
            edge_index=full_graph.edge_index,
            relabel_nodes=True, # Crucial: relabel subgraph nodes from 0
            num_nodes=full_graph.num_nodes
        )
        

        partition_data = Data(
            num_nodes=sub_nodes.size(0),
            x=full_graph.x[sub_nodes],
            edge_index=sub_edge_index,
            y=full_graph.y[sub_nodes]
        )
        

        num_nodes_in_full_graph = full_graph.num_nodes or int(full_graph.edge_index.max().item() + 1)
        mapping = torch.full((num_nodes_in_full_graph,), -1, dtype=torch.long)
        mapping[sub_nodes] = torch.arange(sub_nodes.size(0))


        new_train_mask = torch.zeros(sub_nodes.size(0), dtype=torch.bool)
        

        core_nodes_local_indices = mapping[core_nodes_tensor]


        assert (core_nodes_local_indices != -1).all(), "Logical error: A core node was not found in its own k-hop subgraph."
        
        new_train_mask[core_nodes_local_indices] = True
        partition_data.train_mask = new_train_mask
        

        partition_data.val_mask = full_graph.val_mask[sub_nodes]
        partition_data.test_mask = full_graph.test_mask[sub_nodes]

        partitions.append(partition_data)
        print(f"  - Partition {i} created for Worker-{i}:")
        print(f"    - Subgraph size: {partition_data.num_nodes} nodes, {partition_data.num_edges} edges.")
        print(f"    - Core training nodes: {partition_data.train_mask.sum().item()}")
        print(f"    - Validation nodes in context: {partition_data.val_mask.sum().item()}")
    
    if sum_train_indices != total_core_nodes_partitioned:
        print(f"Warning: Partitioned core nodes ({total_core_nodes_partitioned}) mismatch original training nodes ({sum_train_indices}).")
    
    print(f"\nPartitioning complete.")
    print(f"  - Total original training nodes: {all_train_indices.size(0)}")
    print(f"  - Total partitioned core nodes: {total_core_nodes_partitioned}")
    
    return partitions


if __name__ == '__main__':
    print("===== Running Test for data/partition.py =====")
    
    DATASET = 'Cora'
    NUM_GPUS = 2
    MAX_HOPS_FOR_GNN = 2
    

    _, original_graph = get_dataset(name=DATASET, root='dataset')
    original_train_nodes = original_graph.train_mask.sum().item()
    print(f"\nOriginal '{DATASET}' graph has {original_train_nodes} training nodes.")
    

    partitions = partition_training_graph(
        dataset_name=DATASET, 
        num_partitions=NUM_GPUS, 
        max_hops=MAX_HOPS_FOR_GNN
    )
    

    print("\n--- Verifying Partition Integrity ---")
    assert len(partitions) == NUM_GPUS, f"Expected {NUM_GPUS} partitions, but got {len(partitions)}"
    print("Correct number of partitions created.")
    
    total_partitioned_train_nodes = sum(p.train_mask.sum().item() for p in partitions)
    assert total_partitioned_train_nodes == original_train_nodes, \
        f"Mismatch in training nodes. Original: {original_train_nodes}, Partitioned sum: {total_partitioned_train_nodes}"
    print("Sum of core training nodes matches original count.")
    
    first_partition = partitions[0]
    assert hasattr(first_partition, 'x') and first_partition.x is not None
    assert hasattr(first_partition, 'edge_index') and first_partition.edge_index is not None
    assert hasattr(first_partition, 'train_mask') and first_partition.train_mask is not None
    print("Partitions are valid PyG Data objects with necessary attributes.")
    print("\n===== Test Passed! =====")
