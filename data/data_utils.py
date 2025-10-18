# graphnas_dist/data/data_utils.py

import os
import torch
import shutil
from typing import Tuple
from torch_geometric.datasets import Planetoid, Amazon
from torch_geometric.data import Dataset, Data
import torch_geometric.transforms as T

# List of supported datasets
SUPPORTED_DATASETS = ['Cora', 'Citeseer', 'PubMed', 'Computers', 'Photo']

def get_dataset(name: str, 
                root: str = "dataset", 
                force_reload: bool = False) -> Tuple[Dataset, Data]:
    """
    Automatically downloads and loads the specified graph dataset.

    Downloads from the web if not found locally. Defaults to storing in the 'dataset'
    folder in the project root.

    Args:
        name (str): Name of the dataset (e.g., 'Cora', 'Citeseer').
        root (str): Root directory for storing the dataset.
        force_reload (bool): If True, deletes and re-downloads the dataset if it exists.

    Returns:
        Tuple[Dataset, Data]: The PyG dataset object and the graph data object.
    """
    if name not in SUPPORTED_DATASETS:
        raise ValueError(f"Unsupported dataset: '{name}'. Supported: {SUPPORTED_DATASETS}")

    # Construct the full path to the dataset
    dataset_path = os.path.join(root, name)

    if force_reload and os.path.exists(dataset_path):
        print(f"force_reload=True, removing old data: {dataset_path}")
        shutil.rmtree(dataset_path)

    print(f"Loading/Downloading dataset '{name}' from '{dataset_path}'...")

    if name in ['Cora', 'Citeseer', 'PubMed']:
        # Planetoid datasets are stored in the root/name directory
        dataset = Planetoid(root=root, name=name, transform=T.NormalizeFeatures())
    elif name in ['Computers', 'Photo']:
        # Amazon datasets follow the same pattern
        dataset = Amazon(root=root, name=name, transform=T.NormalizeFeatures())
    else:
        # Placeholder for other dataset types
        raise NotImplementedError(f"Loading logic for dataset '{name}' is not implemented.")
        
    data = dataset[0]  # PyG datasets typically store a single graph in the first element

    print(f"Dataset '{name}' loaded successfully!")
    return dataset, data

if __name__ == '__main__':
    print("===== Running Test for data_utils.py =====")
    

    print("\n[Test Case 1] Loading 'Cora' dataset...")
    try:
        cora_dataset, cora_data = get_dataset(name='Cora', root='dataset')
        print("\n--- Cora Dataset Info ---")
        print(f"  - Dataset object: {cora_dataset}")
        print(f"  - Graph data object: {cora_data}")
        print(f"  - Num nodes: {cora_data.num_nodes}")
        print(f"  - Num edges: {cora_data.num_edges}")
        print(f"  - Num features: {cora_data.num_node_features}")
        print(f"  - Num classes: {cora_dataset.num_classes}")
        print(f"  - Train mask sum: {cora_data.train_mask.sum().item()}")
        print(f"  - Val mask sum: {cora_data.val_mask.sum().item()}")
        print(f"  - Test mask sum: {cora_data.test_mask.sum().item()}")
        print(f"  - Train nodes: {cora_data.train_mask.nonzero().squeeze().tolist()}")
        print("-" * 25)
    except Exception as e:
        print(f"Failed to load 'Cora': {e}")


    print("\n[Test Case 2] Loading 'PubMed' dataset...")
    try:
        pubmed_dataset, pubmed_data = get_dataset(name='PubMed', root='dataset')
        print("\n--- PubMed Dataset Info ---")
        print(f"  - Dataset object: {pubmed_dataset}")
        print(f"  - Graph data object: {pubmed_data}")
        print(f"  - Num nodes: {pubmed_data.num_nodes}")
        print(f"  - Num edges: {pubmed_data.num_edges}")
        print(f"  - Num features: {pubmed_data.num_node_features}")
        print(f"  - Num classes: {pubmed_dataset.num_classes}")
        print(f"  - Train mask sum: {pubmed_data.train_mask.sum().item()}")
        print(f"  - Val mask sum: {pubmed_data.val_mask.sum().item()}")
        print(f"  - Test mask sum: {pubmed_data.test_mask.sum().item()}")
        print("-" * 25)
    except Exception as e:
        print(f"Failed to load 'PubMed': {e}")

    print("\n===== Test for data_utils.py Finished =====")
