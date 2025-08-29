# graphnas_dist/data/data_utils.py

import os
import torch
import shutil
from typing import Tuple
from torch_geometric.datasets import Planetoid, Amazon
from torch_geometric.data import Dataset, Data
import torch_geometric.transforms as T

# 支持的数据集列表
SUPPORTED_DATASETS = ['Cora', 'Citeseer', 'PubMed', 'Computers', 'Photo']

def get_dataset(name: str, 
                root: str = "dataset", 
                force_reload: bool = False) -> Tuple[Dataset, Data]:
    """
    自动下载并加载指定的图数据集。

    如果本地不存在，则从网络下载。默认存储在项目根目录下的 'dataset' 文件夹。

    Args:
        name (str): 数据集名称 (e.g., 'Cora', 'Citeseer', 'Computers').
        root (str): 数据集存储的根目录.
        force_reload (bool): 如果为True，将删除已存在的数据集并重新下载.

    Returns:
        Tuple[Dataset, Data]: PyG的数据集对象和图数据对象.
    """
    if name not in SUPPORTED_DATASETS:
        raise ValueError(f"不支持的数据集: '{name}'. 支持列表: {SUPPORTED_DATASETS}")

    # 构造数据集的完整路径
    dataset_path = os.path.join(root, name)

    if force_reload and os.path.exists(dataset_path):
        print(f"检测到 force_reload=True, 正在删除旧数据: {dataset_path}")
        shutil.rmtree(dataset_path)

    print(f"正在从 '{dataset_path}' 加载/下载数据集 '{name}'...")

    if name in ['Cora', 'Citeseer', 'PubMed']:
        # Planetoid数据集会自动在 root/name 目录下创建文件
        dataset = Planetoid(root=root, name=name, transform=T.NormalizeFeatures())
    elif name in ['Computers', 'Photo']:
        # Amazon数据集同样遵循此模式
        dataset = Amazon(root=root, name=name, transform=T.NormalizeFeatures())
    else:
        # 为未来其他数据集类型预留
        raise NotImplementedError(f"数据集 '{name}' 的加载逻辑尚未实现。")
        
    data = dataset[0]  # PyG 数据集通常将单个图存储在第一个元素

    print(f"'{name}' 数据集加载成功!")
    return dataset, data  # type: ignore

if __name__ == '__main__':
    print("===== Running Test for data_utils.py =====")
    
    # --- 测试用例 1: 加载 Cora 数据集 ---
    print("\n[Test Case 1] Loading 'Cora' dataset...")
    try:
        cora_dataset, cora_data = get_dataset(name='Cora', root='dataset')
        print("\n--- Cora Dataset Info ---")
        print(f"  - Dataset object: {cora_dataset}")
        print(f"  - Graph data object: {cora_data}")
        print(f"  - Number of nodes: {cora_data.num_nodes}")
        print(f"  - Number of edges: {cora_data.num_edges}")
        print(f"  - Number of features: {cora_data.num_node_features}")
        print(f"  - Number of classes: {cora_dataset.num_classes}")
        print(f"  - Train mask sum: {cora_data.train_mask.sum().item()}")
        print(f"  - Validation mask sum: {cora_data.val_mask.sum().item()}")
        print(f"  - Test mask sum: {cora_data.test_mask.sum().item()}")

        # 输出cora的训练节点列表
        print(f"  - Train nodes: {cora_data.train_mask.nonzero().squeeze().tolist()}")
        print("-" * 25)
    except Exception as e:
        print(f"加载 'Cora' 失败: {e}")

    # --- 测试用例 2: 加载 Pubmed 数据集 ---
    print("\n[Test Case 2] Loading 'PubMed' dataset...")
    try:
        pubmed_dataset, pubmed_data = get_dataset(name='PubMed', root='dataset')
        print("\n--- PubMed Dataset Info ---")
        print(f"  - Dataset object: {pubmed_dataset}")
        print(f"  - Graph data object: {pubmed_data}")
        print(f"  - Number of nodes: {pubmed_data.num_nodes}")
        print(f"  - Number of edges: {pubmed_data.num_edges}")
        print(f"  - Number of features: {pubmed_data.num_node_features}")
        print(f"  - Number of classes: {pubmed_dataset.num_classes}")
        print(f"  - Train mask sum: {pubmed_data.train_mask.sum().item()}")
        print(f"  - Validation mask sum: {pubmed_data.val_mask.sum().item()}")
        print(f"  - Test mask sum: {pubmed_data.test_mask.sum().item()}")
        print("-" * 25)
    except Exception as e:
        print(f"加载 'PubMed' 失败: {e}")

    print("\n===== Test for data_utils.py Finished =====")