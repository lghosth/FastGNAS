import os
import torch
import shutil
from typing import Tuple
from torch_geometric.datasets import Planetoid, Amazon, Reddit, Flickr
from ogb.nodeproppred import PygNodePropPredDataset
from torch_geometric.data import Dataset, Data
import torch_geometric.transforms as T
SUPPORTED_DATASETS = ['Cora', 'PubMed', 'Amazon-Computers', 'Reddit', 'ogbn-products', 'ogbn-arxiv', 'Flickr']
DATASET_SPLIT_SEED = 56

def get_dataset(name: str, root: str='dataset', force_reload: bool=False, train_ratio: float=0.0) -> Tuple[Dataset, Data]:
    import random
    import numpy as np
    _torch_rng_state = torch.get_rng_state()
    _random_state = random.getstate()
    _np_state = np.random.get_state()
    torch.manual_seed(DATASET_SPLIT_SEED)
    random.seed(DATASET_SPLIT_SEED)
    np.random.seed(DATASET_SPLIT_SEED)
    if name not in SUPPORTED_DATASETS:
        raise ValueError(f"不支持的数据集: '{name}'. 支持列表: {SUPPORTED_DATASETS}")
    dataset_path = os.path.join(root, name)
    if force_reload and os.path.exists(dataset_path):
        print(f'检测到 force_reload=True, 正在删除旧数据: {dataset_path}')
        shutil.rmtree(dataset_path)
    print(f"正在从 '{dataset_path}' 加载/下载数据集 '{name}'...")
    if name in ['Cora', 'Citeseer', 'PubMed']:
        dataset = Planetoid(root=root, name=name, transform=T.Compose([T.NormalizeFeatures(), T.RandomNodeSplit(split='train_rest', num_val=0.1, num_test=0.1)]))
    elif name in ['Amazon-Computers', 'Photo']:
        if name == 'Amazon-Computers':
            name = f'computers'
        dataset = Amazon(root=root, name=name, transform=T.Compose([T.NormalizeFeatures(), T.RandomNodeSplit(split='train_rest', num_val=0.1, num_test=0.1)]))
    elif name == 'Reddit':
        root = os.path.join(root, name)
        dataset = Reddit(root=root, transform=T.NormalizeFeatures())
    elif name == 'Flickr':
        root = os.path.join(root, name)
        dataset = Flickr(root=root, transform=T.NormalizeFeatures())
    elif name.startswith('ogbn-'):
        dataset = PygNodePropPredDataset(name=name, root=root, transform=T.ToUndirected())
    else:
        raise NotImplementedError(f"数据集 '{name}' 的加载逻辑尚未实现")
    data = dataset[0]
    if hasattr(data, 'y') and isinstance(data.y, torch.Tensor):
        if data.y.dim() > 1:
            data.y = data.y.view(-1)
        data.y = data.y.to(torch.long)
    if name.startswith('ogbn-'):
        split_idx = dataset.get_idx_split()

        def _build_mask(indices_key: str) -> torch.Tensor:
            split_tensor = split_idx.get(indices_key)
            if split_tensor is None:
                raise KeyError(f"OGB 数据集缺少 '{indices_key}' 划分索引")
            if not torch.is_tensor(split_tensor):
                split_tensor = torch.tensor(split_tensor, dtype=torch.long)
            mask = torch.zeros(data.num_nodes, dtype=torch.bool)
            mask[split_tensor] = True
            return mask
        data.train_mask = _build_mask('train')
        data.val_mask = _build_mask('valid')
        data.test_mask = _build_mask('test')
    print(f"'{name}' 数据集加载成功!")
    torch.set_rng_state(_torch_rng_state)
    random.setstate(_random_state)
    np.random.set_state(_np_state)
    return (dataset, data)
if __name__ == '__main__':
    print('===== Running Test for data_utils.py =====')
    print("\n[Test ogbn-arxiv 1] Loading 'Cora' dataset...")
    try:
        cora_dataset, cora_data = get_dataset(name='ogbn-arxiv', root='dataset')
        print('\n--- ogbn-arxiv Dataset Info ---')
        print(f'  - Dataset object: {cora_dataset}')
        print(f'  - Graph data object: {cora_data}')
        print(f'  - Number of nodes: {cora_data.num_nodes}')
        print(f'  - Number of edges: {cora_data.num_edges}')
        print(f'  - Number of features: {cora_data.num_node_features}')
        print(f'  - Number of classes: {cora_dataset.num_classes}')
        print(f'  - Train mask sum: {cora_data.train_mask.sum().item()}')
        print(f'  - Validation mask sum: {cora_data.val_mask.sum().item()}')
        print(f'  - Test mask sum: {cora_data.test_mask.sum().item()}')
        print('-' * 25)
    except Exception as e:
        print(f"加载 'ogbn-arxiv' 失败: {e}")
    print("\n[Test Case 2] Loading 'ogbn-products' dataset...")
    try:
        pubmed_dataset, pubmed_data = get_dataset(name='ogbn-products', root='dataset')
        print('\n--- ogbn-products Dataset Info ---')
        print(f'  - Dataset object: {pubmed_dataset}')
        print(f'  - Graph data object: {pubmed_data}')
        print(f'  - Number of nodes: {pubmed_data.num_nodes}')
        print(f'  - Number of edges: {pubmed_data.num_edges}')
        print(f'  - Number of features: {pubmed_data.num_node_features}')
        print(f'  - Number of classes: {pubmed_dataset.num_classes}')
        print(f'  - Train mask sum: {pubmed_data.train_mask.sum().item()}')
        print(f'  - Validation mask sum: {pubmed_data.val_mask.sum().item()}')
        print(f'  - Test mask sum: {pubmed_data.test_mask.sum().item()}')
        print('-' * 25)
    except Exception as e:
        print(f"加载 'ogbn-products' 失败: {e}")