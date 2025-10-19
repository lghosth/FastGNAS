import torch
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader
from collections import OrderedDict
from typing import Dict, Any

class SimpleBatchManager:

    def __init__(self, k_hop_subgraph: Data, device: torch.device, cache_size: int=5):
        self.graph = k_hop_subgraph
        self.device = device
        self.loader_cache = OrderedDict()
        self.cache_size = cache_size

    def get_loader(self, batch_config: Dict[str, Any], num_layers: int) -> NeighborLoader:
        batch_size = batch_config.get('batch_size', 512)
        cache_key = f'bs_{batch_size}-layers_{num_layers}'
        if cache_key in self.loader_cache:
            self.loader_cache.move_to_end(cache_key)
            return self.loader_cache[cache_key]
        loader = NeighborLoader(data=self.graph, num_neighbors=[-1] * num_layers, batch_size=batch_size, input_nodes=self.graph.train_mask, shuffle=True)
        self.loader_cache[cache_key] = loader
        if len(self.loader_cache) > self.cache_size:
            self.loader_cache.popitem(last=False)
        return loader
if __name__ == '__main__':
    import os
    import sys
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    sys.path.append(project_root)
    from data.data_utils import get_dataset
    from data.partition import PartitionGraph
    print('===== Running Test for batching/simple_batch_manager.py =====')
    print('\n[步骤 1] 创建一个测试用的k-hop子图...')
    DATASET = 'Cora'
    NUM_WORKERS = 2
    MAX_HOPS = 2
    partitions = PartitionGraph.get_partitions(dataset_name=DATASET, num_partitions=NUM_WORKERS, max_hops=MAX_HOPS, strategy='contextual_khop')
    test_subgraph = partitions[0]
    print(f'   -> 测试子图已创建: {test_subgraph.num_nodes} nodes, {test_subgraph.num_edges} edges.')
    print(f'   -> 子图中的训练节点数: {test_subgraph.train_mask.sum().item()}')
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f'  - 使用设备: {device}')
        if device.type == 'cuda':
            test_subgraph = test_subgraph.to(device.index)
        else:
            test_subgraph = test_subgraph.to('cpu')
        batch_manager = SimpleBatchManager(k_hop_subgraph=test_subgraph, device=device, cache_size=3)
        print(f'  - BatchManager初始化成功')
        print(f'    - 缓存大小: {batch_manager.cache_size}')
        print(f'    - 当前缓存项数: {len(batch_manager.loader_cache)}')
    except Exception as e:
        print(f'  初始化BatchManager失败: {e}')
        sys.exit(1)
    print('\n[Test Case 3] Testing loader creation and caching...')
    try:
        test_configs = [({'batch_size': 16}, 2, '小批次2层'), ({'batch_size': 32}, 2, '中等批次2层'), ({'batch_size': 16}, 3, '小批次3层'), ({'batch_size': 32}, 3, '中等批次3层')]
        created_loaders = []
        for i, (batch_config, num_layers, description) in enumerate(test_configs):
            print(f'  - 测试 {description}...')
            loader1 = batch_manager.get_loader(batch_config, num_layers)
            created_loaders.append(loader1)
            print(f'    - 创建成功，缓存项数: {len(batch_manager.loader_cache)}')
            loader2 = batch_manager.get_loader(batch_config, num_layers)
            if loader1 is loader2:
                print(f'    - ✅ 缓存命中成功')
            else:
                print(f'    - ❌ 缓存未命中')
        print(f'  - 最终缓存项数: {len(batch_manager.loader_cache)}')
        print(f'  - 缓存键列表: {list(batch_manager.loader_cache.keys())}')
    except Exception as e:
        print(f'  测试Loader创建失败: {e}')
        sys.exit(1)
    print('\n[Test Case 4] Testing cache size limit...')
    try:
        for i in range(5):
            batch_config = {'batch_size': 8 + i * 4}
            num_layers = 2 + i % 2
            loader = batch_manager.get_loader(batch_config, num_layers)
            print(f"  - 添加配置 {i + 1}: batch_size={batch_config['batch_size']}, layers={num_layers}")
            print(f'    - 当前缓存项数: {len(batch_manager.loader_cache)}')
        assert len(batch_manager.loader_cache) <= batch_manager.cache_size, f'缓存大小 {len(batch_manager.loader_cache)} 超过了限制 {batch_manager.cache_size}'
        print(f'  - ✅ 缓存大小限制测试通过')
    except Exception as e:
        print(f'  测试缓存大小限制失败: {e}')
        sys.exit(1)
    print('\n[Test Case 5] Testing actual data iteration...')
    try:
        batch_config = {'batch_size': 8}
        num_layers = 2
        loader = batch_manager.get_loader(batch_config, num_layers)
        batch_count = 0
        total_nodes = 0
        for batch in loader:
            batch_count += 1
            total_nodes += batch.num_nodes
            print(f'  - 批次 {batch_count}:')
            print(f'    - 节点数: {batch.num_nodes}')
            print(f'    - 边数: {batch.num_edges}')
            print(f'    - 特征形状: {batch.x.shape}')
            print(f'    - 标签形状: {batch.y.shape}')
            if batch_count >= 3:
                break
        print(f'  - 总共处理了 {batch_count} 个批次，{total_nodes} 个节点')
        print(f'  - ✅ 数据迭代测试通过')
    except Exception as e:
        print(f'  测试数据迭代失败: {e}')
        sys.exit(1)
    print('\n[Test Case 6] Testing with real dataset (Cora)...')
    try:
        print('  - 加载Cora数据集...')
        _, cora_data = get_dataset('Cora', root='dataset')
        print('  - 创建Cora子图...')
        sub_nodes = torch.arange(200)
        sub_x = cora_data.x[sub_nodes]
        sub_y = cora_data.y[sub_nodes]
        sub_train_mask = cora_data.train_mask[sub_nodes]
        sub_val_mask = cora_data.val_mask[sub_nodes]
        sub_test_mask = cora_data.test_mask[sub_nodes]
        edge_index = cora_data.edge_index
        mask = (edge_index[0] < 200) & (edge_index[1] < 200)
        sub_edge_index = edge_index[:, mask]
        node_mapping = {old_id.item(): new_id for new_id, old_id in enumerate(sub_nodes)}
        sub_edge_index = torch.tensor([[node_mapping[edge[0].item()], node_mapping[edge[1].item()]] for edge in sub_edge_index.t()], dtype=torch.long).t()
        cora_subgraph = Data(x=sub_x, edge_index=sub_edge_index, y=sub_y, train_mask=sub_train_mask, val_mask=sub_val_mask, test_mask=sub_test_mask)
        print(f'  - Cora子图信息:')
        print(f'    - 节点数: {cora_subgraph.num_nodes}')
        print(f'    - 边数: {cora_subgraph.num_edges}')
        print(f'    - 特征维度: {cora_subgraph.num_node_features}')
        print(f'    - 训练节点数: {cora_subgraph.train_mask.sum().item()}')
        if device.type == 'cuda':
            cora_subgraph = cora_subgraph.to(device.index)
        else:
            cora_subgraph = cora_subgraph.to('cpu')
        real_batch_manager = SimpleBatchManager(k_hop_subgraph=cora_subgraph, device=device, cache_size=2)
        batch_config = {'batch_size': 16}
        num_layers = 2
        loader = real_batch_manager.get_loader(batch_config, num_layers)
        for batch in loader:
            print(f'  - 真实数据批次:')
            print(f'    - 节点数: {batch.num_nodes}')
            print(f'    - 边数: {batch.num_edges}')
            print(f'    - 特征形状: {batch.x.shape}')
            print(f'    - 标签形状: {batch.y.shape}')
            print(f'    - 批次大小: {batch.batch_size}')
            break
        print(f'  - ✅ 真实数据集测试通过')
    except Exception as e:
        print(f'  测试真实数据集失败: {e}')
        import traceback
        traceback.print_exc()
    print('\n===== All tests for simple_batch_manager.py completed successfully! =====')