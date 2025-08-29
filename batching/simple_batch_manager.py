# graphnas_dist/batching/simple_batch_manager.py

import torch
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader
from collections import OrderedDict
from typing import Dict, Any

class SimpleBatchManager:
    """
    批处理管理器，严格遵循两级采样流程。
    功能:
    1.  接收一个已经构造好的、较小的 k-hop 子图作为其操作的"世界"。
    2.  利用 PyG 的 NeighborLoader 在这个子图上高效地进行微观采样(mini-batch)。
    3.  (预留扩展) 未来可在此基础上添加更精细的GPU缓存策略。
        当前实现侧重于高效地复用为特定子图配置好的Loader。
    """

    def __init__(self, 
                 k_hop_subgraph: Data, 
                 device: torch.device, 
                 cache_size: int = 5):
        """
        Args:
            k_hop_subgraph (Data): 从全局图中提取的 k-hop 子图，数据应已在目标GPU设备上。
            device (torch.device): Worker 绑定的GPU设备。
            cache_size (int): Loader实例的缓存大小。
        """
        # 注意：这里的 self.graph 就是那个 k-hop 子图，而不是全局图
        self.graph = k_hop_subgraph
        self.device = device
        self.loader_cache = OrderedDict()
        self.cache_size = cache_size

    def get_loader(self, batch_config: Dict[str, Any], num_layers: int) -> NeighborLoader:
        """
        为当前 k-hop 子图获取一个可迭代的 mini-batch loader。

        Args:
            batch_config (Dict[str, Any]): 包含 'batch_size' 等配置。
            num_layers (int): 模型层数，决定了在子图内采样的邻居跳数。

        Returns:
            NeighborLoader: 一个配置好的、可用于生成 mini-batch 的加载器。
        """
        batch_size = batch_config.get("batch_size", 512)
        
        # 缓存键由 batch_size 和层数共同决定
        cache_key = f"bs_{batch_size}-layers_{num_layers}"

        if cache_key in self.loader_cache:
            self.loader_cache.move_to_end(cache_key)
            return self.loader_cache[cache_key]

        # 在这个 k_hop_subgraph 内部进行采样
        # 注意: input_nodes 现在是子图内的训练节点掩码
        loader = NeighborLoader(
            data=self.graph,
            num_neighbors=[-1] * num_layers,
            batch_size=batch_size,
            input_nodes=self.graph.train_mask, # 使用子图的训练掩码
            shuffle=True
        )

        self.loader_cache[cache_key] = loader
        if len(self.loader_cache) > self.cache_size:
            self.loader_cache.popitem(last=False)
            
        return loader

# ====================
#  独立的测试模块
# ====================
if __name__ == '__main__':
    print("===== Running Test for simple_batch_manager.py =====")
    
    # 导入必要的模块
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    from data.data_utils import get_dataset
    from data.partition import partition_training_graph
    
    # --- 测试用例 1: 创建模拟的k-hop子图 ---
    print("\n[Test Case 1] Creating a simulated k-hop subgraph...")
    try:
        # 创建一个简单的模拟图数据
        num_nodes = 100
        num_features = 16
        num_classes = 3
        
        # 创建随机节点特征
        x = torch.randn(num_nodes, num_features)
        
        # 创建随机边（确保图是连通的）
        edge_index = torch.randint(0, num_nodes, (2, num_nodes * 2))
        # 移除自环
        edge_index = edge_index[:, edge_index[0] != edge_index[1]]
        
        # 创建随机标签
        y = torch.randint(0, num_classes, (num_nodes,))
        
        # 创建训练掩码（前30个节点用于训练）
        train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        train_mask[:30] = True
        
        # 创建验证和测试掩码
        val_mask = torch.zeros(num_nodes, dtype=torch.bool)
        val_mask[30:50] = True
        
        test_mask = torch.zeros(num_nodes, dtype=torch.bool)
        test_mask[50:70] = True
        
        # 创建Data对象
        mock_subgraph = Data(
            x=x,
            edge_index=edge_index,
            y=y,
            train_mask=train_mask,
            val_mask=val_mask,
            test_mask=test_mask
        )
        
        print(f"  - 模拟子图创建成功:")
        print(f"    - 节点数: {mock_subgraph.num_nodes}")
        print(f"    - 边数: {mock_subgraph.num_edges}")
        print(f"    - 特征维度: {mock_subgraph.x.shape[1] if mock_subgraph.x is not None else 'None'}")
        print(f"    - 训练节点数: {mock_subgraph.train_mask.sum().item() if mock_subgraph.train_mask is not None else 0}")
        print(f"    - 验证节点数: {mock_subgraph.val_mask.sum().item() if mock_subgraph.val_mask is not None else 0}")
        print(f"    - 测试节点数: {mock_subgraph.test_mask.sum().item() if mock_subgraph.test_mask is not None else 0}")
        
    except Exception as e:
        print(f"  创建模拟子图失败: {e}")
        sys.exit(1)
    
    # --- 测试用例 2: 初始化SimpleBatchManager ---
    print("\n[Test Case 2] Initializing SimpleBatchManager...")
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"  - 使用设备: {device}")
        
        # 将数据移动到设备上
        if device.type == 'cuda':
            mock_subgraph = mock_subgraph.to(device.index)
        else:
            mock_subgraph = mock_subgraph.to('cpu')
            
        batch_manager = SimpleBatchManager(
            k_hop_subgraph=mock_subgraph,
            device=device,
            cache_size=3
        )
        
        print(f"  - BatchManager初始化成功")
        print(f"    - 缓存大小: {batch_manager.cache_size}")
        print(f"    - 当前缓存项数: {len(batch_manager.loader_cache)}")
        
    except Exception as e:
        print(f"  初始化BatchManager失败: {e}")
        sys.exit(1)
    
    # --- 测试用例 3: 测试Loader创建和缓存功能 ---
    print("\n[Test Case 3] Testing loader creation and caching...")
    try:
        # 测试不同的配置组合
        test_configs = [
            ({'batch_size': 16}, 2, "小批次2层"),
            ({'batch_size': 32}, 2, "中等批次2层"),
            ({'batch_size': 16}, 3, "小批次3层"),
            ({'batch_size': 32}, 3, "中等批次3层"),
        ]
        
        created_loaders = []
        
        for i, (batch_config, num_layers, description) in enumerate(test_configs):
            print(f"  - 测试 {description}...")
            
            # 第一次创建
            loader1 = batch_manager.get_loader(batch_config, num_layers)
            created_loaders.append(loader1)
            
            print(f"    - 创建成功，缓存项数: {len(batch_manager.loader_cache)}")
            
            # 第二次获取（应该从缓存中获取）
            loader2 = batch_manager.get_loader(batch_config, num_layers)
            
            # 验证是否是同一个对象（缓存命中）
            if loader1 is loader2:
                print(f"    - ✅ 缓存命中成功")
            else:
                print(f"    - ❌ 缓存未命中")
        
        print(f"  - 最终缓存项数: {len(batch_manager.loader_cache)}")
        print(f"  - 缓存键列表: {list(batch_manager.loader_cache.keys())}")
        
    except Exception as e:
        print(f"  测试Loader创建失败: {e}")
        sys.exit(1)
    
    # --- 测试用例 4: 测试缓存大小限制 ---
    print("\n[Test Case 4] Testing cache size limit...")
    try:
        # 创建更多不同的配置来测试缓存淘汰
        for i in range(5):
            batch_config = {'batch_size': 8 + i * 4}
            num_layers = 2 + (i % 2)
            
            loader = batch_manager.get_loader(batch_config, num_layers)
            print(f"  - 添加配置 {i+1}: batch_size={batch_config['batch_size']}, layers={num_layers}")
            print(f"    - 当前缓存项数: {len(batch_manager.loader_cache)}")
        
        # 验证缓存大小不超过限制
        assert len(batch_manager.loader_cache) <= batch_manager.cache_size, \
            f"缓存大小 {len(batch_manager.loader_cache)} 超过了限制 {batch_manager.cache_size}"
        
        print(f"  - ✅ 缓存大小限制测试通过")
        
    except Exception as e:
        print(f"  测试缓存大小限制失败: {e}")
        sys.exit(1)
    
    # --- 测试用例 5: 测试实际数据迭代 ---
    print("\n[Test Case 5] Testing actual data iteration...")
    try:
        # 使用一个简单的配置
        batch_config = {'batch_size': 8}
        num_layers = 2
        
        loader = batch_manager.get_loader(batch_config, num_layers)
        
        # 迭代几个批次
        batch_count = 0
        total_nodes = 0
        
        for batch in loader:
            batch_count += 1
            total_nodes += batch.num_nodes
            
            print(f"  - 批次 {batch_count}:")
            print(f"    - 节点数: {batch.num_nodes}")
            print(f"    - 边数: {batch.num_edges}")
            print(f"    - 特征形状: {batch.x.shape}")
            print(f"    - 标签形状: {batch.y.shape}")
            
            # 只测试前3个批次
            if batch_count >= 3:
                break
        
        print(f"  - 总共处理了 {batch_count} 个批次，{total_nodes} 个节点")
        print(f"  - ✅ 数据迭代测试通过")
        
    except Exception as e:
        print(f"  测试数据迭代失败: {e}")
        sys.exit(1)
    
    # --- 测试用例 6: 使用真实数据集测试 ---
    print("\n[Test Case 6] Testing with real dataset (Cora)...")
    try:
        # 加载Cora数据集
        print("  - 加载Cora数据集...")
        _, cora_data = get_dataset('Cora', root='dataset')
        
        # 创建一个小的子图用于测试
        print("  - 创建Cora子图...")
        # 选择前200个节点作为子图
        sub_nodes = torch.arange(200)
        sub_x = cora_data.x[sub_nodes]
        sub_y = cora_data.y[sub_nodes]
        sub_train_mask = cora_data.train_mask[sub_nodes]
        sub_val_mask = cora_data.val_mask[sub_nodes]
        sub_test_mask = cora_data.test_mask[sub_nodes]
        
        # 提取子图的边（只保留两个端点都在子图中的边）
        edge_index = cora_data.edge_index
        mask = (edge_index[0] < 200) & (edge_index[1] < 200)
        sub_edge_index = edge_index[:, mask]
        
        # 重新标记节点ID
        node_mapping = {old_id.item(): new_id for new_id, old_id in enumerate(sub_nodes)}
        sub_edge_index = torch.tensor([[node_mapping[edge[0].item()], node_mapping[edge[1].item()]] 
                                     for edge in sub_edge_index.t()], dtype=torch.long).t()
        
        cora_subgraph = Data(
            x=sub_x,
            edge_index=sub_edge_index,
            y=sub_y,
            train_mask=sub_train_mask,
            val_mask=sub_val_mask,
            test_mask=sub_test_mask
        )
        
        print(f"  - Cora子图信息:")
        print(f"    - 节点数: {cora_subgraph.num_nodes}")
        print(f"    - 边数: {cora_subgraph.num_edges}")
        print(f"    - 特征维度: {cora_subgraph.num_node_features}")
        print(f"    - 训练节点数: {cora_subgraph.train_mask.sum().item()}")
        
        # 将Cora子图移动到设备上
        if device.type == 'cuda':
            cora_subgraph = cora_subgraph.to(device.index)
        else:
            cora_subgraph = cora_subgraph.to('cpu')
            
        # 创建BatchManager
        real_batch_manager = SimpleBatchManager(
            k_hop_subgraph=cora_subgraph,
            device=device,
            cache_size=2
        )
        
        # 测试真实数据
        batch_config = {'batch_size': 16}
        num_layers = 2
        
        loader = real_batch_manager.get_loader(batch_config, num_layers)
        
        # 测试一个批次
        for batch in loader:
            print(f"  - 真实数据批次:")
            print(f"    - 节点数: {batch.num_nodes}")
            print(f"    - 边数: {batch.num_edges}")
            print(f"    - 特征形状: {batch.x.shape}")
            print(f"    - 标签形状: {batch.y.shape}")
            print(f"    - 批次大小: {batch.batch_size}")
            break
        
        print(f"  - ✅ 真实数据集测试通过")
        
    except Exception as e:
        print(f"  测试真实数据集失败: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n===== All tests for simple_batch_manager.py completed successfully! =====")