# graphnas_dist/trainer/worker.py (修改版)

import torch
import torch.nn.functional as F
from typing import Dict, Any, Tuple
import ray

# 导入项目模块
from trainer.builder import ModelBuilder
from batching.hierarchical_cache_manager import HierarchicalCacheManager
from utils.logger import GraphNASLogger
from torch_geometric.data import Data


@ray.remote(num_gpus=1)
class TrainerWorker:
    """
    Worker是一个有状态的数据所有者。
    - 初始化时接收一个固定的数据分区。
    - 接收模型参数，在本地分区上训练，然后返回更新后的参数。
    """
    
    def __init__(self, worker_id: int, gpu_id: int, data_partition: Data):
        self.worker_id = worker_id
        self.gpu_id = gpu_id
        import os
        print(f"[Worker {worker_id}] CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES')}")
        print(f"[Worker {worker_id}] torch.cuda.is_available(): {torch.cuda.is_available()}")
        print(f"[Worker {worker_id}] ray.get_gpu_ids(): {ray.get_gpu_ids() if hasattr(ray, 'get_gpu_ids') else 'N/A'}")
        # print(f"[Worker {worker_id}] LD_LIBRARY_PATH: {os.environ.get('LD_LIBRARY_PATH')}")
        # 在Ray环境中，使用Ray分配的GPU设备
        if torch.cuda.is_available():
            # 获取Ray分配的GPU设备ID
            self.device = torch.device('cuda:0')
            torch.cuda.set_device(self.device) # 确保后续所有CUDA操作都在此设备上
        else:
            self.device = torch.device('cpu')
        
        # 将其拥有的数据分区永久加载到GPU
        self.data_partition = data_partition.to(self.device)
        
        # 为其数据分区初始化一个固定的BatchManager
        # self.batch_manager = SimpleBatchManager(self.data_partition, self.device)

        self.batch_manager = HierarchicalCacheManager(
            subgraph=data_partition,
            device=self.device,
            max_cache_pools=10, # 缓存池数量
            max_pool_size=50 # 每个缓存池的大小
        )
        
        self.logger = GraphNASLogger(name=f"worker_{worker_id}", track_gpu=True)
        
        # 只在CUDA可用时才设置设备
        # if torch.cuda.is_available():
        #     torch.cuda.set_device(self.device)

        # 一个可重用的模型占位符，避免重复创建
        self.model = None 
        self.current_arch_spec = None

        self.logger.logger.info(f"Worker {worker_id} on GPU {gpu_id} initialized with data partition: {self.data_partition}")

    def train_model_on_partition(self, 
                                 arch_spec: Dict[str, Any], 
                                 state_dict: Dict, 
                                 config: Dict[str, Any]
                                ) -> Tuple[float, Dict]:
        """
        接收一个模型，在自己的数据分区上训练，然后返回结果。
        """
        # 1. 检查架构是否改变，如果改变则重建模型
        if self.model is None or str(arch_spec) != str(self.current_arch_spec):
            # Add a check to ensure y is a tensor before calling .max()
            if isinstance(self.data_partition.y, torch.Tensor):
                output_dim = int(self.data_partition.y.max().item()) + 1
            else:
                # Handle cases where y is not a tensor, e.g., set a default or raise an error
                raise TypeError("data_partition.y must be a torch.Tensor to determine output_dim")

            self.model = ModelBuilder.build_model(
                arch_spec, 
                self.data_partition.num_node_features,
                output_dim,
                config
            ).to(self.device)
            self.current_arch_spec = arch_spec

        # 2. 如果提供了state_dict，则加载它 (实现模型迁移)
        if state_dict:
            self.model.load_state_dict(state_dict)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=config.get('lr', 0.01))
        
        # 3. 获取与模型层数匹配的微观采样器
        # num_layers = self._get_num_layers_from_arch(arch_spec)
        # train_loader = self.batch_manager.get_loader(config, num_layers)
        # train_loader = self.batch_manager.get_batch(config.get('batch_size', 16), num_layers)

        # 4. 训练循环
        # for epoch in range(1, config.get('epochs_per_rotation', 10) + 1):
        #     self.model.train()
        #     for batch in train_loader:
        #         optimizer.zero_grad()
        #         out = self.model(batch.x, batch.edge_index)
        #         loss = F.cross_entropy(out[:batch.batch_size], batch.y[:batch.batch_size])
        #         loss.backward()
        #         optimizer.step()
        

        # 4. 训练循环
        # (假设 arch_spec 和 config 已经从 train_model_on_partition 方法的参数中获得)

        # a. 从配置中解析出必要的参数
        epochs = config.get('epochs_per_rotation', 5) # 每个轮转包含的epoch数
        requested_bs = config.get('batch_size', 64)
        # 假设每个epoch包含的训练步数，例如2步
        steps_per_epoch = config.get('steps_per_epoch', 1) 
        # 从模型架构中获取层数
        requested_layers = self._get_num_layers_from_arch(arch_spec)

        for epoch in range(1, epochs + 1):
            self.model.train()
            # b. 内部循环现在控制“训练步数”，而不是迭代加载器
            for step in range(steps_per_epoch):
                # c. 在每个训练步骤中，主动从新管理器获取一个batch
                batch = self.batch_manager.get_batch(
                    requested_bs=requested_bs,
                    requested_layers=requested_layers
                )
                # d. 检查返回的batch是否有效
                if batch is None:
                # 如果缓存和采样都无法提供batch，可以跳过此步骤
                    print(f"警告: 在 epoch {epoch}, step {step+1} 未能获取到batch。")
                    continue

                # e. 后续的训练逻辑保持不变
                optimizer.zero_grad()
                out = self.model(batch.x, batch.edge_index)

                if isinstance(out, torch.Tensor) and isinstance(batch.y, torch.Tensor):
                    loss = F.cross_entropy(out[:batch.batch_size], batch.y[:batch.batch_size])
                    loss.backward()
                    optimizer.step()
                else:
                    print(f"警告: 在 epoch {epoch}, step {step+1} 的输出或标签不是张量。")




        # 5. 评估并返回结果，增加测试
        val_acc, _ = self._evaluate(self.model)
        # 将更新后的模型参数移到CPU，准备传输
        updated_state_dict = {k: v.cpu() for k, v in self.model.state_dict().items()}
        
        return val_acc, updated_state_dict

    def _get_num_layers_from_arch(self, arch_spec: Dict[str, Any]) -> int:
        """
        从架构规范中提取层数，支持layer_wise和cell_based两种格式
        """
        if 'layers' in arch_spec:
            # layer_wise格式
            return len(arch_spec['layers'])
        elif 'cell_spec' in arch_spec:
            # cell_based格式
            return arch_spec.get('num_cells_to_stack', 2)
        else:
            # 默认返回2层
            print(f"警告: 未知的架构格式 {arch_spec.keys()}，使用默认层数2")
            return 2

    @torch.no_grad()
    def _evaluate(self, model) -> Tuple[float, float]:
        """在本地分区的验证集上评估。"""
        model.eval()
        # 注意：评估是在整个本地分区上进行的
        out = model(self.data_partition.x, self.data_partition.edge_index)
        # ... (评估逻辑与之前相同，但操作对象是self.data_partition)
        pred = out.argmax(dim=1)

        # Ensure val_mask and y are Tensors before using them
        if isinstance(self.data_partition.val_mask, torch.Tensor) and \
           isinstance(self.data_partition.y, torch.Tensor):
            val_mask_sum = self.data_partition.val_mask.sum().item()
            if val_mask_sum == 0: return 0.0, 0.0
            val_correct = (pred[self.data_partition.val_mask] == self.data_partition.y[self.data_partition.val_mask]).sum()
            val_acc = int(val_correct) / val_mask_sum
            return val_acc, 0.0 
        else:
            return 0.0, 0.0