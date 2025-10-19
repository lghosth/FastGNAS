from __future__ import annotations
import os
import random
import sys
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import torch
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
from core.controller import ControllerManager
from core.balance_controller import BalanceController

class RLBalancedSearcher:

    def __init__(self, controller_cfg: Optional[Dict[str, Any]]=None, balance_cfg: Optional[Dict[str, Any]]=None, load_predictor: Optional[Any]=None) -> None:
        controller_cfg = dict(controller_cfg or {})
        seed = controller_cfg.pop('random_seed', None)
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
        self.controller = ControllerManager(**controller_cfg)
        self.balance_controller: Optional[BalanceController]
        if balance_cfg is None:
            self.balance_controller = None
        else:
            self.balance_controller = BalanceController(controller_manager=self.controller, load_predictor=load_predictor, **balance_cfg)

    def sample_architectures(self, num_samples: int) -> List[Dict[str, Any]]:
        return self.controller.sample_architectures(num_samples)

    def update_rewards(self, rewards: Dict[str, float], default_reward: float=0.0) -> None:
        self.controller.train_step(rewards, default_reward)

    def best_architecture(self) -> Optional[Tuple[Dict[str, Any], float]]:
        best_spec = getattr(self.controller, 'best_arch_spec', None)
        if best_spec is None:
            return None
        return (best_spec, self.controller.best_arch_accuracy)

    def generate_balanced_batch(self) -> List[Dict[str, Any]]:
        if not self.balance_controller:
            raise RuntimeError('未提供 balance_cfg，无法进行负载均衡采样')
        return self.balance_controller.generate_balanced_architectures()

    def has_balancer(self) -> bool:
        return self.balance_controller is not None

def main() -> int:
    import argparse
    from pprint import pprint
    parser = argparse.ArgumentParser(description='RLBalancedSearcher 测试驱动')
    parser.add_argument('--samples', type=int, default=3, help='采样架构数量')
    parser.add_argument('--use-balancer', action='store_true', help='启用负载均衡批量生成测试')
    parser.add_argument('--balanced-batch-size', type=int, default=4, help='负载均衡批量大小 (仅在启用负载均衡时生效)')
    parser.add_argument('--num-workers', type=int, default=2, help='模拟的 worker 数量')
    args = parser.parse_args()
    controller_cfg = {'search_strategy': 'layer_wise', 'num_layers': 2, 'learning_rate': 0.00035, 'entropy_weight': 0.0001}
    balance_cfg = None
    if args.use_balancer:
        balance_cfg = {'batch_size': args.balanced_batch_size, 'min_architectures': max(1, args.balanced_batch_size // 2), 'tolerance': 0.15, 'num_workers': args.num_workers}
    searcher = RLBalancedSearcher(controller_cfg=controller_cfg, balance_cfg=balance_cfg)
    print('\n=== 采样测试 ===')
    sampled = searcher.sample_architectures(args.samples)
    pprint(sampled)
    if args.use_balancer:
        print('\n=== 负载均衡批量测试 ===')
        balanced_batch = searcher.generate_balanced_batch()
        pprint(balanced_batch)
    return 0
if __name__ == '__main__':
    raise SystemExit(main())