from typing import Dict, Any, Optional
from datetime import datetime
from dataclasses import dataclass, field

@dataclass
class StateEntry:
    state_dict: Dict[str, Any]
    arch_id: str
    best_accuracy: float = 0.0
    save_time: datetime = field(default_factory=datetime.now)

class StateManager:

    def __init__(self, max_memory_states: int=1000):
        self.max_memory_states = max_memory_states
        self._memory_cache: Dict[str, StateEntry] = {}
        self._stats = {'saves': 0, 'loads': 0, 'evictions': 0}
        print(f'StateManager初始化完成 (纯CPU内存版本):')
        print(f'  - 最大状态数量: {max_memory_states}')
        print(f'  - 淘汰策略: 删除精度最低的状态')

    def save_checkpoint(self, arch_id: str, state_dict: Dict[str, Any], accuracy: float=0.0) -> bool:
        if state_dict is None:
            return False
        try:
            self._memory_cache[arch_id] = StateEntry(state_dict=state_dict, arch_id=arch_id, best_accuracy=accuracy, save_time=datetime.now())
            if len(self._memory_cache) > self.max_memory_states:
                self._evict_lowest_accuracy()
            self._stats['saves'] += 1
            return True
        except Exception as e:
            print(f'保存状态失败: {arch_id} - {e}')
            return False

    def load_checkpoint(self, arch_id: str) -> Optional[Dict[str, Any]]:
        if arch_id in self._memory_cache:
            self._stats['loads'] += 1
            entry = self._memory_cache[arch_id]
            return entry.state_dict
        return None

    def _evict_lowest_accuracy(self) -> None:
        if not self._memory_cache:
            return
        lowest_arch_id = min(self._memory_cache.keys(), key=lambda k: self._memory_cache[k].best_accuracy)
        removed_entry = self._memory_cache.pop(lowest_arch_id)
        self._stats['evictions'] += 1

    def update_accuracy(self, arch_id: str, accuracy: float) -> bool:
        if arch_id in self._memory_cache:
            self._memory_cache[arch_id].best_accuracy = accuracy
            return True
        return False

    def clear_cache(self) -> int:
        count = len(self._memory_cache)
        self._memory_cache.clear()
        return count

    def get_statistics(self) -> Dict[str, Any]:
        accuracies = [entry.best_accuracy for entry in self._memory_cache.values()]
        return {'cached_states': len(self._memory_cache), 'max_states': self.max_memory_states, 'usage_ratio': len(self._memory_cache) / self.max_memory_states, 'saves': self._stats['saves'], 'loads': self._stats['loads'], 'evictions': self._stats['evictions'], 'min_accuracy': min(accuracies) if accuracies else 0.0, 'max_accuracy': max(accuracies) if accuracies else 0.0, 'avg_accuracy': sum(accuracies) / len(accuracies) if accuracies else 0.0}

    def has_checkpoint(self, arch_id: str) -> bool:
        return arch_id in self._memory_cache

    def get_cached_architectures(self) -> list:
        return list(self._memory_cache.keys())