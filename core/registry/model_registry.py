from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
import json

@dataclass
class ModelEntry:
    arch_id: str
    arch_spec: Dict[str, Any]
    best_accuracy: float = 0.0
    last_updated: Optional[datetime] = None
    training_epochs: int = 0
    load: float = 1.0
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.last_updated is None:
            self.last_updated = datetime.now()
        if self.metadata is None:
            self.metadata = {}

    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        if isinstance(result['last_updated'], datetime):
            result['last_updated'] = result['last_updated'].isoformat()
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ModelEntry':
        if isinstance(data.get('last_updated'), str):
            data['last_updated'] = datetime.fromisoformat(data['last_updated'])
        return cls(**data)

class ModelRegistry:

    def __init__(self):
        self._registry: Dict[str, ModelEntry] = {}
        self._creation_time = datetime.now()
        print('ModelRegistry初始化完成')

    def register(self, arch_id: str, arch_spec: Dict[str, Any], load: float=1.0, metadata: Optional[Dict[str, Any]]=None) -> None:
        if arch_id in self._registry:
            print(f'警告: 架构 {arch_id} 已存在，将更新')
        entry = ModelEntry(arch_id=arch_id, arch_spec=arch_spec, load=load, metadata=metadata or {})
        self._registry[arch_id] = entry

    def update_performance(self, arch_id: str, accuracy: float, epochs: int=1) -> bool:
        if arch_id not in self._registry:
            print(f'警告: 架构 {arch_id} 不存在，无法更新性能')
            return False
        entry = self._registry[arch_id]
        old_accuracy = entry.best_accuracy
        if accuracy > entry.best_accuracy:
            entry.best_accuracy = accuracy
            print(f'🎉 {arch_id} 新最佳准确率: {accuracy:.4f} (提升: {accuracy - old_accuracy:.4f})')
        entry.training_epochs += epochs
        entry.last_updated = datetime.now()
        return True

    def get_entry(self, arch_id: str) -> Optional[ModelEntry]:
        return self._registry.get(arch_id)

    def get_top_k(self, k: int, sort_by: str='best_accuracy') -> List[ModelEntry]:
        if sort_by not in ['best_accuracy', 'training_epochs', 'load']:
            raise ValueError(f'不支持的排序字段: {sort_by}')
        entries = list(self._registry.values())
        reverse = sort_by != 'load'
        entries.sort(key=lambda x: getattr(x, sort_by), reverse=reverse)
        return entries[:k]

    def get_all_entries(self) -> Dict[str, ModelEntry]:
        return self._registry.copy()

    def get_summary(self) -> Dict[str, Any]:
        if not self._registry:
            return {'total_models': 0, 'best_accuracy': 0.0, 'average_accuracy': 0.0, 'total_training_epochs': 0}
        accuracies = [entry.best_accuracy for entry in self._registry.values()]
        epochs = [entry.training_epochs for entry in self._registry.values()]
        loads = [entry.load for entry in self._registry.values()]
        return {'total_models': len(self._registry), 'best_accuracy': max(accuracies), 'worst_accuracy': min(accuracies), 'average_accuracy': sum(accuracies) / len(accuracies), 'total_training_epochs': sum(epochs), 'average_load': sum(loads) / len(loads), 'creation_time': self._creation_time.isoformat()}

    def export_to_dict(self) -> Dict[str, Any]:
        return {'creation_time': self._creation_time.isoformat(), 'models': {arch_id: entry.to_dict() for arch_id, entry in self._registry.items()}}

    def import_from_dict(self, data: Dict[str, Any]) -> None:
        self._registry.clear()
        if 'creation_time' in data:
            self._creation_time = datetime.fromisoformat(data['creation_time'])
        if 'models' in data:
            for arch_id, entry_data in data['models'].items():
                self._registry[arch_id] = ModelEntry.from_dict(entry_data)
        print(f'导入了 {len(self._registry)} 个模型条目')

    def save_to_file(self, filepath: str) -> bool:
        try:
            data = self.export_to_dict()
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            print(f'ModelRegistry已保存到: {filepath}')
            return True
        except Exception as e:
            print(f'保存失败: {e}')
            return False

    def load_from_file(self, filepath: str) -> bool:
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            self.import_from_dict(data)
            print(f'ModelRegistry已从文件加载: {filepath}')
            return True
        except Exception as e:
            print(f'加载失败: {e}')
            return False

    def clear(self) -> None:
        count = len(self._registry)
        self._registry.clear()
        print(f'已清空注册中心 ({count} 个条目)')

    def exists(self, arch_id: str) -> bool:
        return arch_id in self._registry

    def remove(self, arch_id: str) -> bool:
        if arch_id in self._registry:
            del self._registry[arch_id]
            print(f'已移除架构: {arch_id}')
            return True
        return False

    def get_statistics_by_load_range(self, min_load: float, max_load: float) -> Dict[str, Any]:
        filtered_entries = [entry for entry in self._registry.values() if min_load <= entry.load <= max_load]
        if not filtered_entries:
            return {'count': 0}
        accuracies = [entry.best_accuracy for entry in filtered_entries]
        return {'count': len(filtered_entries), 'best_accuracy': max(accuracies), 'average_accuracy': sum(accuracies) / len(accuracies), 'load_range': f'{min_load:.2f}-{max_load:.2f}'}