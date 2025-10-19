from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime
import ray

@dataclass
class TrainingTask:
    arch_id: str
    arch_spec: Dict[str, Any]
    worker_id: int
    epochs: int
    load: float = 1.0
    priority: int = 0
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

@dataclass
class TrainingResult:
    arch_id: str
    accuracy: float
    training_time: float
    worker_id: int
    state_dict: Optional[Dict] = None
    metrics: Dict[str, float] = None
    error: Optional[str] = None

    def __post_init__(self):
        if self.metrics is None:
            self.metrics = {}

class SchedulerInterface(ABC):

    def __init__(self, num_workers: int):
        self.num_workers = num_workers
        self.is_initialized = False

    @abstractmethod
    def initialize(self, architectures: List[Dict[str, Any]]) -> None:
        pass

    @abstractmethod
    def schedule_next(self) -> List[TrainingTask]:
        pass

    @abstractmethod
    def update_results(self, results: List[TrainingResult]) -> None:
        pass

    @abstractmethod
    def has_more_tasks(self) -> bool:
        pass

    @abstractmethod
    def get_progress(self) -> Dict[str, Any]:
        pass

    @abstractmethod
    def reset(self) -> None:
        pass

    def get_scheduler_type(self) -> str:
        return self.__class__.__name__

    def is_ready(self) -> bool:
        return self.is_initialized

    def validate_architectures(self, architectures: List[Dict[str, Any]]) -> bool:
        if not architectures:
            raise ValueError('架构列表不能为空')
        for i, arch in enumerate(architectures):
            if not isinstance(arch, dict):
                raise ValueError(f'架构 {i} 必须是字典类型')
            if 'arch_id' not in arch:
                raise ValueError(f"架构 {i} 缺少 'arch_id' 字段")
            if 'arch_spec' not in arch:
                raise ValueError(f"架构 {i} 缺少 'arch_spec' 字段")
            if 'load' not in arch:
                arch['load'] = 1.0
        return True