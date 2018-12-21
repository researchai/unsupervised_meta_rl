import abc
from typing import Dict, Any


class Snapshotable(abc.ABC):
    @staticmethod
    @abc.abstractmethod
    def create(params: Dict[str, Any], state: Any = None):
        pass

    @abc.abstractmethod
    def restore(self, state: Any):
        pass

    @property
    @abc.abstractmethod
    def params(self):
        pass

    @property
    @abc.abstractmethod
    def snapshot(self) -> Any:
        pass
