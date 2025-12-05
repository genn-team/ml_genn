from abc import ABC
from typing import Any, Optional

from abc import abstractproperty


class Loss(ABC):
    """Base class for all loss functions"""
    def __init__(self, record_key: Optional[Any] = None):
        self._record_key = record_key
    
    @abstractproperty
    def ground_truth(self) -> str:
        """Gets ground truth class required by this loss function

        Returns:
            str: name of ground truth class
        """
        pass

    @property
    def record_key(self) -> Optional[Any]:
        """Key to record loss with"""
        return self._record_key
