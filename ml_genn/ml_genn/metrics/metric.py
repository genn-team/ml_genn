import numpy as np

from abc import ABC
from typing import Optional
from ..communicators import Communicator

from abc import abstractmethod

class MetricState(ABC):
    @property
    @abstractmethod
    def result(self) -> Optional[np.ndarray]:
        """Quantity calculated by metric"""
        pass

class Metric(ABC):
    """Base class for all metrics"""
    @abstractmethod
    def update(self, state: MetricState, y_true: np.ndarray,
               y_pred: np.ndarray, communicator: Optional[Communicator]):
        """Update metric based on a batch of true and predicted values.

        Args:
            y_true:         'true' values provided to compiled network 
                            evaluate/train method
            y_pred:         predicted values provided by model readout
            communicator:   communicator to use to synchronise metrics
                            across GPUs when doing multi-GPU training.
        """
        pass

    @abstractmethod
    def create_state(self) -> MetricState:
        """Creates new metric state"""
        pass
