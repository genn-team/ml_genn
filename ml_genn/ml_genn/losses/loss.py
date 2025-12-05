from abc import ABC

from abc import abstractproperty


class Loss(ABC):
    """Base class for all loss functions"""
    def __init__(self, record: bool = False):
        self._record = record
    
    @abstractproperty
    def ground_truth(self) -> str:
        """Gets ground truth class required by this loss function

        Returns:
            str: name of ground truth class
        """
        pass

    @property
    def record(self):
        return self._record
