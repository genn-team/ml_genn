from abc import ABC

from abc import abstractproperty


class Loss(ABC):
    """Base class for all loss functions"""
    @abstractproperty
    def ground_truth(self) -> str:
        """Gets ground truth class required by this loss function

        Returns:
            str: name of ground truth class
        """
        pass
