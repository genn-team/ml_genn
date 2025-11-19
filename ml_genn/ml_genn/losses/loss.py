from abc import ABC

from abc import abstractproperty


class Loss(ABC):
    """Base class for all loss functions"""
    @abstractproperty
    def prediction(self) -> str:
        """Gets prediction class this loss function needs

        Returns:
            str: name of loss function
        """
        pass
