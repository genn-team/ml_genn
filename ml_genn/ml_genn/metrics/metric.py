from abc import ABC
from ..communicators import Communicator

from abc import abstractmethod, abstractproperty


class Metric(ABC):
    @abstractmethod
    def update(self, y_true, y_pred, communicator: Communicator):
        pass

    @abstractmethod
    def reset(self):
        pass

    @abstractproperty
    def result(self):
        pass
