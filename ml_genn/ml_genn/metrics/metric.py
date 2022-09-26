from abc import ABC

from abc import abstractmethod, abstractproperty


class Metric(ABC):
    @abstractmethod
    def update(self, y_true, y_pred):
        pass

    @abstractmethod
    def reset(self):
        pass

    @abstractproperty
    def result(self):
        pass
