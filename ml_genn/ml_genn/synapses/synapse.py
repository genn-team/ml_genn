from abc import ABC

from abc import abstractmethod, abstractproperty


class Synapse(ABC):
    @abstractmethod
    def get_model(self, connection, dt):
        pass
