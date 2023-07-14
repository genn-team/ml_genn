from abc import ABC

from abc import abstractmethod, abstractproperty


class Synapse(ABC):
    @abstractmethod
    def get_model(self, connection,
                  dt: float, batch_size: int):
        pass
