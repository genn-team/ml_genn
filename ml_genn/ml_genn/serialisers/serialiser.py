from abc import ABC

from abc import abstractmethod


class Serialiser(ABC):
    @abstractmethod
    def serialise(self, keys, data):
        pass

    @abstractmethod
    def deserialise(self, keys):
        pass

    @abstractmethod
    def deserialise_all(self, keys):
        pass