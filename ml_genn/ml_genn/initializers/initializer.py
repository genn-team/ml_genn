from abc import ABC

from abc import abstractmethod


class Initializer(ABC):
    @abstractmethod
    def get_snippet(self):
        pass
