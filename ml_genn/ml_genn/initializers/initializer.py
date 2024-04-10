from abc import ABC
from ..utils.snippet import InitializerSnippet

from abc import abstractmethod


class Initializer(ABC):
    """Base class for all initializers"""
    @abstractmethod
    def get_snippet(self) -> InitializerSnippet:
        """Gets PyGeNN implementation of initializer"""
        pass
