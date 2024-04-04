import numpy as np

from typing import Dict
from abc import ABC

from abc import abstractmethod


class Serialiser(ABC):
    """Base class for all serialisers"""

    @abstractmethod
    def serialise(self, keys, data: np.ndarray):
        """Serialise a single array to a 'path'
        
        Args:
            keys:   sequence of keys describing path 
                    (keys can be any object that is convertable to string).
            data:   numpy array of data
        """
        pass

    @abstractmethod
    def deserialise(self, keys) -> np.ndarray:
        """Deserialise a single array from a 'path'
        
        Args:
            keys:   sequence of keys describing path 
                    (keys can be any object that is convertable to string).
        """
        pass

    @abstractmethod
    def deserialise_all(self, keys) -> Dict:
        """Recursively deserialises all arrays beneath a 'path'
        
        Args:
            keys:   sequence of keys describing path to recurse from
                    (keys can be any object that is convertable to string).
        """
        pass