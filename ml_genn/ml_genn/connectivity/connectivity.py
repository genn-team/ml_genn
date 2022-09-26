import numpy as np

from abc import ABC
from ..utils.value import InitValue, ValueDescriptor

from abc import abstractmethod
from ..utils.value import is_value_array


class Connectivity(ABC):
    weight = ValueDescriptor()
    delay = ValueDescriptor()

    def __init__(self, weight: InitValue, delay: InitValue):
        self.weight = weight
        self.delay = delay

        # If both weight and delay are arrays, check they are the same shape
        weight_array = is_value_array(self.weight)
        delay_array = is_value_array(self.delay)
        if (weight_array and delay_array
                and np.shape(weight_array) != np.shape(delay_array)):
            raise RuntimeError("If weights and delays are specified as "
                               "arrays, they should be the same shape")

    @abstractmethod
    def connect(self, source, target):
        pass

    @abstractmethod
    def get_snippet(self, connection, prefer_in_memory: bool):
        pass