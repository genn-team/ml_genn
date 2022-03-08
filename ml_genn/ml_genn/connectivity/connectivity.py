import numpy as np

from typing import Sequence
from ..utils import InitValue, Value

from collections import namedtuple

Snippet = namedtuple("Snippet", ["snippet", "matrix_type", 
                                 "weight", "delay"])

class Connectivity:
    def __init__(self, weight: [InitValue], delay: [InitValue]):
        self.weight = Value(weight)
        self.delay = Value(delay)
        
        # If both weight and delay are arrays, check they are the same shape
        weight_array = isinstance(self.weight, (Sequence, np.ndarray))
        delay_array = isinstance(self.delay, (Sequence, np.ndarray))
        if (weight_array and delay_array 
                and np.shape(weight_array) != np.shape(delay_array)):
            raise RuntimeError("If weights and delays are specified as "
                               "arrays, they should be the same shape")
