from numbers import Number
from typing import Sequence, Union

import numpy as np

from ..initializers import Initializer

InitValue = Union[Number, Sequence[Number], Initializer]

class Value(object):
    def __init__(self, value: InitValue):
        self.value = value
        
        if not(isinstance(self.value, (Number, Initializer, 
                                       Sequence, np.ndarray))):
            raise RuntimeError("Value must be initialised to either a number,"
                               "a sequence or with an Initializer object")

    @property
    def is_constant(self):
        return isinstance(self.value, Number)

    def __repr__(self):
        return repr(self.value) 