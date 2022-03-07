import numpy as np
from .. import initializers

from numbers import Number
from typing import Sequence, Union
from ..initializers import Initializer

from .model import get_module_models

# Use Keras-style trick to get dictionary containing default neuron models
_initializers = get_module_models(initializers, Initializer)

InitValue = Union[Number, Sequence[Number], np.ndarray, Initializer]

def _get_value(value):
    # **NOTE** strings are checked first as strings ARE sequences
    if isinstance(value, str):
        if value in _initializers:
            return _initializers[value]
        else:
            raise RuntimeError(f"Initializer '{value}' unknown")
    elif isinstance(value, (Number, Initializer, Sequence, np.ndarray)):
        return value
    else:
        raise RuntimeError(f"Initializers should be specified either as a "
                            "string, a number, a sequence of numbers "
                            "or an Initializer object")
                            
class Value:
    def __init__(self, value: InitValue):
        self.value = _get_value(value)

    @property
    def is_constant(self):
        return isinstance(self.value, Number)
    
    @property
    def is_array(self):
        return isinstance(self.value, (Sequence, np.ndarray))
    
    @property
    def is_initializer(self):
        return isinstance(self.value, Initializer)
    
    def __repr__(self):
        return repr(self.value) 
