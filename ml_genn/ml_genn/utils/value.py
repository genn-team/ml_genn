import numpy as np

from numbers import Number
from typing import Sequence, Union
from ..initializers import Initializer

from copy import deepcopy

from ..initializers import default_initializers

InitValue = Union[Number, Sequence[Number], np.ndarray, Initializer, str]

class ValueDescriptor:
    def __set_name__(self, owner, name):
        self.name = name

    def __get__(self, instance, owner):
        return getattr(instance, f"_{self.name}")

    def __set__(self, instance, value):
        # **NOTE** strings are checked first as strings ARE sequences
        name_internal = f"_{self.name}"
        if isinstance(value, str):
            if value in default_initializers:
                setattr(instance, name_internal, default_initializers[value])  
            else:
                raise RuntimeError(f"Initializer '{value}' unknown")
        elif isinstance(value, (Sequence, np.ndarray)):
            setattr(instance, name_internal, np.asarray(value)) 
        elif isinstance(value, (Number, Initializer)):
            setattr(instance, name_internal, deepcopy(value)) 
        elif value is None:
            setattr(instance, name_internal, None) 
        else:
            raise RuntimeError(f"{self.name} initializers should be "
                               f"specified as a string, a number, a sequence "
                               f"of numbers or an Initializer object")

def is_value_constant(value):
    return isinstance(value, Number)

def is_value_array(value):
    return isinstance(value, (Sequence, np.ndarray))

def is_value_initializer(value):
    return isinstance(value, Initializer)
