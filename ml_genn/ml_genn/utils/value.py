import numpy as np

from numbers import Number
from typing import Sequence, Union
from ..initializers import Initializer

from copy import deepcopy
from inspect import getmembers, isdatadescriptor

from ..initializers import default_initializers

InitValue = Union[Number, Sequence[Number], np.ndarray, Initializer, str]


class ValueDescriptor:
    def __init__(self, genn_name: str = None, value_transform = None):
        self.genn_name = genn_name
        self.value_transform = value_transform
    
    def get_transformed(self, instance, dt):
        # Get value
        val = self.__get__(instance, None)

        # Apply transform if specified and return
        return (val if self.value_transform is None 
                else self.value_transform(val, dt))

    def __set_name__(self, owner, name: str, genn_name: str = None):
        self.name = name

    def __get__(self, instance, owner):
        # If descriptor is accessed as a class attribute, return it
        if instance is None:
            return self
        # Otherwise, return attribute value
        else:
            return getattr(instance, f"_{self.name}")
    
    def __set__(self, instance, value):
        # **NOTE** strings are checked first as strings ARE sequences
        name_internal = f"_{self.name}"
        if (self.value_transform is not None 
                and isinstance(value, (str, Initializer))):
            raise NotImplementedError(f"{self.name} variable has a "
                                      f"transformation and cannot be "
                                      f"initialised using an initializer ")
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


# **THINK** should maybe a method in a base class for Neuron/Synapse etc
def get_values(inst, name_types, dt, vals={}):
    # Get descriptors
    descriptors = getmembers(type(inst), isdatadescriptor)

    # Build dictionary mapping GeNN names to var descriptors
    descriptors = {d.genn_name: d  for n, d in descriptors
                   if (isinstance(d, ValueDescriptor)
                       and d.genn_name is not None)}

    # Return map of GeNN names and transformed values provided by descriptor
    vals.update({v[0]: descriptors[v[0]].get_transformed(inst, dt)
                 for v in name_types
                 if v[0] in descriptors})
    return vals


# **THINK** should maybe a method in a base class for Neuron/Synapse etc
def set_values(inst, vals):
    # Get descriptors
    descriptors = getmembers(type(inst), isdatadescriptor)

    # Build dictionary mapping GeNN names to var descriptors
    descriptors = {d.genn_name: d  for n, d in descriptors
                   if (isinstance(d, ValueDescriptor)
                       and d.genn_name is not None)}
    
    # Loop through values
    for n, v in vals.items():
        # If there is a descriptor matching 
        # this name, use it to set variable
        if n in descriptors:
            descriptors[n].__set__(inst, v)