import numpy as np

from numbers import Number
from typing import Sequence, Union
from ..initializers import Initializer

from copy import deepcopy
from inspect import getmembers, isdatadescriptor

from ..initializers import default_initializers

Value = Union[Number, Sequence[Number], np.ndarray, Initializer]
InitValue = Union[Value, str]

class ValueDescriptor:
    def get_value(self, instance):
        return self.__get__(instance, None)

    def __set_name__(self, owner, name: str, *args):
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

    # Build set of names we care about
    names = set(n[0] for n in name_types)
    
    # Update vals with value descriptor values
    vals.update({n: d.get_value(inst) for n, d in descriptors
                 if isinstance(d, ValueDescriptor) and n in names})
    return vals

# **THINK** should maybe a method in a base class for Neuron/Synapse etc
def get_auto_values(inst, var_names):
    # Get descriptors
    descriptors = getmembers(type(inst), isdatadescriptor)

    # Build set of variable names
    var_names = set(var_names)
    
    # Build dictionaries of var and parameter values from value descriptors
    params = {n: d.get_value(inst) for n, d in descriptors
              if isinstance(d, ValueDescriptor) and n not in var_names}
    vars = {n: d.get_value(inst) for n, d in descriptors
            if isinstance(d, ValueDescriptor) and n in var_names}
    return params, vars

# **THINK** should maybe a method in a base class for Neuron/Synapse etc
def set_values(inst, vals):
    # Get descriptors
    descriptors = getmembers(type(inst), isdatadescriptor)

    # Loop through descriptors
    genn_descriptors = [d for n, d in descriptors
                        if isinstance(d, ValueDescriptor)]
    for n, d in descriptors:
        # If this is a value descriptor, build mapping from
        # GeNN names it is responsible for to descriptor
        if isinstance(d, ValueDescriptor):
            genn_descriptors.update({g[0]: d for g in d.genn_transforms})

    # Loop through values
    for n, v in vals.items():
        # If there is a descriptor matching
        # this name, use it to set variable
        if n in genn_descriptors:
            genn_descriptors[n].__set__(inst, v)
