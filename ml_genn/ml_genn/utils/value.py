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
    def __init__(self, *args):
        # Args are either strings specifying the names of GeNN variables
        # without transforms or tuples specifying names and transforms
        self.genn_transforms = [(a, None) if isinstance(a, str)
                                else a for a in args]

    def get_transformed(self, instance, dt):
        # Get value
        val = self.__get__(instance, None)

        # Apply transforms to get dictionary of GeNN variables
        return {g[0]: (val if g[1] is None else g[1](val, dt))
                for g in self.genn_transforms}

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
        if (any(g[1] is not None for g in self.genn_transforms)
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
def get_genn_var_name(inst, name):
    # Get attribute from instance type
    d = getattr(type(inst), name)

    # If attribute is a value descriptor
    if isinstance(d, ValueDescriptor):
        # Find all untransformed GeNN variables it is responsible for
        t = [g[0] for g in d.genn_transforms
             if g[1] is None]
        
        # If there aren't any, give error
        if len(t) == 0:
            raise RuntimeError(f"There are no GenN variables which "
                               f"directly map to varaible'{name}'")
        # Otherwise, return first
        else:
            return t[0]
    else:
        raise RuntimeError(f"'{name}' is not a ValueDescriptor")


# **THINK** should maybe a method in a base class for Neuron/Synapse etc
def get_values(inst, name_types, dt, vals={}):
    # Get descriptors
    descriptors = getmembers(type(inst), isdatadescriptor)

    # Build set of names we care about
    names = set(n[0] for n in name_types)

    # Loop through descriptors
    for n, d in descriptors:
        # If this is a value descriptor, update values with
        # transformed values for variables we care about
        if isinstance(d, ValueDescriptor):
            vals.update({g: v for g, v in d.get_transformed(inst, dt).items()
                         if g in names})

    return vals


# **THINK** should maybe a method in a base class for Neuron/Synapse etc
def set_values(inst, vals):
    # Get descriptors
    descriptors = getmembers(type(inst), isdatadescriptor)

    # Loop through descriptors
    genn_descriptors = {}
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
