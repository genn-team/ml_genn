"""Serialisers are used to serialise and deserialiser model state
e.g. for saving and resuming from checkpoints. Serialisers provide
a filesystem like data model where data is stored at a 'path' specified by
a sequence of keys (which can be any object that is convertable to string)
"""
from .serialiser import Serialiser
from .numpy import Numpy

from ..utils.module import get_module_classes

default_serialisers = get_module_classes(globals(), Serialiser)

__all__ = ["Serialiser", "Numpy", "default_serialisers"]
