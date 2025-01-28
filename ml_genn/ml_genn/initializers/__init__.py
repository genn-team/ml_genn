"""Initializers are used to define model state variables and parameter
initialization functions which can be offloaded to the GPU"""
from .fluctuation_driven import (FluctuationDrivenExponential,
                                 FluctuationDrivenCentredNormal)
from .initializer import Initializer
from .normal import Normal
from .uniform import Uniform
from .wrapper import Wrapper

from ..utils.module import get_module_classes

default_initializers = get_module_classes(globals(), Initializer)

__all__ = ["FluctuationDrivenExponential", "FluctuationDrivenCentredNormal",
           "Initializer", "Normal", "Uniform", "Wrapper",
           "default_initializers"]
