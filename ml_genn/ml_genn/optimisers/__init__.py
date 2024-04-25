""" Optimisers are used for applying gradient updates provided 
by learning rules to model parameters. They are implemented as GeNN 
custom updates which access both the variable and the gradients via variable references.
"""
from .adam import Adam
from .optimiser import Optimiser

from ..utils.module import get_module_classes

default_optimisers = get_module_classes(globals(), Optimiser)

__all__ = ["Adam", "Optimiser", "default_optimisers"]
