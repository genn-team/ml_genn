from .adam import Adam
from .optimiser import Optimiser

from ..utils.module import get_module_classes

default_optimisers = get_module_classes(globals(), Optimiser)

__all__ = ["Adam", "Optimiser", "default_optimisers"]
