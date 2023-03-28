from .synapse import Synapse
from .delta import Delta
from .exponential import Exponential

from ..utils.module import get_module_classes

default_synapses = get_module_classes(globals(), Synapse)

__all__ = ["Synapse", "Delta", "Exponential", "default_synapses"]
