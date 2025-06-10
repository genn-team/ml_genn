""" Readouts are used to convert the internal state of an output neuron population
into a form that can be used for classification. Unlike ANNs where this this is typically
just the activation of the neurons, with SNNs it can be some function of their internal 
state over time or the spike times.
"""
from .avg_var import AvgVar
from .avg_var_exp_weight import AvgVarExpWeight
from .first_spike_time import FirstSpikeTime
from .max_var import MaxVar
from .readout import Readout
from .spike_count import SpikeCount
from .sum_var import SumVar
from .var import Var

from ..utils.module import get_module_classes

default_readouts = get_module_classes(globals(), Readout)

__all__ = ["AvgVar", "AvgVarExpWeight", "FirstSpikeTime", "MaxVar", "Readout",
           "SpikeCount", "SumVar", "Var", "default_readouts"]
