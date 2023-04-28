from .avg_var import AvgVar
from .avg_var_exp_weight import AvgVarExpWeight
from .max_var import MaxVar
from .readout import Readout
from .spike_count import SpikeCount
from .sum_var import SumVar
from .var import Var

from ..utils.module import get_module_classes

default_readouts = get_module_classes(globals(), Readout)

__all__ = ["AvgVar", "AvgVarExpWeight", "MaxVar", "Readout", "SpikeCount",
           "SumVar", "Var", "default_readouts"]
