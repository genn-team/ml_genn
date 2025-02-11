from __future__ import annotations

import numpy as np

from typing import Union, TYPE_CHECKING
from .synapse import Synapse
from ..utils.auto_model import AutoSynapseModel
from ..utils.model import SynapseModel
from ..utils.value import InitValue, ValueDescriptor

if TYPE_CHECKING:
    from .. import Connection

from ..utils.value import is_value_initializer

from ..utils.decorators import network_default_params

class Exponential(Synapse):
    """Synapse model where inputs produce 
    exponentially decaying currents in target neuron.

    Args:
        tau:    Time constant of input current [ms]
    """
    tau = ValueDescriptor()

    @network_default_params
    def __init__(self, tau: InitValue = 5.0, scale_i : bool = False):
        super(Exponential, self).__init__()

        self.tau = tau
        self.scale_i = scale_i

        if is_value_initializer(self.tau):
            raise NotImplementedError("Exponential synapse model does not "
                                      "currently support tau values specified"
                                      " using Initialiser objects")

    def get_model(self, connection: Connection, dt: float,
                  batch_size: int) -> Union[AutoSynapseModel, SynapseModel]:
        # Build basic model
        genn_model = {"vars": {"I": ("-I / tau", "I + weight")}}
        
        return AutoSynapseModel.from_val_descriptors(genn_model, self)
