from __future__ import annotations

import numpy as np

from typing import TYPE_CHECKING
from .synapse import Synapse
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
    tau = ValueDescriptor(("ExpDecay", lambda val, dt: np.exp(-dt / val)),
                          ("IScale", lambda val, dt: (val / dt) * (1.0 - np.exp(-dt / val))))

    @network_default_params
    def __init__(self, tau: InitValue = 5.0, scale_i : bool = False):
        super(Exponential, self).__init__()

        self.tau = tau
        self.scale_i = scale_i

        if is_value_initializer(self.tau):
            raise NotImplementedError("Exponential synapse model does not "
                                      "currently support tau values specified"
                                      " using Initialiser objects")

    def get_model(self, connection: Connection,
                  dt: float, batch_size: int) -> SynapseModel:
        # Add exponential decay parameter
        genn_model = {"params": [("ExpDecay", "scalar")]}
        
        # If we should scale I, add additional parameter
        # and create sim code to inject scaled current
        if self.scale_i:
            genn_model["params"].append(("IScale", "scalar"))
            genn_model["sim_code"] = "injectCurrent(inSyn * IScale);"
        # Otherwise, create sim code to inject unscaled current
        else:
            genn_model["sim_code"] = "injectCurrent(inSyn);"
        
        # Add standard sim code
        genn_model["sim_code"] += """
        inSyn *= ExpDecay;
        """

        return SynapseModel.from_val_descriptors(genn_model, self, dt)
