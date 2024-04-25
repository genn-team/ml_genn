from __future__ import annotations

import numpy as np

from typing import TYPE_CHECKING
from .synapse import Synapse
from ..utils.model import SynapseModel
from ..utils.value import InitValue, ValueDescriptor

if TYPE_CHECKING:
    from .. import Connection

from ..utils.value import is_value_initializer

genn_model = {
    "params": [("ExpDecay", "scalar")],
    "sim_code":
        """
        injectCurrent(inSyn);
        inSyn *= ExpDecay;
        """}

class Exponential(Synapse):
    """Synapse model where inputs produce 
    exponentially decaying currents in target neuron.

    Args:
        tau:    Time constant of input current [ms]
    """
    tau = ValueDescriptor("ExpDecay", lambda val, dt: np.exp(-dt / val))

    def __init__(self, tau: InitValue = 5.0):
        super(Exponential, self).__init__()

        self.tau = tau

        if is_value_initializer(self.tau):
            raise NotImplementedError("Exponential synapse model does not "
                                      "currently support tau values specified"
                                      " using Initialiser objects")

    def get_model(self, connection: Connection,
                  dt: float, batch_size: int) -> SynapseModel:
        return SynapseModel.from_val_descriptors(genn_model, self, dt)
