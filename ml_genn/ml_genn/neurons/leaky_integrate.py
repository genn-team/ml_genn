from __future__ import annotations

import numpy as np

from typing import Optional, TYPE_CHECKING
from .neuron import Neuron
from ..utils.model import NeuronModel
from ..utils.value import InitValue, ValueDescriptor

if TYPE_CHECKING:
    from .. import Population

from ..utils.decorators import network_default_params

class LeakyIntegrate(Neuron):
    """A leaky-integrator, typically used as an output neuron

    Args:
        v:          Initial value of membrane voltage
        bias:       Initial value of bias curremt
        tau_mem:    Time constant of membrane voltage [ms]
        scale_i:    Should input current ``x`` be scaled by :math:`1-e^\\frac{-dt}{\\tau_\\text{mem}}`?
        readout:    Type of readout to attach to this neuron's output variable
    """
    
    v = ValueDescriptor("V")
    bias = ValueDescriptor("Bias")
    tau_mem = ValueDescriptor(("Alpha", lambda val, dt: np.exp(-dt / val)))

    @network_default_params
    def __init__(self, v: InitValue = 0.0, bias: InitValue = 0.0,
                 tau_mem: InitValue = 20.0, scale_i : bool = False,
                 readout=None):
        super(LeakyIntegrate, self).__init__(readout)

        self.v = v
        self.bias = bias
        self.tau_mem = tau_mem
        self.scale_i = scale_i

    def get_model(self, population: Population,
                  dt: float, batch_size: int) -> NeuronModel:
        genn_model = {
            "vars": [("V", "scalar")],
            "params": [("Alpha", "scalar"), ("Bias", "scalar")]}

        # Define integration code based on whether I should be scaled
        if self.scale_i:
            genn_model["sim_code"] = "V = (Alpha * V) + ((1.0 - Alpha) * Isyn) + Bias;"
        else:
            genn_model["sim_code"] = "V = (Alpha * V) + Isyn + Bias;"

        return NeuronModel.from_val_descriptors(genn_model, "V", self, dt)
