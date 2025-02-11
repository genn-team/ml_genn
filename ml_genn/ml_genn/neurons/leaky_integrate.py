from __future__ import annotations

import numpy as np

from typing import Optional, Union, TYPE_CHECKING
from .neuron import Neuron
from ..utils.auto_model import AutoNeuronModel
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
        readout:    Type of readout to attach to this neuron's output variable
    """
    
    v = ValueDescriptor()
    bias = ValueDescriptor()
    tau_mem = ValueDescriptor()

    @network_default_params
    def __init__(self, v: InitValue = 0.0, bias: InitValue = 0.0,
                 tau_mem: InitValue = 20.0, readout=None):
        super(LeakyIntegrate, self).__init__(readout)

        self.v = v
        self.bias = bias
        self.tau_mem = tau_mem

    def get_model(self, population: Population, dt: float,
                  batch_size: int) -> Union[AutoNeuronModel, NeuronModel]:
        genn_model = {"vars": {"v": ("(-v + i) / tau_mem", None)}}

        # Return model
        return AutoNeuronModel.from_val_descriptors(genn_model, "v", self)
