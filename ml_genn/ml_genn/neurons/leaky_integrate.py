import numpy as np

from typing import Optional
from .neuron import Neuron
from ..utils.model import NeuronModel
from ..utils.value import InitValue, ValueDescriptor

from ..utils.decorators import network_default_params

class LeakyIntegrate(Neuron):
    v = ValueDescriptor("V")
    bias = ValueDescriptor("Bias")
    tau_mem = ValueDescriptor("Alpha", lambda val, dt: np.exp(-dt / val))

    @network_default_params
    def __init__(self, v: InitValue = 0.0, bias: InitValue = 0.0,
                 tau_mem: InitValue = 20.0, scale_i : bool = False,
                 softmax: Optional[bool] = None, readout=None):
        super(LeakyIntegrate, self).__init__(softmax, readout)

        self.v = v
        self.bias = bias
        self.tau_mem = tau_mem
        self.scale_i = scale_i

    def get_model(self, population, dt):
        genn_model = {
            "var_name_types": [("V", "scalar")],
            "param_name_types": [("Alpha", "scalar"), ("Bias", "scalar")],
            "is_auto_refractory_required": False}
        
        # Define integration code based on whether I should be scaled
        if self.scale_i:
            genn_model["sim_code"] = "$(V) = ($(Alpha) * $(V)) + ((1.0 - $(Alpha)) * $(Isyn));"
        else:
            genn_model["sim_code"] = "$(V) = ($(Alpha) * $(V)) + $(Isyn);"

        return NeuronModel.from_val_descriptors(genn_model, "V", self, dt)
