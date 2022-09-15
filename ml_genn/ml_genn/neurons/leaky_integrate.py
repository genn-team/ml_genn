import numpy as np

from .neuron import Neuron
from ..utils.model import NeuronModel
from ..utils.value import InitValue, ValueDescriptor

genn_model = {
    "var_name_types": [("V", "scalar")],
    "param_name_types": [("Alpha", "scalar"), ("Bias", "scalar")],
    "sim_code":
        """
        $(V) = ($(Alpha) * $(V)) + $(Isyn) + $(Bias);
        """,
    "is_auto_refractory_required": False}


class LeakyIntegrate(Neuron):
    v = ValueDescriptor()
    bias = ValueDescriptor()
    tau_mem = ValueDescriptor()

    def __init__(self, v: InitValue = 0.0, bias: InitValue = 0.0,
                 tau_mem: InitValue = 20.0, 
                 softmax: bool = False, output=None):
        super(LeakyIntegrate, self).__init__(softmax, output)

        self.v = v
        self.bias = bias
        self.tau_mem = tau_mem

    def get_model(self, population, dt):
        return NeuronModel(genn_model, "V",
                           {"Alpha": np.exp(-dt / self.tau_mem), 
                            "Bias": self.bias},
                           {"V": self.v})
