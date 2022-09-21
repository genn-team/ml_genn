from .neuron import Neuron
from ..utils.model import NeuronModel
from ..utils.value import InitValue, ValueDescriptor

genn_model = {
    "var_name_types": [("V", "scalar")],
    "param_name_types": [("Vthresh", "scalar"), ("Vreset", "scalar")],
    "sim_code":
        """
        $(V) += $(Isyn);
        """,
    "threshold_condition_code":
        """
        $(V) >= $(Vthresh)
        """,
    "reset_code":
        """
        $(V) = $(Vreset);
        """,
    "is_auto_refractory_required": False}


class IntegrateFire(Neuron):
    v_thresh = ValueDescriptor()
    v_reset = ValueDescriptor()
    v = ValueDescriptor()

    def __init__(self, v_thresh: InitValue = 1.0, v_reset: InitValue = 0.0,
                 v: InitValue = 0.0, softmax: bool = False, readout=None):
        super(IntegrateFire, self).__init__(softmax, readout)

        self.v_thresh = v_thresh
        self.v_reset = v_reset
        self.v = v

    def get_model(self, population, dt):
        return NeuronModel(genn_model, "V",
                           {"Vthresh": self.v_thresh, "Vreset": self.v_reset},
                           {"V": self.v})
