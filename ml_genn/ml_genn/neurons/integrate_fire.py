from typing import Optional
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
    v_thresh = ValueDescriptor("Vthresh")
    v_reset = ValueDescriptor("Vreset")
    v = ValueDescriptor("V")

    def __init__(self, v_thresh: InitValue = 1.0, v_reset: InitValue = 0.0,
                 v: InitValue = 0.0, softmax: Optional[bool] = None,
                 readout=None, **kwargs):
        super(IntegrateFire, self).__init__(softmax, readout, **kwargs)

        self.v_thresh = v_thresh
        self.v_reset = v_reset
        self.v = v

    def get_model(self, population, dt, batch_size):
        return NeuronModel.from_val_descriptors(genn_model, "V", self, dt)
