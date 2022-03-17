from .neuron import Neuron
from ..utils import InitValue, NeuronModel, Value

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
    def __init__(self, v_thresh=1.0, v_reset=0.0, v=0.0, output=None):
        super(IntegrateFire, self).__init__(output)

        self.v_thresh = Value(v_thresh)
        self.v_reset = Value(v_reset)
        self.v = Value(v)

    def get_model(self, population, dt):
        return self.add_output_logic(
            NeuronModel(genn_model, 
                        {"Vthresh": self.v_thresh, "Vreset": self.v_reset},
                        {"V": self.v}), "V")
